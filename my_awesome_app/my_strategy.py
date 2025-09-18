"""
Battery-aware federated learning strategy.

Core idea
---------
- Client selection is energy-aware: clients with higher battery are favored.
- Selection probability is quadratic in the battery level to amplify differences.

Formula
-------
P(client_i) = (battery_level_i^2) / Σ(battery_level_j^2)

"""

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import numpy as np
import wandb
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple, Optional

from .battery_simulator import FleetManager


class BatteryAwareFedAvg(FedAvg):
    """Energy-aware variant of FedAvg with quadratic battery-based selection."""

    def __init__(self, *args, **kwargs) -> None:
        # Receive optional config from server
        self.total_rounds_config = kwargs.pop("total_rounds", None)
        self.local_epochs_config = kwargs.pop("local_epochs", None)
        self.num_supernodes = kwargs.pop("num_supernodes", None)
        min_battery_threshold = float(kwargs.pop("min_battery_threshold", 0.2))

        # Init parent with the remaining kwargs
        super().__init__(*args, **kwargs)

        # Store threshold for later use
        self.min_battery_threshold = min_battery_threshold

        self.fleet_manager = FleetManager()
        self.results_to_save = {}
        self.last_selected_count = 0
        self.last_selected_battery_avg = None
        self.last_selected_battery_min = None

        # Tracking stato per visualizzazione dettagliata su wandb
        self._prev_battery_levels: Dict[str, float] = {}
        # Nessun buffer cumulativo: useremo una tabella separata per ogni round
        self._client_id_order = []  # ordine stabile dei client visti
        self._rounds_since_selected: Dict[str, int] = {}  # contatore round da ultima selezione

        self._init_wandb_run()
        self._print_run_header()

    def _init_wandb_run(self) -> None:
        """Initialize Weights & Biases run with timestamped name."""
        tz = ZoneInfo("Europe/Rome")
        name = datetime.now(tz).strftime("%Y-%m-%d_%H:%M")
        wandb.init(project="FL", name=f"run-{name}")

    def _print_run_header(self) -> None:
        """Print minimal run configuration header to the terminal."""
        if self.num_supernodes is not None:
            print(f"num-supernodes = {self.num_supernodes}")
        if self.total_rounds_config is not None:
            print(f"num-server-rounds = {self.total_rounds_config}")
        if self.local_epochs_config is not None:
            print(f"local-epochs = {self.local_epochs_config}")
        print(f"min-battery-threshold = {self.min_battery_threshold}")



    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, dict]]:
        """Select clients in an energy-aware manner and configure their fit.

        Steps:
        1. Get original client configuration from parent.
        2. Filter clients based on battery threshold.
        3. If none eligible, select top-2 by battery level.
        4. If more than 2 eligible, perform probabilistic selection with quadratic weighting.
        5. Update fleet battery levels post-selection.

        """
        original_config = super().configure_fit(server_round, parameters, client_manager)
        if not original_config:
            return []

        # Available clients and shared config
        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}

        # Eligibility filtering
        eligible_ids = self.fleet_manager.get_eligible_clients([c.cid for c in available_clients], self.min_battery_threshold)
        eligible_clients = [c for c in available_clients if c.cid in eligible_ids]

        # Fallback: pick top-2 by battery if none eligible
        if not eligible_clients:
            levels = [(c, self.fleet_manager.get_battery_level(c.cid)) for c in available_clients]
            levels.sort(key=lambda x: x[1], reverse=True)
            eligible_clients = [c for c, _ in levels[:2]]

        # Probabilistic selection (quadratic weighting) + calcolo probabilità per logging
        prob_map: Dict[str, float] = {}
        if len(eligible_clients) <= 2:
            selected_clients = eligible_clients
            # Assegna probabilità 1 ai selezionati (deterministico) e 0 agli altri disponibili
            for c in available_clients:
                prob_map[c.cid] = 1.0 if c in selected_clients else 0.0
        else:
            weights_map = self.fleet_manager.calculate_selection_weights([c.cid for c in eligible_clients])
            weights = np.array([weights_map[c.cid] for c in eligible_clients])
            if weights.sum() > 0:
                probs = weights / weights.sum()
                num_to_select = max(1, min(len(eligible_clients), len(eligible_clients) // 2))
                if num_to_select < 2 and len(eligible_clients) >= 2:
                    num_to_select = 2
                idx = np.random.choice(len(eligible_clients), size=num_to_select, replace=False, p=probs)
                selected_clients = [eligible_clients[i] for i in idx]
                # Mappa probabilità per tutti gli eleggibili
                for i, c in enumerate(eligible_clients):
                    prob_map[c.cid] = float(probs[i])
                # Non eleggibili -> 0
                for c in available_clients:
                    if c.cid not in prob_map:
                        prob_map[c.cid] = 0.0
            else:
                selected_clients = eligible_clients[:max(1, min(2, len(eligible_clients)))]
                for c in available_clients:
                    prob_map[c.cid] = 1.0 if c in selected_clients else 0.0

        selected_client_ids = [c.cid for c in selected_clients]
        available_client_ids = [c.cid for c in available_clients]

        # Capture stats 
        if selected_client_ids:
            levels = [self.fleet_manager.get_battery_level(cid) for cid in selected_client_ids]
            if levels:
                self.last_selected_battery_avg = sum(levels) / len(levels)
                self.last_selected_battery_min = min(levels)
            else:
                self.last_selected_battery_avg = None
                self.last_selected_battery_min = None
        else:
            self.last_selected_battery_avg = None
            self.last_selected_battery_min = None

        # Logging/update count
        self.last_selected_count = len(selected_clients)

        # Costruzione tabella dettagliata per ogni client disponibile
        # Aggiorna ordine stabile (aggiunge nuovi client mantenendo sort iniziale)
        newly_seen = [c.cid for c in available_clients if c.cid not in self._client_id_order]
        if newly_seen:
            self._client_id_order.extend(newly_seen)
            # Ordine alfabetico/numerico stabile
            self._client_id_order = sorted(self._client_id_order)

        # Prepara dizionario dati presenti in questo round
        present_data: Dict[str, Dict[str, Any]] = {}
        for c in available_clients:
            cid = c.cid
            battery = self.fleet_manager.get_battery_level(cid)
            prev = self._prev_battery_levels.get(cid)
            delta = None if prev is None else battery - prev
            prob = prob_map.get(cid, 0.0)
            present_data[cid] = {
                "battery": round(battery, 4),
                "delta_battery": (round(delta, 4) if delta is not None else np.nan),
                "prob_selection": round(prob, 6),
                "selected": int(cid in selected_client_ids),
                "eligible": int(cid in eligible_ids),
            }

        # Costruisce tabella per il round corrente con ordine fisso
        # Aggiorna contatori rounds_since_selected per client disponibili
        for cid in self._client_id_order:
            if cid in selected_client_ids:
                self._rounds_since_selected[cid] = 0
            else:
                # Incrementa solo se il client è comparso almeno una volta
                if cid in self._rounds_since_selected:
                    self._rounds_since_selected[cid] += 1
                else:
                    # Primo round visto e non selezionato -> 1
                    self._rounds_since_selected[cid] = 1 if cid not in selected_client_ids else 0

        columns = [
            "round",
            "client_id",
            "battery",
            "delta_battery",
            "prob_selection",
            "selected",
            "eligible",
            "rounds_since_selected",
        ]
        round_table = wandb.Table(columns=columns)
        for cid in self._client_id_order:
            if cid in present_data:
                row = [
                    server_round,
                    cid,
                    present_data[cid]["battery"],
                    present_data[cid]["delta_battery"],
                    present_data[cid]["prob_selection"],
                    present_data[cid]["selected"],
                    present_data[cid]["eligible"],
                    self._rounds_since_selected.get(cid, np.nan),
                ]
            else:
                # Client non disponibile in questo round: battery e delta NaN, prob 0, selected 0, eligible 0
                row = [
                    server_round,
                    cid,
                    np.nan,
                    np.nan,
                    0.0,
                    0,
                    0,
                    self._rounds_since_selected.get(cid, np.nan),
                ]
            round_table.add_data(*row)

        # Log tabella per-round con chiave distinta
        wandb.log({
            "round": server_round,
            "selected_clients": self.last_selected_count,
            f"client_status_round_{server_round}": round_table
        }, step=server_round)
        # Aggiorna prev dopo il logging (necessario per calcolare delta corretto al round successivo)
        for c in available_clients:
            self._prev_battery_levels[c.cid] = self.fleet_manager.get_battery_level(c.cid)

        # Fleet update
        self.fleet_manager.update_round(selected_client_ids, available_client_ids)
        return [(client, config) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Aggregate as FedAvg and augment metrics with fleet battery statistics."""

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        battery_metrics = {
            "fleet_avg_battery": fleet_stats.get("avg_battery", 0),
            "fleet_min_battery": fleet_stats.get("min_battery", 0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
        }

        if metrics_aggregated is None:
            metrics_aggregated = {}
        metrics_aggregated.update(battery_metrics)

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Evaluate as FedAvg and enrich metrics and logs with battery stats."""

        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result

        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        battery_metrics = {
            "battery_avg": fleet_stats.get("avg_battery", 0),
            "battery_min": fleet_stats.get("min_battery", 0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
            "eligible_clients": fleet_stats.get("eligible_clients", 0),
        }

        if metrics is None:
            metrics = {}
        metrics.update(battery_metrics)

        analysis_results = {
            "round": server_round,
            "loss": round(loss, 4),
            "accuracy": round(metrics.get("cen_accuracy", 0), 4),
            "battery_avg": round(battery_metrics["battery_avg"], 3),
            "battery_min": round(battery_metrics["battery_min"], 3),
            "fairness_jain": round(battery_metrics["fairness_jain"], 3),
            "total_energy": round(battery_metrics["total_energy_consumed"], 3),
            "eligible_clients": battery_metrics["eligible_clients"],
            "selected_clients": self.last_selected_count,
            "selected_battery_avg": round(self.last_selected_battery_avg, 3) if self.last_selected_battery_avg is not None else None,
            "selected_battery_min": round(self.last_selected_battery_min, 3) if self.last_selected_battery_min is not None else None,
        }

        # Persist results as before
        self.results_to_save[server_round] = analysis_results
        with open("results.json", "w") as f:
            json.dump(self.results_to_save, f, indent=2)

        # Log to WandB keeping the same keys and rounding
        wandb_payload = {
            "loss": round(loss, 4),
            "accuracy": round(metrics.get("cen_accuracy", 0), 4),
            "battery_avg": round(battery_metrics["battery_avg"], 3),
            "battery_min": round(battery_metrics["battery_min"], 3),
            "fairness_jain": round(battery_metrics["fairness_jain"], 3),
            "total_energy": round(battery_metrics["total_energy_consumed"], 3),
            "eligible_clients": battery_metrics["eligible_clients"],
            "selected_clients": self.last_selected_count,
        }
        if self.last_selected_battery_avg is not None:
            wandb_payload["selected_battery_avg"] = round(self.last_selected_battery_avg, 3)
        if self.last_selected_battery_min is not None:
            wandb_payload["selected_battery_min"] = round(self.last_selected_battery_min, 3)
        wandb.log(wandb_payload, step=server_round)

        # Minimal terminal output: evaluation summary
        try:
            acc = metrics.get("cen_accuracy", 0)
            suffix = f"/{self.num_supernodes}" if self.num_supernodes is not None else ""
            print(
                f"[Round {server_round}] loss={loss:.4f} acc={acc:.4f} "
                f"eligible_clients={battery_metrics['eligible_clients']} "
                f"selected_clients={self.last_selected_count}{suffix}"
            )
        except Exception:
            pass

        return loss, metrics
