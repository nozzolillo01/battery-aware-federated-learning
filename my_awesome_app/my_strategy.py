"""Battery-aware federated learning strategy.

Selection probability: P(client_i) = (battery_level_i^2) / Σ(battery_level_j^2)
This quadratic weighting favors clients with higher battery levels.
"""

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import numpy as np
import wandb
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple, Optional

from .battery_simulator import FleetManager

logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("wandb").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)


class BatteryAwareFedAvg(FedAvg):

    def __init__(self, min_battery_threshold: float = 0.2, *args, **kwargs) -> None:
        # Receive optional config from server
        self.total_rounds_config = kwargs.pop("total_rounds", None)
        self.local_epochs_config = kwargs.pop("local_epochs", None)
        self.num_supernodes = kwargs.pop("num_supernodes", None)

        super().__init__(*args, **kwargs)
        self.min_battery_threshold = min_battery_threshold
        self.fleet_manager = FleetManager()
        self.results_to_save = {}
        self.last_selected_count = 0
        self.last_selected_battery_avg = None
        self.last_selected_battery_min = None

        tz = ZoneInfo("Europe/Rome")
        name = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="battery-aware-fl", name=f"run-{name}")

        # Print run header (num-supernodes, num-server-rounds, local-epochs)
        if self.num_supernodes is not None:
            print(f"num-supernodes = {self.num_supernodes}")
        if self.total_rounds_config is not None:
            print(f"num-server-rounds = {self.total_rounds_config}")
        if self.local_epochs_config is not None:
            print(f"local-epochs = {self.local_epochs_config}")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, dict]]:
        original_config = super().configure_fit(server_round, parameters, client_manager)
        if not original_config:
            return []

        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}

        client_ids = [c.cid for c in available_clients]
        eligible_clients_ids = self.fleet_manager.get_eligible_clients(client_ids, self.min_battery_threshold)
        eligible_clients = [c for c in available_clients if c.cid in eligible_clients_ids]

        if not eligible_clients:
            battery_levels = [(c, self.fleet_manager.get_battery_level(c.cid)) for c in available_clients]
            battery_levels.sort(key=lambda x: x[1], reverse=True)
            eligible_clients = [c for c, _ in battery_levels[:2]]

        if len(eligible_clients) <= 2:
            selected_clients = eligible_clients
        else:
            weights_dict = self.fleet_manager.calculate_selection_weights([c.cid for c in eligible_clients])
            weights = np.array([weights_dict[c.cid] for c in eligible_clients])
            if weights.sum() > 0:
                probabilities = weights / weights.sum()
                num_to_select = max(1, min(len(eligible_clients), len(eligible_clients) // 2))
                if num_to_select < 2 and len(eligible_clients) >= 2:
                    num_to_select = 2
                selected_indices = np.random.choice(len(eligible_clients), size=num_to_select, replace=False, p=probabilities)
                selected_clients = [eligible_clients[i] for i in selected_indices]
            else:
                selected_clients = eligible_clients[:max(1, min(2, len(eligible_clients)))]

        selected_client_ids = [c.cid for c in selected_clients]
        available_client_ids = [c.cid for c in available_clients]

        # Capture selected clients' battery stats BEFORE applying consumption/recharge
        try:
            if selected_client_ids:
                levels = [self.fleet_manager.get_battery_level(cid) for cid in selected_client_ids]
                self.last_selected_battery_avg = float(np.mean(levels)) if levels else None
                self.last_selected_battery_min = float(np.min(levels)) if levels else None
            else:
                self.last_selected_battery_avg = None
                self.last_selected_battery_min = None
        except Exception:
            self.last_selected_battery_avg = None
            self.last_selected_battery_min = None

        self.last_selected_count = len(selected_clients)
        wandb.log({
            "round": server_round,
            "selected_clients": self.last_selected_count,
        }, step=server_round)

        self.fleet_manager.update_round(selected_client_ids, available_client_ids)
        return [(client, config) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:

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

        self.results_to_save[server_round] = analysis_results
        with open("results.json", "w") as f:
            json.dump(self.results_to_save, f, indent=2)

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
