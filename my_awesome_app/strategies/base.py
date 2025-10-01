"""Shared base classes for battery-aware Flower strategies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from zoneinfo import ZoneInfo

from ..battery_simulator import FleetManager
from ..selection import ClientSelectionStrategy


class FleetAwareFedAvg(FedAvg):
    """FedAvg variant that tracks battery metrics and delegates client selection."""

    def __init__(
        self,
        *args,
        selection_strategy: ClientSelectionStrategy,
        strategy_name: str,
        min_battery_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.total_rounds_config = kwargs.pop("total_rounds", None)
        self.local_epochs_config = kwargs.pop("local_epochs", None)
        self.num_supernodes = kwargs.pop("num_supernodes", None)

        self.selection_strategy = selection_strategy
        self.strategy = strategy_name
        self.min_battery_threshold = float(min_battery_threshold)

        super().__init__(*args, **kwargs)

        self.fleet_manager = FleetManager()
        self.results_to_save: Dict[str, Any] = {}
        self.last_selected_count = 0
        self.last_selected_battery_avg: Optional[float] = None
        self.last_selected_battery_min: Optional[float] = None
        self.last_deaths_count = 0
        self.current_eligible_count = 0

        self._prev_battery_levels: Dict[str, float] = {}
        self._client_id_order: List[str] = []
        self._rounds_since_selected: Dict[str, int] = {}

        self._init_wandb_run()
        self._print_run_header()
        self._acc_series = {"train": [], "val": [], "test": []}
        self._loss_series = {"train": [], "val": [], "test": []}

    # ------------------------------------------------------------------
    # WandB helpers
    # ------------------------------------------------------------------
    def _wandb_run_prefix(self) -> str:
        return self.strategy.upper()

    def _extra_wandb_config(self) -> Dict[str, Any]:
        return {}

    def _wandb_config(self) -> Dict[str, Any]:
        config = {
            "total_rounds": self.total_rounds_config,
            "local_epochs": self.local_epochs_config,
            "num_supernodes": self.num_supernodes,
            "strategy": self.strategy,
        }
        config.update(self._extra_wandb_config())
        return {k: v for k, v in config.items() if v is not None}

    def _init_wandb_run(self) -> None:
        tz = ZoneInfo("Europe/Rome")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
        try:
            wandb.init(
                project="nuovo",
                name=f"{self._wandb_run_prefix()}-run-{timestamp}",
                config=self._wandb_config(),
            )
            wandb.define_metric("round")
            wandb.define_metric("*", step="round")
        except Exception:
            pass

    def _extra_header_items(self) -> List[Tuple[str, Any]]:
        return []

    def _print_run_header(self) -> None:
        config_items = [
            ("num-supernodes", self.num_supernodes),
            ("num-server-rounds", self.total_rounds_config),
            ("local-epochs", self.local_epochs_config),
            ("strategy", self.strategy),
        ] + self._extra_header_items()

        for name, value in config_items:
            if value is not None:
                print(f"{name} = {value}")

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _extract_available_clients(
        self, original_config: List[Tuple[ClientProxy, Dict[str, Any]]]
    ) -> Tuple[List[ClientProxy], Dict[str, Any]]:
        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}
        return available_clients, config

    def _eligible_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        eligible_ids = self.fleet_manager.get_eligible_clients(
            [c.cid for c in available_clients],
            self.min_battery_threshold,
        )
        return [c for c in available_clients if c.cid in eligible_ids]

    def _ensure_probability_map(
        self,
        probability_map: Optional[Dict[str, float]],
        available_clients: List[ClientProxy],
    ) -> Dict[str, float]:
        if probability_map is None:
            return {c.cid: 0.0 for c in available_clients}
        return {c.cid: probability_map.get(c.cid, 0.0) for c in available_clients}

    def _capture_selected_stats(self, selected_client_ids: List[str]) -> None:
        self.last_selected_count = len(selected_client_ids)

        if selected_client_ids:
            levels = [self.fleet_manager.get_battery_level(cid) for cid in selected_client_ids]
            if levels:
                self.last_selected_battery_avg = sum(levels) / len(levels)
                self.last_selected_battery_min = min(levels)
                return
        self.last_selected_battery_avg = None
        self.last_selected_battery_min = None

    def _log_selection_to_wandb(
        self,
        server_round: int,
        available_clients: List[ClientProxy],
        selected_client_ids: List[str],
        eligible_ids: List[str],
        prob_map: Dict[str, float],
        deaths_ids: List[str],
    ) -> None:
        newly_seen = [c.cid for c in available_clients if c.cid not in self._client_id_order]
        if newly_seen:
            self._client_id_order.extend(newly_seen)
            self._client_id_order = sorted(self._client_id_order)

        present_data: Dict[str, Dict[str, Any]] = {}
        for c in available_clients:
            cid = c.cid
            battery = self.fleet_manager.get_battery_level(cid)
            prev = self._prev_battery_levels.get(cid)
            consumed = self.fleet_manager.client_consumed_battery.get(cid, 0)
            recharged = self.fleet_manager.client_recharged_battery.get(cid, 0)
            prob = prob_map.get(cid, 0.0)
            present_data[cid] = {
                "current_battery_level": round(battery, 4),
                "previous_battery_level": (round(prev, 4) if prev is not None else np.nan),
                "consumed_battery": round(consumed, 4),
                "recharged_battery": round(recharged, 4),
                "prob_selection": round(prob, 6),
                "selected": int(cid in selected_client_ids),
                "eligible": int(cid in eligible_ids),
            }

        for cid in self._client_id_order:
            if cid in selected_client_ids:
                self._rounds_since_selected[cid] = 0
            else:
                if cid in self._rounds_since_selected:
                    self._rounds_since_selected[cid] += 1
                else:
                    self._rounds_since_selected[cid] = 1 if cid not in selected_client_ids else 0

        columns = [
            "round",
            "client_id",
            "device_class",
            "current_battery_level",
            "previous_battery_level",
            "consumed_battery",
            "recharged_battery",
            "prob_selection",
            "selected",
            "eligible",
            "is_dead_during_the_round",
            "rounds_since_selected",
        ]
        round_table = wandb.Table(columns=columns)

        for cid in self._client_id_order:
            if cid in present_data:
                is_dead = int(cid in deaths_ids)
                row = [
                    server_round,
                    cid,
                    self.fleet_manager.get_device_class(cid),
                    present_data[cid]["current_battery_level"],
                    present_data[cid]["previous_battery_level"],
                    present_data[cid]["consumed_battery"],
                    present_data[cid]["recharged_battery"],
                    present_data[cid]["prob_selection"],
                    present_data[cid]["selected"],
                    present_data[cid]["eligible"],
                    is_dead,
                    self._rounds_since_selected.get(cid, np.nan),
                ]
            else:
                row = [
                    server_round,
                    cid,
                    self.fleet_manager.get_device_class(cid),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0.0,
                    0,
                    0,
                    0,
                    self._rounds_since_selected.get(cid, np.nan),
                ]
            round_table.add_data(*row)

        try:
            wandb.log(
                {
                    "round": server_round,
                    "selected_clients": self.last_selected_count,
                    "deaths_clients": self.last_deaths_count,
                    f"client_status_round_{server_round}": round_table,
                },
                step=server_round,
            )
        except Exception:
            pass

        for c in available_clients:
            self._prev_battery_levels[c.cid] = self.fleet_manager.get_battery_level(c.cid)

    # ------------------------------------------------------------------
    # Flower overrides
    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any,
    ) -> List[Tuple[ClientProxy, Dict[str, Any]]]:
        original_config = super().configure_fit(server_round, parameters, client_manager)
        if not original_config:
            return []

        available_clients, config = self._extract_available_clients(original_config)
        eligible_clients = self._eligible_clients(available_clients)
        eligible_ids = [c.cid for c in eligible_clients]
        self.current_eligible_count = len(eligible_ids)

        selected_clients, prob_map = self.selection_strategy.select_clients(
            eligible_clients=eligible_clients,
            available_clients=available_clients,
            fleet_manager=self.fleet_manager,
            num_clients=None,
        )
        prob_map = self._ensure_probability_map(prob_map, available_clients)

        selected_client_ids = [c.cid for c in selected_clients]
        available_client_ids = [c.cid for c in available_clients]

        epochs = int(self.local_epochs_config) if self.local_epochs_config is not None else 1
        deaths = self.fleet_manager.get_dead_clients(selected_client_ids, epochs)
        self.last_deaths_count = len(deaths)

        self._capture_selected_stats(selected_client_ids)
        self.fleet_manager.update_round(selected_client_ids, available_client_ids, epochs)

        self._log_selection_to_wandb(
            server_round,
            available_clients,
            selected_client_ids,
            eligible_ids,
            prob_map,
            list(deaths),
        )

        selected_clients = [c for c in selected_clients if c.cid not in deaths]
        self._selected_for_fit_last_round = [c.cid for c in selected_clients]

        return [(client, config) for client in selected_clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any,
    ) -> List[Tuple[ClientProxy, Dict[str, Any]]]:
        original = super().configure_evaluate(server_round, parameters, client_manager)
        if not original:
            return []

        selected_ids = getattr(self, "_selected_for_fit_last_round", None)
        if not selected_ids:
            return original

        selected_set = set(selected_ids)
        filtered = [(c, cfg) for (c, cfg) in original if c.cid in selected_set]
        return filtered if filtered else original

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _extract_battery_metrics(self) -> Dict[str, float]:
        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        return {
            "fleet_avg_battery": fleet_stats.get("avg_battery", 0.0),
            "fleet_min_battery": fleet_stats.get("min_battery", 0.0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        battery_metrics = self._extract_battery_metrics()
        if metrics_aggregated is None:
            metrics_aggregated = {}
        metrics_aggregated.update(battery_metrics)

        try:
            self.last_train_accuracy = metrics_aggregated.get("train_accuracy")
            self.last_train_loss = metrics_aggregated.get("train_loss")
        except Exception:
            pass

        return parameters_aggregated, metrics_aggregated

    def _prepare_evaluation_metrics(
        self,
        server_round: int,
        loss: float,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        battery_metrics = {
            "battery_avg": fleet_stats.get("avg_battery", 0.0),
            "battery_min": fleet_stats.get("min_battery", 0.0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
            "eligible_clients": getattr(
                self,
                "current_eligible_count",
                fleet_stats.get("eligible_clients", 0),
            ),
        }

        if metrics is None:
            metrics = {}
        metrics.update(battery_metrics)

        is_server_eval = "test_accuracy_server" in metrics

        results_dict: Dict[str, Any] = {
            "round": server_round,
            "battery_avg": round(battery_metrics["battery_avg"], 3),
            "battery_min": round(battery_metrics["battery_min"], 3),
            "fairness_jain": round(battery_metrics["fairness_jain"], 3),
            "total_energy": round(battery_metrics["total_energy_consumed"], 3),
            "eligible_clients": battery_metrics["eligible_clients"],
            "selected_clients": self.last_selected_count,
            "selected_battery_avg": round(self.last_selected_battery_avg, 3)
            if self.last_selected_battery_avg is not None
            else None,
            "selected_battery_min": round(self.last_selected_battery_min, 3)
            if self.last_selected_battery_min is not None
            else None,
            "deaths_clients": self.last_deaths_count,
        }

        if is_server_eval:
            results_dict["test_accuracy_server"] = round(float(metrics.get("test_accuracy_server", 0.0)), 4)
            results_dict["test_loss_server"] = round(loss, 4)
        else:
            results_dict["val_accuracy_client"] = round(float(metrics.get("accuracy", 0.0)), 4)
            results_dict["val_loss_client"] = round(loss, 4)

        if getattr(self, "last_train_accuracy", None) is not None:
            results_dict["train_accuracy_client"] = float(self.last_train_accuracy)
        if getattr(self, "last_train_loss", None) is not None:
            results_dict["train_loss_client"] = float(self.last_train_loss)

        return results_dict

    def _log_results_to_wandb(self, server_round: int, analysis_results: Dict[str, Any]) -> None:
        wandb_payload = {k: v for k, v in analysis_results.items() if k not in ["round"] and v is not None}
        try:
            wandb.log(wandb_payload, step=server_round)
        except Exception:
            pass

        try:
            def _append(target: List[Optional[float]], value: Optional[Any]) -> None:
                target.append(float(value) if value is not None else None)

            _append(self._acc_series["train"], analysis_results.get("train_accuracy_client"))
            _append(self._acc_series["val"], analysis_results.get("val_accuracy_client"))
            _append(self._acc_series["test"], analysis_results.get("test_accuracy_server"))

            _append(self._loss_series["train"], analysis_results.get("train_loss_client"))
            _append(self._loss_series["val"], analysis_results.get("val_loss_client"))
            _append(self._loss_series["test"], analysis_results.get("test_loss_server"))

            xs = list(range(1, len(self._acc_series["train"]) + 1))
            acc_plot = wandb.plot.line_series(
                xs=xs,
                ys=[
                    self._acc_series["train"],
                    self._acc_series["val"],
                    self._acc_series["test"],
                ],
                keys=[
                    "train_accuracy_client",
                    "val_accuracy_client",
                    "test_accuracy_server",
                ],
                title="Accuracy per round",
                xname="round",
            )
            loss_plot = wandb.plot.line_series(
                xs=xs,
                ys=[
                    self._loss_series["train"],
                    self._loss_series["val"],
                    self._loss_series["test"],
                ],
                keys=[
                    "train_loss_client",
                    "val_loss_client",
                    "test_loss_server",
                ],
                title="Loss per round",
                xname="round",
            )
            wandb.log(
                {
                    "chart/accuracy_per_round": acc_plot,
                    "chart/loss_per_round": loss_plot,
                },
                step=server_round,
            )
        except Exception:
            pass

    def _print_evaluation_summary(
        self,
        server_round: int,
        loss: float,
        metrics: Dict[str, Any],
    ) -> None:
        try:
            val_acc = metrics.get("val_accuracy_client")
            val_loss = metrics.get("val_loss_client")
            test_acc = metrics.get("test_accuracy_server")
            test_loss = metrics.get("test_loss_server")
            test_acc_label = (
                f"test_accuracy_server={test_acc:.4f}" if test_acc is not None else "test_accuracy_server=NA"
            )
            test_loss_label = (
                f"test_loss_server={test_loss:.4f}" if test_loss is not None else "test_loss_server=NA"
            )
            val_acc_label = (
                f"val_accuracy_client={val_acc:.4f}" if val_acc is not None else "val_accuracy_client=NA"
            )
            val_loss_label = (
                f"val_loss_client={val_loss:.4f}" if val_loss is not None else "val_loss_client=NA"
            )
            eligible = metrics.get("eligible_clients", 0)
            total = self.num_supernodes if self.num_supernodes is not None else "?"
            selected = metrics.get("selected_clients", self.last_selected_count)
            deaths = metrics.get("deaths_clients", self.last_deaths_count)

            print(
                f"[Round {server_round}] {test_acc_label} {test_loss_label} {val_acc_label} {val_loss_label} "
                f"eligible_clients={eligible} "
                f"selected_clients={selected}/{total} "
                f"deaths={deaths}/{selected}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Evaluation overrides
    # ------------------------------------------------------------------
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result
        self._last_server_eval = {"loss": loss, **metrics}
        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        try:
            agg = super().aggregate_evaluate(server_round, results, failures)
        except Exception:
            agg = (None, {})

        loss, metrics = agg
        if loss is None or metrics is None:
            return agg

        analysis_results = self._prepare_evaluation_metrics(server_round, loss, metrics)

        try:
            if hasattr(self, "_last_server_eval") and isinstance(self._last_server_eval, dict):
                if "test_accuracy_server" in self._last_server_eval:
                    analysis_results["test_accuracy_server"] = round(
                        float(self._last_server_eval["test_accuracy_server"]),
                        4,
                    )
                if "loss" in self._last_server_eval:
                    analysis_results["test_loss_server"] = round(
                        float(self._last_server_eval["loss"]),
                        4,
                    )
        except Exception:
            pass
        finally:
            if hasattr(self, "_last_server_eval"):
                delattr(self, "_last_server_eval")

        self._log_results_to_wandb(server_round, analysis_results)
        self._print_evaluation_summary(server_round, loss, analysis_results)

        return loss, metrics
