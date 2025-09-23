"""
base strategy with FedAvg with logging 

"""

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import numpy as np
import wandb

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple, Optional

from .battery_simulator import FleetManager


class BaseStrategy(FedAvg):

    def __init__(self, *args, **kwargs) -> None:
        # Receive optional config from server
        self.total_rounds_config = kwargs.pop("total_rounds", None)
        self.local_epochs_config = kwargs.pop("local_epochs", None)
        self.num_supernodes = kwargs.pop("num_supernodes", None)
        self.min_battery_threshold = float(kwargs.pop("min_battery_threshold", 0.0))
        self.strategy = kwargs.pop("strategy", "base")

        # Init parent with the remaining kwargs
        super().__init__(*args, **kwargs)

        self.fleet_manager = FleetManager()
        self.results_to_save = {}
        self.last_selected_count = 0
        self.last_selected_battery_avg = None
        self.last_selected_battery_min = None

        # Tracking state for detailed wandb visualization
        self._prev_battery_levels: Dict[str, float] = {}
        self._client_id_order = [] 
        self._rounds_since_selected: Dict[str, int] = {} 

        # Initialize Weights & Biases run and print header
        self._init_wandb_run()
        self._print_run_header()

    def _init_wandb_run(self) -> None:
        tz = ZoneInfo("Europe/Rome")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(
            project="Federated Learning", 
            name=f"BASE-run-{timestamp}",
                        config={
                "min_battery_threshold": self.min_battery_threshold,
                "total_rounds": self.total_rounds_config,
                "local_epochs": self.local_epochs_config,
                "num_supernodes": self.num_supernodes,
                "strategy": self.strategy
            }
        )

    def _print_run_header(self) -> None:
        config_items = [
            ("num-supernodes", self.num_supernodes),
            ("num-server-rounds", self.total_rounds_config),
            ("local-epochs", self.local_epochs_config),
            ("min-battery-threshold", self.min_battery_threshold),
            ("strategy", self.strategy)
        ]
        
        for name, value in config_items:
            if value is not None:
                print(f"{name} = {value}")



    def _extract_available_clients(self, original_config: List[Tuple[ClientProxy, Dict]]) -> Tuple[List[ClientProxy], Dict]:
        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}
        return available_clients, config

    def _eligible_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        #return all the clients with battery level >= min_battery_threshold
        eligible_ids = self.fleet_manager.get_eligible_clients(
            [c.cid for c in available_clients], self.min_battery_threshold)
        return [c for c in available_clients if c.cid in eligible_ids]

    def _probabilistic_selection(self, eligible_clients: List[ClientProxy],available_clients: List[ClientProxy]) -> Tuple[List[ClientProxy], Dict[str, float]]:
        # Create probability map with 1.0 for all clients eligible and 0 otherwise

        selected_clients = eligible_clients

        prob_map: Dict[str, float] = {}
        for c in available_clients:
            prob_map[c.cid] = 1.0 if c in selected_clients else 0.0

        return selected_clients, prob_map

    def _capture_selected_stats(self, selected_client_ids: List[str]) -> None:

        self.last_selected_count = len(selected_client_ids)
        
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

    def _log_selection_to_wandb(
        self, 
        server_round: int, 
        available_clients: List[ClientProxy], 
        selected_client_ids: List[str], 
        eligible_ids: List[str], 
        prob_map: Dict[str, float]
    ) -> None:
        
        # Update stable client ordering (add new clients while maintaining sort)
        newly_seen = [c.cid for c in available_clients if c.cid not in self._client_id_order]
        if newly_seen:
            self._client_id_order.extend(newly_seen)
            self._client_id_order = sorted(self._client_id_order)  # Stable alphanumeric ordering

        # Prepare current round data for all available clients
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

        # Update rounds_since_selected counters
        for cid in self._client_id_order:
            if cid in selected_client_ids:
                self._rounds_since_selected[cid] = 0
            else:
                # Increment only if client has been seen before
                if cid in self._rounds_since_selected:
                    self._rounds_since_selected[cid] += 1
                else:
                    # First time seen and not selected -> 1
                    self._rounds_since_selected[cid] = 1 if cid not in selected_client_ids else 0

        # Create a table for the current round with fixed column order
        columns = [
            "round", "client_id", "current_battery_level", "previous_battery_level", "consumed_battery", "recharged_battery", 
            "prob_selection", "selected", "eligible", "rounds_since_selected",
        ]
        round_table = wandb.Table(columns=columns)
        
        for cid in self._client_id_order:
            if cid in present_data:
                row = [
                    server_round,
                    cid,
                    present_data[cid]["current_battery_level"],
                    present_data[cid]["previous_battery_level"],
                    present_data[cid]["consumed_battery"],
                    present_data[cid]["recharged_battery"],
                    present_data[cid]["prob_selection"],
                    present_data[cid]["selected"],
                    present_data[cid]["eligible"],
                    self._rounds_since_selected.get(cid, np.nan),
                ]
            else:
                # Client not available this round: NaN for battery related values, 0 for others
                row = [server_round, cid, np.nan, np.nan, np.nan, np.nan, 0.0, 0, 0, self._rounds_since_selected.get(cid, np.nan)]
            round_table.add_data(*row)

        # Log table with distinct key for each round
        wandb.log({
            "round": server_round,
            "selected_clients": self.last_selected_count,
            f"client_status_round_{server_round}": round_table
        }, step=server_round)
        
        # Update previous battery levels for comparison in next round
        for c in available_clients:
            self._prev_battery_levels[c.cid] = self.fleet_manager.get_battery_level(c.cid)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, dict]]:

        original_config = super().configure_fit(server_round, parameters, client_manager)
        if not original_config:
            return []

        # Get available clients and config
        available_clients, config = self._extract_available_clients(original_config)
        
        # In FedAvg, all available clients are eligible and selected
        eligible_clients = self._eligible_clients(available_clients)
        eligible_ids = [c.cid for c in eligible_clients]
        
        # Select all clients
        selected_clients, prob_map = self._probabilistic_selection(eligible_clients, available_clients)
        selected_client_ids = [c.cid for c in selected_clients]
        available_client_ids = [c.cid for c in available_clients]

        self.current_eligible_count = len(eligible_ids)
        
        # Capture statistics about selected clients
        self._capture_selected_stats(selected_client_ids)
        
        # Log detailed information to Weights & Biases
        self._log_selection_to_wandb(server_round, available_clients, selected_client_ids, eligible_ids, prob_map)
        
        # Update fleet battery levels
        self.fleet_manager.update_round(selected_client_ids, available_client_ids)
        
        return [(client, config) for client in selected_clients]

    def _extract_battery_metrics(self) -> Dict[str, float]:

        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        return {
            "fleet_avg_battery": fleet_stats.get("avg_battery", 0),
            "fleet_min_battery": fleet_stats.get("min_battery", 0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:

        # Let parent class handle the actual parameter aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Add battery metrics to the aggregated metrics
        battery_metrics = self._extract_battery_metrics()
        
        # Ensure metrics_aggregated is a dictionary
        if metrics_aggregated is None:
            metrics_aggregated = {}
            
        # Update with battery metrics
        metrics_aggregated.update(battery_metrics)

        return parameters_aggregated, metrics_aggregated

    def _prepare_evaluation_metrics(self, server_round: int, loss: float, metrics: Dict[str, Any]) -> Dict[str, Any]:

        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        battery_metrics = {
            "battery_avg": fleet_stats.get("avg_battery", 0),
            "battery_min": fleet_stats.get("min_battery", 0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
            "eligible_clients": getattr(self, "current_eligible_count", fleet_stats.get("eligible_clients", 0)),
        }
        
        # Ensure metrics is a dictionary
        if metrics is None:
            metrics = {}
        metrics.update(battery_metrics)

        
        # Create complete results dictionary with proper rounding
        return {
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
     
    def _log_results_to_wandb(self, server_round: int, analysis_results: Dict[str, Any]) -> None:

        # Create a copy of the results for WandB
        wandb_payload = {
            k: v for k, v in analysis_results.items() 
            if k not in ["round"] and v is not None
        }
        
        wandb.log(wandb_payload, step=server_round)
        
    def _print_evaluation_summary(self, server_round: int, loss: float, metrics: Dict[str, Any]) -> None:
        try:
            acc = metrics.get("cen_accuracy", 0)
            eligible = metrics.get("eligible_clients", 0)
            suffix = f"/{self.num_supernodes}" if self.num_supernodes is not None else ""
            
            print(
                f"[Round {server_round}] loss={loss:.4f} acc={acc:.4f} "
                f"eligible_clients={eligible} "
                f"selected_clients={self.last_selected_count}{suffix}"
            )
        except Exception:
            pass

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
   
        # Let parent class handle the actual evaluation
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result
        
        # Prepare comprehensive evaluation metrics
        analysis_results = self._prepare_evaluation_metrics(server_round, loss, metrics)
        
        # Log to Weights & Biases
        self._log_results_to_wandb(server_round, analysis_results)
        
        # Print summary to terminal
        self._print_evaluation_summary(server_round, loss, metrics)

        return loss, metrics
