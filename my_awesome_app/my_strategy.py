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
from typing import Any, Dict, List, Tuple, Optional, cast

from .battery_simulator import FleetManager


class BatteryAwareFedAvg(FedAvg):
    """Energy-aware variant of FedAvg with quadratic battery-based selection."""

    def __init__(self, *args, **kwargs) -> None:
        # Receive optional config from server
        self.total_rounds_config = kwargs.pop("total_rounds", None)
        self.local_epochs_config = kwargs.pop("local_epochs", None)
        self.num_supernodes = kwargs.pop("num_supernodes", None)
        self.min_battery_threshold = float(kwargs.pop("min_battery_threshold", 0.0))

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
        """
        Initialize Weights & Biases run with timestamped name.
        Creates a unique run name based on current date and time.
        """
        tz = ZoneInfo("Europe/Rome")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(
            project="FL", 
            name=f"run-{timestamp}",
            config={
                "min_battery_threshold": self.min_battery_threshold,
                "total_rounds": self.total_rounds_config,
                "local_epochs": self.local_epochs_config,
                "num_supernodes": self.num_supernodes
            }
        )

    def _print_run_header(self) -> None:
        """
        Print minimal run configuration header to the terminal.
        Displays key parameters of the current federated learning run.
        """
        config_items = [
            ("num-supernodes", self.num_supernodes),
            ("num-server-rounds", self.total_rounds_config),
            ("local-epochs", self.local_epochs_config),
            ("min-battery-threshold", self.min_battery_threshold)
        ]
        
        for name, value in config_items:
            if value is not None:
                print(f"{name} = {value}")



    def _extract_available_clients(self, original_config: List[Tuple[ClientProxy, Dict]]) -> Tuple[List[ClientProxy], Dict]:
        """
        Extract available clients and shared configuration from original config.
        
        Args:
            original_config: Configuration from parent class
            
        Returns:
            tuple: (available_clients, shared_config)
        """
        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}
        return available_clients, config

    def _eligible_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """
        Filter clients based on battery threshold.
        
        Args:
            available_clients: List of available clients
            
        Returns:
            List[ClientProxy]: Clients with sufficient battery level
        """
        eligible_ids = self.fleet_manager.get_eligible_clients(
            [c.cid for c in available_clients], 
            self.min_battery_threshold
        )
        return [c for c in available_clients if c.cid in eligible_ids]

    def _fallback_topk_by_battery(self, available_clients: List[ClientProxy], k: int = 2) -> List[ClientProxy]:
        """
        Select top-k clients by battery level as fallback.
        
        Args:
            available_clients: List of available clients
            k: Number of clients to select
            
        Returns:
            List[ClientProxy]: Top-k clients by battery level
        """
        levels = [(c, self.fleet_manager.get_battery_level(c.cid)) for c in available_clients]
        levels.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in levels[:k]]

    def _probabilistic_selection(
        self, 
        eligible_clients: List[ClientProxy], 
        available_clients: List[ClientProxy]
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:
        """
        Perform probabilistic client selection with quadratic battery weighting.
        
        Args:
            eligible_clients: List of clients eligible for selection
            available_clients: List of all available clients
            
        Returns:
            tuple: (selected_clients, probability_map)
        """
        prob_map: Dict[str, float] = {}
        
        # If we have very few eligible clients, select all of them
        if len(eligible_clients) <= 2:
            selected_clients = eligible_clients
            # Assign probability 1.0 to selected clients (deterministic) and 0.0 to others
            for c in available_clients:
                prob_map[c.cid] = 1.0 if c in selected_clients else 0.0
            return selected_clients, prob_map
        
        # Calculate quadratic weights for eligible clients
        weights_map = self.fleet_manager.calculate_selection_weights([c.cid for c in eligible_clients])
        weights = np.array([weights_map[c.cid] for c in eligible_clients])
        
        # Normalize weights to probabilities
        probs = weights / weights.sum()
        
        # Select at least 1, at most half of eligible clients, but minimum 2 if possible
        num_to_select = max(1, min(len(eligible_clients), len(eligible_clients) // 2))
        if num_to_select < 2 and len(eligible_clients) >= 2:
            num_to_select = 2
            
        # Probabilistic selection based on battery weights
        idx = np.random.choice(len(eligible_clients), size=num_to_select, replace=False, p=probs)
        selected_clients = [eligible_clients[i] for i in idx]
        
        # Map probabilities for all eligible clients
        for i, c in enumerate(eligible_clients):
            prob_map[c.cid] = float(probs[i])
            
        # Non-eligible clients have 0.0 probability
        for c in available_clients:
            if c.cid not in prob_map:
                prob_map[c.cid] = 0.0
                
        return selected_clients, prob_map

    def _capture_selected_stats(self, selected_client_ids: List[str]) -> None:
        """
        Capture statistics about selected clients.
        
        Stores count, average battery level and minimum battery level
        of selected clients for reporting and analysis.
        
        Args:
            selected_client_ids: List of IDs of selected clients
        """
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
        """
        Log detailed client selection data to Weights & Biases.
        
        Args:
            server_round: Current federated learning round
            available_clients: List of available clients
            selected_client_ids: List of IDs of selected clients
            eligible_ids: List of IDs of eligible clients
            prob_map: Map of client IDs to selection probabilities
        """
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
            delta = None if prev is None else battery - prev
            prob = prob_map.get(cid, 0.0)
            present_data[cid] = {
                "battery": round(battery, 4),
                "delta_battery": (round(delta, 4) if delta is not None else np.nan),
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
            "round", "client_id", "battery", "delta_battery", 
            "prob_selection", "selected", "eligible", "rounds_since_selected",
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
                # Client not available this round: NaN for battery and delta, 0 for others
                row = [server_round, cid, np.nan, np.nan, 0.0, 0, 0, self._rounds_since_selected.get(cid, np.nan)]
            round_table.add_data(*row)

        # Log table with distinct key for each round
        wandb.log({
            "round": server_round,
            "selected_clients": self.last_selected_count,
            f"client_status_round_{server_round}": round_table
        }, step=server_round)
        
        # Update previous battery levels for delta calculation in next round
        for c in available_clients:
            self._prev_battery_levels[c.cid] = self.fleet_manager.get_battery_level(c.cid)

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

        # Get available clients and config
        available_clients, config = self._extract_available_clients(original_config)
        
        # Filter clients by battery threshold
        eligible_clients = self._eligible_clients(available_clients)
        eligible_ids = [c.cid for c in eligible_clients]
        
        # Fallback to top-2 by battery if none are eligible
        if not eligible_clients:
            eligible_clients = self._fallback_topk_by_battery(available_clients, k=2)
        
        # Perform client selection with battery-weighted probabilities
        selected_clients, prob_map = self._probabilistic_selection(eligible_clients, available_clients)
        selected_client_ids = [c.cid for c in selected_clients]
        available_client_ids = [c.cid for c in available_clients]
        
        # Capture statistics about selected clients
        self._capture_selected_stats(selected_client_ids)
        
        # Log detailed information to Weights & Biases
        self._log_selection_to_wandb(server_round, available_clients, selected_client_ids, eligible_ids, prob_map)
        
        # Update fleet battery levels
        self.fleet_manager.update_round(selected_client_ids, available_client_ids)
        
        return [(client, config) for client in selected_clients]

    def _extract_battery_metrics(self) -> Dict[str, float]:
        """
        Extract battery-related metrics from fleet statistics.
        
        Returns:
            Dict[str, float]: Dictionary with battery metrics
        """
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
        """
        Aggregate as FedAvg and augment metrics with fleet battery statistics.
        
        Args:
            server_round: The current round of federated learning
            results: List of successful client results
            failures: List of failed client results
            
        Returns:
            tuple: (aggregated_parameters, metrics_dict)
        """
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
        """
        Prepare detailed metrics for evaluation results, including battery statistics.
        
        Args:
            server_round: Current round number
            loss: Evaluation loss
            metrics: Base metrics from evaluation
            
        Returns:
            Dict[str, Any]: Enhanced metrics with battery statistics
        """
        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        battery_metrics = {
            "battery_avg": fleet_stats.get("avg_battery", 0),
            "battery_min": fleet_stats.get("min_battery", 0),
            "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
            "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
            "eligible_clients": fleet_stats.get("eligible_clients", 0),
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
        """
        Log evaluation results to Weights & Biases.
        
        Args:
            server_round: Current round number
            analysis_results: Results to log
        """
        # Create a copy of the results for WandB
        wandb_payload = {
            k: v for k, v in analysis_results.items() 
            if k not in ["round"] and v is not None
        }
        
        wandb.log(wandb_payload, step=server_round)
        
    def _print_evaluation_summary(self, server_round: int, loss: float, metrics: Dict[str, Any]) -> None:
        """
        Print minimal evaluation summary to terminal.
        
        Args:
            server_round: Current round number
            loss: Evaluation loss
            metrics: Evaluation metrics
        """
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
        """
        Evaluate as FedAvg and enrich metrics and logs with battery stats.
        
        Args:
            server_round: The current round of federated learning
            parameters: Model parameters to evaluate
            
        Returns:
            Optional[Tuple[float, Dict[str, Any]]]: Evaluation results or None
        """
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