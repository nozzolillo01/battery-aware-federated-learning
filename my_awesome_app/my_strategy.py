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
import os
from datetime import datetime
from typing import Dict, List, Tuple

from .battery_simulator import FleetManager

logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("wandb").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)


class BatteryAwareFedAvg(FedAvg):

    def __init__(self, min_battery_threshold: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.min_battery_threshold = min_battery_threshold
        self.fleet_manager = FleetManager()
        self.results_to_save = {}
        self.total_federation_size = 10
        
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="battery-aware-fl", name=f"simple-strategy-{name}")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, dict]]:
        
        original_config = super().configure_fit(server_round, parameters, client_manager)
        
        if not original_config:
            return []
        
        available_clients = [client for client, _ in original_config]
        config = original_config[0][1] if original_config else {}
        
        client_ids = [c.cid for c in available_clients]
        eligible_clients_ids = self.fleet_manager.get_eligible_clients(
            client_ids, self.min_battery_threshold
        )
        
        eligible_clients = [c for c in available_clients if c.cid in eligible_clients_ids]
        
        if not eligible_clients:
            battery_levels = [(c, self.fleet_manager.get_battery_level(c.cid)) 
                            for c in available_clients]
            battery_levels.sort(key=lambda x: x[1], reverse=True)
            eligible_clients = [c for c, _ in battery_levels[:2]]
        
        if len(eligible_clients) <= 2:
            selected_clients = eligible_clients
        else:
            weights_dict = self.fleet_manager.calculate_selection_weights(
                [c.cid for c in eligible_clients]
            )
            
            weights = np.array([weights_dict[c.cid] for c in eligible_clients])
            
            if weights.sum() > 0:
                probabilities = weights / weights.sum()
                
                num_to_select = max(1, min(len(eligible_clients), len(eligible_clients) // 2))
                if num_to_select < 2 and len(eligible_clients) >= 2:
                    num_to_select = 2
                
                selected_indices = np.random.choice(
                    len(eligible_clients),
                    size=num_to_select,
                    replace=False,
                    p=probabilities
                )
                selected_clients = [eligible_clients[i] for i in selected_indices]
            else:
                selected_clients = eligible_clients[:max(1, min(2, len(eligible_clients)))]
        
        self.fleet_manager.update_round([c.cid for c in selected_clients], 
                                      [c.cid for c in available_clients])
        
        return [(client, config) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        
        battery_metrics = {
            "fleet_avg_battery": fleet_stats.get("avg_battery", 0),
            "fleet_min_battery": fleet_stats.get("min_battery", 0),
            "participation_rate": fleet_stats.get("participation_rate", 0.0),
        }
        
        if metrics_aggregated is None:
            metrics_aggregated = {}
        metrics_aggregated.update(battery_metrics)
        
        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
            
        loss, metrics = result
        
        fleet_stats = self.fleet_manager.get_fleet_stats(self.min_battery_threshold)
        
        battery_metrics = {
            "battery_avg": fleet_stats.get("avg_battery", 0),
            "battery_min": fleet_stats.get("min_battery", 0),
            "participation_rate": fleet_stats.get("participation_rate", 0.0),
        }
        
        if metrics is None:
            metrics = {}
        metrics.update(battery_metrics)
        
        analysis_results = {
            "loss": round(loss, 4),
            "accuracy": round(metrics.get('cen_accuracy', 0), 4),
            "battery_avg": round(battery_metrics['battery_avg'], 3),
            "battery_min": round(battery_metrics['battery_min'], 3),
            "participation_rate": round(battery_metrics['participation_rate'], 3),
        }
        
        self.results_to_save[server_round] = analysis_results
        
        with open("results.json", "w") as f:
            json.dump(self.results_to_save, f, indent=2)
        
        wandb.log({
            "loss": round(loss, 4),
            "accuracy": round(metrics.get('cen_accuracy', 0), 4),
            "battery_avg": round(battery_metrics['battery_avg'], 3),
            "battery_min": round(battery_metrics['battery_min'], 3),
            "participation_rate": round(battery_metrics['participation_rate'], 3),
        }, step=server_round)
        
        return loss, metrics
