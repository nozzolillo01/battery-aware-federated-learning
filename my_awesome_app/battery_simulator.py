"""Simple battery simulation for federated learning client selection."""

import random
from typing import Dict, List


class BatterySimulator:
    """
    Simulates battery behavior for a federated learning client device.
    
    Tracks battery levels, consumption during training, and recharging when idle.
    """
    
    def __init__(self, client_id: str, sensor_type: str = None):
        self.client_id = client_id
        self.battery_level = random.uniform(0.3, 1.0)
        self.consumption_rate = random.uniform(0.15, 0.35)
        self.total_consumption = 0.0
        self.training_rounds = 0
        if sensor_type is None:
            self.sensor_type = random.choice(["standard", "high_eff", "low_eff"])
        else:
            self.sensor_type = sensor_type
        if self.sensor_type == "high_eff":
            self.harvesting_capability = random.uniform(0.06, 0.12)
        elif self.sensor_type == "low_eff":
            self.harvesting_capability = random.uniform(0.01, 0.04)
        else:
            self.harvesting_capability = random.uniform(0.03, 0.08)

    def recharge_battery(self):
        """
        Simulates battery recharging through energy harvesting.
        Increases battery by a random amount between 0 and harvesting_capability.
        """
        harvested = random.uniform(0, self.harvesting_capability)
        self.battery_level = min(1.0, self.battery_level + harvested)
        return self.battery_level

        
    def can_participate(self, min_threshold: float = 0.2) -> bool:
        """Check if device has enough battery to participate in training."""
        return self.battery_level >= min_threshold
    
    def consume_battery(self):
        """Simulate battery consumption during model training."""
        consumption = self.consumption_rate + random.uniform(-0.02, 0.02)
        # Update battery level
        self.battery_level = max(0.0, self.battery_level - consumption)
        self.total_consumption += consumption
        self.training_rounds += 1
    
    def get_energy_efficiency(self) -> float:
        """Returns the average energy consumption per training round."""
        if self.training_rounds == 0:
            return 0.0
        return self.total_consumption / self.training_rounds


class FleetManager:
    """
    Manages a fleet of client devices with battery simulation.
    
    Handles client selection, weight calculation, and battery updates.
    """
    
    def __init__(self):
        self.clients: Dict[str, BatterySimulator] = {}
        self.unique_clients_ever_used = set()
        self.client_participation_count = {}
    
    def add_client(self, client_id: str) -> BatterySimulator:
        """Add a new client to the fleet or return existing one."""
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]
    
    def get_battery_level(self, client_id: str) -> float:
        """Get the current battery level for a client."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level
    
    def get_eligible_clients(self, client_ids: List[str], min_threshold: float = 0.2) -> List[str]:
        """Return list of clients with battery above minimum threshold."""
        eligible = []
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            if self.clients[client_id].can_participate(min_threshold):
                eligible.append(client_id)
        return eligible
    
    def calculate_selection_weights(self, client_ids: List[str]) -> Dict[str, float]:
        """Calculate client selection weights based on battery levels."""
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** 2  # Quadratic weighting
        
        return weights
    
    def update_round(self, selected_clients: List[str], all_clients: List[str]):
        """Update battery levels after a training round."""
        # Update each client's battery
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            if client_id in selected_clients:
                # Consume battery for selected clients
                self.clients[client_id].consume_battery()
                self.unique_clients_ever_used.add(client_id)
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1
            else:
                # Recharge battery for idle clients
                self.clients[client_id].recharge_battery()
    
    def get_fleet_stats(self, min_threshold: float = 0.2) -> Dict:
        """Get basic statistics about the fleet's battery status."""
        if not self.clients:
            return {
                "avg_battery": 0, 
                "min_battery": 0,
                "max_battery": 0,
                "eligible_clients": 0,
                "total_clients": 0,
                "fairness_jain": 0.0,
                "total_energy_consumed": 0.0,
                "avg_energy_efficiency": 0.0
            }
        
        # Battery levels
        battery_levels = [client.battery_level for client in self.clients.values()]
        eligible = sum(1 for client in self.clients.values() if client.can_participate(min_threshold))
        
        # Fairness (Jain's index) over participation counts across all known clients
        total_clients = len(self.clients)
        counts = [self.client_participation_count.get(cid, 0) for cid in self.clients]
        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)
        if total_clients > 0 and sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (total_clients * sum_x2)
        else:
            fairness_jain = 0.0
        
        # Energy metrics (only for clients that have participated)
        total_energy = sum(client.total_consumption for client in self.clients.values())
        active_clients = [client for client in self.clients.values() if client.training_rounds > 0]
        
        if active_clients:
            avg_efficiency = sum(client.get_energy_efficiency() for client in active_clients) / len(active_clients)
        else:
            avg_efficiency = 0.0
        
        return {
            "avg_battery": sum(battery_levels) / len(battery_levels),
            "min_battery": min(battery_levels),
            "max_battery": max(battery_levels),
            "eligible_clients": eligible,
            "total_clients": total_clients,
            "fairness_jain": fairness_jain,
            "total_energy_consumed": total_energy,
            "avg_energy_efficiency": avg_efficiency
        }