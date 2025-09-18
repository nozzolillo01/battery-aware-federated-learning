"""Simple battery simulation for federated learning client selection."""

import random
from typing import Dict, List


class BatterySimulator:
    """
    Simulates battery behavior for a federated learning client device.
    
    Tracks battery levels, consumption during training, and recharging when idle.
    """
    
    # Constants for sensor types
    SENSOR_TYPES = ["standard", "high_eff", "low_eff"]
    HARVESTING_RANGES = {
        "high_eff": (0.06, 0.12),
        "low_eff": (0.01, 0.04),
        "standard": (0.03, 0.08)
    }
    
    def __init__(self, client_id: str, sensor_type: str = None):
        self.client_id = client_id
        self.battery_level = random.uniform(0.3, 1.0)
        self.consumption_rate = random.uniform(0.15, 0.35)
        self.total_consumption = 0.0
        self.training_rounds = 0
        
        # Set sensor type and harvesting capability
        self.sensor_type = sensor_type if sensor_type in self.SENSOR_TYPES else random.choice(self.SENSOR_TYPES)
        min_harvest, max_harvest = self.HARVESTING_RANGES[self.sensor_type]
        self.harvesting_capability = random.uniform(min_harvest, max_harvest)

    def recharge_battery(self) -> float:
        """
        Simulates battery recharging through energy harvesting.
        Increases battery by a random amount between 0 and harvesting_capability.
        
        Returns:
            float: The updated battery level.
        """
        harvested = random.uniform(0, self.harvesting_capability)
        self.battery_level = min(1.0, self.battery_level + harvested)
        return self.battery_level
        
    def can_participate(self, min_threshold: float = 0.0) -> bool:
        """
        Check if device has enough battery to participate in training.
        
        Args:
            min_threshold: Minimum battery level required for participation.
            
        Returns:
            bool: True if battery is sufficient, False otherwise.
        """
        return self.battery_level >= min_threshold
    
    def consume_battery(self) -> None:
        """
        Simulate battery consumption during model training.
        """
        consumption = self.consumption_rate
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
    Tracks participation statistics for fairness evaluation.
    """
    
    def __init__(self):
        self.clients: Dict[str, BatterySimulator] = {}
        self.unique_clients_ever_used = set()
        self.client_participation_count: Dict[str, int] = {}
    
    def add_client(self, client_id: str) -> BatterySimulator:
        """
        Add a new client to the fleet or return existing one.
        
        Args:
            client_id: The unique identifier for the client.
            
        Returns:
            BatterySimulator: The client's battery simulator instance.
        """
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]
    
    def get_battery_level(self, client_id: str) -> float:
        """
        Get the current battery level for a client.
        
        Args:
            client_id: The client's unique identifier.
            
        Returns:
            float: Current battery level between 0.0 and 1.0.
        """
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level
    
    def get_eligible_clients(self, client_ids: List[str], min_threshold: float = 0.0) -> List[str]:
        """
        Return list of clients with battery above minimum threshold.
        
        Args:
            client_ids: List of client IDs to check.
            min_threshold: Minimum battery level required for eligibility.
            
        Returns:
            List[str]: IDs of clients with sufficient battery.
        """
        eligible = []
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            if self.clients[client_id].can_participate(min_threshold):
                eligible.append(client_id)
        return eligible
    
    def calculate_selection_weights(self, client_ids: List[str]) -> Dict[str, float]:
        """
        Calculate client selection weights based on battery levels.
        Uses quadratic weighting to prioritize clients with higher battery.
        
        Args:
            client_ids: List of client IDs to calculate weights for.
            
        Returns:
            Dict[str, float]: Dictionary mapping client IDs to their selection weights.
        """
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** 2  # Quadratic weighting
        
        return weights
    
    def update_round(self, selected_clients: List[str], all_clients: List[str]) -> None:
        """
        Update battery levels after a training round.
        Selected clients consume battery, while idle clients recharge.
        
        Args:
            selected_clients: IDs of clients selected for training.
            all_clients: IDs of all available clients in this round.
        """
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
    
    def get_fleet_stats(self, min_threshold: float = 0.0) -> Dict[str, float]:
        """
        Get comprehensive statistics about the fleet's battery status.
        
        Args:
            min_threshold: Minimum battery threshold for eligibility.
            
        Returns:
            Dict[str, float]: Dictionary with fleet statistics.
        """
        # Default values if no clients exist
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
        
        # Calculate battery statistics
        battery_levels = [client.battery_level for client in self.clients.values()]
        eligible = sum(1 for client in self.clients.values() if client.can_participate(min_threshold))
        
        # Calculate fairness using Jain's index
        total_clients = len(self.clients)
        counts = [self.client_participation_count.get(cid, 0) for cid in self.clients]
        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)
        
        fairness_jain = 0.0
        if total_clients > 0 and sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (total_clients * sum_x2)
        
        # Calculate energy efficiency metrics
        total_energy = sum(client.total_consumption for client in self.clients.values())
        active_clients = [client for client in self.clients.values() if client.training_rounds > 0]
        
        avg_efficiency = 0.0
        if active_clients:
            avg_efficiency = sum(client.get_energy_efficiency() for client in active_clients) / len(active_clients)
        
        # Return comprehensive statistics
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