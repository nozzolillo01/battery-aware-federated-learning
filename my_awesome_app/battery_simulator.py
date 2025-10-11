"""Battery simulation for federated learning clients."""

import random
from typing import Dict, List


class BatterySimulator:
    """
    Simulates battery behavior for a federated learning client device.
    Tracks battery levels, consumption during training, and recharging when idle.
    """

    DEVICE_CLASSES: Dict[str, Dict[str, tuple]] = {
        "low_power_sensor": {
            "consumption_range": (0.005, 0.015),
            "harvesting_range": (0.0, 0.010),
        },
        "mid_edge_device": {
            "consumption_range": (0.020, 0.030),
            "harvesting_range": (0.0, 0.025),
        },
        "high_power_gateway": {
            "consumption_range": (0.040, 0.060),
            "harvesting_range": (0.0, 0.050),
        },
    }
    
    def __init__(self, client_id: str, device_class: str = None):
        """Initialize battery simulator with random device class and energy parameters."""
        self.client_id = client_id
        self.battery_level = random.uniform(0.1, 1.0)
        self.total_consumption = 0.0

        if device_class not in self.DEVICE_CLASSES:
            device_class = random.choice(list(self.DEVICE_CLASSES.keys()))
        self.device_class = device_class
        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]

        self.consumption_for_epochs = random.uniform(cmin, cmax)
        self.harvesting_capability = random.uniform(hmin, hmax)

    def recharge_battery(self, local_epochs: int = 1) -> float:
        """Recharge battery through energy harvesting and return effective harvested amount."""
        previous_level = self.battery_level
        epochs = max(1, int(local_epochs))
        harvested = self.harvesting_capability * epochs
        self.battery_level = min(1.0, previous_level + harvested)
        effective_harvested = self.battery_level - previous_level
        return effective_harvested

    def can_participate(self, min_threshold: float = 0.0) -> bool:
        """Check if client has more battery than the minimum threshold"""
        return self.battery_level >= min_threshold 

    def enough_battery_for_training(self, local_epochs: int) -> bool:
        """Check if battery level is sufficient to complete the training epochs for one round."""
        epochs = max(1, int(local_epochs))
        needed = self.consumption_for_epochs * epochs
        return self.battery_level >= needed

    def consume_battery(self, local_epochs: int) -> bool:
        """
        Consume battery for training and return whether training completed successfully.
        
        Battery consumption is proportional to the number of local training epochs.
        If battery runs out during training, consumes all remaining battery and returns False.
        """
        epochs = max(1, int(local_epochs))
        needed = self.consumption_for_epochs * epochs
        
        if self.battery_level >= needed:
            # Sufficient battery: complete training
            self.battery_level = max(0.0, self.battery_level - needed)
            self.total_consumption += needed
            return True
        else:
            # Insufficient battery: consume all remaining and fail
            consumed = self.battery_level
            self.battery_level = 0.0
            self.total_consumption += consumed
            return False


class FleetManager:
    """
    Manages a fleet of client devices with battery simulation.
    
    Handles client selection, weight calculation, and battery updates.
    Tracks participation statistics for fairness evaluation.
    """
    
    def __init__(self):
        """Initialize fleet manager with empty client tracking structures."""
        self.clients: Dict[str, BatterySimulator] = {}
        self.client_participation_count: Dict[str, int] = {}
        self.client_recharged_battery: Dict[str, float] = {}
        self.client_consumed_battery: Dict[str, float] = {}
    
    def add_client(self, client_id: str) -> BatterySimulator:
        """Add a new client with battery simulator if not already registered."""
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]

    def get_battery_level(self, client_id: str) -> float:
        """Get current battery level for a specific client."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level

    def get_device_class(self, client_id: str) -> str:
        """Get device class type for a specific client."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return getattr(self.clients[client_id], "device_class", "unknown")

    def get_dead_clients(self, selected_clients: List[str], local_epochs: int) -> List[str]:
        """Identify the devices that did not complete the training because they ran out of battery during it."""
        return [
            cid for cid in selected_clients 
            if not self.clients[cid].enough_battery_for_training(local_epochs)
        ]

    def get_eligible_clients(self, client_ids: List[str], min_threshold: float = 0.0) -> List[str]:
        """Filter clients that meet minimum battery threshold for participation."""
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
        
        return [
            cid for cid in client_ids 
            if self.clients[cid].can_participate(min_threshold)
        ]

    def calculate_selection_weights(self, client_ids: List[str], alpha: float = 2.0) -> Dict[str, float]:
        """Calculate selection weights based on battery levels with exponential scaling."""
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** alpha
        return weights

    def update_round(self, selected_clients: List[str], all_clients: List[str], local_epochs: int) -> None:
        """Update battery levels after training round and track participation statistics."""
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)

            self.client_consumed_battery[client_id] = 0.0
            
            if client_id in selected_clients:
                previous_level = self.clients[client_id].battery_level
                self.clients[client_id].consume_battery(local_epochs)
                consumed = previous_level - self.clients[client_id].battery_level
                self.client_consumed_battery[client_id] = consumed
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1

            recharged = self.clients[client_id].recharge_battery(local_epochs)
            self.client_recharged_battery[client_id] = recharged
  
    def get_fleet_stats(self, min_threshold: float = 0.0) -> Dict[str, float]:
        """Calculate comprehensive fleet statistics including battery levels and fairness metrics."""
        if not self.clients:
            return {
                "avg_battery": 0,
                "min_battery": 0,
                "eligible_clients": 0,
                "fairness_jain": 0.0,
                "total_energy_consumed": 0.0,
            }

        battery_levels = [client.battery_level for client in self.clients.values()]
        eligible = sum(1 for client in self.clients.values() if client.can_participate(min_threshold))

        total_clients = len(self.clients)
        counts = [self.client_participation_count.get(cid, 0) for cid in self.clients]
        sum_x = sum(counts)
        sum_x2 = sum(c * c for c in counts)

        fairness_jain = 0.0
        if total_clients > 0 and sum_x2 > 0:
            fairness_jain = (sum_x * sum_x) / (total_clients * sum_x2)

        total_energy = sum(client.total_consumption for client in self.clients.values())

        return {
            "avg_battery": sum(battery_levels) / len(battery_levels),
            "min_battery": min(battery_levels),
            "eligible_clients": eligible,
            "fairness_jain": fairness_jain,
            "total_energy_consumed": total_energy,
        }