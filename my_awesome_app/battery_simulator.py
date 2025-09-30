"""Simple battery simulation for federated learning client selection."""

import random
from typing import Dict, List


class BatterySimulator:
    """
    Simulates battery behavior for a federated learning client device.
    
    Tracks battery levels, consumption during training, and recharging when idle.
    """
    

    # Hardware device classes with distinct energy profiles
    DEVICE_CLASSES: Dict[str, Dict[str, tuple]] = {
        "raspberry": {
            "consumption_range": (0.008, 0.015),   
            "harvesting_range": (0.010, 0.025),    
        },
        "edgegpu": {
            "consumption_range": (0.015, 0.030),   
            "harvesting_range": (0.006, 0.015),   
        },
        "lowpowermcu": {
            "consumption_range": (0.003, 0.008),  
            "harvesting_range": (0.012, 0.030),    
        },
    }
    
    def __init__(self, client_id: str, device_class: str = None):
        self.client_id = client_id
        self.battery_level = random.uniform(0.1, 1.0)
        self.total_consumption = 0.0
        self.training_rounds = 0

        # Assign a hardware device class and set consumption/harvesting ranges accordingly
        if device_class not in self.DEVICE_CLASSES:
            device_class = random.choice(list(self.DEVICE_CLASSES.keys()))
        self.device_class = device_class
        cmin, cmax = self.DEVICE_CLASSES[self.device_class]["consumption_range"]
        hmin, hmax = self.DEVICE_CLASSES[self.device_class]["harvesting_range"]
        # Per-epoch consumption depends on device class
        self.consumption_rate = random.uniform(cmin, cmax)
        # Harvesting capability also depends on device class
        self.harvesting_capability = random.uniform(hmin, hmax)

    def recharge_battery(self) -> float:
        """
        Simulates battery recharging through energy harvesting.
        Increases battery by a random amount between 0 and harvesting_capability.
        
        Returns:
            float: The amount of battery harvested this cycle.
        """
        previous_level = self.battery_level
        harvested = random.uniform(0, self.harvesting_capability)
        self.battery_level = min(1.0, previous_level + harvested)
        effective_harvested = self.battery_level - previous_level
        return effective_harvested

    def can_participate(self, min_threshold: float = 0.0) -> bool:
        """
        Check if device has enough battery to participate in training.
        
        Args:
            min_threshold: Minimum battery level required for participation.
            
        Returns:
            bool: True if battery is sufficient, False otherwise.
        """
        return self.battery_level >= min_threshold 

    def _effective_consumption(self, local_epochs) -> float:
        """
        Compute effective energy consumption for the current round as
        per-epoch consumption multiplied by the number of local epochs.

        Args:
            local_epochs: Number of local training epochs planned for this round.

        Returns:
            float: Effective battery fraction to be consumed.
        """
        epochs = max(1, int(local_epochs))
        return self.consumption_rate * epochs

    def enough_battery_for_training(self, local_epochs) -> bool:
        """
        Check if device has enough battery to perform training.
        
        Args:
            local_epochs: Number of local training epochs planned for this round.
        
        Returns:
            bool: True if battery is sufficient for training, False otherwise.
        """
        return self.battery_level >= self._effective_consumption(local_epochs)

    def consume_battery(self, local_epochs) -> bool:
        """
        Simulate battery consumption during model training.
        Decreases battery by (consumption_rate * local_epochs) if enough battery is available.
        If not enough battery, drains battery to 0.

        Returns:
            bool: True if training was completed, False if battery drained.

        """

        effective_needed = self._effective_consumption(local_epochs)
        if self.enough_battery_for_training(local_epochs):
            consumption = effective_needed
            self.battery_level = max(0.0, self.battery_level - consumption)
            self.total_consumption += consumption
            self.training_rounds += 1
            return True
        else:
            consumption = self.battery_level  
            self.battery_level = 0.0
            self.total_consumption += consumption
            self.training_rounds += 1
        return False




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
        self.client_recharged_battery: Dict[str, float] = {}
        self.client_consumed_battery: Dict[str, float] = {}
    
    def add_client(self, client_id: str) -> BatterySimulator:
        """
        Add a new client to the fleet or return existing one.
        
        Args:
            client_id: The unique identifier for the client.
            
        Returns:
            BatterySimulator: The client's battery simulator instance.
        """
        if client_id not in self.clients:
            # Create a simulator with a randomly assigned device class
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

    def get_device_class(self, client_id: str) -> str:
        """Return the device class assigned to the client (e.g., raspberry, edgegpu, lowpowermcu)."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return getattr(self.clients[client_id], "device_class", "unknown")

    def get_dead_clients(self, selected_clients: List[str], local_epochs) -> List[str]:
        """
        Get a list of selected clients that don't have enough battery to
        successfully complete the local training step in this round.

        Note:
            A client can be considered "dead" even if its current battery is > 0,
            as long as it is strictly less than the required consumption_rate
            for completing the training step. In that case, during consumption
            the battery will be drained to 0.0 and the client will not finish
            training (and must not exchange weights with the server).

        Args:
            selected_clients: Client IDs selected to attempt training this round.
            local_epochs: Number of local training epochs planned for this round.

        Returns:
            List[str]: IDs of clients without sufficient energy to complete training.
        """
        dead_clients = []
        for client_id in selected_clients:
            if not self.clients[client_id].enough_battery_for_training(local_epochs):
                dead_clients.append(client_id)
        return dead_clients

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

    def calculate_selection_weights(self, client_ids: List[str], alpha: float = 2.0) -> Dict[str, float]:
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
            weights[client_id] = battery_level ** alpha 
        
        return weights

    def update_round(self, selected_clients: List[str], all_clients: List[str], local_epochs) -> None:
        """
        Update battery levels after a training round.
        Selected clients consume battery, while idle clients recharge.
        
        Args:
            selected_clients: IDs of clients selected for training.
            all_clients: IDs of all available clients in this round.
            local_epochs: Number of local training epochs executed per selected client in this round.
        """
        
        
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)

            # Initialize battery consumption to 0
            self.client_consumed_battery[client_id] = 0.0
            
            if client_id in selected_clients:
                # Save battery level before consumption
                previous_level = self.clients[client_id].battery_level
                
                # Consume battery for selected clients
                self.clients[client_id].consume_battery(local_epochs)

                # Calculate the amount of battery consumed
                consumed = previous_level - self.clients[client_id].battery_level
                self.client_consumed_battery[client_id] = consumed
                
                self.unique_clients_ever_used.add(client_id)
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1
                
            
            # Recharge battery for all clients
            recharged = self.clients[client_id].recharge_battery()
            #save the recharged value in this round for wandb logging in the table
            self.client_recharged_battery[client_id] = recharged
  
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
                "eligible_clients": 0,
                "fairness_jain": 0.0,
                "total_energy_consumed": 0.0,
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

        
        # Return comprehensive statistics
        return {
            "avg_battery": sum(battery_levels) / len(battery_levels),
            "min_battery": min(battery_levels),
            "eligible_clients": eligible,
            "fairness_jain": fairness_jain,
            "total_energy_consumed": total_energy,
        }