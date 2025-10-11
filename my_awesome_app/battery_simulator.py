"""Battery simulation for federated learning clients."""

import random


class BatterySimulator:
    """Simulates battery behavior for a client device."""

    DEVICE_CLASSES = {
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
    
    def __init__(self, client_id, device_class=None):
        """Initialize battery simulator."""
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

    def recharge_battery(self, local_epochs=1):
        """Recharge battery through energy harvesting."""
        previous_level = self.battery_level
        epochs = max(1, int(local_epochs))
        harvested = self.harvesting_capability * epochs
        self.battery_level = min(1.0, previous_level + harvested)
        effective_harvested = self.battery_level - previous_level
        return effective_harvested

    def can_participate(self, min_threshold=0.0):
        """Check if client has sufficient battery."""
        return self.battery_level >= min_threshold 

    def enough_battery_for_training(self, local_epochs):
        """Check if battery is sufficient for training."""
        epochs = max(1, int(local_epochs))
        needed = self.consumption_for_epochs * epochs
        return self.battery_level >= needed

    def consume_battery(self, local_epochs):
        """Consume battery for training."""
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
    """Manages a fleet of client devices with battery simulation."""
    
    def __init__(self):
        """Initialize fleet manager."""
        self.clients = {}
        self.client_participation_count = {}
        self.client_recharged_battery = {}
        self.client_consumed_battery = {}
    
    def add_client(self, client_id):
        """Add a new client."""
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]

    def get_battery_level(self, client_id):
        """Get battery level."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level

    def get_device_class(self, client_id):
        """Get device class."""
        if client_id not in self.clients:
            self.add_client(client_id)
        return getattr(self.clients[client_id], "device_class", "unknown")

    def get_dead_clients(self, selected_clients, local_epochs):
        """Get clients that ran out of battery."""
        return [
            cid for cid in selected_clients 
            if not self.clients[cid].enough_battery_for_training(local_epochs)
        ]

    def get_eligible_clients(self, client_ids, min_threshold=0.0):
        """Filter clients by battery threshold."""
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
        
        return [
            cid for cid in client_ids 
            if self.clients[cid].can_participate(min_threshold)
        ]

    def calculate_selection_weights(self, client_ids, alpha=2.0):
        """Calculate selection weights (battery^alpha)."""
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** alpha
        return weights

    def update_round(self, selected_clients, all_clients, local_epochs):
        """Update battery levels after training round."""
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
  
    def get_fleet_stats(self, min_threshold=0.0):
        """Calculate fleet statistics."""
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