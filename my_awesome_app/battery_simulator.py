"""Simple battery simulation for federated learning client selection."""

import random
from typing import Dict, List


class BatterySimulator:
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.battery_level = random.uniform(0.3, 1.0)
        self.consumption_rate = random.uniform(0.15, 0.35)
        self.recharge_rate = random.uniform(0.02, 0.08)
        
    def can_participate(self, min_threshold: float = 0.2) -> bool:
        return self.battery_level >= min_threshold
    
    def consume_battery(self):
        consumption = self.consumption_rate + random.uniform(-0.05, 0.05)
        self.battery_level = max(0.0, self.battery_level - consumption)
    
    def recharge_battery(self):
        recharge = self.recharge_rate + random.uniform(-0.01, 0.01)
        self.battery_level = min(1.0, self.battery_level + recharge)


class FleetManager:
    
    def __init__(self):
        self.clients: Dict[str, BatterySimulator] = {}
        self.unique_clients_ever_used = set()
        self.client_participation_count = {}
    
    def add_client(self, client_id: str) -> BatterySimulator:
        if client_id not in self.clients:
            self.clients[client_id] = BatterySimulator(client_id)
        return self.clients[client_id]
    
    def get_battery_level(self, client_id: str) -> float:
        if client_id not in self.clients:
            self.add_client(client_id)
        return self.clients[client_id].battery_level
    
    def get_eligible_clients(self, client_ids: List[str], min_threshold: float = 0.2) -> List[str]:
        eligible = []
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            if self.clients[client_id].can_participate(min_threshold):
                eligible.append(client_id)
        return eligible
    
    def calculate_selection_weights(self, client_ids: List[str]) -> Dict[str, float]:
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            battery_level = self.clients[client_id].battery_level
            weights[client_id] = battery_level ** 2
        
        return weights
    
    def update_round(self, selected_clients: List[str], all_clients: List[str]):
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            if client_id in selected_clients:
                self.clients[client_id].consume_battery()
                self.unique_clients_ever_used.add(client_id)
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1
            else:
                self.clients[client_id].recharge_battery()
    
    def get_fleet_stats(self, min_threshold: float = 0.2) -> Dict:
        if not self.clients:
            return {
                "avg_battery": 0, 
                "eligible_clients": 0,
                "unique_clients_used": 0,
                "participation_rate": 0.0,
                "active_clients_this_round": 0
            }
        
        battery_levels = [client.battery_level for client in self.clients.values()]
        eligible = sum(1 for client in self.clients.values() if client.can_participate(min_threshold))
        
        unique_used = len(self.unique_clients_ever_used)
        total_known_clients = len(self.clients)
        participation_rate = unique_used / total_known_clients if total_known_clients > 0 else 0.0
        
        return {
            "avg_battery": sum(battery_levels) / len(battery_levels),
            "min_battery": min(battery_levels),
            "max_battery": max(battery_levels),
            "eligible_clients": eligible,
            "unique_clients_used": unique_used,
            "participation_rate": participation_rate,
            "active_clients_this_round": total_known_clients,
            "client_participation_counts": dict(self.client_participation_count)
        }
    
    def get_eligible_clients(self, client_ids: List[str], min_threshold: float = 0.2) -> List[str]:
        """Get list of clients eligible for training."""
        eligible = []
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            if self.clients[client_id].can_participate(min_threshold):
                eligible.append(client_id)
        return eligible
    
    def calculate_selection_weights(self, client_ids: List[str]) -> Dict[str, float]:
        """Calculate selection weights based on battery levels."""
        weights = {}
        for client_id in client_ids:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            battery_level = self.clients[client_id].battery_level
            # Higher battery = higher selection probability
            weights[client_id] = battery_level ** 2
        
        return weights
    
    def update_round(self, selected_clients: List[str], all_clients: List[str]):
        """Update battery levels after a training round."""
        for client_id in all_clients:
            if client_id not in self.clients:
                self.add_client(client_id)
            
            if client_id in selected_clients:
                self.clients[client_id].consume_battery()
                # Track participation
                self.unique_clients_ever_used.add(client_id)
                self.client_participation_count[client_id] = self.client_participation_count.get(client_id, 0) + 1
            else:
                self.clients[client_id].recharge_battery()
    
    def get_fleet_stats(self, min_threshold: float = 0.2) -> Dict:
        """Get simple fleet statistics."""
        if not self.clients:
            return {
                "avg_battery": 0, 
                "eligible_clients": 0,
                "unique_clients_used": 0,
                "participation_rate": 0.0,
                "active_clients_this_round": 0
            }
        
        battery_levels = [client.battery_level for client in self.clients.values()]
        eligible = sum(1 for client in self.clients.values() if client.can_participate(min_threshold))
        
        # Calculate participation statistics
        unique_used = len(self.unique_clients_ever_used)
        total_known_clients = len(self.clients)
        participation_rate = unique_used / total_known_clients if total_known_clients > 0 else 0.0
        
        return {
            "avg_battery": sum(battery_levels) / len(battery_levels),
            "min_battery": min(battery_levels),
            "max_battery": max(battery_levels),
            "eligible_clients": eligible,
            "unique_clients_used": unique_used,
            "participation_rate": participation_rate,
            "active_clients_this_round": total_known_clients,
            "client_participation_counts": dict(self.client_participation_count)
        }
