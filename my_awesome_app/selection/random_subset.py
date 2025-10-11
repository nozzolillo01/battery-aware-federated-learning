"""Random selection strategy."""

import random
from .base import SelectionRegistry


@SelectionRegistry.register("random")
def select_random(available_clients, fleet_manager, params):
    """Select clients uniformly at random."""
    if not available_clients:
        return [], {}

    sample_fraction = float(params.get("sample_fraction", 0.5))
    num_clients = max(1, int(len(available_clients) * sample_fraction))

    selected_clients = random.sample(available_clients, k=num_clients)
    probability_map = {
        client.cid: 1.0 if client in selected_clients else 0.0 
        for client in available_clients
    }
    
    return selected_clients, probability_map
