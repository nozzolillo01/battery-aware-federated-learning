"""Battery-aware selection strategy."""

import numpy as np
from .base import SelectionRegistry


@SelectionRegistry.register("battery_aware")
def select_battery_aware(available_clients, fleet_manager, params):
    """Select clients weighted by battery level (weight = battery^alpha)."""
    if not available_clients:
        return [], {}

    alpha = float(params.get("alpha", 2.0))
    sample_fraction = float(params.get("sample_fraction", 0.5))
    min_battery_threshold = float(params.get("min_battery_threshold", 0.2))

    # Filter eligible clients by battery threshold
    eligible_ids = fleet_manager.get_eligible_clients(
        [c.cid for c in available_clients], 
        min_battery_threshold
    )
    eligible_clients = [c for c in available_clients if c.cid in eligible_ids]

    # Fallback if no eligible clients: select top 2 by battery
    if not eligible_clients:        
        levels = [
            (client, fleet_manager.get_battery_level(client.cid)) 
            for client in available_clients
        ]
        levels.sort(key=lambda item: item[1], reverse=True)
        eligible_clients = [client for client, _ in levels[:min(2, len(levels))]]
        
        if not eligible_clients:
            return [], {c.cid: 0.0 for c in available_clients}

    # Calculate battery-based weights
    weights_map = fleet_manager.calculate_selection_weights(
        [client.cid for client in eligible_clients], 
        alpha
    )
    weights = np.array(
        [weights_map.get(client.cid, 0.0) for client in eligible_clients], 
        dtype=float
    )
    
    # Ensure valid weights
    if weights.sum() <= 0:
        weights = np.ones(len(eligible_clients), dtype=float)

    # Normalize to probabilities
    probabilities = weights / weights.sum()

    # Determine number of clients to select
    desired = max(1, int(len(available_clients) * sample_fraction))
    desired = min(desired, len(eligible_clients))

    # Weighted random sampling
    indices = np.random.choice(
        len(eligible_clients), 
        size=desired, 
        replace=False, 
        p=probabilities
    )
    selected_clients = [eligible_clients[index] for index in indices]

    # Build probability map for all available clients
    probability_map = {client.cid: 0.0 for client in available_clients}
    for client, prob in zip(eligible_clients, probabilities):
        probability_map[client.cid] = float(prob)

    return selected_clients, probability_map
