"""Battery-aware selection strategy - weighted probabilistic sampling."""

from flwr.server.client_proxy import ClientProxy
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple

from ..battery_simulator import FleetManager
from .base import SelectionRegistry
import numpy as np

@SelectionRegistry.register("battery_aware")
def select_battery_aware(
    available_clients: List[ClientProxy],
    fleet_manager: Optional["FleetManager"],
    params: Dict[str, any],
) -> Tuple[List[ClientProxy], Dict[str, float]]:
    """Select clients with probability weighted by battery level.
    
    Clients with higher battery levels have higher probability of selection.
    The strength of this preference is controlled by the alpha parameter.
    
    This strategy filters clients by min_battery_threshold to get eligible clients,
    then applies weighted sampling. If no clients are eligible, it falls back to
    selecting the top 2 clients with highest battery levels.
    
    Args:
        available_clients: All available clients in the round.
        fleet_manager: Fleet manager to access battery levels.
        params: Configuration parameters:
            - alpha (float): Battery weight exponent (default: 2.0).
                Higher values = stronger preference for high battery.
                weight = battery_level ^ alpha
            - sample_fraction (float): Fraction of available clients to select (default: 0.5).
            - min_battery_threshold (float): Minimum battery level for eligibility (default: 0.2).
    
    Returns:
        Tuple of (selected_clients, probability_map) where:
            - selected_clients: List of selected clients (sampled by weighted probability).
            - probability_map: Dict mapping client_id to normalized selection probability.
    """
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
    probability_map: Dict[str, float] = {client.cid: 0.0 for client in available_clients}
    for client, prob in zip(eligible_clients, probabilities):
        probability_map[client.cid] = float(prob)

    return selected_clients, probability_map
