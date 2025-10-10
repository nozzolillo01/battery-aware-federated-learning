"""Random subset selection strategy - uniform random sampling."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from flwr.server.client_proxy import ClientProxy

from .base import SelectionRegistry

if TYPE_CHECKING:  # pragma: no cover
    from ..battery_simulator import FleetManager


@SelectionRegistry.register("random")
def select_random(
    available_clients: List[ClientProxy],
    fleet_manager: Optional["FleetManager"],
    params: Dict[str, any],
) -> Tuple[List[ClientProxy], Dict[str, float]]:
    """Select clients uniformly at random from ALL available clients.
    
    This is the baseline selection strategy that completely ignores battery levels
    and eligibility criteria. It samples a random subset of ALL available clients,
    making it a true baseline for comparison with battery-aware strategies.
    
    Args:
        available_clients: All available clients in the round (selection pool).
        fleet_manager: Fleet manager (unused in random selection).
        params: Configuration parameters:
            - sample_fraction (float): Fraction of available clients to select (default: 0.5).
    
    Returns:
        Tuple of (selected_clients, probability_map) where:
            - selected_clients: List of randomly selected clients from ALL available.
            - probability_map: Dict mapping client_id to selection probability (1.0 if selected, 0.0 otherwise).
    
    Note:
        This strategy intentionally ignores battery levels and min_battery_threshold
        to serve as a pure random baseline. Use battery_aware for energy-conscious selection.
    """
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
