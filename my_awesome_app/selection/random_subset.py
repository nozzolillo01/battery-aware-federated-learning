"""Random subset selection strategy - uniform random sampling."""


import random
from flwr.server.client_proxy import ClientProxy
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple

from ..battery_simulator import FleetManager
from .base import SelectionRegistry


@SelectionRegistry.register("random")
def select_random(
    available_clients: List[ClientProxy],
    fleet_manager: Optional["FleetManager"],
    params: Dict[str, any],
) -> Tuple[List[ClientProxy], Dict[str, float]]:
    """Select clients uniformly at random from ALL available clients.
    
    This is the baseline selection strategy that completely ignores battery levels
    and eligibility criteria. It samples a random subset of ALL available clients.
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
