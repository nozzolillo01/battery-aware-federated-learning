from typing import Dict
from flwr.server.client_proxy import ClientProxy
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..battery_simulator import FleetManager
from .base import SelectionRegistry

@SelectionRegistry.register("all_available")
def select_all_available(
    available_clients: List[ClientProxy],
    fleet_manager: Optional["FleetManager"],
    params: Dict[str, any],
) -> Tuple[List[ClientProxy], Dict[str, float]]:
    """Select all available clients, ignoring battery levels and eligibility."""
    selected_clients = available_clients
    probability_map = {client_id: 1.0 for client_id in available_clients}
    return selected_clients, probability_map