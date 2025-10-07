"""Random subset selection strategy."""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from flwr.server.client_proxy import ClientProxy
from .base import ClientSelectionStrategy

if TYPE_CHECKING:  # pragma: no cover
    from ..battery_simulator import FleetManager


class RandomSubsetSelection(ClientSelectionStrategy):
    """Randomly selects a subset of eligible clients."""

    def __init__(self, sample_fraction: float = 0.5) -> None:
        self.sample_fraction = sample_fraction

    def select_clients(
        self,
        eligible_clients: List[ClientProxy],
        available_clients: List[ClientProxy],
        *,
        fleet_manager: Optional["FleetManager"] = None,  # Unused
        num_clients: Optional[int] = None,
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:
        if not eligible_clients:
            return [], {client.cid: 0.0 for client in available_clients}

        if num_clients is None:
            num_clients = max(1, int(round(len(eligible_clients) * self.sample_fraction)))
        num_clients = max(1, min(num_clients, len(eligible_clients)))

        selected_clients = random.sample(eligible_clients, k=num_clients)
        probability_map = {
            client.cid: 1.0 if client in selected_clients else 0.0 for client in available_clients
        }
        return selected_clients, probability_map
