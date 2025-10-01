"""Core abstractions for client selection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:  # pragma: no cover
    from ..battery_simulator import FleetManager


class ClientSelectionStrategy(ABC):
    """Interface for strategies that select clients for a federated round."""

    @abstractmethod
    def select_clients(
        self,
        eligible_clients: List[ClientProxy],
        available_clients: List[ClientProxy],
        *,
        fleet_manager: Optional["FleetManager"] = None,
        num_clients: Optional[int] = None,
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:
        """Select a subset of clients among those available.

        Args:
            eligible_clients: Clients that satisfy the eligibility criteria.
            available_clients: All clients available for the current round.
            fleet_manager: Fleet manager instance to retrieve battery metadata.
            num_clients: Desired number of clients to sample.

        Returns:
            A tuple (selected_clients, probability_map) where:
                selected_clients: ordered list of chosen clients.
                probability_map: mapping from client id to sampling probability.
        """
        raise NotImplementedError
