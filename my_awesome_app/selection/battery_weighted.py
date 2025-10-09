"""Battery-aware selection strategy."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from flwr.server.client_proxy import ClientProxy
from .base import ClientSelectionStrategy

if TYPE_CHECKING:  # pragma: no cover
    from ..battery_simulator import FleetManager


class BatteryWeightedSelection(ClientSelectionStrategy):
    """Selects clients with probability based on their battery level."""

    def __init__(
        self, 
        alpha: float = 2.0, 
        sample_fraction: float = 0.5
    ) -> None:
        self.alpha = alpha
        self.sample_fraction = sample_fraction

    def _fallback_best_available(
        self,
        available_clients: List[ClientProxy],
        fleet_manager: "FleetManager",
        num_fallback: int = 2
    ) -> List[ClientProxy]:
        """Select clients with highest battery when none meet threshold."""
        if not available_clients:
            return []
        levels = [(client, fleet_manager.get_battery_level(client.cid)) for client in available_clients]
        levels.sort(key=lambda item: item[1], reverse=True)
        return [client for client, _ in levels[:min(num_fallback, len(levels))]]

    def _build_probability_map(
        self,
        available_clients: List[ClientProxy],
        selected_pool: List[ClientProxy],
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        prob_map: Dict[str, float] = {client.cid: 0.0 for client in available_clients}
        for client, prob in zip(selected_pool, probabilities):
            prob_map[client.cid] = float(prob)
        return prob_map

    def select_clients(
        self,
        eligible_clients: List[ClientProxy],
        available_clients: List[ClientProxy],
        *,
        fleet_manager: Optional["FleetManager"] = None,
        num_clients: Optional[int] = None,
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:


        if not eligible_clients:
            eligible_clients = self._fallback_best_available(available_clients, fleet_manager, num_fallback=2)
            if not eligible_clients:
                return [], {c.cid: 0.0 for c in available_clients}

        weights_map = fleet_manager.calculate_selection_weights([client.cid for client in eligible_clients], self.alpha)
        weights = np.array([weights_map.get(client.cid, 0.0) for client in eligible_clients], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(eligible_clients), dtype=float)

        probabilities = weights / weights.sum()

        if num_clients is None:
            desired = int(len(available_clients) * self.sample_fraction)
        else:
            desired = num_clients
        desired = max(1, min(desired, len(eligible_clients)))

        indices = np.random.choice(len(eligible_clients), size=desired, replace=False, p=probabilities)
        selected_clients = [eligible_clients[index] for index in indices]
        probability_map = self._build_probability_map(available_clients, eligible_clients, probabilities)
        return selected_clients, probability_map
