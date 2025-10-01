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

    def __init__(self, alpha: float = 2.0, fallback_top_k: int = 2, min_selected: int = 2) -> None:
        self.alpha = alpha
        self.fallback_top_k = fallback_top_k
        self.min_selected = min_selected

    def _fallback_topk_by_battery(
        self,
        available_clients: List[ClientProxy],
        fleet_manager: "FleetManager",
    ) -> List[ClientProxy]:
        if not available_clients:
            return []
        levels = [(client, fleet_manager.get_battery_level(client.cid)) for client in available_clients]
        levels.sort(key=lambda item: item[1], reverse=True)
        top_k = self.fallback_top_k if self.fallback_top_k > 0 else len(available_clients)
        return [client for client, _ in levels[:top_k]]

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
        if fleet_manager is None:
            raise ValueError("BatteryWeightedSelection requires a FleetManager instance")

        working_clients = eligible_clients
        if not working_clients:
            working_clients = self._fallback_topk_by_battery(available_clients, fleet_manager)

        if not working_clients:
            return [], {client.cid: 0.0 for client in available_clients}

        weights_map = fleet_manager.calculate_selection_weights([client.cid for client in working_clients], self.alpha)
        weights = np.array([max(weights_map.get(client.cid, 0.0), 0.0) for client in working_clients], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(working_clients), dtype=float)

        probabilities = weights / weights.sum()

        if num_clients is None:
            desired = max(len(working_clients) // 2, self.min_selected)
        else:
            desired = num_clients
        desired = max(1, min(desired, len(working_clients)))

        indices = np.random.choice(len(working_clients), size=desired, replace=False, p=probabilities)
        selected_clients = [working_clients[index] for index in indices]
        probability_map = self._build_probability_map(available_clients, working_clients, probabilities)
        return selected_clients, probability_map
