"""Battery-aware federated learning application package."""

from .strategies import BatteryAwareClientFedAvg, FleetAwareFedAvg, RandomClientFedAvg
from .selection import BatteryWeightedSelection, ClientSelectionStrategy, RandomSubsetSelection

__all__ = [
    "BatteryAwareClientFedAvg",
    "FleetAwareFedAvg",
    "RandomClientFedAvg",
    "ClientSelectionStrategy",
    "BatteryWeightedSelection",
    "RandomSubsetSelection",
]
