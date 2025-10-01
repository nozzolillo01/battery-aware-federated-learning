"""FedAvg strategy using random client selection."""

from __future__ import annotations

from typing import Any

from ..selection import RandomSubsetSelection
from .base import FleetAwareFedAvg


class RandomClientFedAvg(FleetAwareFedAvg):
    """Random baseline strategy that ignores battery levels."""

    def __init__(self, *args: Any, sample_fraction: float = 0.5, **kwargs: Any) -> None:
        strategy_name = kwargs.pop("strategy", "random_baseline")
        selection_strategy = RandomSubsetSelection(sample_fraction=sample_fraction)
        super().__init__(
            *args,
            selection_strategy=selection_strategy,
            strategy_name=strategy_name,
            min_battery_threshold=0.0,
            **kwargs,
        )
        self.sample_fraction = sample_fraction
