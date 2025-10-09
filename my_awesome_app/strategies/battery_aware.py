"""Battery-aware FedAvg strategy implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..selection import BatteryWeightedSelection
from .base import FleetAwareFedAvg


class BatteryAwareClientFedAvg(FleetAwareFedAvg):
    """Energy-aware strategy that prioritizes clients with higher battery levels."""

    def __init__(
        self,
        *args: Any,
        alpha: float = 2.0,
        sample_fraction: float = 0.5,
        min_battery_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        strategy_name = kwargs.pop("strategy", "battery_aware")
        self.alpha = float(alpha)
        self.sample_fraction = float(sample_fraction)
        self.min_battery_threshold = float(min_battery_threshold)
        selection_strategy = BatteryWeightedSelection(
            alpha=self.alpha,
            sample_fraction=sample_fraction,
        )
        super().__init__(
            *args,
            selection_strategy=selection_strategy,
            strategy_name=strategy_name,
            min_battery_threshold=min_battery_threshold,
            **kwargs,
        )

    def _extra_wandb_config(self) -> Dict[str, Any]:
        config = super()._extra_wandb_config()
        config.update(
            {
                "min_battery_threshold": self.min_battery_threshold,
                "alpha": self.alpha,
                "sample_fraction": self.sample_fraction,
            }
        )
        return config

    def _extra_header_items(self) -> List[Tuple[str, Any]]:
        extra = super()._extra_header_items()
        extra.extend(
            [
                ("min-battery-threshold", self.min_battery_threshold),
                ("alpha", self.alpha),
                ("sample-fraction", self.sample_fraction),
            ]
        )
        return extra
