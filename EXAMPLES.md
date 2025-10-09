# 📚 Battery-Aware FL Framework - Examples

## Basic Usage Examples

### Example 1: Random Baseline (No Battery Awareness)

```toml
# pyproject.toml
[tool.flwr.app.config]
strategy = 0                  # Random selection
sample-fraction = 0.5         # Select 50% of clients
num-server-rounds = 10
local-epochs = 5
```

**Behavior**: Selects 50% of clients uniformly at random, ignoring battery levels.

---

### Example 2: Battery-Aware with Moderate Selection Pressure

```toml
# pyproject.toml
[tool.flwr.app.config]
strategy = 1                  # Battery-aware
sample-fraction = 0.5         # Select 50% of clients
alpha = 2.0                   # Quadratic weighting (default)
min-battery-threshold = 0.2   # Only clients with ≥20% battery
num-server-rounds = 25
local-epochs = 5
```

**Behavior**: 
- Excludes clients below 20% battery
- Clients with 80% battery have ~7x higher probability than 30% battery clients
- Selects 50% of available clients using weighted probabilities

---

### Example 3: Aggressive Battery Conservation

```toml
# pyproject.toml
[tool.flwr.app.config]
strategy = 1
sample-fraction = 0.3         # Select only 30% of clients
alpha = 3.0                   # Cubic weighting (very selective)
min-battery-threshold = 0.4   # Only clients with ≥40% battery
num-server-rounds = 50
local-epochs = 3
```

**Behavior**:
- Very conservative: only high-battery clients selected
- Clients with 80% battery have ~20x higher probability than 50% clients
- Lower participation rate allows more time for recharging
- Suitable for energy-constrained scenarios

---

### Example 4: Maximum Fairness

```toml
# pyproject.toml
[tool.flwr.app.config]
strategy = 1
sample-fraction = 0.2         # Small selection fraction
alpha = 1.0                   # Linear weighting (mild preference)
min-battery-threshold = 0.1   # Very low threshold
num-server-rounds = 100       # Many rounds
local-epochs = 2              # Short training
```

**Behavior**:
- Weak battery preference (more fair across all clients)
- Low participation rate spreads load over many rounds
- Short training sessions = less energy per round
- Maximizes Jain's fairness index

---

## Creating a Custom Selection Strategy

### Step 1: Create Your Selection Logic

Create `my_awesome_app/selection/priority_based.py`:

```python
"""Priority-based selection strategy example."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from flwr.server.client_proxy import ClientProxy
from .base import ClientSelectionStrategy

if TYPE_CHECKING:
    from ..battery_simulator import FleetManager


class PriorityBasedSelection(ClientSelectionStrategy):
    """Selects clients based on longest time since last selection."""

    def __init__(self, sample_fraction: float = 0.5):
        self.sample_fraction = sample_fraction
        self.last_selected_round: Dict[str, int] = {}
        self.current_round = 0

    def select_clients(
        self,
        eligible_clients: List[ClientProxy],
        available_clients: List[ClientProxy],
        *,
        fleet_manager: Optional["FleetManager"] = None,
        num_clients: Optional[int] = None,
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:
        """Select clients who haven't been selected recently."""
        
        self.current_round += 1
        
        if not eligible_clients:
            return [], {c.cid: 0.0 for c in available_clients}

        # Calculate rounds since last selection
        priorities = []
        for client in eligible_clients:
            last_round = self.last_selected_round.get(client.cid, 0)
            rounds_waiting = self.current_round - last_round
            priorities.append((client, rounds_waiting))

        # Sort by priority (most rounds waiting first)
        priorities.sort(key=lambda x: x[1], reverse=True)

        # Determine how many to select
        if num_clients is None:
            desired = int(len(available_clients) * self.sample_fraction)
        else:
            desired = num_clients
        desired = max(1, min(desired, len(eligible_clients)))

        # Select top priority clients
        selected_clients = [client for client, _ in priorities[:desired]]

        # Update tracking
        for client in selected_clients:
            self.last_selected_round[client.cid] = self.current_round

        # Create probability map (1.0 for selected, 0.0 for others)
        probability_map = {
            c.cid: 1.0 if c in selected_clients else 0.0 
            for c in available_clients
        }

        return selected_clients, probability_map
```

### Step 2: Create Your Strategy

Create `my_awesome_app/strategies/priority_strategy.py`:

```python
"""Priority-based FedAvg strategy."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

from ..selection import PriorityBasedSelection
from .base import FleetAwareFedAvg


class PriorityFedAvg(FleetAwareFedAvg):
    """Strategy that prioritizes clients not selected recently."""

    def __init__(
        self,
        *args: Any,
        sample_fraction: float = 0.5,
        min_battery_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        strategy_name = kwargs.pop("strategy", "priority_based")
        self.sample_fraction = sample_fraction
        
        selection_strategy = PriorityBasedSelection(
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
        config.update({
            "sample_fraction": self.sample_fraction,
            "min_battery_threshold": self.min_battery_threshold,
        })
        return config

    def _extra_header_items(self) -> List[Tuple[str, Any]]:
        extra = super()._extra_header_items()
        extra.extend([
            ("sample-fraction", self.sample_fraction),
            ("min-battery-threshold", self.min_battery_threshold),
        ])
        return extra
```

### Step 3: Register in Selection Module

Update `my_awesome_app/selection/__init__.py`:

```python
"""Client selection strategy implementations."""

from .base import ClientSelectionStrategy
from .battery_weighted import BatteryWeightedSelection
from .random_subset import RandomSubsetSelection
from .priority_based import PriorityBasedSelection  # Add this

__all__ = [
    "ClientSelectionStrategy",
    "BatteryWeightedSelection",
    "RandomSubsetSelection",
    "PriorityBasedSelection",  # Add this
]
```

### Step 4: Register in Strategies Module

Update `my_awesome_app/strategies/__init__.py`:

```python
"""Federated strategy implementations."""

from .base import FleetAwareFedAvg
from .battery_aware import BatteryAwareClientFedAvg
from .random_client import RandomClientFedAvg
from .priority_strategy import PriorityFedAvg  # Add this

__all__ = [
    "FleetAwareFedAvg",
    "BatteryAwareClientFedAvg",
    "RandomClientFedAvg",
    "PriorityFedAvg",  # Add this
]
```

### Step 5: Add to Server Configuration

Update `my_awesome_app/server_app.py` in the `server_fn` function:

```python
def server_fn(context: Context) -> ServerAppComponents:
    # ... existing code ...
    
    strategy_id = context.run_config.get("strategy", 0)
    
    if strategy_id == 1:
        # Battery-aware strategy
        strategy_impl = BatteryAwareClientFedAvg(...)
    elif strategy_id == 2:  # Add this
        # Priority-based strategy
        strategy_impl = PriorityFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_weighted_average,
            evaluate_fn=lambda sr, pn, c: evaluate_global_model(sr, pn, c, testloader, "cpu"),
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            num_supernodes=num_supernodes,
            sample_fraction=sample_fraction,
            min_battery_threshold=min_battery_threshold,
        )
    else:
        # Random baseline
        strategy_impl = RandomClientFedAvg(...)
    
    # ... rest of code ...
```

### Step 6: Update Configuration

```toml
# pyproject.toml
[tool.flwr.app.config]
strategy = 2                  # 0=random, 1=battery_aware, 2=priority_based
sample-fraction = 0.5
min-battery-threshold = 0.2
```

### Step 7: Run Your Custom Strategy

```bash
flwr run . --run-config 'strategy=2 sample-fraction=0.4'
```

---

## Tips for Custom Strategies

1. **Always inherit from `ClientSelectionStrategy`** for selection logic
2. **Always inherit from `FleetAwareFedAvg`** for FL strategy
3. **Use `fleet_manager`** to access battery levels and device info
4. **Return probability map** for logging/analysis
5. **Handle edge cases**: empty eligible list, all clients depleted, etc.
6. **Test with small simulations** first (10-50 clients)
7. **Monitor W&B metrics**: fairness, energy, deaths per round

---

## Performance Tuning

### High Throughput (Fast Training)
```toml
sample-fraction = 0.8         # Select many clients
alpha = 1.5                   # Mild preference
local-epochs = 3              # Short training
```

### Energy Efficiency (Preserve Battery)
```toml
sample-fraction = 0.3         # Select few clients
alpha = 3.0                   # Strong preference
min-battery-threshold = 0.5   # High minimum
local-epochs = 2              # Very short training
```

### Maximum Fairness (Equal Participation)
```toml
strategy = 0                  # Random selection
sample-fraction = 0.2         # Low participation
local-epochs = 1              # Minimal energy per client
num-server-rounds = 200       # Spread over many rounds
```

### Balanced (Recommended Starting Point)
```toml
strategy = 1
sample-fraction = 0.5
alpha = 2.0
min-battery-threshold = 0.2
local-epochs = 5
num-server-rounds = 25
```
