# Battery-Aware Federated Learning

Federated learning with realistic battery simulator. The goal is to study how energy impacts client selection and training performance.

## Project intent
- Simulate per-device battery consumption and energy harvesting
- Experiment with client selection policies 
- Measure accuracy, energy usage, and system fairness

## Architecture
The framework decouples client selection logic from the FL strategy and simulates battery dynamics per client.

Flow:
- Server uses a strategy (FedAvg-based) that delegates client selection to a pluggable selection policy.
- FleetManager tracks battery for all clients (consume on training, recharge per round).
- Selected clients train locally; server aggregates and proceeds to next round.



## Code structure
```
my_awesome_app/
├── battery_simulator.py      # BatterySimulator + FleetManager
├── selection/
│   ├── base.py               # SelectionRegistry (auto-discovery system)
│   ├── random_subset.py      # Built-in: random selection function
│   ├── battery_weighted.py   # Built-in: battery-aware selection function
│   └── __init__.py           # Auto-imports all selection functions
├── task.py                   # Model/dataset utilities (e.g., CIFAR-10)
├── client_app.py             # Flower client app
└── server_app.py             # Universal server (works with any selection function)
```

## Quick start
```bash
pip install -e .
flwr run .                                            # random by default
flwr run . --run-config 'selection="battery_aware"'   # battery_aware strategy
flwr run . large-simulation                           # 200 clients
```

## Config (in pyproject.toml)
- **selection**: name of selection function ("random" | "battery_aware" | your custom)
- **sample-fraction**: fraction of clients to select (default 0.5)
- **alpha**: battery weight exponent for battery_aware (default 2.0)
- **min-battery-threshold**: minimum battery to be eligible (default 0.2)

Override from CLI:
```bash
flwr run . --run-config 'selection="random"'
flwr run . --run-config 'selection="battery_aware" alpha=3.0 sample-fraction=0.3'
```

## Battery model
```
consumption = consumption_per_epoch   × local_epochs
harvesting  = harvesting_per_epoch    × local_epochs
update      = battery - consumption + harvesting (clamped to [0, 1])
```
Device classes: low_power_sensor, mid_edge_device, high_power_gateway.

## How to add a custom selection policy

**Just 1 file needed! Auto-discovered and registered.**

Create `selection/my_custom.py`:

```python
"""My custom client selection policy."""

from typing import Dict, List, Optional, Tuple
from flwr.server.client_proxy import ClientProxy
from .base import SelectionRegistry

@SelectionRegistry.register("my_custom")  # ← Auto-registers with this name
def select_my_way(
    available_clients: List[ClientProxy],
    fleet_manager,
    params: Dict[str, any],  # ← All config params passed here
) -> Tuple[List[ClientProxy], Dict[str, float]]:
    """
    Your selection logic here.
    
    Args:
        available_clients: All available clients in the round.
        fleet_manager: Access battery via fleet_manager.get_battery_level(client_id).
        params: All config parameters from pyproject.toml.
    
    Returns:
        (selected_clients, probability_map)
    """
    # Get parameters
    sample_fraction = params.get("sample_fraction", 0.5)
    min_threshold = params.get("min_battery_threshold", 0.2)
    my_param = params.get("my_param", 1.0)  # Your custom parameter
    
    # Example: Filter eligible clients, then select top-K by battery
    eligible = [
        c for c in available_clients 
        if fleet_manager.get_battery_level(c.cid) >= min_threshold
    ]
    
    if not eligible:
        return [], {c.cid: 0.0 for c in available_clients}
    
    k = max(1, int(len(available_clients) * sample_fraction))
    k = min(k, len(eligible))
    
    # Sort by battery and take top K
    client_batteries = [
        (c, fleet_manager.get_battery_level(c.cid)) for c in eligible
    ]
    client_batteries.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in client_batteries[:k]]
    
    # Build probability map
    prob_map = {
        c.cid: 1.0 / len(selected) if c in selected else 0.0 
        for c in available_clients
    }
    
    return selected, prob_map
```

**Configure in `pyproject.toml`:**
```toml
[tool.flwr.app.config]
selection = "my_custom"  # ← Your registered name
my-param = 1.5           # ← Custom parameters (passed to params dict)
sample-fraction = 0.5
```

**Run:**
```bash
flwr run . --run-config 'selection=my_custom my-param=2.0'
```

**That's it!**.

**List available strategies:**
```bash
python -c "from my_awesome_app.selection import SelectionRegistry; print(SelectionRegistry.list_all())"
# Output: ['battery_aware', 'my_custom', 'random']
```


## Monitoring
Integration with Weights & Biases (W&B): per-round accuracy/loss, battery levels, selection probabilities, and fairness.
