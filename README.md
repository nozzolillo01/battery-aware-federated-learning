# Battery-Aware Federated Learning

Federated learning with realistic battery constraints for IoT/mobile devices. The goal is to study how residual energy impacts client selection and training performance.

## Project intent
- Simulate per-device battery consumption and energy harvesting
- Experiment with client selection policies (random vs battery-aware)
- Measure accuracy, energy usage, and system fairness

## Architecture
The framework decouples client selection logic from the FL strategy and simulates battery dynamics per client.

Flow:
- Server uses a strategy (FedAvg-based) that delegates client selection to a pluggable selection policy.
- FleetManager tracks battery for all clients (consume on training, recharge per round).
- Selected clients train locally; server aggregates and proceeds to next round.

High-level diagram:
```
Flower Server (FedAvg)
   └─ Strategy (FleetAwareFedAvg)
        ├─ Selection Policy (pluggable)
        └─ FleetManager (battery state)
Clients (train/eval) ←──────────────┘
```

## Code structure
```
my_awesome_app/
├── battery_simulator.py      # BatterySimulator + FleetManager
├── selection/
│   ├── base.py               # ClientSelectionStrategy interface
│   ├── random_subset.py      # Uniform random selection
│   └── battery_weighted.py   # Battery-aware weighted selection
├── strategies/
│   ├── base.py               # FedAvg wrapper integrating selection + fleet
│   ├── random_client.py      # FedAvg + random selection
│   └── battery_aware.py      # FedAvg + battery-aware selection
├── task.py                   # Model/dataset utilities (e.g., CIFAR-10)
├── client_app.py             # Flower client app
└── server_app.py             # Flower server app
```

## Quick start
```
pip install -e .
flwr run .                           # battery-aware by default
flwr run . --run-config 'strategy=0' # random baseline
flwr run . large-simulation          # 200 clients
```

## Config (in pyproject.toml)
- strategy: 0 (random) | 1 (battery-aware)
- alpha: strength of battery preference (default 2.0)
- sample-fraction: fraction of clients to select (default 0.5)
- min-battery-threshold: minimum battery to be eligible (default 0.2)

Override from CLI:
```
flwr run . --run-config 'alpha=3.0 sample-fraction=0.3 min-battery-threshold=0.2'
```

## Battery model
```
consumption = consumption_per_epoch   × local_epochs
harvesting  = harvesting_per_epoch    × local_epochs
update      = battery - consumption + harvesting (clamped to [0, 1])
```
Device classes: low_power_sensor, mid_edge_device, high_power_gateway.

## How to add a custom policy for client selection

***Step 1: Create your selection policy***

Implement `ClientSelectionStrategy` in `selection/my_policy.py`:

```python
from typing import Dict, List, Optional, Tuple
from flwr.server.client_proxy import ClientProxy
from .base import ClientSelectionStrategy

class MyCustomSelection(ClientSelectionStrategy):
    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param
    
    def select_clients(
        self,
        eligible_clients: List[ClientProxy],
        available_clients: List[ClientProxy],
        *,
        fleet_manager=None,
        num_clients: Optional[int] = None,
    ) -> Tuple[List[ClientProxy], Dict[str, float]]:
        """Your selection logic here. Access battery via fleet_manager.get_battery(cid)."""
        k = num_clients or max(1, int(0.5 * len(available_clients)))
        k = min(k, len(eligible_clients))
        
        # Example: select top-K by battery level
        client_batteries = [(c, fleet_manager.get_battery(c.cid)) for c in eligible_clients]
        client_batteries.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in client_batteries[:k]]
        
        prob_map = {c.cid: 1.0 / len(selected) for c in selected} if selected else {}
        return selected, prob_map
  ```
           
 ***Step 2: Create a strategy using your policy***

 Create strategies/my_strategy.py:
 ```python
from .base import FleetAwareFedAvg
from ..selection.my_policy import MyCustomSelection

class MyCustomFedAvg(FleetAwareFedAvg):
    def __init__(self, *args, my_param: float = 1.0, **kwargs):
        selection_strategy = MyCustomSelection(my_param=my_param)
        super().__init__(*args, selection_strategy=selection_strategy, strategy_name="my_custom", **kwargs)
 ```

***Step 3: Register in server_app.py***
```python
from .strategies.my_strategy import MyCustomFedAvg

strategy_map = {
    0: RandomClientFedAvg,
    1: BatteryAwareClientFedAvg,
    2: MyCustomFedAvg,  # Add this
}
 ```

 ***Step 4: run***
  ```flwr run . --run-config 'strategy=2 my-param=1.5' ```


## Monitoring
Integration with Weights & Biases (W&B): per-round accuracy/loss, battery levels, selection probabilities, and fairness.
