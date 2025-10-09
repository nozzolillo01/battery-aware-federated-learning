# 🔋 Battery-Aware Federated Learning

Federated Learning system with battery-aware client selection, designed for large-scale simulations in mobile/IoT scenarios. It includes a realistic battery simulator, selection strategies, rich logging to Weights & Biases (W&B), and full reproducibility from `pyproject.toml`.

## 🌟 What it does (at a glance)

- Battery-based client selection (weight ∝ battery_level^α, default α=2)
- Configurable minimum battery threshold for eligibility
- Realistic per-client consumption/charging simulation
- Complete metrics: client-side training/validation and centralized server-side test
- Ready-made W&B charts: “Accuracy per round” and “Loss per round”

## 📦 Project structure

```
app_battery/
├── my_awesome_app/
│   ├── task.py              # CNN + CIFAR‑10 data loading + federated partitioning
│   ├── battery_simulator.py # Battery simulation and fleet management
│   ├── selection/           # Client selection policies (random subset, battery-weighted)
│   ├── strategies/          # Flower FedAvg variants built on selection policies
│   ├── client_app.py        # Flower client 
│   ├── server_app.py        # Flower server 
│   └── __init__.py
├── pyproject.toml           # Flower/strategy/environment configuration
└── README.md
```

## 🧠 Data and splits: what we actually measure

- Dataset: CIFAR‑10 from Hugging Face (`uoft-cs/cifar10`).
- Federated partitioning: Dirichlet with `alpha=0.1` (non‑IID) on the training data via `flwr_datasets.FederatedDataset`.
- Client-side validation: for each client, its shard is split `train/test` 80/20 (seed=42). Our “val” is therefore 20% of the client’s local training shard.
- Server-side test: centralized CIFAR‑10 “test” split (10,000 images) evaluated in `server_app.get_evaluate_fn`.

## ▶️ How to run

Requirements: Python 3.8+, pip. Optional: W&B account.

1) Install (recommended in a virtualenv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2) Start the simulation (uses `pyproject.toml` defaults)
```bash
flwr run .
```

## ⚙️ Configuration Parameters

All parameters are configured in `pyproject.toml` under `[tool.flwr.app.config]`:

### Training Parameters
- **`num-server-rounds`** (int): Number of federated learning rounds
- **`fraction-fit`** (float): Fraction of clients sampled by Flower's base FedAvg (typically 1.0)
- **`local-epochs`** (int): Number of local training epochs per client per round

### Client Selection Strategy
- **`strategy`** (int): Selection strategy to use
  - `0` = `random_baseline`: Uniform random selection (ignores battery)
  - `1` = `battery_aware`: Battery-weighted probabilistic selection

### Battery-Aware Selection Parameters (when `strategy=1`)
- **`sample-fraction`** (float, 0.0-1.0): Fraction of available clients to select per round
  - Example: With 200 clients and `sample-fraction=0.5`, selects 100 clients
  - Minimum 1 client always selected (if any available)
  - Default: `0.5` (50%)
  
- **`alpha`** (float, ≥0): Battery weight exponent for selection probability
  - Controls how strongly battery level influences selection
  - `alpha=1.0`: Linear weighting (mild preference for high battery)
  - `alpha=2.0`: Quadratic weighting (strong preference, **default**)
  - `alpha=3.0`: Cubic weighting (very strong preference)
  - Higher values = more aggressive battery-based selection
  
- **`min-battery-threshold`** (float, 0.0-1.0): Minimum battery level for participation
  - Clients below this threshold are excluded from selection
  - If no clients meet threshold, selects 2 clients with highest battery (fallback)
  - Default: `0.2` (20%)
  - Example: `0.0` = no minimum, `0.5` = only clients with ≥50% battery

### Device Classes
The simulator uses three device classes with different energy profiles:
- **`low_power_sensor`**: Low consumption (0.5-1.5%), low harvesting (0-1%)
- **`mid_edge_device`**: Medium consumption (2-3%), medium harvesting (0-2.5%)  
- **`high_power_gateway`**: High consumption (4-6%), high harvesting (0-5%)

### Federation Size
```toml
[tool.flwr.federations]
default = "small-simulation"

[tool.flwr.federations.small-simulation]
options.num-supernodes = 10

[tool.flwr.federations.medium-simulation]
options.num-supernodes = 50

[tool.flwr.federations.large-simulation]
options.num-supernodes = 200
```

### Running with Different Configurations

**Change federation size:**
```bash
flwr run . medium-simulation  # 50 clients
flwr run . large-simulation   # 200 clients
```

**Override parameters from CLI:**
```bash
flwr run . --run-config 'strategy=1 sample-fraction=0.3 alpha=3.0 num-server-rounds=25'
```

**Custom number of supernodes:**
```bash
flwr run . --num-supernodes 100
```

### Weights & Biases (W&B)
- **Enable**: Run `wandb login` for online charts
- **Disable**: Set `export WANDB_DISABLED=true`
- Note: `WANDB_SILENT=true` is set in code to reduce console noise

## 📊 Monitoring on W&B

You’ll find:
- chart/accuracy_per_round: train_accuracy_client, val_accuracy_client, test_accuracy_server
- chart/loss_per_round: train_loss_client, val_loss_client, test_loss_server
- Per-round tables with selection probabilities, battery levels, selected clients, deaths, etc.

Run names are `RANDOM_BASELINE-run-YYYY-mm-dd_HH:MM:SS` or `BATTERY_AWARE-run-...` depending on the strategy.

## 🔧 How Battery-Aware Selection Works

When `strategy=1` (battery_aware), the selection process follows these steps:

1. **Filter eligible clients**: Only clients with `battery_level >= min-battery-threshold` are eligible
2. **Calculate weights**: Each eligible client gets weight = `battery_level^alpha`
3. **Normalize probabilities**: Convert weights to probabilities that sum to 1.0
4. **Sample clients**: Select `sample-fraction * available_clients` using weighted random sampling
5. **Remove depleted**: Clients without enough battery for training are removed
6. **Train survivors**: Remaining clients train and consume battery
7. **Recharge all**: All clients (selected or not) recharge via energy harvesting

**Key insight**: Higher `alpha` creates stronger preference for high-battery clients, while `sample-fraction` controls how many clients train per round.

## 🎨 Extending the Framework

To add your own client selection strategy:

1. **Create selection policy** in `my_awesome_app/selection/`:
```python
from .base import ClientSelectionStrategy

class MyCustomSelection(ClientSelectionStrategy):
    def select_clients(self, eligible_clients, available_clients, 
                      fleet_manager=None, num_clients=None):
        # Your custom selection logic
        selected = ...  # List[ClientProxy]
        prob_map = {...}  # Dict[str, float]
        return selected, prob_map
```

2. **Create strategy** in `my_awesome_app/strategies/`:
```python
from .base import FleetAwareFedAvg
from ..selection import MyCustomSelection

class MyCustomStrategy(FleetAwareFedAvg):
    def __init__(self, *args, **kwargs):
        selection_strategy = MyCustomSelection()
        super().__init__(*args, selection_strategy=selection_strategy,
                        strategy_name="my_custom", **kwargs)
```

3. **Register in server_app.py** and add to config options

📘 **See [EXAMPLES.md](EXAMPLES.md) for complete step-by-step tutorial with code examples!**

## � Recent Changes

See [CHANGES.md](CHANGES.md) for a detailed summary of framework improvements including:
- Configurable `sample-fraction` parameter
- Comprehensive parameter documentation
- Descriptive device class names
- Complete custom strategy tutorial

## �📄 License

Apache‑2.0. See `LICENSE`.

## 🔨 Stack

- [Flower](https://flower.ai/) • [PyTorch](https://pytorch.org/) • [flwr-datasets](https://github.com/adap/flower/tree/main/baselines/flwr_datasets) • [Weights & Biases](https://wandb.ai/) • CIFAR‑10

—

Optimized for experimenting with energy‑aware federated learning strategies with clear metrics and reproducibility.