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

Useful options (from `pyproject.toml`)
```toml
[tool.flwr.app.config]
num-server-rounds = 10
fraction-fit = 1.0
local-epochs = 3
min-battery-threshold = 0.2
alpha = 2
strategy = 1  # 0=random_baseline, 1=battery_aware

[tool.flwr.federations]
default = "small-simulation"

[tool.flwr.federations.small-simulation]
options.num-supernodes = 50
```

You can pick a different federation with `--federation` (e.g., `medium-simulation`). The number of simulated clients is read from `options.num-supernodes`. Alternatively, pass `--num-supernodes N` on the CLI (the server parses it in `server_app.get_num_supernodes_from_config`).

Disable W&B: export `WANDB_DISABLED=true` or run `wandb login` to have online charts. In code, `WANDB_SILENT` is set to avoid noisy console logs.

## 📊 Monitoring on W&B

You’ll find:
- chart/accuracy_per_round: train_accuracy_client, val_accuracy_client, test_accuracy_server
- chart/loss_per_round: train_loss_client, val_loss_client, test_loss_server
- Per-round tables with selection probabilities, battery levels, selected clients, deaths, etc.

Run names are `RANDOM_BASELINE-run-YYYY-mm-dd_HH:MM:SS` or `BATTERY_AWARE-run-...` depending on the strategy.

## 🔧 Quick customizations

- Strategy: change `strategy` (0 base, 1 battery‑aware); for the battery‑aware strategy, tune `alpha` (higher ⇒ more bias toward high-battery clients).
- Battery threshold: `min-battery-threshold`.
- Transforms: see `task.get_transforms()` and add augmentation for train only.

## 📄 License

Apache‑2.0. See `LICENSE`.

## 🔨 Stack

- [Flower](https://flower.ai/) • [PyTorch](https://pytorch.org/) • [flwr-datasets](https://github.com/adap/flower/tree/main/baselines/flwr_datasets) • [Weights & Biases](https://wandb.ai/) • CIFAR‑10

—

Optimized for experimenting with energy‑aware federated learning strategies with clear metrics and reproducibility.