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
│   ├── base_strategy.py     # Base strategy (random-like selection with energy constraints)
│   ├── battery_strategy.py  # Battery‑aware strategy (weights ∝ b^α)
│   ├── client_app.py        # Flower client (local train + validation)
│   ├── server_app.py        # Flower server (centralized test)
│   └── __init__.py
├── pyproject.toml           # Flower/strategy/environment configuration
├── run_sweep.sh             # (opt.) sweep script
├── sweep_runs.csv           # (opt.) sweep history
└── README.md
```

## 🧠 Data and splits: what we actually measure

- Dataset: CIFAR‑10 from Hugging Face (`uoft-cs/cifar10`).
- Federated partitioning: Dirichlet with `alpha=0.1` (non‑IID) on the training data via `flwr_datasets.FederatedDataset`.
- Client-side validation: for each client, its shard is split `train/test` 80/20 (seed=42). Our “val” is therefore 20% of the client’s local training shard.
- Server-side test: centralized CIFAR‑10 “test” split (10,000 images) evaluated in `server_app.get_evaluate_fn`.

Key metrics (names as in W&B logs):
- train_accuracy_client, train_loss_client: from `aggregate_fit` (sample-weighted average across client results)
- val_accuracy_client, val_loss_client: from clients’ `aggregate_evaluate` (client key `accuracy` → renamed to `val_accuracy_client`)
- test_accuracy_server, test_loss_server: from centralized server evaluation
- Others: battery_avg, battery_min, fairness_jain, total_energy, selected_clients, deaths_clients, etc.

Why are val_accuracy_client and test_accuracy_server often similar?
- They use the same global model and the same evaluation transforms (Normalize/ToTensor).
- The aggregated client validation (weighted average over many clients) tends to approximate an “average” distribution that is close to the centralized test. Seeing similar trends is expected.

Why is train accuracy high from the very beginning?
- Shards are non‑IID (Dirichlet α=0.1) → class imbalance and easier local distributions.
- Multiple local epochs on the same shard cause local overfitting (train accuracy can exceed 0.9) while val/test remain around 0.2–0.4.

How to differentiate curves / reduce overfitting
- Reduce `local-epochs` (1–2) or increase clients per round.
- Add data augmentation for train only; keep eval with normalization only.
- Add regularization (e.g., SGD `weight_decay`) or dropout.

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
strategy = 1  # 0=base, 1=battery_aware

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

Run names are `BASE-run-YYYY-mm-dd_HH:MM:SS` or `BATTERY-run-...` depending on the strategy.

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