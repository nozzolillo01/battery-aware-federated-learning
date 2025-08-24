# 🔋 Battery-Aware Federated Learning

A sophisticated federated learning system that incorporates **battery-aware client selection** to optimize energy efficiency in mobile and IoT environments.

## 🌟 Features

### 🎯 **Smart Client Selection**
- **Quadratic weighting**: Selection probability ∝ `battery_level²`
- **Minimum threshold**: 20% battery requirement for participation
- **Emergency fallback**: Automatic selection of highest battery clients when needed
- **Fairness monitoring**: Tracks participation rates across all clients

### 🔬 **Realistic Battery Simulation**
- **Dynamic consumption**: 15-35% battery drain during training
- **Natural recharging**: 2-8% battery recovery when idle
- **Individual profiles**: Each client has unique consumption patterns
- **Fleet management**: Comprehensive battery health monitoring

### 🚀 **Clean Execution**
- **Silent operation**: Minimal terminal output (`Loading project configuration... Success`)
- **Complete logging**: All metrics saved to `results.json`
- **Real-time monitoring**: Wandb integration for live visualization
- **Error resilience**: Robust fallback mechanisms

## 📊 Architecture

```
app_battery/
├── my_awesome_app/
│   ├── task.py              # CNN model + Fashion-MNIST data handling
│   ├── battery_simulator.py # Battery simulation & fleet management
│   ├── my_strategy.py       # Battery-aware client selection strategy
│   ├── client_app.py        # Federated learning client
│   ├── server_app.py        # Federated learning server
│   └── __init__.py
├── pyproject.toml           # Project configuration
├── README.md
└── .gitignore
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nozzolillo01/battery-aware-federated-learning.git
cd battery-aware-federated-learning
```

2. **Create and activate a virtual environment**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

### Run Simulation
```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the federated simulation
flwr run .
```

**Expected output:**
```
Loading project configuration... 
Success
```

This minimal output confirms the application is running correctly! The simulation is executing in the background.

**Results saved to:**
- `results.json` - Complete metrics for all rounds
- Wandb dashboard - Real-time visualization (requires Wandb account)

**First-time Wandb setup:**
When running for the first time, you'll be prompted to log in to Wandb:
1. Create account at [wandb.ai](https://wandb.ai) if you don't have one
2. Generate API key at [wandb.ai/authorize](https://wandb.ai/authorize)
3. Enter `2` when prompted for "Use an existing W&B account"
4. Paste your API key when requested

## 📈 Sample Results

```json
{
  "1": {
    "loss": 0.6252,
    "accuracy": 0.7709,
    "battery_avg": 0.520,
    "battery_min": 0.357,
    "participation_rate": 0.4
  },
  "5": {
    "loss": 0.4036,
    "accuracy": 0.8499,
    "battery_avg": 0.336,
    "battery_min": 0.202,
    "participation_rate": 0.778
  }
}
```

**Key insights:**
- **Accuracy**: Improves from ~10% to ~85% over 5 rounds
- **Battery health**: Gradually decreases as expected
- **Participation**: Increases from 40% to 77.8% clients involved

## ⚙️ Configuration

### Training Parameters (`pyproject.toml`)
```toml
[tool.flwr.app.config]
num-server-rounds = 5    # Number of federated rounds
fraction-fit = 0.5       # Fraction of clients per round (50%)
local-epochs = 3         # Local training epochs per client
```

### Battery Strategy (`server_app.py`)
```python
strategy = BatteryAwareFedAvg(
    min_battery_threshold=0.2,  # 20% minimum battery
    # ... other FL parameters
)
```

### Common Issues

**1. Command not found: flwr**
```
bash: flwr: command not found
```
**Solution**: Make sure you have activated the virtual environment:
```bash
source venv/bin/activate
```

**2. Missing wandb module**
```
AttributeError: module 'wandb' has no attribute 'init'
```
**Solution**: Install wandb manually if not automatically installed:
```bash
pip install wandb
```

## 🔬 Technical Details

### Machine Learning
- **Model**: LeNet-style CNN (6→16 conv channels, 3 FC layers)
- **Dataset**: Fashion-MNIST (10 classes, federated partitioning)
- **Optimizer**: Adam with CrossEntropy loss
- **Framework**: PyTorch 2.6.0

### Federated Learning
- **Framework**: Flower 1.15.2+ with simulation mode
- **Strategy**: Extended FedAvg with battery-aware selection
- **Aggregation**: Weighted averaging based on dataset sizes
- **Evaluation**: Centralized on server with federated metrics

### Battery Simulation
- **Energy model**: Quadratic weighting for selection probability
- **Consumption**: Realistic drain during training participation
- **Recovery**: Natural recharging when idle
- **Tracking**: Comprehensive fleet statistics and participation monitoring

## 📊 Key Metrics Tracked

### Core Performance
- **`loss`**: Model training loss
- **`accuracy`**: Classification accuracy
- **`battery_avg`**: Average fleet battery level
- **`battery_min`**: Minimum battery level across fleet
- **`participation_rate`**: Percentage of clients that have participated

## 🎛️ Customization

### Modify Battery Behavior
Edit `battery_simulator.py`:
```python
consumption_rate = random.uniform(0.15, 0.35)  # Adjust drain rate
recharge_rate = random.uniform(0.02, 0.08)     # Adjust recharge rate
```

### Change Selection Strategy
Edit `my_strategy.py`:
```python
weights[client_id] = battery_level ** 2  # Try linear, cubic, etc.
```

### Adjust FL Parameters
Edit `pyproject.toml`:
```toml
num-server-rounds = 10    # More rounds
fraction-fit = 0.3        # Fewer clients per round
local-epochs = 5          # More local training
```

## 🔍 Monitoring

### Real-time (Wandb)
- Live accuracy/loss curves
- Battery level trends
- Participation rate evolution

To view the Wandb dashboard:
1. Log in to [wandb.ai](https://wandb.ai)
2. Go to your projects
3. Select the "battery-aware-fl" project
4. Click on the latest run to view real-time metrics

### Post-training (results.json)
- Complete round-by-round metrics
- Statistical summaries
- Performance trends

You can visualize the results.json file using any JSON viewer or with Python:
```python
import json
import matplotlib.pyplot as plt

# Load results
with open("results.json", "r") as f:
    results = json.load(f)

# Extract metrics
rounds = [int(r) for r in results.keys()]
accuracy = [results[str(r)]["accuracy"] for r in rounds]
battery_avg = [results[str(r)]["battery_avg"] for r in rounds]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rounds, accuracy, 'b-o', label='Accuracy')
plt.plot(rounds, battery_avg, 'g-o', label='Avg Battery')
plt.xlabel('Rounds')
plt.legend()
plt.grid(True)
plt.show()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

Apache-2.0 License

## 🏗️ Built With

- **[Flower](https://flower.ai/)** - Federated learning framework
- **[PyTorch](https://pytorch.org/)** - Machine learning library
- **[Wandb](https://wandb.ai/)** - Experiment tracking
- **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** - Dataset

---

**⚡ Optimized for energy-efficient federated learning in resource-constrained environments**
