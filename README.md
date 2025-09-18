# 🔋 Battery-Aware Federated Learning

A sophisticated federated learning system that incorporates **battery-aware client selection** to optimize energy efficiency in mobile and IoT environments. This project demonstrates how intelligent client selection strategies can enhance battery life while maintaining model performance in resource-constrained federated learning scenarios.

## 🌟 Key Features

### 🎯 Smart Client Selection
- **Quadratic Battery Weighting**: Selection probability ∝ `battery_level²` to prioritize devices with higher battery levels
- **Configurable Thresholds**: Minimum 20% battery requirement for participation (customizable)
- **Emergency Fallback**: Automatic selection of highest battery clients when needed
- **Fairness Monitoring**: Comprehensive tracking of participation rates across all clients via Jain's index

### 🔬 Realistic Battery Simulation
- **Dynamic Energy Consumption**: 15-35% battery drain during model training phases
- **Natural Recharging**: 2-8% battery recovery when clients are idle
- **Individual Device Profiles**: Each client has unique consumption patterns
- **Fleet Management**: Centralized tracking of all device battery states

### 📊 Comprehensive Monitoring
- **Complete Logging**: All metrics automatically saved to `results.json`
- **Real-time Visualization**: Weights & Biases (W&B) integration for live experiment tracking
- **Detailed Battery Analytics**: Monitor min/avg battery levels, energy consumption, and fairness metrics
- **Client Participation Analysis**: Track selection probabilities and rounds since last selection

## 📋 Project Structure

```
app_battery/
├── my_awesome_app/
│   ├── task.py              # CNN model + Fashion-MNIST/CIFAR10 data handling
│   ├── battery_simulator.py # Battery simulation & fleet management
│   ├── my_strategy.py       # Battery-aware client selection strategy
│   ├── client_app.py        # Federated learning client implementation
│   ├── server_app.py        # Federated learning server coordinator
│   └── __init__.py
├── pyproject.toml           # Project configuration
├── README.md
└── .gitignore
```

## 🧰 Technical Architecture

### Battery-Aware Selection Strategy (`my_strategy.py`)
The `BatteryAwareFedAvg` class extends Flower's `FedAvg` strategy with energy-aware client selection:

```python
# Selection probability is proportional to the square of battery level
P(client_i) = (battery_level_i^2) / Σ(battery_level_j^2)
```

#### Core Components:
- **Client Selection Pipeline**: 
  - Extract available clients → Filter by battery threshold → Apply probabilistic selection
  - Fallback to deterministic selection when necessary
  
- **Monitoring & Evaluation**: 
  - Battery metrics extraction → Results preparation → W&B logging
  - Fairness analysis using Jain's index

### Battery Simulation (`battery_simulator.py`)
Two key classes manage device energy states:

1. **BatterySimulator**: Handles individual client battery levels with:
   - Realistic discharge rates during training
   - Recovery when idle
   - Randomized consumption profiles

2. **FleetManager**: Centralized service that:
   - Tracks all client battery states
   - Identifies eligible clients
   - Calculates selection weights
   - Provides fleet-wide statistics

### Machine Learning Components (`task.py`)
- **CNN Architecture**: LeNet-style convolutional neural network
- **Datasets**: Fashion-MNIST (default) and CIFAR10 support
- **Training Logic**: Local training epochs, optimizers, and data loading

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/battery-aware-federated-learning.git
cd battery-aware-federated-learning
```

2. **Create and activate a virtual environment**
```bash
# Create virtual environment
python -m venv venv

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

### Configuration

Edit `pyproject.toml` to customize:
```toml
[tool.flwr.app.config]
num-server-rounds = 5    # Number of federated rounds
fraction-fit = 0.5       # Fraction of clients per round (50%)
local-epochs = 3         # Local training epochs per client
min-battery-threshold = 0.2 # Minimum battery level required
```

## 📈 Results Analysis

### Sample Metrics
```json
{
  "1": {
    "loss": 0.6252,
    "accuracy": 0.7709,
    "battery_avg": 0.520,
    "battery_min": 0.357,
    "fairness_jain": 0.62
  },
  "5": {
    "loss": 0.4036,
    "accuracy": 0.8499,
    "battery_avg": 0.336,
    "battery_min": 0.202,
    "fairness_jain": 0.84
  }
}
```

**Key Metrics Explained:**
- **Loss**: Model loss on centralized test set
- **Accuracy**: Classification accuracy
- **Battery Metrics**: Average and minimum battery levels across all devices
- **Fairness Index**: Jain's fairness index for client participation (1.0 = perfectly fair)

## 🔍 Advanced Monitoring

### Real-time (Weights & Biases)
- Live accuracy and loss curves
- Battery level tracking across rounds
- Fairness (Jain's index) evolution
- Client selection visualization

## 🔧 Customization Options

### Modify Battery Behavior
Edit `battery_simulator.py`:
```python
consumption_rate = random.uniform(0.15, 0.35)  # Adjust drain rate
recharge_rate = random.uniform(0.02, 0.08)     # Adjust recharge rate
```

### Adjust Federated Learning Parameters
Edit `pyproject.toml`:
```toml
num-server-rounds = 10    # More rounds
fraction-fit = 0.3        # Fewer clients per round
local-epochs = 5          # More local training
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## 🔧 Built With

- **[Flower](https://flower.ai/)** - Federated Learning framework
- **[PyTorch](https://pytorch.org/)** - Machine Learning library
- **[Weights & Biases](https://wandb.ai/)** - Experiment tracking
- **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** - Dataset

---

**⚡ Optimized for energy-efficient federated learning in resource-constrained environments**