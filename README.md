# 🔋 Battery-Aware Federated Learning

Federated learning con vincoli di batteria realistici per dispositivi IoT/mobile. Obiettivo: studiare come l’energia residua influenzi la selezione dei client e le prestazioni del training.

## � Intento del progetto
- Simulare consumo e ricarica della batteria per ogni device
- Sperimentare politiche di selezione client (random vs battery-aware)
- Misurare accuratezza, consumo energetico e fairness del sistema

## 🏗️ Architettura (in breve)
```
ClientSelectionStrategy (interfaccia)
        │
        ├── RandomSubsetSelection
        └── BatteryWeightedSelection (prob ∝ battery^alpha)

FleetAwareFedAvg (strategia FedAvg che usa la policy di selezione)
BatterySimulator + FleetManager (stato energia e metriche)
```

## 📁 Struttura codice
```
my_awesome_app/
├── battery_simulator.py      # Modello energetico + gestione fleet
├── selection/                # Algoritmi di selezione (plug-in)
│   ├── base.py               # Interfaccia
│   ├── random_subset.py      # Baseline
│   └── battery_weighted.py   # Battery-aware
├── strategies/               # Varianti FedAvg
│   ├── base.py               # FleetAwareFedAvg
│   ├── random_client.py
│   └── battery_aware.py
├── task.py                   # CNN + CIFAR‑10
├── client_app.py
└── server_app.py
```

## ⚡ Quick start
```
pip install -e .
flwr run .                           # battery-aware di default
flwr run . --run-config 'strategy=0' # random baseline
flwr run . large-simulation          # 200 client
```

## 🔧 Config (principali)
- strategy: 0 (random) | 1 (battery-aware)
- alpha: forza preferenza batteria (default 2.0)
- sample-fraction: quota di client selezionati (default 0.5)
- min-battery-threshold: soglia minima per essere eleggibili (default 0.2)

Override da CLI:
```
flwr run . --run-config 'alpha=3.0 sample-fraction=0.3 min-battery-threshold=0.2'
```

## 🪫 Modello batteria (semplificato)
```
consumo   = consumo_per_epoch   × local_epochs
ricarica  = harvesting_per_epoch × local_epochs
update    = battery - consumo + ricarica (clamp 0..1)
```
Classi device: low_power_sensor, mid_edge_device, high_power_gateway.

## 🧩 Estendere con una policy custom
```
# selection/my_strategy.py
class MySelection(ClientSelectionStrategy):
    def select_clients(self, eligible, available, fleet_manager=None, num_clients=None):
        # logica custom...
        return selected, prob_map
```
Registra poi una strategia in `strategies/` che usa la tua policy e aggiungila in `server_app.py`.

## 📈 Monitoraggio
Integrato con Weights & Biases (W&B): accuracy/loss per round, livelli batteria, probabilità di selezione, fairness. 
Abilita con `wandb login` o disabilita con `export WANDB_DISABLED=true`.

—
Design: semplice, modulare, sperimentale.