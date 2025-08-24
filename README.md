# Client selection strategy in Federated Learning

## Strategia

- **Selezione pesata**: Probabilità ∝ battery_level²
- **Soglia minima**: 20% batteria richiesta
- **Gestione emergenza**: Fallback sui client con batteria più alta
- **Fairness**: Monitoraggio partecipazione e bilanciamento carichi


### Esecuzione Pulita
Durante l'esecuzione di `flwr run .`, il terminale rimane pulito senza output.
Tutte le informazioni vengono salvate automaticamente in un file di output:

## File Output

### results.json
```json
{
  "1": {
    "loss": 1.3196,
    "accuracy": 0.4874,
    "battery_avg": 0.525,
    "battery_min": 0.306,
    "participation_rate": 0.4,
    "clients_never_used": 8
  }
}
```

### Visualizzazione
- **Grafici Wandb**: Analisi in tempo reale durante il training
- **File JSON**: Dati completi per analisi post-training

## Come Usare

1. **Esegui simulazione**:
   ```bash
   flwr run .
   ```

2. **Visualizza risultati**:
   - Grafici in tempo reale su Wandb durante il training
   - File `results.json` con tutti i dati numerici al completamento

## Configurazione

In `pyproject.toml`:
```toml
[tool.flwr.federations.default]
options.num-server-rounds = 5  # Numero di round
options.fraction-fit = 0.5     # Frazione client per round
```

In `server_app.py`:
```python
strategy = BatteryAwareFedAvg(
    min_battery_threshold=0.2,  # 20% soglia batteria
    # ... altri parametri
)
```

## Metriche Tracciate

### Core Metrics
- **Loss/Accuracy**: Performance ML
- **Battery Health**: Avg/Min livelli batteria
- **Participation Rate**: % client che hanno partecipato almeno una volta


