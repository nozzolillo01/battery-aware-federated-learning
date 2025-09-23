#!/bin/bash

# Esegue 30 ripetizioni per ciascuna combinazione di:
# - federations: small-simulation | medium-simulation | big-simulation
# - strategy: 0 | 1
# - num-server-rounds: 5 | 10 | 50
# - local-epochs: 1 | 3 | 5
# Esempio singola run: flwr run . small-simulation --run-config "strategy=0 num-server-rounds=5 local-epochs=1"

set -euo pipefail

# Parametri configurabili via env
REPEATS=${REPEATS:-30}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-1}
LOG_CSV=${LOG_CSV:-"sweep_runs.csv"}   # per disattivare logging CSV: export LOG_CSV=""
DRY_RUN=${DRY_RUN:-0}                  # se 1, stampa soltanto i comandi senza eseguirli
RESUME=${RESUME:-0}                    # se 1, salta le run già completate (letto da LOG_CSV)

echo "Preparazione ambiente..."

# Attiva l'ambiente virtuale se esiste, oppure aggiunge ~/.local/bin al PATH
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../venv/bin/activate
elif [ -x "$HOME/.local/bin/flwr" ]; then
  export PATH="$PATH:$HOME/.local/bin"
fi

# Verifica che il comando flwr sia disponibile
if ! command -v flwr >/dev/null 2>&1; then
  echo "Errore: 'flwr' non trovato nel PATH. Attiva il venv o installa Flower (pip install flwr)."
  exit 1
fi

# Possibile filtrare via env: FEDERATIONS, STRATEGIES, ROUNDS, EPOCHS (spazio-separati)
if [ -n "${FEDERATIONS:-}" ]; then IFS=' ' read -r -a federations <<< "$FEDERATIONS"; else federations=("small-simulation" "medium-simulation" "big-simulation"); fi
if [ -n "${STRATEGIES:-}" ]; then IFS=' ' read -r -a strategies <<< "$STRATEGIES"; else strategies=(0 1); fi
if [ -n "${ROUNDS:-}" ]; then IFS=' ' read -r -a rounds_list <<< "$ROUNDS"; else rounds_list=(5 10 50); fi
if [ -n "${EPOCHS:-}" ]; then IFS=' ' read -r -a epochs_list <<< "$EPOCHS"; else epochs_list=(1 3 5); fi

total_combos=$(( ${#federations[@]} * ${#strategies[@]} * ${#rounds_list[@]} * ${#epochs_list[@]} ))
echo "Esecuzione sweep: ${total_combos} combinazioni x ${REPEATS} ripetizioni (totale $(( total_combos * REPEATS )) run)"

# Header CSV se richiesto e file non esiste
if [ -n "$LOG_CSV" ] && [ ! -f "$LOG_CSV" ]; then
  echo "timestamp,federation,strategy,num_server_rounds,local_epochs,repeat,exit_code,duration_sec" > "$LOG_CSV"
fi

combo_idx=0
for fed in "${federations[@]}"; do
  for strat in "${strategies[@]}"; do
    for rounds in "${rounds_list[@]}"; do
      for epochs in "${epochs_list[@]}"; do
        combo_idx=$((combo_idx + 1))
        echo ""
        echo "====================================================================="
        echo "Combinazione ${combo_idx}/${total_combos}: fed=${fed} strategy=${strat} rounds=${rounds} epochs=${epochs}"
        echo "====================================================================="
        echo ""

        for rep in $(seq 1 "$REPEATS"); do
          # Se RESUME attivo e la run risulta già completata nel CSV, la saltiamo
          if [ "$RESUME" -eq 1 ] && [ -n "$LOG_CSV" ] && [ -f "$LOG_CSV" ]; then
            if awk -F, -v f="${fed}" -v s="${strat}" -v r="${rounds}" -v e="${epochs}" -v rep="${rep}" '($2==f && $3==s && $4==r && $5==e && $6==rep && $7==0){found=1} END{exit (found?0:1)}' "$LOG_CSV"; then
              echo "[RESUME] già completata: fed=${fed} strategy=${strat} rounds=${rounds} epochs=${epochs} rep=${rep} → skip"
              continue
            fi
          fi

          echo "[${fed}] strategy=${strat} rounds=${rounds} epochs=${epochs} | Ripetizione ${rep}/${REPEATS}"
          echo "Comando: flwr run . ${fed} --run-config \"strategy=${strat} num-server-rounds=${rounds} local-epochs=${epochs}\""
          start_ts=$(date +%s)
          if [ "$DRY_RUN" -eq 1 ]; then
            echo "(dry-run) flwr run . ${fed} --run-config \"strategy=${strat} num-server-rounds=${rounds} local-epochs=${epochs}\""
            exit_code=0
          elif flwr run . "${fed}" --run-config "strategy=${strat} num-server-rounds=${rounds} local-epochs=${epochs}"; then
            exit_code=0
            echo "→ Run completata con successo"
          else
            exit_code=$?
            echo "→ Run terminata con errore (exit code=${exit_code})"
          fi
          end_ts=$(date +%s)
          duration=$(( end_ts - start_ts ))

          # Log CSV opzionale
          if [ -n "$LOG_CSV" ]; then
            ts_iso=$(date -Iseconds)
            echo "${ts_iso},${fed},${strat},${rounds},${epochs},${rep},${exit_code},${duration}" >> "$LOG_CSV"
          fi

          # Pausa tra le ripetizioni (se non è l'ultima)
          if [ "$rep" -lt "$REPEATS" ]; then
            sleep "$SLEEP_BETWEEN"
          fi
        done

        # Pausa breve tra combinazioni
        sleep "$SLEEP_BETWEEN"
      done
    done
  done
done

echo ""
echo "Sweep completato: ${total_combos} combinazioni x ${REPEATS} ripetizioni."
if [ -n "$LOG_CSV" ]; then
  echo "Log CSV: ${LOG_CSV}"
fi
