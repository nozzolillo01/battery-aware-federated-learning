#!/bin/bash

# Script per eseguire flwr run . per 10 volte consecutive
# aspettando che ogni esecuzione termini prima di avviarne un'altra

echo "Avvio di 10 esecuzioni sequenziali di 'flwr run .'"

# Prova ad attivare l'ambiente virtuale se esiste
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
elif [ -f ~/.local/bin/flwr ]; then
  export PATH=$PATH:~/.local/bin
fi

for i in {1..10}
do
  echo ""
  echo "==============================================="
  echo "Avvio esecuzione $i di 10..."
  echo "==============================================="
  echo ""
  
  # Esegui il comando
  echo "Esecuzione del comando: flwr run ."
  flwr run .
  
  # Verifica il codice di uscita
  if [ $? -eq 0 ]; then
    echo ""
    echo "Esecuzione $i completata con successo!"
  else
    echo ""
    echo "Esecuzione $i terminata con errore (codice: $?)"
  fi
  
  # Se non è l'ultima esecuzione, attendi un momento prima di avviare la prossima
  if [ $i -lt 10 ]; then
    echo "Preparazione per la prossima esecuzione..."
    sleep 2
  fi
done

echo ""
echo "==============================================="
echo "Tutte le 10 esecuzioni sono state completate!"
echo "==============================================="