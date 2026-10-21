#!/bin/bash

# Definisci i modelli
# MODELS=("unsloth/Qwen3-4B-Base-unsloth-bnb-4bit" "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit") # Aggiungi altri qui separati da spazio
# MODEL_KEYS=("qwen_4b" "qwen_8b")

MODELS=("unsloth/Qwen3-4B-Base-unsloth-bnb-4bit") # Aggiungi altri qui separati da spazio
MODEL_KEYS=("qwen_4b")


# Definisci le temperature
TEMPERATURES=(0.5 1 1.5) # Esempio: (0.2 1.0)

# Definisci le run e i seed (associati per indice)
# RUN_IDS=("run0" "run1" "run2") # Esempio: (run0 run1 run2)
# SEEDS=(42 123 999)

RUN_IDS=("run0") # Esempio: (run0 run1 run2)
SEEDS=(42)


# Ciclo sui modelli
for i in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$i]}"
    MODEL_KEY="${MODEL_KEYS[$i]}"

    # Ciclo sulle temperature
    for TEMP in "${TEMPERATURES[@]}"; do
        
        # Ciclo sulle run/seed
        for j in "${!RUN_IDS[@]}"; do
            RUN_ID="${RUN_IDS[$j]}"
            SEED="${SEEDS[$j]}"

            echo "--------------------------------------------------------"
            echo "Bash Launcher: Avvio Sessione $MODEL_KEY | T: $TEMP | $RUN_ID"
            echo "--------------------------------------------------------"

            # Esecuzione script Python come processo isolato
            python run_experiment.py \
                --model_id "$MODEL_ID" \
                --model_key "$MODEL_KEY" \
                --temp "$TEMP" \
                --run_id "$RUN_ID" \
                --seed "$SEED" \

            # Piccola pausa opzionale per far "respirare" la GPU e il filesystem
            echo "Sessione terminata. Pulizia cache in corso..."
            rm -rf runs/*
            rm -rf unsloth_compiled_cache/*
            # La cache di HuggingFace (~/.cache/huggingface/hub) ti consiglio di NON 
            # cancellarla ogni volta se i modelli sono gli stessi, altrimenti perdi 
            # tempo a riscaricarli. Se vuoi, scommenta la riga sotto:
            # rm -rf ~/.cache/huggingface/hub 
            
            sleep 5
        done
    done
done

echo "TUTTE LE SPERIMENTAZIONI SONO STATE COMPLETATE."