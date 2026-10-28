#!/usr/bin/env bash
set -e

export VLLM_ATTENTION_BACKEND=FLASHINFER
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==============================================================================
# CONFIGURAZIONE AMBIENTI E PERCORSI PYTHON
# ==============================================================================
PYTHON_GEN="$HOME/vllm_venv/bin/python"
PYTHON_FT="$(pwd)/unsloth_venv/bin/python"

export HF_XET_HIGH_PERFORMANCE=1

# Percorsi espliciti delle cache
HF_BASE_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_BASE_CACHE}/hub"
HF_DATASETS_CACHE="${HF_BASE_CACHE}/datasets"

# ==============================================================================
# PARAMETRI GLOBALI DELLA RICERCA
# ==============================================================================
BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
BASE_TAG="Qwen2.5-Coder-7B-Instruct-AWQ"
DATA_LANG="java"
TOTAL_GENERATIONS=10

TEMPERATURES=(0.5 1.0 1.5)
RUN_IDS=(0)
HF_USER="stefanocarrera"

# ==============================================================================
# FUNZIONI DI PULIZIA PROFONDA
# ==============================================================================

# 1. Rimuove uno specifico repository da Hub (modelli o dataset)
clean_hf_repo() {
    local repo_id="$1"
    if [ -z "$repo_id" ]; then
        return
    fi

    local clean_name="${repo_id#*/}"
    local dir_pattern
    dir_pattern=$(echo "$repo_id" | sed 's/\//--/g')

    # Rimozione snapshot, blobs e lock da HF Hub
    rm -rf "${HF_HUB_CACHE}/models--${dir_pattern}" 2>/dev/null || true
    rm -rf "${HF_HUB_CACHE}/datasets--${dir_pattern}" 2>/dev/null || true
    rm -rf "${HF_HUB_CACHE}/models--${clean_name}" 2>/dev/null || true
    rm -rf "${HF_HUB_CACHE}/datasets--${clean_name}" 2>/dev/null || true
    
    # Rimuove eventuali file lock orfani associati al repo
    rm -f "${HF_HUB_CACHE}/.locks/models*${dir_pattern}*" 2>/dev/null || true
    rm -f "${HF_HUB_CACHE}/.locks/datasets*${dir_pattern}*" 2>/dev/null || true
}

# 2. Pulizia di sistema da eseguire tra una generazione e l'altra
deep_clean_disk() {
    echo ">>> [DISK DEEP CLEANUP] Svuotamento cache Arrow, lock e file temporanei..."
    
    # 1. Pulizia processi orfani su GPU per sbloccare la VRAM
    fuser -v /dev/nvidia* 2>&1 | awk '{print $NF}' | xargs -r kill -9 2>/dev/null || true

    # 2. Rimuove la cache locale Arrow dei dataset generati
    rm -rf "${HF_DATASETS_CACHE}"/* 2>/dev/null || true

    # 3. Rimuove download incompleti e lock
    rm -rf "${HF_HUB_CACHE}/.incomplete" 2>/dev/null || true
    rm -rf "${HF_HUB_CACHE}/.locks" 2>/dev/null || true

    # 4. Rimuove i modelli intermedi mantenendo SOLO il modello base Qwen3-8B
    find "${HF_HUB_CACHE}" -maxdepth 1 -name "models--*" ! -name "*${BASE_TAG}*" -exec rm -rf {} + 2>/dev/null || true

    # 5. Rimuove i dataset intermedi mantenendo SOLO il dataset originario LeetCode
    find "${HF_HUB_CACHE}" -maxdepth 1 -name "datasets--*" ! -name "*leetcode*" -exec rm -rf {} + 2>/dev/null || true

    # 6. Elimina cartelle temporanee locali
    rm -rf ./tmp_ft_checkpoint_* 2>/dev/null || true
    rm -rf /tmp/tmp* 2>/dev/null || true
    rm -rf /tmp/torch* 2>/dev/null || true
    rm -rf /tmp/triton* 2>/dev/null || true

    # 7. Sincronizzazione filesystem
    sync
}

echo "=========================================================="
echo "AVVIO RUN MATRIX: ${#TEMPERATURES[@]} Temperature x ${#RUN_IDS[@]} Run (${RUN_IDS[*]})"
echo "=========================================================="

for TEMP in "${TEMPERATURES[@]}"
do
    for run_id in "${RUN_IDS[@]}"
    do
        echo "##########################################################"
        echo ">>> AVVIO ESPERIMENTO: Temp=${TEMP} | Run ID=${run_id}"
        echo "##########################################################"

        CURRENT_MODEL="$BASE_MODEL"
        PREV_MODEL=""

        # Arriva fino a g <= TOTAL_GENERATIONS per valutare l'ultimo modello prodotto
        for (( g=0; g<=TOTAL_GENERATIONS; g++ ))
        do
            NEXT_G=$((g + 1))
            echo "----------------------------------------------------------"
            if [ "$g" -eq "$TOTAL_GENERATIONS" ]; then
                echo ">>> [Temp: ${TEMP} | Run: ${run_id}] Generazione finale solo TEST (g${g})"
            else
                echo ">>> [Temp: ${TEMP} | Run: ${run_id}] Dataset g${g} -> Modello g${NEXT_G}"
            fi
            echo "----------------------------------------------------------"

            # ------------------------------------------------------------------
            # 1. Step Generazione vLLM
            # ------------------------------------------------------------------
            echo ">>> [vLLM] Generazione campioni (g${g})..."
            $PYTHON_GEN gen_n_push.py \
                --data_lang "$DATA_LANG" \
                --model_id "$CURRENT_MODEL" \
                --temperature "$TEMP" \
                --g "$g" \
                --tot_gen "$TOTAL_GENERATIONS" \
                --run_id "$run_id" \
                --base_tag "$BASE_TAG" \
                --max_new_t 512 \
                --top_p 0.95

            TEST_DATASET_NAME="kic_D_test_${DATA_LANG}_${BASE_TAG}_t${TEMP}_g${g}_run${run_id}"

            # Pulizia immediata cache locale del dataset di test
            clean_hf_repo "$TEST_DATASET_NAME"
            clean_hf_repo "${HF_USER}/${TEST_DATASET_NAME}"

            deep_clean_disk
            sleep 3

            # Se siamo all'ultima generazione, non facciamo fine-tuning: usciamo dal loop g
            if [ "$g" -eq "$TOTAL_GENERATIONS" ]; then
                echo ">>> Raggiunta l'ultima generazione (g${g}): test set generato, FT saltato."
                break
            fi

            # ------------------------------------------------------------------
            # 2. Step Fine-Tuning
            # ------------------------------------------------------------------
            TRAIN_DATASET_NAME="kic_D_train_${DATA_LANG}_${BASE_TAG}_t${TEMP}_g${g}_run${run_id}"
            HF_TRAIN_DATASET="${HF_USER}/${TRAIN_DATASET_NAME}"
            NEXT_MODEL="${HF_USER}/kic_M_${BASE_TAG}_${DATA_LANG}_t${TEMP}_g${NEXT_G}_${run_id}"

            echo ">>> [FT] Fine-tuning su dataset g${g} per produrre modello g${NEXT_G}..."
            $PYTHON_FT ft_n_push.py \
                --base_model_id "$CURRENT_MODEL" \
                --data_id "$HF_TRAIN_DATASET" \
                --g "$NEXT_G" \
                --data_lang "$DATA_LANG" \
                --temperature "$TEMP" \
                --run_id "$run_id" \
                --base_tag "$BASE_TAG" \
                --lr 1e-5 \
                --merge_weights

            # ------------------------------------------------------------------
            # 3. Pulizia selettiva post-training
            # ------------------------------------------------------------------
            clean_hf_repo "$TRAIN_DATASET_NAME"
            clean_hf_repo "$HF_TRAIN_DATASET"

            if [ -n "$PREV_MODEL" ] && [ "$PREV_MODEL" != "$BASE_MODEL" ]; then
                clean_hf_repo "$PREV_MODEL"
            fi

            deep_clean_disk

            PREV_MODEL="$CURRENT_MODEL"
            CURRENT_MODEL="$NEXT_MODEL"

            echo ">>> Fine giro g${g}. Spazio effettivo su disco:"
            df -h / | awk 'NR==1 || NR==2'
            sleep 5
        done

        # ----------------------------------------------------------------------
        # Pulizia post-run: elimina l'ultimo modello rimasto in cache locale
        # ----------------------------------------------------------------------
        if [ "$CURRENT_MODEL" != "$BASE_MODEL" ]; then
            clean_hf_repo "$CURRENT_MODEL"
        fi
        deep_clean_disk

        echo ">>> Conclusa run ${run_id} per Temp ${TEMP}."
        echo ""
    done
done

deep_clean_disk
echo "=== TUTTE LE RUN COMPLETATE. DISCO PERFETTAMENTE PULITO ==="