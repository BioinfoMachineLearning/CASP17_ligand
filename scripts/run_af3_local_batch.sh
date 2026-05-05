#!/usr/bin/env bash
# Run AF3 locally via Docker on GPU 1 (A100 80GB)
# Usage: bash scripts/run_af3_local_batch.sh [start_idx] [end_idx]
#   Default: 0 99 (first 100 targets)

set -euo pipefail

cd /bmlfast/Lyuwei/0.Projects/CASP17_ligand

START_IDX=${1:-0}
END_IDX=${2:-99}

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
INPUTS_DIR="$PROJECT_ROOT/data/test_cases/casp16_l3000_struct/af3_inputs_msa"
OUTPUTS_DIR="$PROJECT_ROOT/outputs/alphafold3/casp16_l3000_struct"
WEIGHTS_DIR="/bml/Lyuwei/Alphafold3_weights"
GPU_DEVICE=1

mkdir -p "$OUTPUTS_DIR"

mapfile -t JSON_FILES < <(find "$INPUTS_DIR" -name "*.json" | sort)
TOTAL=${#JSON_FILES[@]}

echo "=== AF3 Local Batch ==="
echo "  Total targets : $TOTAL"
echo "  Running       : index $START_IDX to $END_IDX"
echo "  GPU           : $GPU_DEVICE (A100 80GB)"
echo "  Start time    : $(date)"
echo ""

DONE=0
FAIL=0

for i in $(seq "$START_IDX" "$END_IDX"); do
    if [ "$i" -ge "$TOTAL" ]; then
        echo "  Index $i >= total $TOTAL, stopping."
        break
    fi

    JSON_FILE="${JSON_FILES[$i]}"
    TARGET=$(basename "$JSON_FILE" .json)
    TARGET_OUT="$OUTPUTS_DIR/$TARGET"

    # Skip if already done
    if find "$TARGET_OUT" -name "*ranking_scores*" 2>/dev/null | grep -q .; then
        echo "[$((i+1))/$((END_IDX+1))] $TARGET — already complete, skipping."
        DONE=$((DONE + 1))
        continue
    fi

    echo "[$((i+1))/$((END_IDX+1))] $TARGET — starting at $(date '+%H:%M:%S')..."
    mkdir -p "$TARGET_OUT"

    TMP_INPUT=$(mktemp -d)
    cp "$JSON_FILE" "$TMP_INPUT/"

    if docker run --rm \
        --gpus "\"device=$GPU_DEVICE\"" \
        --memory=200g --memory-swap=200g --cpus=32.0 --shm-size=8g \
        --user="$(id -u):$(id -g)" \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e TF_FORCE_UNIFIED_MEMORY=1 \
        -e XLA_CLIENT_MEM_FRACTION=3.2 \
        -v "$TMP_INPUT":/tmp/af_input \
        -v "$TARGET_OUT":/tmp/af_output \
        -v "$WEIGHTS_DIR":/tmp/models \
        alphafold3 \
        python /app/alphafold/run_alphafold.py \
            --input_dir=/tmp/af_input \
            --model_dir=/tmp/models \
            --output_dir=/tmp/af_output \
            --norun_data_pipeline \
            --num_diffusion_samples=5 \
            --num_recycles=10 \
            --conformer_max_iterations=10000 \
            --gpu_device=0 \
            --force_output_dir=true \
        2>&1 | tee -a "logs/af3_local_${TARGET}.log"; then
        echo "  $TARGET — done at $(date '+%H:%M:%S')"
        DONE=$((DONE + 1))
    else
        echo "  $TARGET — FAILED at $(date '+%H:%M:%S')"
        FAIL=$((FAIL + 1))
    fi

    rm -rf "$TMP_INPUT"
done

echo ""
echo "=== Batch complete at $(date) ==="
echo "  Done: $DONE  Failed: $FAIL"
