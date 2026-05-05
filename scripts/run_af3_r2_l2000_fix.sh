#!/bin/bash
# Fix AF3 R2 for L2000: rerun failed seeds
# L2001: all 10 seeds (11-20), L2002: seeds 11-17 (18-20 done)
# GPU 1, XLA limited

set -uo pipefail

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
INPUT_DATA_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l2000_struct_r2/_data_inputs"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l2000_struct_r2"
MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
DOCKER_IMAGE="alphafold3_casp17"
GPU_DEVICE=0

MEMORY_LIMIT="100g"
CPUS="96.0"
SHM_SIZE="8g"

XLA_PREALLOCATE="false"
TF_UNIFIED="1"
XLA_MEM_FRACTION="0.25"

SEEDS=(11 12 13 14 15 16 17 18 19 20)

UID_VAL=$(id -u)
GID_VAL=$(id -g)

echo "=== AF3 R2 L2000 fix (seeds 11-20) ==="

for data_json in "${INPUT_DATA_DIR}"/*_data.json; do
    fname=$(basename "$data_json" _data.json)

    for seed in "${SEEDS[@]}"; do
        existing=$(find "${OUTPUT_DIR}" -name "${fname}_seed-${seed}_sample-*_model.cif" 2>/dev/null | wc -l)
        if [ "${existing}" -ge 5 ]; then
            echo "  [${fname}] seed=${seed}: skip (${existing} CIFs)"
            continue
        fi

        echo "  [${fname}] seed=${seed}: running..."
        docker run --rm \
            --gpus "\"device=${GPU_DEVICE}\"" \
            --memory="${MEMORY_LIMIT}" \
            --memory-swap="${MEMORY_LIMIT}" \
            --cpus="${CPUS}" \
            --shm-size="${SHM_SIZE}" \
            --user="${UID_VAL}:${GID_VAL}" \
            -e "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOCATE}" \
            -e "TF_FORCE_UNIFIED_MEMORY=${TF_UNIFIED}" \
            -e "XLA_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION}" \
            -v "${INPUT_DATA_DIR}:/tmp/af_input" \
            -v "${OUTPUT_DIR}:/tmp/af_output" \
            -v "${MODEL_DIR}:/tmp/models" \
            -v "${DB_DIR}:/public_databases" \
            "${DOCKER_IMAGE}" \
            python run_alphafold.py \
            --json_path="/tmp/af_input/${fname}_data.json" \
            --model_dir=/tmp/models \
            --output_dir=/tmp/af_output \
            --jax_compilation_cache_dir=/tmp/af_output/.jax_cache \
            --model_seed="${seed}" \
            --norun_data_pipeline \
            2>&1 | tee -a "${OUTPUT_DIR}/af3_r2_${fname}_fix.log"

        ret=$?
        if [ $ret -ne 0 ]; then
            echo "  [${fname}] seed=${seed}: FAILED (exit code $ret)"
        fi
    done
done

echo "=== AF3 R2 L2000 fix complete ==="
