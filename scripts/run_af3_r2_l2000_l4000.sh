#!/bin/bash
# Run AF3 Round 2 for L2000 (2 targets) and L4000 (20 targets)
# Seeds 11-20 (original used 1-10), MSA reuse via _data.json + --norun_data_pipeline
# GPU 1, XLA limited to 40%

set -uo pipefail

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
DOCKER_IMAGE="alphafold3_casp17"
GPU_DEVICE=1

MEMORY_LIMIT="100g"
CPUS="96.0"
SHM_SIZE="8g"

XLA_PREALLOCATE="false"
TF_UNIFIED="1"
XLA_MEM_FRACTION="0.4"

SEEDS=(11 12 13 14 15 16 17 18 19 20)

UID_VAL=$(id -u)
GID_VAL=$(id -g)

run_af3_r2() {
    local ORIG_DIR="$1"
    local OUTPUT_DIR="$2"
    local DATASET_NAME="$3"

    mkdir -p "${OUTPUT_DIR}"
    local INPUT_DATA_DIR="${OUTPUT_DIR}/_data_inputs"
    mkdir -p "${INPUT_DATA_DIR}"

    # Copy _data.json files and remove modelSeeds
    echo "  Copying _data.json from ${ORIG_DIR}..."
    find "${ORIG_DIR}" -name "*_data.json" | while read djson; do
        fname=$(basename "$djson")
        cp -n "$djson" "${INPUT_DATA_DIR}/${fname}" 2>/dev/null
    done

    # Remove modelSeeds from all copied data.json
    python3 -c "
import json, os, glob
for f in glob.glob('${INPUT_DATA_DIR}/*.json'):
    d = json.load(open(f))
    if 'modelSeeds' in d:
        del d['modelSeeds']
        json.dump(d, open(f, 'w'), indent=2)
"

    local json_count=$(ls ${INPUT_DATA_DIR}/*.json 2>/dev/null | wc -l)
    echo "  ${DATASET_NAME}: ${json_count} targets, seeds ${SEEDS[0]}-${SEEDS[-1]}"

    for data_json in "${INPUT_DATA_DIR}"/*_data.json; do
        local fname=$(basename "$data_json" _data.json)

        for seed in "${SEEDS[@]}"; do
            existing=$(find "${OUTPUT_DIR}/${fname}" -name "seed-${seed}_sample-*_model.cif" 2>/dev/null | wc -l)
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
                2>&1 | tee -a "${OUTPUT_DIR}/af3_r2_${fname}.log"

            ret=$?
            if [ $ret -ne 0 ]; then
                echo "  [${fname}] seed=${seed}: FAILED (exit code $ret)"
            fi
        done
    done
}

echo "=== AF3 Round 2: L2000 + L4000 (GPU ${GPU_DEVICE}, seeds 11-20) ==="
echo ""

echo "--- L2000 struct (2 targets) ---"
run_af3_r2 \
    "${PROJECT_ROOT}/outputs/alphafold3/casp16_l2000_struct" \
    "${PROJECT_ROOT}/outputs/alphafold3/casp16_l2000_struct_r2" \
    "L2000"

echo ""
echo "--- L4000 (20 targets) ---"
run_af3_r2 \
    "${PROJECT_ROOT}/outputs/alphafold3/casp16_l4000" \
    "${PROJECT_ROOT}/outputs/alphafold3/casp16_l4000_r2" \
    "L4000"

echo ""
echo "=== AF3 R2 L2000+L4000 complete ==="
