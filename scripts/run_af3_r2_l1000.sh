#!/bin/bash
# Run AF3 Round 2 for casp16_l1000: seeds 52-61 × 5 samples = 50 new models/target
# Reuse existing MSA from first wave _data.json files with --norun_data_pipeline
# GPU 1

set -uo pipefail

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
ORIG_OUTPUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l1000"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l1000_r2"
MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
DOCKER_IMAGE="alphafold3_casp17"
GPU_DEVICE=1

MEMORY_LIMIT="100g"
CPUS="96.0"
SHM_SIZE="8g"

# Limited XLA memory to coexist with Boltz2 on same GPU
XLA_PREALLOCATE="false"
TF_UNIFIED="1"
XLA_MEM_FRACTION="0.4"

SEEDS=(52 53 54 55 56 57 58 59 60 61)

UID_VAL=$(id -u)
GID_VAL=$(id -g)

mkdir -p "${OUTPUT_DIR}"

echo "=== AF3 Round 2 (L1000, 17 targets, seeds 52-61, MSA reuse) ==="
echo "Output: ${OUTPUT_DIR}"
echo ""

# Step 1: Copy _data.json (with MSA) from original outputs as input for each target
# AF3 reads the _data.json which contains pre-computed MSA, then we skip data pipeline
INPUT_DATA_DIR="${OUTPUT_DIR}/_data_inputs"
mkdir -p "${INPUT_DATA_DIR}"

for target_dir in "${ORIG_OUTPUT_DIR}"/l*/; do
    target_lower=$(basename "${target_dir}")
    data_json="${target_dir}/${target_lower}_data.json"
    if [ -f "${data_json}" ]; then
        cp -n "${data_json}" "${INPUT_DATA_DIR}/${target_lower}_data.json" 2>/dev/null
    else
        echo "WARNING: no data.json for ${target_lower}"
    fi
done

echo "Copied $(ls ${INPUT_DATA_DIR}/*.json 2>/dev/null | wc -l) data.json files"
echo ""

for data_json in "${INPUT_DATA_DIR}"/*_data.json; do
    target_lower=$(basename "${data_json}" _data.json)
    target_upper=$(echo "$target_lower" | tr '[:lower:]' '[:upper:]')

    for seed in "${SEEDS[@]}"; do
        # Check if this seed already done (5 samples)
        existing=$(find "${OUTPUT_DIR}/${target_lower}" -name "seed-${seed}_sample-*_model.cif" 2>/dev/null | wc -l)
        if [ "${existing}" -ge 5 ]; then
            echo "  [${target_upper}] seed=${seed}: skip (${existing} CIFs)"
            continue
        fi

        echo "  [${target_upper}] seed=${seed}: running..."
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
            --json_path="/tmp/af_input/${target_lower}_data.json" \
            --model_dir=/tmp/models \
            --output_dir=/tmp/af_output \
            --jax_compilation_cache_dir=/tmp/af_output/.jax_cache \
            --model_seed="${seed}" \
            --norun_data_pipeline \
            2>&1 | tee -a "${OUTPUT_DIR}/af3_r2_${target_lower}.log"

        ret=$?
        if [ $ret -ne 0 ]; then
            echo "  [${target_upper}] seed=${seed}: FAILED (exit code $ret)"
        fi
    done
done

echo ""
echo "=== AF3 R2 L1000 complete ==="
