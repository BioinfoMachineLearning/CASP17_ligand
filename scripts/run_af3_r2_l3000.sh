#!/bin/bash
# Run AF3 Round 2 for casp16_l3000_struct: seeds 11-20 × 5 samples = 50 new models/target
# Reuse existing MSA from _data.json + --norun_data_pipeline
# Uses --num_seeds=10 with modelSeeds=[11] in JSON → generates seeds 11-20
#
# Usage:
#   GPU_DEVICE=0 bash scripts/run_af3_r2_l3000.sh                    # run all targets on GPU 0
#   GPU_DEVICE=1 bash scripts/run_af3_r2_l3000.sh                    # run all targets on GPU 1
#   GPU_DEVICE=0 bash scripts/run_af3_r2_l3000.sh l3001 l3002 l3003  # run specific targets
#
# XLA memory: set XLA_MEM_FRACTION env var to override (default 0.4)

set -uo pipefail

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
ORIG_OUTPUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l3000_struct"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp16_l3000_struct_r2"
MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
DOCKER_IMAGE="${DOCKER_IMAGE:-alphafold3_new}"

# Configurable via env vars (with defaults)
GPU_DEVICE="${GPU_DEVICE:-0}"
MEMORY_LIMIT="${MEMORY_LIMIT:-100g}"
CPUS="${CPUS:-96.0}"
SHM_SIZE="${SHM_SIZE:-8g}"
XLA_PREALLOCATE="false"
TF_UNIFIED="1"
XLA_MEM_FRAC="${XLA_MEM_FRACTION:-0.4}"

NUM_SEEDS=10  # generates seeds 11-20 from modelSeeds=[11]

UID_VAL=$(id -u)
GID_VAL=$(id -g)

# Step 1: Prepare _data.json inputs (copy from original outputs, set modelSeeds=[11])
INPUT_DATA_DIR="${OUTPUT_DIR}/_data_inputs"
mkdir -p "${INPUT_DATA_DIR}"

echo "=== AF3 R2 L3000 struct (seeds 11-20, GPU ${GPU_DEVICE}, XLA_MEM=${XLA_MEM_FRAC}) ==="
echo "Docker image: ${DOCKER_IMAGE}"
echo "Copying _data.json files..."

find "${ORIG_OUTPUT_DIR}" -name "*_data.json" | while read djson; do
    fname=$(basename "$djson")
    if [ ! -f "${INPUT_DATA_DIR}/${fname}" ]; then
        cp "$djson" "${INPUT_DATA_DIR}/${fname}"
    fi
done

# Set modelSeeds=[11] and fix version for all data.json
python3 -c "
import json, glob
for f in glob.glob('${INPUT_DATA_DIR}/*.json'):
    d = json.load(open(f))
    d['modelSeeds'] = [11]
    if d.get('version', 1) > 3:
        d['version'] = 1
    json.dump(d, open(f, 'w'), indent=2)
"

TOTAL=$(ls ${INPUT_DATA_DIR}/*.json 2>/dev/null | wc -l)
echo "Prepared ${TOTAL} data.json files (modelSeeds=[11])"
echo ""

# Step 2: Determine target list
if [ $# -gt 0 ]; then
    TARGETS=("$@")
    echo "Running ${#TARGETS[@]} specified targets: ${TARGETS[*]}"
else
    TARGETS=()
    for f in "${INPUT_DATA_DIR}"/*_data.json; do
        t=$(basename "$f" _data.json)
        TARGETS+=("$t")
    done
    echo "Running all ${#TARGETS[@]} targets"
fi
echo ""

# Step 3: Run inference (one docker call per target, --num_seeds=10 generates all 10 seeds)
for target in "${TARGETS[@]}"; do
    # Check if already done (10 seeds × 5 samples = 50 CIFs)
    existing=$(find "${OUTPUT_DIR}/${target}" -name "*_model.cif" 2>/dev/null | wc -l)
    if [ "${existing}" -ge 50 ]; then
        echo "  [${target}]: skip (${existing} CIFs)"
        continue
    fi

    data_json="${INPUT_DATA_DIR}/${target}_data.json"
    if [ ! -f "${data_json}" ]; then
        echo "  [${target}]: no data.json, skipping"
        continue
    fi

    echo "  [${target}]: running (existing: ${existing} CIFs)..."
    docker run --rm \
        --gpus "\"device=${GPU_DEVICE}\"" \
        --memory="${MEMORY_LIMIT}" \
        --memory-swap="${MEMORY_LIMIT}" \
        --cpus="${CPUS}" \
        --shm-size="${SHM_SIZE}" \
        --user="${UID_VAL}:${GID_VAL}" \
        -e "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOCATE}" \
        -e "TF_FORCE_UNIFIED_MEMORY=${TF_UNIFIED}" \
        -e "XLA_CLIENT_MEM_FRACTION=${XLA_MEM_FRAC}" \
        -v "${INPUT_DATA_DIR}:/tmp/af_input" \
        -v "${OUTPUT_DIR}:/tmp/af_output" \
        -v "${MODEL_DIR}:/tmp/models" \
        -v "${DB_DIR}:/public_databases" \
        "${DOCKER_IMAGE}" \
        python run_alphafold.py \
        --json_path="/tmp/af_input/${target}_data.json" \
        --model_dir=/tmp/models \
        --output_dir=/tmp/af_output \
        --jax_compilation_cache_dir=/tmp/af_output/.jax_cache \
        --num_seeds="${NUM_SEEDS}" \
        --norun_data_pipeline \
        2>&1 | tee -a "${OUTPUT_DIR}/af3_r2_${target}.log"

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "  [${target}]: FAILED (exit code $ret)"
    fi
done

echo ""
echo "=== AF3 R2 L3000 complete ==="
