#!/bin/bash
# Run AF3 on chembl35_A dataset (10 series, 559 compounds) on GPU 0.
# MSA reuse: run data pipeline once per series, then inference-only for all compounds.
#
# Usage: bash scripts/run_af3_chembl35_A.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="chembl35_A"
INPUT_DIR="$PROJECT_ROOT/data/test_cases/$DATASET/af3_inputs"
MSA_INPUT_DIR="$PROJECT_ROOT/data/test_cases/$DATASET/af3_inputs_msa"
OUTPUT_DIR="$PROJECT_ROOT/outputs/alphafold3/$DATASET"
MANIFEST="$PROJECT_ROOT/data/test_cases/$DATASET/af3_manifest.json"
LOGFILE="$PROJECT_ROOT/outputs/alphafold3/${DATASET}_run.log"

DOCKER_IMAGE="alphafold3_new"
GPU_DEVICE=0
NUM_SEEDS=2

# GPU 0 is A100 40GB → use half VRAM, unified memory for overflow
XLA_PREALLOCATE=true
TF_UNIFIED_MEM=1
XLA_MEM_FRACTION=0.5

# Resource limits (similar to documented config)
MEMORY=100g
MEMORY_SWAP=100g
CPUS=96.0
SHM_SIZE=8g

MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
JAX_CACHE="/tmp/af3_jax_cache_chembl35"

UID_VAL=$(id -u)
GID_VAL=$(id -g)

mkdir -p "$OUTPUT_DIR" "$MSA_INPUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

run_docker() {
    local json_path="$1"
    local out_dir="$2"
    local extra_flags="${3:-}"
    local input_dir
    input_dir="$(dirname "$json_path")"
    local json_name
    json_name="$(basename "$json_path")"

    docker run --rm \
        --gpus "\"device=$GPU_DEVICE\"" \
        --memory=$MEMORY --memory-swap=$MEMORY_SWAP \
        --cpus=$CPUS --shm-size=$SHM_SIZE \
        --user="${UID_VAL}:${GID_VAL}" \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PREALLOCATE \
        -e TF_FORCE_UNIFIED_MEMORY=$TF_UNIFIED_MEM \
        -e XLA_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION \
        -v "$input_dir":/tmp/af_input \
        -v "$out_dir":/tmp/af_output \
        -v "$MODEL_DIR":/tmp/models \
        -v "$DB_DIR":/public_databases \
        $DOCKER_IMAGE \
        python run_alphafold.py \
        --json_path="/tmp/af_input/$json_name" \
        --model_dir=/tmp/models \
        --output_dir=/tmp/af_output \
        --jax_compilation_cache_dir=/tmp/af_output/.jax_cache \
        $extra_flags
}

inject_msa() {
    # Extract MSA from reference data JSON and inject into target JSONs
    local ref_data_json="$1"
    local series_id="$2"

    python3 -c "
import json, glob, sys

ref_path = '$ref_data_json'
series_id = '$series_id'
input_dir = '$INPUT_DIR'
msa_dir = '$MSA_INPUT_DIR'

# Extract MSA from reference output
with open(ref_path) as f:
    ref = json.load(f)

msa_data = {}
for seq in ref['sequences']:
    if 'protein' in seq:
        p = seq['protein']
        for key in ('unpairedMsa', 'pairedMsa', 'templates'):
            if key in p:
                msa_data[key] = p[key]
        break

if not msa_data:
    print(f'ERROR: No MSA found in {ref_path}', file=sys.stderr)
    sys.exit(1)

print(f'  MSA keys: {list(msa_data.keys())}')

# Inject MSA into all JSONs for this series
import pathlib
jsons = sorted(pathlib.Path(input_dir).glob(f'{series_id}_*.json'))
for jp in jsons:
    with open(jp) as f:
        data = json.load(f)
    for seq in data['sequences']:
        if 'protein' in seq:
            seq['protein'].update(msa_data)
    out = pathlib.Path(msa_dir) / jp.name
    out.write_text(json.dumps(data, indent=2))

print(f'  Injected MSA into {len(jsons)} JSONs for {series_id}')
"
}

# ─── Main pipeline ───────────────────────────────────────────────
log "=== AF3 chembl35_A pipeline: 10 series, 559 compounds ==="
log "GPU: $GPU_DEVICE (A100 40GB, unified memory)"
log "Seeds: $NUM_SEEDS × 5 samples = $((NUM_SEEDS * 5)) models/compound"

# Read series from manifest
SERIES_LIST=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(sorted(m.keys())))")
log "Series: $SERIES_LIST"

for SERIES in $SERIES_LIST; do
    # Get first compound name for this series
    FIRST_COMPOUND=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m['$SERIES'][0])")
    COMPOUND_COUNT=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(len(m['$SERIES']))")

    log ""
    log "━━━ Series $SERIES ($COMPOUND_COUNT compounds) ━━━"

    # Phase 1: Run data pipeline only on first compound (to get MSA)
    FIRST_JSON="$INPUT_DIR/${FIRST_COMPOUND}.json"
    FIRST_OUTPUT_DIR="$OUTPUT_DIR"

    # Find data JSON (AF3 may create timestamped subdir or direct dir, case-preserved)
    FOUND_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -name "${FIRST_COMPOUND}_data.json" 2>/dev/null | head -1)

    if [ -n "$FOUND_DATA" ]; then
        log "[Phase 1] MSA already exists for $SERIES ($FOUND_DATA), skipping data pipeline"
        FIRST_DATA_JSON="$FOUND_DATA"
    else
        log "[Phase 1] Running data pipeline for $FIRST_COMPOUND (MSA generation)..."
        if run_docker "$FIRST_JSON" "$FIRST_OUTPUT_DIR" "--norun_inference" 2>&1 | tee -a "$LOGFILE"; then
            log "[Phase 1] Data pipeline done for $FIRST_COMPOUND"
        else
            log "[Phase 1] ERROR: Data pipeline failed for $FIRST_COMPOUND, skipping series"
            continue
        fi
        FOUND_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -name "${FIRST_COMPOUND}_data.json" 2>/dev/null | head -1)
        if [ -z "$FOUND_DATA" ]; then
            log "[Phase 1] ERROR: Cannot find data JSON for $FIRST_COMPOUND after pipeline, skipping series"
            continue
        fi
        FIRST_DATA_JSON="$FOUND_DATA"
    fi

    # Phase 2: Inject MSA into all JSONs for this series
    MSA_CHECK="$MSA_INPUT_DIR/${FIRST_COMPOUND}.json"
    if [ -f "$MSA_CHECK" ]; then
        log "[Phase 2] MSA-injected JSONs already exist for $SERIES, skipping injection"
    else
        log "[Phase 2] Injecting MSA into $COMPOUND_COUNT JSONs for $SERIES..."
        inject_msa "$FIRST_DATA_JSON" "$SERIES" 2>&1 | tee -a "$LOGFILE"
    fi

    # Phase 3: Run inference for all compounds (skip data pipeline)
    log "[Phase 3] Running inference for $COMPOUND_COUNT compounds in $SERIES..."
    COMPOUNDS=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(' '.join(m['$SERIES']))")

    for COMPOUND in $COMPOUNDS; do
        # Check if already completed — search all dirs matching this compound name
        SEED_COUNT=$(find "$OUTPUT_DIR" -maxdepth 2 -type d -name 'seed-*' -path "*${COMPOUND}*" 2>/dev/null | wc -l)
        if [ "$SEED_COUNT" -ge "$((NUM_SEEDS * 5))" ]; then
            log "  [$COMPOUND] Already done, skipping"
            continue
        fi

        MSA_JSON="$MSA_INPUT_DIR/${COMPOUND}.json"
        if [ ! -f "$MSA_JSON" ]; then
            log "  [$COMPOUND] ERROR: No MSA JSON found, skipping"
            continue
        fi

        log "  [$COMPOUND] Running inference..."
        if run_docker "$MSA_JSON" "$OUTPUT_DIR" "--norun_data_pipeline" 2>&1 | tee -a "$LOGFILE"; then
            log "  [$COMPOUND] Done"
        else
            log "  [$COMPOUND] FAILED (exit code $?)"
        fi
    done

    log "[Series $SERIES] Completed"
done

log ""
log "=== All series completed ==="
