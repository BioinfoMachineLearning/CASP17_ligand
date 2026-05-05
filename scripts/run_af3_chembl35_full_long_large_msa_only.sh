#!/bin/bash
# Phase-1-only MSA computation for 13 long_large targets that don't yet have
# _data.json. Runs on CPU only (no GPU). Sequential per target.
#
# After all 13 complete, use inject_long_large_ready.py (or the equivalent
# snippet in the Hellbender instructions) to inject MSA into their compound
# JSONs and ship another batch to Hellbender.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_long_large_msa_only.txt"
MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_long_large.json"
INPUT_DIR="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_inputs"
OUTPUT_DIR="$PROJECT_ROOT/outputs/alphafold3/chembl35_full"
LOGFILE="$PROJECT_ROOT/outputs/alphafold3/chembl35_full_long_large_msa_only.log"

DOCKER_IMAGE=alphafold3_casp17
GPU_DEVICE=0        # MSA pipeline is CPU-only; still need --gpus for the container
MEMORY=30g
MEMORY_SWAP=30g
CPUS=36             # jackhmmer: 4 databases × 8 threads + slack
SHM_SIZE=8g

MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
JAX_CACHE_HOST="/tmp/af3_jax_cache_chembl35_long_large_msa"

UID_VAL=$(id -u); GID_VAL=$(id -g)
mkdir -p "$OUTPUT_DIR" "$JAX_CACHE_HOST"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [long_large_msa_only] $*" | tee -a "$LOGFILE"; }

log "=== MSA-only pipeline for 13 long_large targets ==="

mapfile -t SERIES_LIST < <(awk 'NF {print $1}' "$TARGETS_FILE")
log "Targets: ${#SERIES_LIST[@]}"

for SERIES in "${SERIES_LIST[@]}"; do
    FIRST_COMPOUND=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$SERIES'][0])")
    FIRST_JSON="$INPUT_DIR/${FIRST_COMPOUND}.json"

    FOUND_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -iname "${FIRST_COMPOUND}_data.json" 2>/dev/null | head -1 || true)
    if [ -n "$FOUND_DATA" ]; then
        log "[$SERIES] MSA already exists ($FOUND_DATA), skip"
        continue
    fi

    log "[$SERIES] Running data pipeline for $FIRST_COMPOUND ..."
    if docker run --rm \
        --gpus "\"device=$GPU_DEVICE\"" \
        --memory="$MEMORY" --memory-swap="$MEMORY_SWAP" \
        --cpus="$CPUS" --shm-size="$SHM_SIZE" \
        --user="${UID_VAL}:${GID_VAL}" \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e TF_FORCE_UNIFIED_MEMORY=1 \
        -e XLA_CLIENT_MEM_FRACTION=0.3 \
        -v "$INPUT_DIR":/tmp/af_input \
        -v "$OUTPUT_DIR":/tmp/af_output \
        -v "$MODEL_DIR":/tmp/models \
        -v "$DB_DIR":/public_databases \
        -v "$JAX_CACHE_HOST":/tmp/jax_cache \
        "$DOCKER_IMAGE" \
        python run_alphafold.py \
            --json_path="/tmp/af_input/${FIRST_COMPOUND}.json" \
            --model_dir=/tmp/models \
            --output_dir=/tmp/af_output \
            --jax_compilation_cache_dir=/tmp/jax_cache \
            --model_seed=1 \
            --norun_inference >>"$LOGFILE" 2>&1; then
        log "[$SERIES] Data pipeline done"
    else
        log "[$SERIES] FAILED"
    fi
done

log "=== All done ==="
