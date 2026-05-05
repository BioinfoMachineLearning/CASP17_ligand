#!/bin/bash
# Parametric AF3 runner for chembl35_full subsets (short/long).
#
# Required env vars:
#   DATASET_TAG        e.g. chembl35_full_short | chembl35_full_long
#   TARGETS_FILE       TSV "target_id<TAB>seq_len<TAB>num_compounds"
#                      — iteration order follows the file, so pre-sort it
#                        (short→long for GPU0, long→short for GPU1).
#   MANIFEST           JSON {target_id: [compound_names,...]}
#   GPU_DEVICE         0 | 1
#   XLA_PREALLOCATE    true | false
#   TF_UNIFIED_MEM     0 | 1
#   XLA_MEM_FRACTION   e.g. 0.5 | 0.9
#   MEMORY             e.g. 80g
#   CPUS               e.g. 60
# Optional:
#   NUM_SEEDS          default 2  (→ 2×5=10 conformers/compound)
#   SHM_SIZE           default 8g
#   DOCKER_IMAGE       default alphafold3_new
#
set -euo pipefail

: "${DATASET_TAG:?}"
: "${TARGETS_FILE:?}"
: "${MANIFEST:?}"
: "${GPU_DEVICE:?}"
: "${XLA_PREALLOCATE:?}"
: "${TF_UNIFIED_MEM:?}"
: "${XLA_MEM_FRACTION:?}"
: "${MEMORY:?}"
: "${CPUS:?}"
NUM_SEEDS="${NUM_SEEDS:-2}"
SHM_SIZE="${SHM_SIZE:-8g}"
DOCKER_IMAGE="${DOCKER_IMAGE:-alphafold3_new}"
MEMORY_SWAP="${MEMORY_SWAP:-$MEMORY}"
# Old images (alphafold3_new / alphafold3) reject --model_seed and require
# modelSeeds in the JSON; newer images (alphafold3_casp17) require the CLI flag.
# Override to empty string ("" ) when running against the old image.
MODEL_SEED_FLAG="${MODEL_SEED_FLAG---model_seed=1}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Shared dataset directory: MSA cache lives under chembl35_full so both halves
# can (in principle) share the data pipeline output per target.
DATASET_ROOT="chembl35_full"
INPUT_DIR="$PROJECT_ROOT/data/test_cases/$DATASET_ROOT/af3_inputs"
MSA_INPUT_DIR="$PROJECT_ROOT/data/test_cases/$DATASET_ROOT/af3_inputs_msa"
OUTPUT_DIR="$PROJECT_ROOT/outputs/alphafold3/$DATASET_ROOT"
LOGFILE="$PROJECT_ROOT/outputs/alphafold3/${DATASET_TAG}_run.log"

MODEL_DIR="/bml/Lyuwei/Alphafold3_weights"
DB_DIR="/bmlfast/databases"
JAX_CACHE_HOST="/tmp/af3_jax_cache_${DATASET_TAG}"

UID_VAL=$(id -u); GID_VAL=$(id -g)
mkdir -p "$OUTPUT_DIR" "$MSA_INPUT_DIR" "$JAX_CACHE_HOST"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$DATASET_TAG] $*" | tee -a "$LOGFILE"; }

run_docker() {
    local json_path="$1"
    local out_dir="$2"
    local extra_flags="${3:-}"
    local input_dir; input_dir="$(dirname "$json_path")"
    local json_name; json_name="$(basename "$json_path")"

    docker run --rm \
        --gpus "\"device=$GPU_DEVICE\"" \
        --memory="$MEMORY" --memory-swap="$MEMORY_SWAP" \
        --cpus="$CPUS" --shm-size="$SHM_SIZE" \
        --user="${UID_VAL}:${GID_VAL}" \
        -e XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PREALLOCATE" \
        -e TF_FORCE_UNIFIED_MEMORY="$TF_UNIFIED_MEM" \
        -e XLA_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION" \
        -v "$input_dir":/tmp/af_input \
        -v "$out_dir":/tmp/af_output \
        -v "$MODEL_DIR":/tmp/models \
        -v "$DB_DIR":/public_databases \
        -v "$JAX_CACHE_HOST":/tmp/jax_cache \
        "$DOCKER_IMAGE" \
        python run_alphafold.py \
        --json_path="/tmp/af_input/$json_name" \
        --model_dir=/tmp/models \
        --output_dir=/tmp/af_output \
        --jax_compilation_cache_dir=/tmp/jax_cache \
        $extra_flags
}

inject_msa() {
    local ref_data_json="$1" series_id="$2"
    python3 - <<PY
import json, pathlib, sys
ref = json.load(open("$ref_data_json"))
msa = {}
for s in ref["sequences"]:
    if "protein" in s:
        for k in ("unpairedMsa","pairedMsa","templates"):
            if k in s["protein"]:
                msa[k] = s["protein"][k]
        break
if not msa:
    print("ERROR: no MSA in $ref_data_json", file=sys.stderr); sys.exit(1)
input_dir = pathlib.Path("$INPUT_DIR")
msa_dir   = pathlib.Path("$MSA_INPUT_DIR")
msa_dir.mkdir(parents=True, exist_ok=True)
jsons = sorted(input_dir.glob("${series_id}_*.json"))
for jp in jsons:
    data = json.load(open(jp))
    for s in data["sequences"]:
        if "protein" in s:
            s["protein"].update(msa)
    (msa_dir / jp.name).write_text(json.dumps(data, indent=2))
print(f"  injected MSA into {len(jsons)} JSONs for $series_id")
PY
}

log "=== AF3 $DATASET_TAG pipeline ==="
log "GPU=$GPU_DEVICE  XLA_MEM_FRACTION=$XLA_MEM_FRACTION  UNIFIED_MEM=$TF_UNIFIED_MEM"
log "MEMORY=$MEMORY  CPUS=$CPUS  NUM_SEEDS=$NUM_SEEDS ($((NUM_SEEDS*5)) models/compound)"
log "TARGETS_FILE=$TARGETS_FILE"

# Read target list in file order
mapfile -t SERIES_LIST < <(awk 'NF {print $1}' "$TARGETS_FILE")
log "Targets: ${#SERIES_LIST[@]}"

for SERIES in "${SERIES_LIST[@]}"; do
    COMPOUND_COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))['$SERIES']))")
    FIRST_COMPOUND=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$SERIES'][0])")

    log ""
    log "━━━ Series $SERIES (${COMPOUND_COUNT} compounds) ━━━"

    # Phase 1: data pipeline once per series
    FIRST_JSON="$INPUT_DIR/${FIRST_COMPOUND}.json"
    # AF3 writes outputs with the compound name lowercased — use -iname.
    FOUND_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -iname "${FIRST_COMPOUND}_data.json" 2>/dev/null | head -1 || true)

    if [ -n "$FOUND_DATA" ]; then
        log "[Phase 1] MSA already exists ($FOUND_DATA), skip data pipeline"
        FIRST_DATA_JSON="$FOUND_DATA"
    else
        log "[Phase 1] Running data pipeline for $FIRST_COMPOUND"
        if run_docker "$FIRST_JSON" "$OUTPUT_DIR" "--norun_inference $MODEL_SEED_FLAG" >>"$LOGFILE" 2>&1; then
            log "[Phase 1] Data pipeline done"
        else
            log "[Phase 1] ERROR: data pipeline failed for $FIRST_COMPOUND, skip series"
            continue
        fi
        FOUND_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -iname "${FIRST_COMPOUND}_data.json" 2>/dev/null | head -1 || true)
        if [ -z "$FOUND_DATA" ]; then
            log "[Phase 1] ERROR: no data JSON after pipeline, skip series"
            continue
        fi
        FIRST_DATA_JSON="$FOUND_DATA"
    fi

    # Phase 2: inject MSA
    MSA_CHECK="$MSA_INPUT_DIR/${FIRST_COMPOUND}.json"
    if [ -f "$MSA_CHECK" ]; then
        log "[Phase 2] MSA-injected JSONs already exist, skip"
    else
        log "[Phase 2] Injecting MSA into $COMPOUND_COUNT JSONs"
        inject_msa "$FIRST_DATA_JSON" "$SERIES" 2>&1 | tee -a "$LOGFILE"
    fi

    # Phase 3: inference per compound (skip data pipeline)
    log "[Phase 3] Running inference for $COMPOUND_COUNT compounds"
    COMPOUNDS=$(python3 -c "import json; print(' '.join(json.load(open('$MANIFEST'))['$SERIES']))")
    EXPECT=$((NUM_SEEDS * 5))
    for COMPOUND in $COMPOUNDS; do
        # Flat output layout: AF3 writes "<lower>_ranking_scores.csv" at OUTPUT_DIR
        # top level only after all seeds × samples finish. Its existence is a
        # reliable "compound complete" marker (case-insensitive via -iname).
        if find "$OUTPUT_DIR" -maxdepth 1 -iname "${COMPOUND}_ranking_scores.csv" 2>/dev/null | grep -q .; then
            log "  [$COMPOUND] already done, skip"
            continue
        fi

        MSA_JSON="$MSA_INPUT_DIR/${COMPOUND}.json"
        if [ ! -f "$MSA_JSON" ]; then
            log "  [$COMPOUND] ERROR: no MSA JSON, skip"
            continue
        fi

        log "  [$COMPOUND] inference..."
        if run_docker "$MSA_JSON" "$OUTPUT_DIR" "--norun_data_pipeline $MODEL_SEED_FLAG --num_seeds=$NUM_SEEDS" >>"$LOGFILE" 2>&1; then
            log "  [$COMPOUND] done"
        else
            log "  [$COMPOUND] FAILED"
        fi
    done
    log "[Series $SERIES] completed"
done

log ""
log "=== $DATASET_TAG finished ==="
