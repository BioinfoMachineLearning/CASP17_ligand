#!/usr/bin/env bash
# AF3 for CASP17_T targets on **daisy**: MSA once + N rounds x 50 models.
#
#   bash casp17/scripts/run_af3_daisy_casp17_T.sh T2455            # MSA + r1 r2 = 100
#   bash casp17/scripts/run_af3_daisy_casp17_T.sh T2451 r3 r4      # extra rounds, MSA reused
#   GPU=1 CPUS=8 bash casp17/scripts/run_af3_daisy_casp17_T.sh T2455
#
# Idempotent: an existing MSA data.json is reused, and any round already holding
# 50 model.cif is skipped -- safe to re-run after an interruption.
#
# Why a daisy-specific script: scripts/run_af3_casp17_{R,T,M}.sh hardcode the
# image `alphafold3_new`, which only exists on one of our hosts.
# daisy has `alphafold3-3.0.3_casp17`, whose folding_input.py is half-patched:
# _validate_keys drops 'modelSeeds' (rejects the key) while the loader below
# still requires it -> catch-22. We mount casp17/patches/folding_input_daisy.py
# (that one line un-commented) to break the deadlock.
#
# Params are the image defaults, so no flags are passed:
#   num_diffusion_samples = 5   (run_alphafold.py:349)
#   step_scale            = 1.5 (diffusion_head.py:106)
#   seeds taken verbatim from JSON modelSeeds (no --num_seeds/--model_seed,
#   which would respectively re-expand and overwrite the seed range)
#   rN -> seeds (N-1)*10+1 .. N*10, so each round is 10 seeds x 5 = 50 models.
# Diversity comes from independent seeds, NOT from more samples per seed.
#
# Memory: XLA_PYTHON_CLIENT_PREALLOCATE=false -> on-demand alloc, ~8.6 GB for a
# 2x255 aa dimer, so the card stays shareable. Do NOT
# set XLA_PYTHON_CLIENT_MEM_FRACTION: it breaks GPU init in this image.

set -euo pipefail

# AF3 databases + weights. `docker run -v` does NOT fail on a missing host path:
# it creates it as root and mounts it empty, so AF3 dies much later with an
# error that points nowhere near the real cause. Resolve and check up front.
AF3_DB_DIR="${AF3_DB_DIR:-/bmlfast/databases}"
AF3_WEIGHTS_DIR="${AF3_WEIGHTS_DIR:-/bml/Lyuwei/Alphafold3_weights}"
for _d in "$AF3_DB_DIR" "$AF3_WEIGHTS_DIR"; do
  [ -d "$_d" ] && [ -n "$(ls -A "$_d" 2>/dev/null)" ] || {
    echo "[ERR] not a non-empty directory: $_d" >&2
    echo "      set AF3_DB_DIR / AF3_WEIGHTS_DIR first (see env.example)" >&2
    exit 1
  }
done

TARGET=${1:?TARGET (e.g. T2455)}; shift || true
GPU=${GPU:-0}
CPUS=${CPUS:-8}
MSA_CPUS=${MSA_CPUS:-32}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=${AF3_IMAGE:-alphafold3-3.0.3_casp17}
PATCH="${PROJECT_ROOT}/casp17/patches/folding_input_daisy.py"
[ -f "$PATCH" ] || bash "${PROJECT_ROOT}/casp17/scripts/make_af3_patch.sh"

ROUNDS=("$@")
[ ${#ROUNDS[@]} -gt 0 ] || ROUNDS=(r1 r2)

cd "$PROJECT_ROOT"
R1_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp17_T_r1/${TARGET}"

log() { echo "[$(date '+%F %T')] $*"; }
count_cif() { find "$1" -path "*/seed-*_sample-*" -name "*_model.cif" 2>/dev/null | wc -l || true; }

# Per-target input dir wins over the series-wide one (T2451 lives in its own).
INPUT_DIR="${PROJECT_ROOT}/data/test_cases/casp17_${TARGET}/af3_inputs"
[ -f "${INPUT_DIR}/${TARGET}.json" ] || INPUT_DIR="${PROJECT_ROOT}/data/test_cases/casp17_T/af3_inputs"
[ -f "${INPUT_DIR}/${TARGET}.json" ] || { log "ERR: no ${TARGET}.json under data/test_cases/casp17_{${TARGET},T}/af3_inputs"; exit 1; }
log "input dir: $INPUT_DIR"

docker_af3() {   # caller sets: OUT_DIR, GPU_FLAG, MOUNTS_EXTRA, JSON_IN_CONTAINER, NCPU
  docker run --rm "${GPU_FLAG[@]}" \
    --cpus="${NCPU}" --memory=64g --memory-swap=64g --shm-size=8g \
    --user="$(id -u):$(id -g)" \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e TF_FORCE_UNIFIED_MEMORY=0 \
    -v "$OUT_DIR":/tmp/af_output \
    -v "$AF3_WEIGHTS_DIR":/tmp/models \
    -v "$AF3_DB_DIR":/public_databases \
    -v "/tmp/af3jax_g${GPU}":/tmp/jax_cache \
    -v "$PATCH":/app/alphafold/src/alphafold3/common/folding_input.py:ro \
    "${MOUNTS_EXTRA[@]}" \
    "$IMAGE" \
    python run_alphafold.py \
      --json_path="$JSON_IN_CONTAINER" \
      --model_dir=/tmp/models --db_dir=/public_databases \
      --output_dir=/tmp/af_output \
      --jax_compilation_cache_dir=/tmp/jax_cache \
      --force_output_dir \
      "$@"
}

flatten_inner() {   # $1=out_dir  $2=json "name" field (image always makes an inner dir)
  local inner
  for inner in "$1/$2" "$1/$(echo "$2" | tr '[:upper:]' '[:lower:]')"; do
    if [ -d "$inner" ]; then ( shopt -s dotglob; mv "$inner"/* "$1"/ ); rmdir "$inner"; break; fi
  done
}

# ================= phase 1: MSA (CPU only, once per target) =================
mkdir -p "$R1_DIR" "/tmp/af3jax_g${GPU}"
MSA_JSON=$(find "$R1_DIR" -maxdepth 2 -iname "${TARGET}_r1_data.json" 2>/dev/null | head -1)
if [ -z "$MSA_JSON" ]; then
  log "[msa] no data.json yet -> running AF3 data pipeline (CPU, ~30-60 min)"
  python3 - "${INPUT_DIR}/${TARGET}.json" "${INPUT_DIR}/${TARGET}_r1.json" "${TARGET}_r1" <<'PY'
import json, sys
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
j = json.load(open(src))
j['name'] = name
j['modelSeeds'] = list(range(1, 11))
json.dump(j, open(dst, 'w'), indent=2)
chains = [(next(iter(s)), s[next(iter(s))].get('id')) for s in j['sequences']]
print(f"  wrote {dst} (name={name}, seeds 1..10, chains={chains})")
PY
  OUT_DIR="$R1_DIR"
  GPU_FLAG=()                                   # data pipeline is jackhmmer -> CPU
  MOUNTS_EXTRA=(-v "$INPUT_DIR":/tmp/af_input)
  JSON_IN_CONTAINER="/tmp/af_input/${TARGET}_r1.json"
  NCPU="$MSA_CPUS"
  docker_af3 --norun_inference > "${PROJECT_ROOT}/outputs/alphafold3/casp17_T_r1/${TARGET}_phase1.log" 2>&1
  flatten_inner "$R1_DIR" "${TARGET}_r1"
  MSA_JSON=$(find "$R1_DIR" -maxdepth 2 -iname "${TARGET}_r1_data.json" 2>/dev/null | head -1)
  [ -n "$MSA_JSON" ] || { log "ERR: data pipeline produced no data.json"; exit 1; }
fi
log "[msa] ready: $MSA_JSON"
python3 - "$MSA_JSON" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
for s in j['sequences']:
    v = s[next(iter(s))]
    if 'unpairedMsa' in v:
        n_un = sum(1 for L in v['unpairedMsa'].split('\n') if L.startswith('>'))
        n_pa = sum(1 for L in v.get('pairedMsa', '').split('\n') if L.startswith('>'))
        print(f"  chain {v.get('id')}: unpaired {n_un} seqs, paired {n_pa}, templates {len(v.get('templates') or [])}")
PY

# ================= phase 2: inference rounds =================
for ROUND in "${ROUNDS[@]}"; do
  [[ "$ROUND" =~ ^r[0-9]+$ ]] || { log "ERR: bad round '$ROUND' (want rN)"; exit 1; }
  N=${ROUND#r}
  SEED_START=$(( (N - 1) * 10 + 1 ))
  OUT_DIR="${PROJECT_ROOT}/outputs/alphafold3/casp17_T_${ROUND}/${TARGET}"

  HAVE=$(count_cif "$OUT_DIR")
  if [ "$HAVE" = "50" ]; then log "[$ROUND] already has 50 model.cif -- skipping"; continue; fi

  mkdir -p "$OUT_DIR"
  JSON_BASE="${TARGET}_${ROUND}_data.json"
  if [ "$ROUND" = "r1" ]; then
    JSON_BASE="$(basename "$MSA_JSON")"          # phase1 output, seeds 1..10 already set
  else
    python3 - "$MSA_JSON" "${OUT_DIR}/${JSON_BASE}" "${TARGET}_${ROUND}" "$SEED_START" <<'PY'
import json, sys
src, dst, name, s0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
j = json.load(open(src))
j['name'] = name
j['modelSeeds'] = list(range(s0, s0 + 10))
json.dump(j, open(dst, 'w'), indent=2)
print(f"  wrote {dst} (name={name}, modelSeeds={j['modelSeeds']}, MSA reused)")
PY
  fi

  LOG="${PROJECT_ROOT}/outputs/alphafold3/casp17_T_${ROUND}/${TARGET}_phase2.log"
  log "[$ROUND] inference on GPU$GPU (seeds ${SEED_START}..$((SEED_START+9))) -> $LOG"
  GPU_FLAG=(--gpus "\"device=${GPU}\"")
  MOUNTS_EXTRA=()
  JSON_IN_CONTAINER="/tmp/af_output/${JSON_BASE}"
  NCPU="$CPUS"
  docker_af3 --norun_data_pipeline --num_diffusion_samples=5 > "$LOG" 2>&1
  # --force_output_dir (set in docker_af3) is load-bearing: without it AF3 sees
  # the non-empty output_dir and silently redirects to a timestamped SIBLING dir
  # outside the bind mount -> all 50 models vanish with --rm, exit 0, no error.
  flatten_inner "$OUT_DIR" "${TARGET}_${ROUND}"
  M=$(count_cif "$OUT_DIR")
  log "[$ROUND] DONE: $M model.cif"
  [ "$M" = "50" ] || log "[$ROUND] WARN: expected 50 cif, got $M"
  grep -q "output written to /tmp/af_output$" "$LOG" \
    || log "[$ROUND] WARN: '$(grep -o 'output written to .*' "$LOG" | tail -1)' -- must be exactly /tmp/af_output"
done

TOTAL=0
for ROUND in "${ROUNDS[@]}"; do
  TOTAL=$(( TOTAL + $(count_cif "${PROJECT_ROOT}/outputs/alphafold3/casp17_T_${ROUND}/${TARGET}") ))
done
log "ALL DONE: $TARGET rounds ${ROUNDS[*]} = $TOTAL models"
