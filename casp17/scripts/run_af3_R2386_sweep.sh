#!/usr/bin/env bash
# AF3 inference for one R2386 ion-stoichiometry point (solvent-model target).
#
# Usage:
#   bash casp17/scripts/run_af3_R2386_sweep.sh <POINT> <GPU> [CPUS]
#   e.g. bash casp17/scripts/run_af3_R2386_sweep.sh S1 0
#
# Prerequisite: phase 1 (RNA MSA search) must have produced
#   /bml/Lyuwei/alphafold_results/casp17_R_R2386/msa/*_data.json
# Run it once with R2386_msa.json + --norun_inference; all four S points share it.
#
# This script builds a "ready" JSON = the MSA-bearing RNA chain from phase 1
# + the ion entities and seeds from data/test_cases/casp17_R/af3_inputs/R2386_<POINT>.json,
# then runs inference only.
#
# Machine-specific facts (re-verified 2026-08-01):
#   - image alphafold3-3.0.3_casp17; the older alphafold3_casp17 has a broken cpp ext
#   - the stock image REJECTS the modelSeeds JSON key -> must bind-mount the patched
#     folding_input.py; with the patch and no --model_seed, JSON modelSeeds is read as-is
#   - XLA_PYTHON_CLIENT_MEM_FRACTION breaks GPU init in this image; use PREALLOCATE=false
#   - /bmlfast is 100% full: outputs go to /bml, scratch to /tmp

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

POINT=${1:?POINT (S1|S2|S3|S4)}
GPU=${2:?GPU device id}
CPUS=${3:-10}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH="${ROOT}/casp17/patches/folding_input_daisy.py"
[ -f "$PATCH" ] || bash "${ROOT}/casp17/scripts/make_af3_patch.sh"
OUT_BASE="${AF3_RESULTS_DIR:-/bml/Lyuwei/alphafold_results}/casp17_R_R2386"
SPEC="${ROOT}/data/test_cases/casp17_R/af3_inputs/R2386_${POINT}.json"
READY_DIR="/tmp/af3_R2386_ready_${POINT}"
OUT_DIR="${OUT_BASE}/${POINT}"
LOG="${OUT_BASE}/${POINT}_inference.log"

[ -f "$SPEC" ] || { echo "[ERR] missing $SPEC" >&2; exit 1; }

DATA_JSON=$(find "${OUT_BASE}/msa" -maxdepth 2 -iname "*_data.json" 2>/dev/null | head -1)
[ -n "$DATA_JSON" ] || { echo "[ERR] phase-1 data.json not found under ${OUT_BASE}/msa" >&2; exit 2; }

mkdir -p "$READY_DIR" "$OUT_DIR" "/tmp/af3jax_g${GPU}"

python3 - "$DATA_JSON" "$SPEC" "$READY_DIR/R2386_${POINT}.json" "$POINT" <<'PY'
import json, sys
data_json, spec_json, out_path, point = sys.argv[1:5]
data = json.load(open(data_json))
spec = json.load(open(spec_json))

# RNA chain carrying the searched MSA, from phase 1
rna = next(s for s in data["sequences"] if "rna" in s)
n_hits = sum(1 for L in rna["rna"].get("unpairedMsa", "").split("\n") if L.startswith(">"))
if n_hits == 0:
    raise SystemExit(f"[ERR] phase-1 RNA MSA is empty in {data_json}")

# ion entities from the stoichiometry spec
ions = [s for s in spec["sequences"] if "ligand" in s]
n_ion = sum(len(s["ligand"]["id"]) for s in ions)

out = {
    "name": f"R2386_{point}",
    "modelSeeds": spec["modelSeeds"],
    "sequences": [rna] + ions,
    "dialect": "alphafold3",
    "version": 1,
}
json.dump(out, open(out_path, "w"), indent=2)
print(f"[ready] {out_path}: MSA {n_hits} hits, {n_ion} ions, seeds {out['modelSeeds']}")
PY

echo "[$(date '+%H:%M:%S')] R2386/${POINT} inference on GPU${GPU} -> $LOG"

docker run --rm --gpus "\"device=${GPU}\"" \
  --cpus="${CPUS}" --memory=64g --memory-swap=64g --shm-size=8g \
  --user="$(id -u):$(id -g)" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e TF_FORCE_UNIFIED_MEMORY=0 \
  -v "${READY_DIR}":/tmp/af_input:ro \
  -v "${OUT_DIR}":/tmp/af_output \
  -v "$AF3_WEIGHTS_DIR":/tmp/models \
  -v "$AF3_DB_DIR":/public_databases \
  -v "/tmp/af3jax_g${GPU}":/tmp/jax_cache \
  -v "${PATCH}":/app/alphafold/src/alphafold3/common/folding_input.py:ro \
  alphafold3-3.0.3_casp17 \
  python run_alphafold.py \
    --json_path="/tmp/af_input/R2386_${POINT}.json" \
    --output_dir=/tmp/af_output --model_dir=/tmp/models --db_dir=/public_databases \
    --jax_compilation_cache_dir=/tmp/jax_cache \
    --norun_data_pipeline --force_output_dir --num_diffusion_samples=5 \
  > "$LOG" 2>&1

N_CIF=$(find "$OUT_DIR" -name "*model.cif" 2>/dev/null | wc -l)
echo "[$(date '+%H:%M:%S')] R2386/${POINT} DONE: ${N_CIF} model.cif (expect 25 + 1 top-ranked)"
[ "$N_CIF" -ge 25 ] || { echo "[WARN] expected >=25 cifs, got $N_CIF" >&2; exit 5; }
