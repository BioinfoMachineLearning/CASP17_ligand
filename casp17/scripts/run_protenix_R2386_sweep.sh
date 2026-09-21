#!/usr/bin/env bash
# Protenix ion-stoichiometry sweep for R2386, one checkpoint per invocation.
#
#   bash casp17/scripts/run_protenix_R2386_sweep.sh <default|applied> <GPU>
#
# Runs all four stoichiometry points S1..S4 sequentially on one GPU, 5 seeds x
# 5 samples = 25 models each. Launch the two checkpoints on two GPUs in
# parallel to get the dual-ckpt comparison in one wall-clock pass.
#
# On checkpoint choice for THIS target: protenix_base_20250630 ("applied") has a
# 2025-06-30 data cutoff, so PDB 9C6I -- the 2.56 A cryo-EM structure of this
# exact 417 nt construct, deposited 2024-06-07 -- is inside its training window.
# That makes 9C6I useless as a fair discriminator between the two checkpoints,
# and leakage does not by itself imply the applied weights are better here.
# The honest label-free comparison is Mg2+ coordination geometry (real Mg sits
# 2.0-2.2 A from an RNA oxygen in a regular octahedron) plus a look for gross
# errors; pair_iptm is reported too but is of doubtful meaning when the
# "partner chains" are 83 single-atom ions.

set -uo pipefail

CKPT_KIND=${1:?default|applied}
GPU=${2:?GPU device id}

case "$CKPT_KIND" in
  default) MODEL=protenix_base_default_v1.0.0 ;;
  applied) MODEL=protenix_base_20250630_v1.0.0 ;;
  *) echo "[ERR] first arg must be 'default' or 'applied'" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/protenix

# Seed block per point, matching gen_R2386_solvent_inputs.py
declare -A SEEDS=( [S1]="101,102,103,104,105" [S2]="106,107,108,109,110"
                   [S3]="111,112,113,114,115" [S4]="116,117,118,119,120" )

echo "===== Protenix ${CKPT_KIND} (${MODEL}) on GPU${GPU} ====="; date

for P in S1 S2 S3 S4; do
  JSON="data/test_cases/casp17_R/protenix_inputs/R2386_${P}.json"
  if ! grep -q unpairedMsaPath "$JSON"; then
    echo "[ERR] $JSON has no unpairedMsaPath — run casp17/scripts/inject_rna_msa_R2386.py first." >&2
    echo "      Without it Protenix silently runs the RNA single-sequence." >&2
    exit 2
  fi
  LOG="logs/protenix/R2386_${P}_${CKPT_KIND}.log"
  echo "[$(date '+%H:%M:%S')] ${P} seeds=${SEEDS[$P]} -> $LOG"
  PROTENIX_ROOT_DIR="$ROOT/weights/protenix" \
  conda run --no-capture-output -n casp17_ligand python \
    casp17_ligand/models/protenix_inference.py \
    --config-name protenix_inference_casp17_R2386 \
    "targets=[R2386_${P}]" \
    seeds=\'"${SEEDS[$P]}"\' \
    model_name="${MODEL}" \
    gpu_device="${GPU}" \
    output_dir="outputs/protenix/casp17_R2386_${CKPT_KIND}" \
    > "$LOG" 2>&1
  RC=$?
  N=$(find "outputs/protenix/casp17_R2386_${CKPT_KIND}/R2386_${P}" -name "*.cif" 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M:%S')] ${P} exit=$RC  cifs=$N"
done

echo "===== Protenix ${CKPT_KIND} done ====="; date
find "outputs/protenix/casp17_R2386_${CKPT_KIND}" -name "*.cif" 2>/dev/null | wc -l | xargs echo "total cifs:"
