#!/usr/bin/env bash
# Wait for the R2386 AF3 phase-1 RNA MSA search to finish, then launch two
# stoichiometry points concurrently, one per GPU.
#
#   bash casp17/scripts/autostart_af3_R2386.sh S1 S2      # default GPUs 0 and 1
#   bash casp17/scripts/autostart_af3_R2386.sh S3 S4 0 1
#
# Phase 1 writes <OUT_BASE>/msa/*_data.json only at the very end, so the file
# appearing is the completion signal. We additionally require the RNA chain to
# carry a non-empty unpairedMsa before starting inference, otherwise a partial
# or failed search would silently degrade all 25 models to single-sequence.

set -uo pipefail

P_A=${1:-S1}
P_B=${2:-S2}
GPU_A=${3:-0}
GPU_B=${4:-1}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_BASE="${AF3_RESULTS_DIR:-/bml/Lyuwei/alphafold_results}/casp17_R_R2386"
cd "$ROOT"

echo "[$(date '+%F %T')] waiting for phase-1 MSA under ${OUT_BASE}/msa ..."
while true; do
  DATA_JSON=$(find "${OUT_BASE}/msa" -maxdepth 2 -iname "*_data.json" 2>/dev/null | head -1)
  if [ -n "$DATA_JSON" ]; then
    HITS=$(python3 - "$DATA_JSON" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    rna = next(s for s in d["sequences"] if "rna" in s)
    print(sum(1 for L in rna["rna"].get("unpairedMsa", "").split("\n") if L.startswith(">")))
except Exception:
    print(0)
PY
)
    if [ "${HITS:-0}" -gt 0 ]; then
      echo "[$(date '+%F %T')] phase-1 done: $DATA_JSON with $HITS MSA hits"
      break
    fi
  fi
  if ! pgrep -f run_alphafold >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] [ERR] no AF3 process alive and no usable data.json — phase 1 failed." >&2
    echo "  check ${OUT_BASE}/phase1_msa.log" >&2
    exit 2
  fi
  sleep 60
done

echo "[$(date '+%F %T')] launching ${P_A} on GPU${GPU_A} and ${P_B} on GPU${GPU_B}"
bash casp17/scripts/run_af3_R2386_sweep.sh "$P_A" "$GPU_A" 10 &
PID_A=$!
sleep 20   # stagger so the two containers do not race on the same JAX cache warm-up
bash casp17/scripts/run_af3_R2386_sweep.sh "$P_B" "$GPU_B" 10 &
PID_B=$!

wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
echo "[$(date '+%F %T')] ${P_A} exit=$RC_A  ${P_B} exit=$RC_B"
for P in "$P_A" "$P_B"; do
  N=$(find "${OUT_BASE}/${P}" -name "*model.cif" 2>/dev/null | wc -l)
  echo "  ${P}: ${N} model.cif"
done
