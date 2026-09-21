#!/bin/bash
# CASP17 RNA single-target local pipeline (locked in 2026-06-01):
#   Stage 1: Boltz-2 r1+r2 parallel on GPU 0+1
#   Stage 2: Protenix dual-ckpt r1 parallel on GPU 0+2 → winner r2 on GPU of winner
#
# Sequential between stages (GPU 0 is reused by both Boltz-2 r1 and Protenix default).
# Total wall: ~2-3 hours per target (60-100 nt RNA).
#
# Pre-req:
#   1. ensemble_inputs.csv has the target row
#   2. Three input prep JSONs/YAMLs exist (boltz2/af3/protenix)
#   3. RNA MSA pre-searched and patched into protenix JSON
#      (run_rna_msa_only.py auto-patches unpairedMsaPath)
#
# Fail-isolated: Boltz-2 failure does NOT block Protenix; both rcs reported at end.
#
# Usage:
#   bash casp17/scripts/run_local_pipeline_R.sh <TARGET>
# Example:
#   bash casp17/scripts/run_local_pipeline_R.sh R2365
#
# Top-level log: logs/local_pipeline/<TARGET>.log

set -uo pipefail

TARGET="${1:?target required, e.g. R2365}"

cd "$(dirname "$0")/../.."
mkdir -p logs/local_pipeline logs/boltz2 logs/protenix/dualckpt

TOP_LOG="logs/local_pipeline/${TARGET}.log"
exec > >(tee -a "$TOP_LOG") 2>&1

echo "=========================================="
echo "=== Local pipeline $TARGET ==="
echo "=== started: $(date) ==="
echo "=========================================="

# ─── Stage 1: Boltz-2 r1+r2 (GPU 0+1) ────────────────────────────
echo; echo "----- Stage 1: Boltz-2 r1+r2 (GPU 0+1) -----"; date
bash casp17/scripts/run_boltz2_r1r2_casp17_R.sh "$TARGET" 0 1
RC_BOLTZ2=$?
echo "Stage 1 (Boltz-2) exit=$RC_BOLTZ2"

# ─── Stage 2: Protenix dual-ckpt r1 + winner r2 (GPU 0+2) ────────
echo; echo "----- Stage 2: Protenix dual-ckpt (GPU 0=default, 2=applied) -----"; date
bash casp17/scripts/run_protenix_rna_dualckpt.sh "$TARGET" 0 2
RC_PROTENIX=$?
echo "Stage 2 (Protenix dual-ckpt) exit=$RC_PROTENIX"

echo
echo "=========================================="
echo "=== Local pipeline $TARGET DONE ==="
echo "=== finished: $(date) ==="
echo "=== Boltz-2 rc=$RC_BOLTZ2  Protenix rc=$RC_PROTENIX ==="
echo "=========================================="

[ "$RC_BOLTZ2" = "0" ] && [ "$RC_PROTENIX" = "0" ]
