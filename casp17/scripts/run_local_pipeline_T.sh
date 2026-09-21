#!/bin/bash
# CASP17 T-series (protein-ligand) single-target local pipeline (locked-in 2026-06-13):
#   Stage 1: Boltz-2 r1+r2 parallel on GPU 0+1
#   Stage 2: Protenix dual-ckpt r1 parallel on GPU 0+2 → winner r2 on winner's GPU
#
# AF3 and SeedFold are NOT run here:
#   - AF3 cannot run on V100 → handled remotely by user
#   - SeedFold submitted via web API by user
#
# Sequential between stages (GPU 0 is reused by both Boltz-2 r1 and Protenix
# default). Total wall depends on target size; T2383 (1142 aa) is large — may
# take several hours per stage. Memory: V100-32GB may OOM at this size; if so,
# user will provide larger GPUs (no parameter shrink per user directive).
#
# Pre-req:
#   1. Three input prep JSONs/YAMLs exist (boltz2/af3/protenix)
#      (af3 prep is run for completeness even though AF3 is remote)
#   2. NO MSA pre-search needed — Boltz-2 and Protenix run their own internal
#      protein MSA + template search; Protenix `enable_cache:true` shares it
#      across r1/r2/dual-ckpt invocations.
#
# Fail-isolated: Boltz-2 failure does NOT block Protenix; both rcs reported at end.
#
# Usage:
#   bash casp17/scripts/run_local_pipeline_T.sh <TARGET>
# Example:
#   bash casp17/scripts/run_local_pipeline_T.sh T2383
#
# Top-level log: logs/local_pipeline/<TARGET>.log

set -uo pipefail

TARGET="${1:?target required, e.g. T2383}"

cd "$(dirname "$0")/../.."
mkdir -p logs/local_pipeline logs/boltz2 logs/protenix/dualckpt

TOP_LOG="logs/local_pipeline/${TARGET}.log"
exec > >(tee -a "$TOP_LOG") 2>&1

echo "=========================================="
echo "=== Local T-series pipeline $TARGET ==="
echo "=== started: $(date) ==="
echo "=========================================="

# ─── Stage 1: Boltz-2 r1+r2 (GPU 0+1) ────────────────────────────
echo; echo "----- Stage 1: Boltz-2 r1+r2 (GPU 0+1) -----"; date
bash casp17/scripts/run_boltz2_r1r2_casp17_T.sh "$TARGET" 0 1
RC_BOLTZ2=$?
echo "Stage 1 (Boltz-2) exit=$RC_BOLTZ2"

# ─── Stage 2: Protenix dual-ckpt r1 + winner r2 (GPU 0+2) ────────
echo; echo "----- Stage 2: Protenix dual-ckpt (GPU 0=default, 2=applied) -----"; date
bash casp17/scripts/run_protenix_dualckpt_casp17_T.sh "$TARGET" 0 2
RC_PROTENIX=$?
echo "Stage 2 (Protenix dual-ckpt) exit=$RC_PROTENIX"

echo
echo "=========================================="
echo "=== Local T-series pipeline $TARGET DONE ==="
echo "=== finished: $(date) ==="
echo "=== Boltz-2 rc=$RC_BOLTZ2  Protenix rc=$RC_PROTENIX ==="
echo "=========================================="

[ "$RC_BOLTZ2" = "0" ] && [ "$RC_PROTENIX" = "0" ]
