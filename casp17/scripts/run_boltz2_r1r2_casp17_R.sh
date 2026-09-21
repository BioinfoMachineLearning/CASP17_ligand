#!/bin/bash
# CASP17 RNA Boltz-2 r1+r2 fixed pipeline (locked in 2026-06-01):
#   - Two GPUs, r1+r2 parallel
#   - step_scale=1.5 both rounds (CASP17 RNA-tuned)
#   - Auto symlink to casp17_R_r{1,2}/ via round_tag (handled by boltz2_inference.py)
#   - Skip-if-exists by default (set OVERRIDE=true to rerun)
#
# Usage:
#   bash casp17/scripts/run_boltz2_r1r2_casp17_R.sh <TARGET> [GPU_R1=0] [GPU_R2=1]
# Example:
#   bash casp17/scripts/run_boltz2_r1r2_casp17_R.sh R2365
#   bash casp17/scripts/run_boltz2_r1r2_casp17_R.sh R2365 0 1
#
# Logs: logs/boltz2/<TARGET>_r{1,2}.log

set -uo pipefail

TARGET="${1:?target required, e.g. R2365}"
GPU_R1="${2:-0}"
GPU_R2="${3:-1}"
OVERRIDE="${OVERRIDE:-false}"

cd "$(dirname "$0")/../.."
mkdir -p logs/boltz2

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "true" ] && OVERRIDE_FLAG="override=true"

echo "===== Boltz-2 r1+r2 for $TARGET (GPU $GPU_R1=r1, GPU $GPU_R2=r2) ====="; date

CUDA_VISIBLE_DEVICES=$GPU_R1 conda run --no-capture-output -n casp17_ligand python \
    casp17_ligand/models/boltz2_inference.py \
    dataset=casp17_R round_tag=r1 targets=${TARGET} step_scale=1.5 $OVERRIDE_FLAG \
    > logs/boltz2/${TARGET}_r1.log 2>&1 &
PID_R1=$!

CUDA_VISIBLE_DEVICES=$GPU_R2 conda run --no-capture-output -n casp17_ligand python \
    casp17_ligand/models/boltz2_inference.py \
    dataset=casp17_R round_tag=r2 targets=${TARGET} step_scale=1.5 $OVERRIDE_FLAG \
    > logs/boltz2/${TARGET}_r2.log 2>&1 &
PID_R2=$!

wait $PID_R1; RC_R1=$?
wait $PID_R2; RC_R2=$?

echo "===== Boltz-2 done for $TARGET ====="; date
echo "r1 exit=$RC_R1  r2 exit=$RC_R2"

[ "$RC_R1" = "0" ] && [ "$RC_R2" = "0" ]
