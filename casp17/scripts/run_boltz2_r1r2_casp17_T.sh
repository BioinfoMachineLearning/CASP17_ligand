#!/bin/bash
# CASP17 T-series Boltz-2 r1+r2 fixed pipeline (locked-in 2026-06-13):
#   - Two GPUs, r1+r2 parallel
#   - step_scale=1.5 both rounds (CASP17 RNA-tuned; reused for protein-ligand,
#     same as casp17_R: it worked on the RNA side, and nothing is cut back for OOM)
#   - Auto symlink to casp17_T_r{1,2}/ via round_tag (boltz2_inference.py logic)
#
# Usage:
#   bash casp17/scripts/run_boltz2_r1r2_casp17_T.sh <TARGET> [GPU_R1=0] [GPU_R2=1]
# Example:
#   bash casp17/scripts/run_boltz2_r1r2_casp17_T.sh T2383
#
# Logs: logs/boltz2/<TARGET>_r{1,2}.log

set -uo pipefail

TARGET="${1:?target required, e.g. T2383}"
GPU_R1="${2:-0}"
GPU_R2="${3:-1}"
OVERRIDE="${OVERRIDE:-false}"

cd "$(dirname "$0")/../.."
mkdir -p logs/boltz2

OVERRIDE_FLAG=""
[ "$OVERRIDE" = "true" ] && OVERRIDE_FLAG="override=true"

run_round() {
    local round="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n casp17_ligand python \
        casp17_ligand/models/boltz2_inference.py \
        dataset=casp17_T round_tag=$round targets=${TARGET} step_scale=1.5 $OVERRIDE_FLAG \
        > logs/boltz2/${TARGET}_${round}.log 2>&1
}

# Same GPU for both rounds => run them back-to-back instead of stacking two
# diffusion jobs on one card. Two concurrent boltz processes each size their
# VRAM against the WHOLE card, so co-scheduling them OOMs rather than sharing.
if [ "$GPU_R1" = "$GPU_R2" ]; then
    echo "===== Boltz-2 r1->r2 SERIAL for $TARGET (GPU $GPU_R1) ====="; date
    run_round r1 "$GPU_R1"; RC_R1=$?
    echo "r1 exit=$RC_R1 ($(date '+%H:%M:%S'))"
    run_round r2 "$GPU_R2"; RC_R2=$?
else
    echo "===== Boltz-2 r1+r2 for $TARGET (GPU $GPU_R1=r1, GPU $GPU_R2=r2) ====="; date
    run_round r1 "$GPU_R1" & PID_R1=$!
    run_round r2 "$GPU_R2" & PID_R2=$!
    wait $PID_R1; RC_R1=$?
    wait $PID_R2; RC_R2=$?
fi

echo "===== Boltz-2 done for $TARGET ====="; date
echo "r1 exit=$RC_R1  r2 exit=$RC_R2"

[ "$RC_R1" = "0" ] && [ "$RC_R2" = "0" ]
