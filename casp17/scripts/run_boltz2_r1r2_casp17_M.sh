#!/usr/bin/env bash
# Boltz-2 for the CASP17 M series (multi-chain RNA + protein + ligand).
#
#   bash casp17/scripts/run_boltz2_r1r2_casp17_M.sh <TARGET> [GPU=0]
#
# Why this is not run_boltz2_r1r2_casp17_T.sh with a different dataset: a
# 13-chain complex OOMs at the default batch width, so the models are collected
# in 25 narrow rounds instead of 2 wide ones.
#
# Root cause of the earlier OOMs (proven): Boltz's --max_parallel_samples
# defaults to 5, so every round pushed a 5-wide chunk through the structure
# module → ~80GB → OOM, INDEPENDENT of --diffusion_samples (12/15/20/25/50 all
# OOM'd identically). diffusion_samples=4 makes the natural chunk = min(5,4)=4,
# which peaks at ~44GB and completes (validated: 4 CIF, 0 failed).
#
# So: diffusion_samples=4 × 25 rounds = 100 models. step_scale=1.2 is the
# documented small-batch setting (more diversity and higher
# best-of-N at small batch). recycling_steps=10, sampling_steps=200,
# use_potentials, use_msa_server all standard. Nothing quality-related touched.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TGT="${1:?target required, e.g. M2415}"
GPU="${2:-0}"
NS=4
SS=1.2
PRED_DIR_BASE="outputs/boltz2/casp17_M"
ROUNDS="r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25"

cif_count() {  # $1=round_tag
  find "${PRED_DIR_BASE}/${TGT}_$1/boltz_results_${TGT}_input/predictions" \
    -name '*.cif' 2>/dev/null | wc -l
}

echo "==================== $TGT Boltz-2 4x25 (GPU$GPU, ss=$SS) ===================="
TOTAL=0
FIRST=1
for rt in $ROUNDS; do
  echo "[$(date '+%H:%M:%S')] $rt: diffusion_samples=$NS step_scale=$SS recycling=10 sampling=200 potentials=on"
  CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/models/boltz2_inference.py \
    dataset=casp17_M round_tag="$rt" targets=$TGT \
    diffusion_samples=$NS step_scale=$SS override=true
  n=$(cif_count "$rt")
  TOTAL=$((TOTAL + n))
  echo "[$(date '+%H:%M:%S')] $rt → $n CIF   (cumulative $TOTAL / 100)"
  if [ "$FIRST" = "1" ]; then
    FIRST=0
    if [ "$n" -eq 0 ]; then
      echo "!!! r1 produced 0 CIF unexpectedly (4-parallel was validated to work). ABORTING — investigate."
      exit 1
    fi
  fi
  if [ "$TOTAL" -ge 100 ]; then
    echo "==================== DONE: reached $TOTAL CIF at $rt ===================="
    exit 0
  fi
done
echo "==================== FINISHED all rounds: total=$TOTAL CIF (target 100) ===================="
