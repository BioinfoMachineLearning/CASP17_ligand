#!/bin/bash
# CASP17 RNA dual-ckpt protocol (locked in 2026-05-05):
#   1. r1: run BOTH ckpts (default + applied) in parallel on two GPUs (50 model each)
#   2. select winner by mean iptm (locked metric, no fallback)
#   3. r2: run ONLY winner ckpt on one GPU (50 models, fresh seeds)
#   4. symlink winner r1+r2 -> canonical paths so ensemble pipeline picks them up
#
# Fail-fast: any stage crash aborts the script; user must intervene.
#
# Usage:
#   bash casp17/scripts/run_protenix_rna_dualckpt.sh <TARGET> <GPU_A> <GPU_B>
# Example:
#   bash casp17/scripts/run_protenix_rna_dualckpt.sh R2317 0 2
#
# Pre-req:  RNA MSA already pre-searched and unpairedMsaPath patched into
#           data/test_cases/casp17_R/protenix_inputs/<TARGET>.json
#           via casp17/scripts/run_rna_msa_only.py (locked flow).

set -euo pipefail

TARGET="${1:?target required, e.g. R2317}"
GPU_A="${2:?GPU for default ckpt required}"
GPU_B="${3:?GPU for applied ckpt required}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Per-ckpt output dirs (winner gets a r2 dir later)
DEFAULT_R1_DIR="outputs/protenix/casp17_R_default_r1"
APPLIED_R1_DIR="outputs/protenix/casp17_R_applied_r1"
DEFAULT_R2_DIR="outputs/protenix/casp17_R_default_r2"
APPLIED_R2_DIR="outputs/protenix/casp17_R_applied_r2"

# Canonical dirs (symlinked from winner)
CANONICAL_R1="outputs/protenix/casp17_R_r1"
CANONICAL_R2="outputs/protenix/casp17_R_r2"

AUDIT_CSV="casp17/audit/protenix_ckpt_selection.csv"
LOG_DIR="logs/protenix/dualckpt"
mkdir -p "$LOG_DIR"

INPUT_JSON="data/test_cases/casp17_R/protenix_inputs/${TARGET}.json"

# ── Pre-flight ────────────────────────────────────────────────────────────
echo "=== Protenix dual-ckpt for $TARGET (GPU $GPU_A=default, GPU $GPU_B=applied) ==="
[ -f "$INPUT_JSON" ] || { echo "FAIL: input JSON missing: $INPUT_JSON"; exit 2; }

if ! grep -q '"unpairedMsaPath"' "$INPUT_JSON"; then
    echo "FAIL: $INPUT_JSON has no unpairedMsaPath."
    echo "Run RNA MSA pre-search first:"
    echo "  PROTENIX_ROOT_DIR=$PROJECT_ROOT/weights/protenix \\"
    echo "    conda run --no-capture-output -n protenix python \\"
    echo "    casp17/scripts/run_rna_msa_only.py \\"
    echo "    --input $INPUT_JSON \\"
    echo "    --out_dir data/test_cases/casp17_R/protenix_msa"
    exit 2
fi

# ── Stage 1: r1 dual-ckpt parallel ────────────────────────────────────────
echo
echo "=== Stage 1: r1 dual-ckpt parallel ==="

run_r1() {
    local ckpt="$1" gpu="$2" out_dir="$3" log_path="$4"
    PROTENIX_ROOT_DIR="$PROJECT_ROOT/weights/protenix" \
        conda run --no-capture-output -n casp17_ligand python \
        casp17_ligand/models/protenix_inference.py \
        --config-name protenix_inference_casp17_R_r1 \
        model_name="$ckpt" \
        output_dir="$out_dir" \
        "targets=[$TARGET]" \
        gpu_device="$gpu" \
        > "$log_path" 2>&1
}

R1_DEFAULT_LOG="$LOG_DIR/${TARGET}_r1_default_gpu${GPU_A}.log"
R1_APPLIED_LOG="$LOG_DIR/${TARGET}_r1_applied_gpu${GPU_B}.log"

run_r1 protenix_base_default_v1.0.0   "$GPU_A" "$DEFAULT_R1_DIR" "$R1_DEFAULT_LOG" &
PID_DEFAULT=$!
run_r1 protenix_base_20250630_v1.0.0  "$GPU_B" "$APPLIED_R1_DIR" "$R1_APPLIED_LOG" &
PID_APPLIED=$!

# Fail-fast: wait both, propagate any non-zero
set +e
wait "$PID_DEFAULT"; RC_DEFAULT=$?
wait "$PID_APPLIED"; RC_APPLIED=$?
set -e

if [ "$RC_DEFAULT" != "0" ] || [ "$RC_APPLIED" != "0" ]; then
    echo "FAIL: r1 stage failed. default exit=$RC_DEFAULT  applied exit=$RC_APPLIED"
    echo "Check logs:"
    echo "  $R1_DEFAULT_LOG"
    echo "  $R1_APPLIED_LOG"
    exit 3
fi

# ── Stage 2: select winner by mean iptm ───────────────────────────────────
echo
echo "=== Stage 2: ckpt selection (mean iptm, locked) ==="
SEL_OUT="$LOG_DIR/${TARGET}_selection.txt"
conda run --no-capture-output -n casp17_ligand python \
    casp17_ligand/utils/select_protenix_ckpt.py \
    --target "$TARGET" \
    --default-dir "$DEFAULT_R1_DIR" \
    --applied-dir "$APPLIED_R1_DIR" \
    --audit-csv "$AUDIT_CSV" \
    --expected-n 50 \
    | tee "$SEL_OUT"

WINNER=$(grep '^WINNER=' "$SEL_OUT" | head -1 | cut -d= -f2)
[ -n "$WINNER" ] || { echo "FAIL: selector produced no WINNER"; exit 4; }
echo "Winner: $WINNER"

if [ "$WINNER" = "default" ]; then
    WINNER_CKPT="protenix_base_default_v1.0.0"
    WINNER_R1_DIR="$DEFAULT_R1_DIR"
    WINNER_R2_DIR="$DEFAULT_R2_DIR"
    WINNER_GPU="$GPU_A"
else
    WINNER_CKPT="protenix_base_20250630_v1.0.0"
    WINNER_R1_DIR="$APPLIED_R1_DIR"
    WINNER_R2_DIR="$APPLIED_R2_DIR"
    WINNER_GPU="$GPU_B"
fi

# ── Stage 3: r2 on winner ─────────────────────────────────────────────────
echo
echo "=== Stage 3: r2 on winner ($WINNER ckpt, GPU $WINNER_GPU) ==="
R2_LOG="$LOG_DIR/${TARGET}_r2_${WINNER}_gpu${WINNER_GPU}.log"
PROTENIX_ROOT_DIR="$PROJECT_ROOT/weights/protenix" \
    conda run --no-capture-output -n casp17_ligand python \
    casp17_ligand/models/protenix_inference.py \
    --config-name protenix_inference_casp17_R_r2 \
    model_name="$WINNER_CKPT" \
    output_dir="$WINNER_R2_DIR" \
    "targets=[$TARGET]" \
    gpu_device="$WINNER_GPU" \
    > "$R2_LOG" 2>&1

# ── Stage 4: symlink winner -> canonical so ensemble picks it up ──────────
echo
echo "=== Stage 4: symlink winner -> canonical r1/r2 paths ==="
mkdir -p "$CANONICAL_R1" "$CANONICAL_R2"
link_winner() {
    local canon_dir="$1" winner_dir="$2"
    local link_path="$canon_dir/$TARGET"
    # Replace any existing entry (dir, file, or stale symlink)
    if [ -e "$link_path" ] || [ -L "$link_path" ]; then
        rm -rf "$link_path"
    fi
    # Relative symlink so trees can move together
    local rel
    rel=$(realpath --relative-to="$canon_dir" "$winner_dir/$TARGET")
    ln -s "$rel" "$link_path"
    echo "  $link_path -> $rel"
}
link_winner "$CANONICAL_R1" "$WINNER_R1_DIR"
link_winner "$CANONICAL_R2" "$WINNER_R2_DIR"

echo
echo "=== Done: $TARGET / winner=$WINNER ==="
echo "  audit: $AUDIT_CSV"
echo "  canonical r1: $CANONICAL_R1/$TARGET"
echo "  canonical r2: $CANONICAL_R2/$TARGET"
