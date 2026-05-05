#!/bin/bash
# R1+R2 ensemble pipeline runner.
#
# Stages:
#   1) ensemble_generation (pocket_plddt_4.5 prefilter top-50 per method, PB + LG chem validate)
#   2) evaluate_ensemble   (OpenStructure Docker → lDDT-PLI score_cache)
#   3) evaluate_topn_clusters (Butina + consensus_pb + maxclust 0.80:60:0.05:0.30)
#   4) compare_r1_vs_r1r2  (vs r1-only baseline)
#
# Usage:
#   bash scripts/run_ensemble_r1r2_pipeline.sh <dataset_short> [num_workers] [--yes]
#
# Examples:
#   bash scripts/run_ensemble_r1r2_pipeline.sh l2000 4
#   bash scripts/run_ensemble_r1r2_pipeline.sh l3000 16
#
set -euo pipefail

SHORT="${1:?dataset short name required: l1000 | l2000 | l3000 | l4000}"
NUM_WORKERS="${2:-8}"
AUTO_YES="${3:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

case "$SHORT" in
    l1000)
        LOGICAL="casp16_l1000_r1r2"
        REF_DIR="data/casp16_data/struct/L1000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L1000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        ;;
    l2000)
        LOGICAL="casp16_l2000_r1r2"
        REF_DIR="data/casp16_data/struct/L2000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L2000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        ;;
    l3000)
        LOGICAL="casp16_l3000_r1r2"
        REF_DIR="data/casp16_data/struct/L3000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L3000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        ;;
    l4000)
        LOGICAL="casp16_l4000_r1r2"
        REF_DIR="data/casp16_data/struct/L4000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L4000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        ;;
    *)
        echo "ERROR: unknown dataset short name '$SHORT'. Use l1000 | l2000 | l3000 | l4000"
        exit 2
        ;;
esac

CFG_NAME="ensemble_generation_r1r2_${SHORT}"
CFG_PATH="configs/model/${CFG_NAME}.yaml"
ENSEMBLE_DIR="outputs/ensemble_r1r2/${LOGICAL}"
COMPARE_OUT="outputs/ensemble_r1r2/compare_${SHORT}.csv"

if [ ! -f "$CFG_PATH" ]; then
    echo "ERROR: config not found: $CFG_PATH"
    exit 2
fi

# ── Pre-flight check ───────────────────────────────────────────────────────
echo "========================================================================"
echo "  R1+R2 ensemble pipeline: $LOGICAL"
echo "  Config:        $CFG_PATH"
echo "  Workers:       $NUM_WORKERS"
echo "  Ensemble out:  $ENSEMBLE_DIR"
echo "  Compare out:   $COMPARE_OUT"
echo "========================================================================"
echo
echo "--- System load ---"
uptime
echo
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo
fi

LOAD_1M=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)
LOAD_INT=${LOAD_1M%.*}
if [ "$LOAD_INT" -gt 500 ] 2>/dev/null; then
    echo "WARNING: 1-min load average is ${LOAD_1M} — very high."
    echo "         Consider running on another machine or reducing NUM_WORKERS."
fi

if [ "$AUTO_YES" != "--yes" ]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# ── Stage 1: ensemble_generation ──────────────────────────────────────────
echo
echo "=== Stage 1: ensemble_generation ($LOGICAL) ==="
conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/models/ensemble_generation.py \
        --config-name="$CFG_NAME" \
        num_workers=$NUM_WORKERS

# ── Stage 2: evaluate_ensemble (lDDT Docker) ──────────────────────────────
echo
echo "=== Stage 2: evaluate_ensemble (OpenStructure lDDT Docker) ==="
conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/analysis/evaluate_ensemble.py \
        --ensemble_dir "$ENSEMBLE_DIR" \
        --reference_dir "$REF_DIR" \
        --smiles_dir "$SMILES_DIR" \
        --num_workers "$NUM_WORKERS"

# ── Stage 3: cluster eval ─────────────────────────────────────────────────
echo
echo "=== Stage 3: evaluate_topn_clusters (consensus_pb, maxclust 0.80:60:0.05:0.30) ==="
conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/analysis/evaluate_topn_clusters.py \
        --strategy consensus_pb --maxclust 0.80:60:0.05:0.30 \
        --datasets "$LOGICAL" \
        --base-dir outputs/ensemble_r1r2 \
        --workers "$NUM_WORKERS"

# ── Stage 4: compare to r1-only baseline ──────────────────────────────────
echo
echo "=== Stage 4: compare r1 vs r1+r2 ==="
if [ -f "$R1_BASELINE_CSV" ]; then
    conda run --no-capture-output -n casp17_ligand \
        python casp17_ligand/analysis/compare_r1_vs_r1r2.py \
            --r1 "$R1_BASELINE_CSV" \
            --r1r2 outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv \
            --output "$COMPARE_OUT"
else
    echo "R1 baseline CSV not found: $R1_BASELINE_CSV"
    echo "Skipping compare stage."
fi

echo
echo "========================================================================"
echo "  Pipeline complete."
echo "  Results:"
echo "    - $ENSEMBLE_DIR/ranking_summary.csv"
echo "    - $ENSEMBLE_DIR/evaluation_summary.csv"
echo "    - outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv"
echo "    - $COMPARE_OUT"
echo "========================================================================"
