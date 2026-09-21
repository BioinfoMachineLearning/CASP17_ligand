#!/bin/bash
# R1+R2 ensemble pipeline runner.
#
# Stages:
#   1) ensemble_generation (pair_iptm prefilter top-50 per method, PB + LG chem validate)
#   2) evaluate_ensemble   (OpenStructure Docker → lDDT-PLI score_cache; SKIP for CASP17)
#   3) evaluate_topn_clusters (Butina + consensus_pb + maxclust 0.80:60:0.05:0.30)
#   4) compare_r1_vs_r1r2  (vs r1-only baseline; SKIP for CASP17)
#
# Modes (auto-detected from $1):
#   CASP16 (multi-target, has GT):  bash ... <l1000|l2000|l3000|l4000> [num_workers] [--yes]
#   CASP17 (per-target, no GT):     bash ... <Rxxxx> [num_workers] [--yes]
#                                   (e.g. R2317 → series=casp17_R, target=R2317; reads
#                                    data/test_cases/casp17_R/ensemble_inputs_<TGT>.csv,
#                                    auto-derives it from ensemble_inputs.csv if missing)
#
set -euo pipefail

# ── OpenMM thread pinning (do NOT remove) ────────────────────────────────────
# Stage 1's PB-fail relax runs many OpenMM contexts in parallel. OpenMM's CPU
# platform defaults to "use every core it can see", so each of the N relax
# workers grabs all 80 cores and they thrash: T2451 showed load ~613 and a
# relax roughly two orders of magnitude slower than it should be. The rule is
# `workers x threads <= physical cores`, and since the pipeline already
# parallelises across poses, threads must be 1.
# Exported (not just set) because the relax happens in child python processes.
export OPENMM_CPU_THREADS="${OPENMM_CPU_THREADS:-1}"

SHORT="${1:?target required: a CASP17 id such as T2409 or R2314 (the l1000..l4000 CASP16 benchmark branches need data not shipped with this repo)}"
NUM_WORKERS="${2:-8}"
AUTO_YES="${3:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default flags
SKIP_STAGE2=0
SKIP_STAGE4=0
INPUT_CSV=""
OUTPUT_SUFFIX=""
TARGET=""

case "$SHORT" in
    l1000)
        LOGICAL="casp16_l1000_r1r2"
        REF_DIR="data/casp16_data/struct/L1000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L1000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        CFG_NAME="ensemble_generation_r1r2_${SHORT}"
        ;;
    l2000)
        LOGICAL="casp16_l2000_r1r2"
        REF_DIR="data/casp16_data/struct/L2000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L2000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        CFG_NAME="ensemble_generation_r1r2_${SHORT}"
        ;;
    l3000)
        LOGICAL="casp16_l3000_r1r2"
        REF_DIR="data/casp16_data/struct/L3000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L3000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        CFG_NAME="ensemble_generation_r1r2_${SHORT}"
        ;;
    l4000)
        LOGICAL="casp16_l4000_r1r2"
        REF_DIR="data/casp16_data/struct/L4000_prepared"
        SMILES_DIR="data/casp16_data/smiles/L4000"
        R1_BASELINE_CSV="outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv"
        CFG_NAME="ensemble_generation_r1r2_${SHORT}"
        ;;
    R[0-9]*|T[0-9]*|M[0-9]*)
        # CASP17 R-series (RNA-ligand), T-series (protein-ligand), or
        # M-series (multi-chain RNA+protein+ligand complex):
        # SHORT is the target itself. Branch on prefix to pick series.
        TARGET="$SHORT"
        case "$TARGET" in
            R*) SERIES="casp17_R" ;;
            T*) SERIES="casp17_T" ;;
            M*) SERIES="casp17_M" ;;
        esac
        LOGICAL="${SERIES}_r1r2"
        CFG_NAME="ensemble_generation_r1r2_${SERIES}"
        REF_DIR=""             # no GT
        SMILES_DIR="data/casp17_data/smiles"
        R1_BASELINE_CSV=""     # no r1 baseline for CASP17
        SKIP_STAGE2=1          # no GT → skip lDDT
        SKIP_STAGE4=1          # no r1-only baseline → skip compare
        OUTPUT_SUFFIX="_${TARGET}"

        # Per-target Stage 1 input CSV. Auto-derive from full ensemble_inputs.csv
        # by filtering the target's row, so the user only maintains one CSV.
        FULL_CSV="data/test_cases/${SERIES}/ensemble_inputs.csv"
        INPUT_CSV="data/test_cases/${SERIES}/ensemble_inputs_${TARGET}.csv"
        if [ ! -f "$INPUT_CSV" ]; then
            if [ ! -f "$FULL_CSV" ]; then
                echo "ERROR: neither $INPUT_CSV nor $FULL_CSV exists. Add ${TARGET} row to $FULL_CSV first."
                exit 2
            fi
            HEADER=$(head -1 "$FULL_CSV")
            ROW=$(awk -F, -v t="$TARGET" 'NR>1 && $1==t {print; exit}' "$FULL_CSV")
            if [ -z "$ROW" ]; then
                echo "ERROR: target ${TARGET} not found in $FULL_CSV (column 1). Add it and retry."
                exit 2
            fi
            printf '%s\n%s\n' "$HEADER" "$ROW" > "$INPUT_CSV"
            echo "Auto-derived $INPUT_CSV from $FULL_CSV (1 row)."
        fi
        ;;
    *)
        echo "ERROR: unknown short name '$SHORT'. Use l1000|l2000|l3000|l4000 (CASP16) or Rxxxx/Txxxx (CASP17)."
        exit 2
        ;;
esac

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
echo "  OPENMM threads: $OPENMM_CPU_THREADS (must be 1; see top of script)"
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
STAGE1_ARGS=(--config-name="$CFG_NAME" num_workers=$NUM_WORKERS)
if [ -n "$INPUT_CSV" ]; then
    STAGE1_ARGS+=("input_csv=$INPUT_CSV")
fi
conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/models/ensemble_generation.py "${STAGE1_ARGS[@]}"

# ── Stage 2: evaluate_ensemble (lDDT Docker) — skip when no GT ────────────
if [ "$SKIP_STAGE2" = "1" ]; then
    echo
    echo "=== Stage 2: SKIP (CASP17 / no ground truth) ==="
else
    echo
    echo "=== Stage 2: evaluate_ensemble (OpenStructure lDDT Docker) ==="
    conda run --no-capture-output -n casp17_ligand \
        python casp17_ligand/analysis/evaluate_ensemble.py \
            --ensemble_dir "$ENSEMBLE_DIR" \
            --reference_dir "$REF_DIR" \
            --smiles_dir "$SMILES_DIR" \
            --num_workers "$NUM_WORKERS"
fi

# ── Stage 3: cluster eval ─────────────────────────────────────────────────
echo
echo "=== Stage 3: evaluate_topn_clusters (consensus_pb, maxclust 0.80:60:0.05:0.30) ==="
STAGE3_ARGS=(--strategy consensus_pb --maxclust 0.80:60:0.05:0.30
             --datasets "$LOGICAL" --base-dir outputs/ensemble_r1r2
             --workers "$NUM_WORKERS")
if [ -n "$OUTPUT_SUFFIX" ]; then
    STAGE3_ARGS+=(--output-suffix "$OUTPUT_SUFFIX")
fi
conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/analysis/evaluate_topn_clusters.py "${STAGE3_ARGS[@]}"

# CASP17 multi-target dir → split combined CSV into per-target latest files.
if [ -n "$TARGET" ]; then
    SUFFIXED_CSV="outputs/ensemble_r1r2/cluster_experiment_detail_latest${OUTPUT_SUFFIX}.csv"
    if [ -f "$SUFFIXED_CSV" ]; then
        SRC="$SUFFIXED_CSV" python3 - <<'PY'
import csv, os
src = os.environ['SRC']
with open(src) as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0].keys())
for r in rows:
    out = f"outputs/ensemble_r1r2/cluster_experiment_detail_latest_{r['target']}.csv"
    with open(out, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=fields); w.writeheader(); w.writerow(r)
    print('split →', out)
PY
    fi
fi

# ── Stage 4: compare to r1-only baseline — skip when no baseline ──────────
if [ "$SKIP_STAGE4" = "1" ]; then
    echo
    echo "=== Stage 4: SKIP (CASP17 / no r1 baseline) ==="
else
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
fi

echo
echo "========================================================================"
echo "  Pipeline complete."
echo "  Results:"
echo "    - $ENSEMBLE_DIR/ranking_summary.csv"
if [ "$SKIP_STAGE2" != "1" ]; then
    echo "    - $ENSEMBLE_DIR/evaluation_summary.csv"
fi
if [ -n "$TARGET" ]; then
    echo "    - outputs/ensemble_r1r2/cluster_experiment_detail_latest_${TARGET}.csv"
else
    echo "    - outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv"
fi
if [ "$SKIP_STAGE4" != "1" ]; then
    echo "    - $COMPARE_OUT"
fi
echo "========================================================================"
