#!/bin/bash
# long_large inference on aster GPU 1 (80GB), 50GB budget.
# Shares GPU 1 with short_a's 20GB preallocation → 70GB of 80GB total, fits.
# Order: ascending length (482 → 1073 aa) per targets_long_large.txt.
# All 19 targets already have Phase 1 MSAs produced (from long_large_msa_only
# + earlier aborted run), so run_af3_chembl35_full.sh will skip Phase 1 and go
# straight to Phase 2 inject → Phase 3 inference.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_long_large"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_long_large.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_long_large.json"
export GPU_DEVICE=1
export XLA_PREALLOCATE=true
export TF_UNIFIED_MEM=0
export XLA_CLIENT_MEM_FRACTION=0.625   # 0.625 * 80GB = 50GB
export XLA_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export MEMORY=40g
export MEMORY_SWAP=40g
export CPUS=48
export NUM_SEEDS=2
export DOCKER_IMAGE=alphafold3_casp17

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
