#!/bin/bash
# GPU 1 (A100 80GB): long-sequence half, descending length order (long→short).
# 90% VRAM, no unified memory needed.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_long"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_long.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_long.json"
export GPU_DEVICE=1
export XLA_PREALLOCATE=true
export TF_UNIFIED_MEM=0
export XLA_MEM_FRACTION=0.9
export MEMORY=80g
export MEMORY_SWAP=80g
export CPUS=60
export NUM_SEEDS=2     # 2 seeds × 5 samples = 10 conformers/compound
export DOCKER_IMAGE=alphafold3_casp17

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
