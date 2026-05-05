#!/bin/bash
# GPU 1 (A100 80GB, ~28GB free, other user owns ~53GB): second half of short
# targets, ~20GB budget → mem_fraction=0.25 (0.25 * 80GB = 20GB). 80GB card so
# no unified memory needed.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_short_b"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_short_b.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_short_b.json"
export GPU_DEVICE=0
export XLA_PREALLOCATE=true
export TF_UNIFIED_MEM=1
export XLA_CLIENT_MEM_FRACTION=0.9   # 0.9 * 40GB = 36GB (post-swap)
export XLA_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export MEMORY=40g
export MEMORY_SWAP=40g
export CPUS=48
export NUM_SEEDS=2
export DOCKER_IMAGE=alphafold3_casp17

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
