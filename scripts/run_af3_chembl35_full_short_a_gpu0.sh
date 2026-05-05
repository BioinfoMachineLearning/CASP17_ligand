#!/bin/bash
# GPU 0 (A100 40GB, ~28GB free, boltz owns rest): half of short targets,
# ~20GB budget → mem_fraction=0.5 + unified memory overflow.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_short_a"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_short_a.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_short_a.json"
export GPU_DEVICE=1
export XLA_PREALLOCATE=true
export TF_UNIFIED_MEM=0
export XLA_CLIENT_MEM_FRACTION=0.25  # 0.25 * 80GB = 20GB (post-swap)
export XLA_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export MEMORY=40g
export MEMORY_SWAP=40g
export CPUS=48
export NUM_SEEDS=2
export DOCKER_IMAGE=alphafold3_casp17

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
