#!/bin/bash
# GPU 0 (A100 40GB): short-sequence half, ascending length order (short→long).
# 50% VRAM + unified memory overflow per AGENTS.md for 40GB card.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_short"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_short.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_short.json"
export GPU_DEVICE=0
export XLA_PREALLOCATE=true
export TF_UNIFIED_MEM=1
export XLA_MEM_FRACTION=0.5
export MEMORY=80g
export MEMORY_SWAP=80g
export CPUS=60
export NUM_SEEDS=2     # 2 seeds × 5 samples = 10 conformers/compound
export DOCKER_IMAGE=alphafold3_casp17

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
