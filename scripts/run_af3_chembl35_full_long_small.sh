#!/bin/bash
# Long-half "small" subset (351..476 aa, 18 targets, 1208 compounds).
# Intended to run on ANOTHER machine — adjust GPU_DEVICE / VRAM / MEMORY / DOCKER_IMAGE
# to that host's resources before launching.
#
# Example launch on a host with a free 80GB GPU:
#   export GPU_DEVICE=0 XLA_MEM_FRACTION=0.9
#   bash scripts/run_af3_chembl35_full_long_small.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_long_small"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_long_small.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_long_small.json"

# --- tune these to the target host ---
export GPU_DEVICE="${GPU_DEVICE:-0}"
export XLA_PREALLOCATE="${XLA_PREALLOCATE:-true}"
export TF_UNIFIED_MEM="${TF_UNIFIED_MEM:-0}"         # 1 if 40GB card w/ little free
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.9}"
export XLA_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export MEMORY="${MEMORY:-80g}"
export MEMORY_SWAP="${MEMORY_SWAP:-$MEMORY}"
export CPUS="${CPUS:-48}"
export NUM_SEEDS="${NUM_SEEDS:-2}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-alphafold3_casp17}"
# --------------------------------------

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
