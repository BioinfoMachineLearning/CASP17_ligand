#!/bin/bash
# Second long_large lane on GPU 0 (40GB A100, ours alone since short_b finished).
# Walks targets + compounds in REVERSE order — both lanes share OUTPUT_DIR and
# rely on the per-compound `<lower>_ranking_scores.csv` skip to avoid races.
# They will eventually meet around the middle of P0DTD1 (727 compound series).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DATASET_TAG="chembl35_full_long_large_rev"
export TARGETS_FILE="$PROJECT_ROOT/data/test_cases/chembl35_full/targets_long_large_desc.txt"
export MANIFEST="$PROJECT_ROOT/data/test_cases/chembl35_full/af3_manifest_long_large_desc.json"
# All resource params overridable from env. Defaults assume aster GPU 0 (40GB
# A100, mostly-free). On other hosts override per local situation, e.g.:
#   GPU_DEVICE=2 XLA_CLIENT_MEM_FRACTION=0.9 TF_UNIFIED_MEM=0 bash $0
export GPU_DEVICE="${GPU_DEVICE:-0}"
export XLA_PREALLOCATE="${XLA_PREALLOCATE:-true}"
export TF_UNIFIED_MEM="${TF_UNIFIED_MEM:-1}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.7}"
export XLA_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export MEMORY="${MEMORY:-40g}"
export MEMORY_SWAP="${MEMORY_SWAP:-$MEMORY}"
export CPUS="${CPUS:-48}"
export NUM_SEEDS="${NUM_SEEDS:-2}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-alphafold3_casp17}"

exec bash "$PROJECT_ROOT/scripts/run_af3_chembl35_full.sh"
