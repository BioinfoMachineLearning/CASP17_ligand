#!/bin/bash
# One-shot: rebuild protenix conda env + run L3000 struct inference for 5 remaining targets
# GPU 1 (A100 80GB), seeds 101-110, 5 samples/seed = 50 models/target
#
# Targets:
#   L3081, L3138 — need seed_103 only
#   L3019, L3118, L3134 — need all 10 seeds (846aa + 2 ligands, need 40GB+)

set -e
cd /bmlfast/Lyuwei/0.Projects/CASP17_ligand

CONDA_BASE=/bmlfast/Lyuwei/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

LOG() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Create protenix conda env ─────────────────────────────────────────
if conda env list | grep -q "^protenix "; then
    LOG "protenix env already exists, skipping create"
else
    LOG "Creating protenix conda env (python=3.11)..."
    conda create -n protenix python=3.11 -y
    LOG "Done creating env"
fi

# ── Step 2: Install protenix package ──────────────────────────────────────────
LOG "Installing protenix (pip install -e forks/Protenix)..."
conda run -n protenix pip install -e forks/Protenix 2>&1 | tail -5
LOG "Done installing protenix"

# ── Step 3: Install CUDA build tools (for layer_norm CUDA ext if needed) ──────
LOG "Installing CUDA build tools (cuda-nvcc, cuda-toolkit, cuda-cudart-dev 12.6)..."
conda install -n protenix -c conda-forge \
    cuda-nvcc=12.6.77 cuda-toolkit=12.6.2 cuda-cudart-dev=12.6.77 -y 2>&1 | tail -3
LOG "Done CUDA tools"

# ── Step 4: Verify ─────────────────────────────────────────────────────────────
LOG "Verifying protenix import..."
conda run -n protenix python -c "import protenix; print('protenix import OK')"
LOG "Verifying CUDA..."
conda run -n protenix python -c "
import torch
print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 1: {torch.cuda.get_device_name(1)}')" \
    2>&1 || LOG "WARN: cuda check failed, proceeding anyway"

# ── Step 5: Run inference ──────────────────────────────────────────────────────
LOG "Starting protenix inference for L3019,L3081,L3118,L3134,L3138 on GPU 1..."
PYTHONPATH="$PWD" conda run --no-capture-output -n casp17_ligand \
    python casp17_ligand/models/protenix_inference.py \
    dataset=casp16_l3000_struct \
    gpu_device=1 \
    "targets=[L3019,L3081,L3118,L3134,L3138]"

LOG "All done."
