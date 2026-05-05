#!/usr/bin/env bash
# Set up conda env `pbcnet` per forks/PBCNet2.0/README.md
set -euo pipefail
LOG=/bmlfast/Lyuwei/0.Projects/CASP17_ligand/logs/pbcnet_env_setup.log
mkdir -p "$(dirname "$LOG")"

source /bmlfast/Lyuwei/miniconda3/etc/profile.d/conda.sh

echo "=== creating pbcnet env (python 3.8) ===" | tee "$LOG"
conda create -n pbcnet python=3.8 -y >> "$LOG" 2>&1

conda activate pbcnet

echo "=== installing torch ===" | tee -a "$LOG"
pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 >> "$LOG" 2>&1

echo "=== installing pip deps ===" | tee -a "$LOG"
pip install --no-cache-dir \
    pandas packaging PyYAML pydantic scipy matplotlib \
    rdkit networkx psutil tqdm scikit-learn bio rootutils >> "$LOG" 2>&1

echo "=== installing dgl 1.0.2 (cu118) ===" | tee -a "$LOG"
pip install --no-cache-dir dgl==1.0.2+cu118 \
    -f https://data.dgl.ai/wheels/cu118/repo.html --no-deps >> "$LOG" 2>&1 \
    || pip install --no-cache-dir dgl==1.0.2 \
       -f https://data.dgl.ai/wheels/cu113/repo.html --no-deps >> "$LOG" 2>&1

echo "=== verifying imports ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, dgl, rdkit, Bio, scipy, numpy
print(f"  torch  {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  dgl    {dgl.__version__}")
print(f"  rdkit  {rdkit.__version__}")
print(f"  bio    {Bio.__version__}")
PY

echo "=== done ===" | tee -a "$LOG"
