# CASP17 Ligand

CASP17 Ligand is an orchestration framework designed for predicting protein-ligand co-folding structures for CASP17 targets. It integrates multiple state-of-the-art models—including **Boltz-2**, **Protenix**, AlphaFold3, and RoseTTAFold3—into a unified pipeline for data preparation, isolated inference, and ensemble ranking.

## Table of Contents
- [Architecture & Setup](#architecture--setup)
- [Data Preparation](#data-preparation)
- [Running Boltz-2](#running-boltz-2)
- [Running Protenix](#running-protenix)
- [Ranking & Evaluation](#ranking--evaluation)

---

## Architecture & Setup

The framework uses a lightweight orchestration environment to coordinate various model-specific environments, avoiding dependency conflicts. 

1. **Orchestration Environment**: 
   Create the base environment used to run data preparation and inference dispatch scripts.
   ```bash
   conda create -n casp17_ligand python=3.10
   conda activate casp17_ligand
   pip install hydra-core omegaconf pandas rootutils
   ```
2. **Weights Setup**: Use the provided script to symlink necessary weights (e.g., Boltz-2, AF3) to their respective directories.
   ```bash
   bash scripts/setup_weights.sh
   ```

## Data Preparation

The project utilizes a unified data pipeline to minimize redundancy. 

1. Provide an `ensemble_inputs.csv` in your dataset directory: `data/test_cases/{dataset}/`.
2. The unified components will automatically handle sequence parsing, ligand SMILES extraction, and protonation.
3. Method-specific inputs (YAML for Boltz-2, JSON for Protenix) are generated from this shared CSV.

---

## Running Boltz-2

Boltz-2 inference runs via its own dedicated Conda environment (e.g., `boltz`). 

**1. Input Preparation:**
```bash
python casp17_ligand/data/boltz2_input_preparation.py dataset=YOUR_DATASET series=YOUR_SERIES
```
> **Multi-ligand / Dimer Handling**: For targets with multiple ligands, the preparation script automatically computes 3D centers via RDKit and applies exact `<4.5Å` `contact` constraints. This is critical to prevent Boltz-2 from generating divergent, unbound poses.

**2. Inference:**
```bash
python casp17_ligand/models/boltz2_inference.py dataset=YOUR_DATASET
```

### Optimized Parameters (Recommended)
Based on rigorous internal testing, the following hyperparameters yield the best performance:
- `diffusion_samples`: **50** (Default best).
  - *Fallback:* If Out-Of-Memory (OOM) occurs for ultra-large proteins (>1000 aa), lower this to **25**. 10 or fewer is not recommended due to the diversity collapse.
- `step_scale`: **1.5** (Default).
  - *Note on Fallback:* If using 25 samples, you can optionally lower `step_scale` to **1.2**. Our experiments showed that `1.5` and `1.2` perform almost identically, though `1.2` showed a very marginal edge (+0.05 lDDT-PLI) in smaller batch validations. Using either is perfectly fine.
- `sampling_steps`: **200**
- `recycling_steps`: **10**
- `use_potentials`: **true**
- `use_msa_server`: **true**. 
  - *Efficiency Tip:* If running multiple ligands against the **same protein**, only use the MSA server for the first molecule. Copy the generated MSA to `msa/{series}_msa.csv` and specify this path in your subsequent YAMLs to skip redundant MSA generation.
- **Affinity**: Disabled by default for speed. Structural generation is the primary focus.

---

## Running Protenix

Protenix (v1) inference is executed through a local conda environment, avoiding Docker overhead.

**1. Input Preparation:**
```bash
python casp17_ligand/data/protenix_input_preparation.py dataset=YOUR_DATASET series=YOUR_SERIES
```
> **Efficiency Tip**: The Protenix preparation scripts are configured to reuse previously generated MSA and Template data (from AlphaFold3 runs). This significantly accelerates the pipeline.

**2. Inference:**
```bash
python casp17_ligand/models/protenix_inference.py dataset=YOUR_DATASET
```

### Optimized Parameters / Notes
- **Hardware Requirements**: Protenix requires GPUs with substantial VRAM (e.g., V100 32GB or A100).
- **Sampling Strategy**: For optimal coverage and high-quality prediction ensembles, it is recommended to run multiple seeds per target. A standard configuration is **10 seeds × 5 samples = 50 models** per target.
- **High-Precision Settings**: To push for the best structural accuracy, use the following parameters (override the fast defaults):
  - `cycle`: **10** (Recycling steps)
  - `step`: **200** (Diffusion sampling steps)
  - `dtype`: **bf16** (Use mixed precision to save VRAM and maintain numerical stability)
  - **Feature Flags**: Ensure `use_msa: true` and `use_template: true` are enabled to utilize structural priors.

---

## Ranking & Evaluation

After generating predictions across your chosen methods, run the ensemble generation script to rank all poses.

```bash
python casp17_ligand/models/ensemble_generation.py dataset=YOUR_DATASET
```
The pipeline evaluates structures using a dual-ranking system:
1. **RMSD Clustering** to group similar geometric poses.
2. **SuCOS** (Shape and Color Similarity) to evaluate physicochemical interactions, with a fallback to MCS (Maximum Common Substructure) to repair bond-order issues.
