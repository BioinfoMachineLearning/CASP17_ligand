# CASP17_ligand

A consensus pipeline for ligand-pose prediction. Several co-folding methods run
independently on the same target; their poses are checked for physical validity,
clustered by shape overlap, and the five largest clusters become the five
submitted models. This repository drives AlphaFold3, Boltz-2 and Protenix; we
also used SeedFold, which has no public release.

This is the code MULTICOM used in CASP17: 12 RNA-ligand targets, 11
protein-ligand targets, one 13-chain RNA-protein complex, and fragment screens
over 1209 and 647 compounds.

## Why four methods

Across the 20 pose targets, the model we ranked first came from Protenix 9
times, AlphaFold3 4, SeedFold 4, and Boltz-2 3. None of them is reliably best,
and confidence scores do not tell you which one to trust: on T2414 Boltz-2
reported an interface ipTM of 0.953 and still put the ligand 9–10 Å from the
pocket the other three methods agreed on.

So we do not pick a method. We generate 100 models per method per target, throw
away the ones that are physically impossible, and let the remaining poses vote
by clustering. A pose that three methods land on independently is a better bet
than a pose one method is confident about.

## Requirements

Each method keeps its own conda environment; this package only dispatches to
them and post-processes what comes back.

```bash
conda create -n casp17_ligand python=3.10
conda activate casp17_ligand
conda install -c conda-forge pymol-open-source   # pip has no working build
pip install -e .
```

You also need, separately:

- **Boltz-2** and **Protenix** in their own conda environments (`boltz`,
  `protenix`). `casp17_ligand/utils/env_utils.py` finds them via
  `conda info --base`.
- **AlphaFold3** as a docker image, plus its databases and weights. Weights
  require a request to DeepMind — see
  [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).
- A **PoseBench** environment if you want the relax step.
  `casp17_ligand/utils/relax_pose.py` and `casp17_ligand/utils/minimal_relax.py`
  run there under OpenMM, not in this one.

Copy `env.example` to `.env` and fill in where your databases, weights and CASP
group code live:

```bash
cp env.example .env
set -a; source .env; set +a
```

`CASP_GROUP_ID` has no default — submissions filed under the wrong group code
fail silently, so the scripts refuse to guess.

Boltz-2 downloads its own weights on first run; `bash scripts/setup_weights.sh`
links them into `weights/boltz/`. Protenix expects its checkpoints under
`weights/protenix/` (override with `protenix_root_dir` in the model config).

## Running one target end to end

A target needs two files from the prediction center and one CSV row. The two
files are what the per-method input generators read; the CSV row is what the
ensemble stage reads.

```bash
TGT=T2409
SERIES=CASP17_T

mkdir -p data/casp17_data/sequences/$SERIES data/casp17_data/smiles/$SERIES

# Sequence. Check view=sequence against view=all by hand before trusting it --
# mirrors and server feeds have been wrong.
curl -s "https://predictioncenter.org/casp17/target.cgi?target=$TGT&view=sequence" \
  > data/casp17_data/sequences/$SERIES/$TGT.txt

# Ligand SMILES (tab-separated: ID, Name, SMILES, Task)
curl -s "https://predictioncenter.org/download_area/CASP17/extra_experiments/ligands/$TGT.smiles.txt" \
  > data/casp17_data/smiles/$SERIES/$TGT.tsv
```

Then add one row to `data/test_cases/casp17_T/ensemble_inputs.csv`:

```csv
target,entity_type,protein_sequence,ligand_name,ligand_smiles,experimental_affinity,notes
T2409,protein,MNTSGLGWMSATEMAAQ...,U04,CCOC(=O)NC1=CC=CC=C1,,472 aa amidase
```

RNA targets use the same schema with `rna_sequence` in place of
`protein_sequence`, `rna` as the entity type, and `CASP17_R` as the series.

```bash
# 1. Per-method inputs (YAML for Boltz-2, JSON for AF3 and Protenix)
python casp17_ligand/data/boltz2_input_preparation.py   --config-name boltz2_input_preparation_casp17_T
python casp17_ligand/data/af3_input_preparation.py      --config-name af3_input_preparation_casp17_T
python casp17_ligand/data/protenix_input_preparation.py --config-name protenix_input_preparation_casp17_T

# 2. Inference: two rounds x 50 models per method
bash casp17/scripts/run_local_pipeline_T.sh $TGT        # Boltz-2 + Protenix, two GPUs
bash casp17/scripts/run_af3_casp17_T.sh $TGT            # AlphaFold3, MSA once then r1 r2

# 3. Filter, cluster, rank
bash scripts/run_ensemble_r1r2_pipeline.sh $TGT 16 --yes

# 4. Write the five CASP LG submission files
python casp17/scripts/generate_submission.py \
  --cluster-csv outputs/ensemble_r1r2/cluster_experiment_detail_latest_$TGT.csv \
  --output-dir casp17/submissions/casp17_T \
  --smiles-dir data/casp17_data/smiles/CASP17_T
```

Step 2 assumes the AF3 image is named `alphafold3-3.0.3_casp17`; set
`AF3_IMAGE` if yours differs. RNA targets run through
`run_local_pipeline_R.sh` and `run_protenix_rna_dualckpt.sh` instead.

Multi-chain complexes (the M series) skip step 1: a 13-chain assembly does not
fit the one-row schema, so its inputs are written by hand. Copy
`data/test_cases/casp17_M/boltz2_inputs/M2415_input.yaml` and the Protenix JSON
next to it as templates, check them with
`python casp17/scripts/verify_target_inputs.py`, then run
`run_boltz2_r1r2_casp17_M.sh` and `run_protenix_dualckpt_casp17_M.sh`. Boltz-2
needs the narrow-batch route there: 13 chains OOM at the default batch width, so
that driver collects 4 models per round over 25 rounds.

Step 3 does the work. Each predicted structure is split into receptor PDB and
ligand SDF, run through PoseBusters, and relaxed only if it fails — a relax that
does not fix the failure is discarded rather than shipped. Surviving poses are
clustered with Butina on SuCOS shape overlap, and cluster representatives are
ordered by cluster size.

One step is not optional and not automated: look at the poses in PyMOL before
submitting.

```bash
python casp17_ligand/analysis/generate_alignment_zips.py --targets $TGT
```

This writes a `.pml` that loads all five models aligned on the receptor. We
caught a wrong-pocket MODEL 1 this way more than once.

## Layout

```
casp17_ligand/       the package: input prep, inference dispatch, ensemble, analysis
casp17/scripts/      inference drivers, submission writer, CASP format validator
scripts/             ensemble pipeline entry point, weight setup
scripts/tests/       unit tests (pytest)
configs/             Hydra configs for the R and T series
data/test_cases/     per-target input manifests
```

## Upstream

We ran these from their own checkouts rather than vendoring them:

| | | |
|---|---|---|
| AlphaFold3 | google-deepmind/alphafold3 | `4ca8a65` |
| Boltz-2 | jwohlwend/boltz | `cb04aec` |
| Protenix | bytedance/Protenix | `21c0b8b` |
| PoseBench | BioinfoMachineLearning/PoseBench | `c5d728d` |
| RTMScore | sc8668/RTMScore | `3280de1` |

`casp17/scripts/LG_validation.py` is the CASP organisers' format checker, patched
to accept receptor lines and to take its paths from the environment.
