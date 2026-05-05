"""
Compare Protenix lDDT on L4000 with vs without Cys145 pocket anchoring.

With Cys145:    outputs/protenix/casp16_l4000/          (already evaluated)
Without Cys145: outputs/protenix/casp16_l4000_no_Cys145/ (evaluated here)

Results saved to: outputs/protenix_cys145_comparison/
"""

import os
import sys
import json
import glob
import tempfile
import subprocess
import logging
import concurrent.futures
import threading

import numpy as np
import pandas as pd

# ── project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from casp17_ligand.models.ensemble_generation import cif_to_pdb_sdf, _detect_ligand_chains
from casp17_ligand.analysis.evaluate_ensemble import run_ost_compare, _extract_best_scores, read_smiles_from_tsv
from casp17_ligand.utils.data_utils import extract_protein_and_ligands_with_prody

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
TARGETS = [
    "L4001", "L4002", "L4003", "L4004", "L4011", "L4013", "L4014", "L4015",
    "L4016", "L4017", "L4018", "L4019", "L4020", "L4022", "L4023", "L4024",
    "L4025", "L4026", "L4027", "L4028",
]

WITH_CYS_CACHE_DIR   = "outputs/ensemble/casp16_l4000/targets"
NO_CYS_CIF_DIR       = "outputs/protenix/casp16_l4000_no_Cys145"
COMPARISON_DIR       = "outputs/protenix_cys145_comparison"
SMILES_DIR           = "data/casp16_data/smiles/L4000"
REF_DIR              = "data/casp16_data/struct/L4000_prepared"

DOCKER_THREADS = 8   # parallel Docker calls per target


# ── Step 1: Read "with Cys145" scores directly from existing cache ─────────────
def read_with_cys145_scores(target: str) -> dict:
    """Return {model_key: lddt_pli} for all protenix models, with Cys145."""
    cache_path = os.path.join(WITH_CYS_CACHE_DIR, target, "score_cache_docker.json")
    if not os.path.exists(cache_path):
        log.warning(f"[{target}] score_cache missing: {cache_path}")
        return {}
    with open(cache_path) as f:
        cache = json.load(f)
    return {
        k: v["lddt_pli"]
        for k, v in cache.items()
        if k.startswith("protenix") and v.get("lddt_pli") is not None
    }


# ── Step 2: Evaluate "no Cys145" protenix models with Docker ─────────────────
def _prepare_ref_sdfs(target: str, tmp_dir: str, smiles_dir: str, ref_dir: str):
    """Convert reference ligand PDBs to SDFs. Returns list of SDF paths."""
    ref_target_dir = os.path.join(ref_dir, target)
    ref_lig_pdbs = sorted(glob.glob(os.path.join(ref_target_dir, "ligand_*.pdb")))
    smiles_file = os.path.join(smiles_dir, f"{target}.tsv")
    ligands = read_smiles_from_tsv(smiles_file)
    smiles_by_name = {name: smi for name, smi in ligands}

    ref_sdf_paths = []
    for ref_lig_pdb in ref_lig_pdbs:
        basename = os.path.basename(ref_lig_pdb)
        parts = basename.replace(".pdb", "").split("_")
        resname = parts[1] if len(parts) >= 3 else ""
        lig_smiles = smiles_by_name.get(resname) or (ligands[0][1] if ligands else None)
        if not lig_smiles:
            continue
        ref_sdf = os.path.join(tmp_dir, basename.replace(".pdb", ".sdf"))
        try:
            extract_protein_and_ligands_with_prody(
                input_pdb_file=ref_lig_pdb,
                protein_output_pdb_file=None,
                ligands_output_sdf_file=ref_sdf,
                ligand_smiles=lig_smiles,
                load_hetatms_as_ligands=True,
                generify_resnames=False,
                write_output_files=True,
            )
            if os.path.exists(ref_sdf):
                ref_sdf_paths.append(ref_sdf)
        except Exception as e:
            log.warning(f"[{target}] ref SDF failed for {basename}: {e}")
    return ref_sdf_paths


def evaluate_no_cys145_target(target: str) -> dict:
    """
    Evaluate all protenix CIFs in no_Cys145 dir for one target.

    For dimer targets (L4000 all have 2 ligand chains), extract BOTH ligand
    chains from each CIF and evaluate against all reference ligands.
    Take max lDDT across both chains as the model score.

    Returns {model_key: lddt_pli}.
    """
    # Output cache (to avoid re-running Docker if script is restarted)
    out_dir = os.path.join(COMPARISON_DIR, target)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "score_cache_no_cys145.json")
    score_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            score_cache = json.load(f)

    # Find CIF files
    cif_paths = sorted(glob.glob(
        os.path.join(NO_CYS_CIF_DIR, target, "seed_*", "predictions", f"{target}_sample_*.cif")
    ))
    if not cif_paths:
        log.warning(f"[{target}] No CIF files found in no_Cys145 dir")
        return {}

    # Reference protein PDB
    ref_pdb = os.path.join(REF_DIR, target, "protein_aligned.pdb")
    if not os.path.exists(ref_pdb):
        log.warning(f"[{target}] Reference PDB missing: {ref_pdb}")
        return {}

    # Prepare reference SDFs (done once per target)
    tmp_root = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    ref_sdf_paths = _prepare_ref_sdfs(target, tmp_root, SMILES_DIR, REF_DIR)
    if not ref_sdf_paths:
        log.warning(f"[{target}] No reference SDFs prepared")
        return {}

    # CIF → PDB/SDF conversion directory
    cif_conv_dir = os.path.join(out_dir, "cif_converted")
    os.makedirs(cif_conv_dir, exist_ok=True)

    # Detect ligand chains from first CIF (same for all models)
    first_cif = cif_paths[0]
    lig_info = _detect_ligand_chains(first_cif, prot_chain="A")
    if not lig_info:
        log.warning(f"[{target}] Could not detect ligand chains from {first_cif}")
        return {}

    # For dimer: largest n_atoms chains are the main ligands
    max_atoms = max(n for _, n in lig_info)
    # Threshold: chains with atom count >= 50% of max are "main ligands"
    main_lig_chains = [ch for ch, n in lig_info if n >= max_atoms * 0.5]
    log.info(f"[{target}] Ligand chains: {lig_info} → evaluating: {main_lig_chains}")

    # SMILES for each chain (use the first SMILES from TSV for all chains)
    smiles_file = os.path.join(SMILES_DIR, f"{target}.tsv")
    ligands = read_smiles_from_tsv(smiles_file)
    # Use the SMILES of the main ligand (first unique SMILES by heavy atom count)
    smiles_by_atoms = {}
    for name, smi in ligands:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ha = mol.GetNumHeavyAtoms()
            if ha not in smiles_by_atoms:
                smiles_by_atoms[ha] = smi
    chain_smiles = {}
    for ch, n in lig_info:
        if ch in main_lig_chains:
            chain_smiles[ch] = smiles_by_atoms.get(n) or (ligands[0][1] if ligands else None)

    cache_lock = threading.Lock()

    def _eval_one_cif(idx_cif):
        idx, cif_path = idx_cif
        model_key = f"protenix_model{idx}"

        with cache_lock:
            if model_key in score_cache:
                return model_key, score_cache[model_key]

        name_base = f"{target}_protenix_model_{idx}"
        best_lddt = None

        for lig_ch in main_lig_chains:
            smi = chain_smiles.get(lig_ch)
            chain_conv_dir = os.path.join(cif_conv_dir, f"ch_{lig_ch}")
            os.makedirs(chain_conv_dir, exist_ok=True)
            name = f"{name_base}_lig{lig_ch}"

            try:
                pdb, sdf = cif_to_pdb_sdf(
                    cif_path, chain_conv_dir, name,
                    smiles=smi, prot_chain="A", lig_chain=lig_ch,
                    err_log_path=os.path.join(out_dir, "err.log"),
                )
            except Exception as e:
                log.warning(f"[{target}] cif_to_pdb_sdf failed for {model_key} ch {lig_ch}: {e}")
                continue

            if not pdb or not sdf:
                continue

            out_json = os.path.join(tmp_root, f"ost_{model_key}_ch{lig_ch}.json")
            try:
                run_ost_compare(pdb, sdf, ref_pdb, ref_sdf_paths, out_json)
                with open(out_json) as f:
                    metrics = json.load(f)
                lddt_pli, _, valid = _extract_best_scores(metrics)
                if valid and lddt_pli is not None:
                    if best_lddt is None or lddt_pli > best_lddt:
                        best_lddt = lddt_pli
            except Exception as e:
                log.warning(f"[{target}] Docker eval failed for {model_key} ch {lig_ch}: {e}")

        result = best_lddt
        with cache_lock:
            score_cache[model_key] = result
            with open(cache_path, "w") as f:
                json.dump(score_cache, f, indent=2)
        return model_key, result

    # Build list of (idx, cif_path) pairs
    indexed_cifs = list(enumerate(cif_paths))

    log.info(f"[{target}] Evaluating {len(indexed_cifs)} CIFs × {len(main_lig_chains)} chains with Docker ({DOCKER_THREADS} threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOCKER_THREADS) as executor:
        for model_key, lddt in executor.map(_eval_one_cif, indexed_cifs):
            pass  # results already saved to cache

    # Return valid scores
    return {k: v for k, v in score_cache.items() if v is not None}


# ── Step 3: Aggregate and compare ─────────────────────────────────────────────
def compare_targets():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Initialize PyMOL (required for cif_to_pdb_sdf / _detect_ligand_chains)
    import pymol
    pymol.finish_launching(["pymol", "-qc"])

    rows = []
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Processing {target}")
        log.info(f"{'='*60}")

        # With Cys145 (existing)
        with_scores = read_with_cys145_scores(target)
        with_vals = [v for v in with_scores.values() if v is not None]

        # Without Cys145 (evaluate now)
        no_scores = evaluate_no_cys145_target(target)
        no_vals = [v for v in no_scores.values() if v is not None]

        row = {
            "target": target,
            "with_cys145_n": len(with_vals),
            "with_cys145_top1": max(with_vals) if with_vals else None,
            "with_cys145_avg":  sum(with_vals) / len(with_vals) if with_vals else None,
            "no_cys145_n": len(no_vals),
            "no_cys145_top1": max(no_vals) if no_vals else None,
            "no_cys145_avg":  sum(no_vals) / len(no_vals) if no_vals else None,
        }
        row["delta_top1"] = (
            (row["with_cys145_top1"] - row["no_cys145_top1"])
            if row["with_cys145_top1"] is not None and row["no_cys145_top1"] is not None
            else None
        )
        row["delta_avg"] = (
            (row["with_cys145_avg"] - row["no_cys145_avg"])
            if row["with_cys145_avg"] is not None and row["no_cys145_avg"] is not None
            else None
        )
        rows.append(row)
        log.info(
            f"[{target}] With: top1={row['with_cys145_top1']:.4f} avg={row['with_cys145_avg']:.4f} | "
            f"No: top1={row['no_cys145_top1']:.4f} avg={row['no_cys145_avg']:.4f} | "
            f"Δtop1={row['delta_top1']:+.4f} Δavg={row['delta_avg']:+.4f}"
            if all(v is not None for v in [
                row['with_cys145_top1'], row['with_cys145_avg'],
                row['no_cys145_top1'], row['no_cys145_avg']
            ])
            else f"[{target}] incomplete scores"
        )

    df = pd.DataFrame(rows)

    # Print summary table
    print("\n" + "="*100)
    print("Protenix lDDT comparison: WITH vs WITHOUT Cys145 pocket anchoring (L4000, 20 targets)")
    print("="*100)
    fmt_cols = ["target",
                "with_cys145_top1", "no_cys145_top1", "delta_top1",
                "with_cys145_avg",  "no_cys145_avg",  "delta_avg"]
    print(df[fmt_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print macro averages
    valid = df.dropna(subset=["with_cys145_top1", "no_cys145_top1"])
    if len(valid):
        print("\n" + "-"*60)
        print(f"Macro average over {len(valid)} targets:")
        print(f"  Top-1 lDDT:  With={valid['with_cys145_top1'].mean():.4f}  "
              f"No={valid['no_cys145_top1'].mean():.4f}  "
              f"Δ={valid['delta_top1'].mean():+.4f}")
        print(f"  Avg lDDT:    With={valid['with_cys145_avg'].mean():.4f}  "
              f"No={valid['no_cys145_avg'].mean():.4f}  "
              f"Δ={valid['delta_avg'].mean():+.4f}")
        print("-"*60)

    # Save CSV
    out_csv = os.path.join(COMPARISON_DIR, "protenix_cys145_comparison.csv")
    df.to_csv(out_csv, index=False)
    log.info(f"\nResults saved → {out_csv}")

    return df


if __name__ == "__main__":
    compare_targets()
