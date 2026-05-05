#!/usr/bin/env python3
"""Direct lddt-pli evaluation for Protenix outputs (oracle metrics).

Usage:
    python casp17_ligand/analysis/eval_protenix_lddt.py \\
        --protenix_dir outputs/protenix/casp16_l3000_struct_zn \\
        --reference_dir data/casp16_data/struct/L3000_prepared \\
        --smiles_dir data/casp16_data/smiles/L3000 \\
        --output_csv outputs/ensemble/casp16_l3000_struct_zn/eval_protenix_zn_oracle.csv \\
        --n_eval 50 \\
        --num_workers 12

For each target, evaluates all N models and computes ORACLE metrics:
  - oracle_top1_lddt:      max lddt-pli across all N models
  - oracle_top5_avg_lddt:  average of top-5 models sorted by lddt-pli (not confidence)

ZN/NAG chains are automatically excluded via CIF chain auto-detection.
Already-computed OST scores are cached, so re-runs only score new models.
"""

import argparse
import concurrent.futures
import csv
import glob
import json
import logging
import os
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CIF chain detection & conversion (reuse from ensemble_generation)
# ─────────────────────────────────────────────────────────────

def _detect_chains_from_cif(cif_path):
    """Parse CIF atom_site to get chain atom counts without PyMOL."""
    chains = {}
    try:
        in_loop = False
        headers = []
        asym_col = None
        with open(cif_path) as f:
            for line in f:
                line_s = line.strip()
                if line_s.startswith("loop_"):
                    in_loop = True
                    headers = []
                    asym_col = None
                    continue
                if in_loop and line_s.startswith("_atom_site."):
                    col_name = line_s.split(".")[1]
                    headers.append(col_name)
                    if col_name == "label_asym_id":
                        asym_col = len(headers) - 1
                elif in_loop and line_s.startswith(("ATOM", "HETATM")):
                    parts = line_s.split()
                    if asym_col is not None and asym_col < len(parts):
                        ch = parts[asym_col]
                        chains[ch] = chains.get(ch, 0) + 1
                elif in_loop and line_s and not line_s.startswith(("_", "#")):
                    if lines_s.startswith(("ATOM", "HETATM")):
                        pass
                    else:
                        in_loop = False
    except Exception:
        pass
    return chains


def detect_protein_and_ligand_chains(cif_path):
    """Return (prot_chain, lig_chain) by parsing CIF atom_site rows."""
    # Read chain atom counts from CIF
    chains = {}
    try:
        in_loop = False
        headers = []
        asym_col = None
        comp_col = None
        chain_resnames = {}  # chain -> set of residue names

        with open(cif_path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("loop_"):
                    in_loop = True
                    headers = []
                    asym_col = None
                    comp_col = None
                    continue
                if in_loop and s.startswith("_atom_site."):
                    col = s.split(".", 1)[1]
                    headers.append(col)
                    if col == "label_asym_id":
                        asym_col = len(headers) - 1
                    if col == "label_comp_id":
                        comp_col = len(headers) - 1
                elif in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
                    parts = s.split()
                    if asym_col is not None and asym_col < len(parts):
                        ch = parts[asym_col]
                        chains[ch] = chains.get(ch, 0) + 1
                        if comp_col is not None and comp_col < len(parts):
                            resname = parts[comp_col]
                            chain_resnames.setdefault(ch, set()).add(resname)
                elif in_loop and s and not s.startswith(("_", "#", "loop_", "ATOM", "HETATM", ";")):
                    in_loop = False
    except Exception as e:
        log.warning(f"Failed to parse chains from {cif_path}: {e}")
        return "A", "B"

    if not chains:
        return "A", "B"

    # Protein chain = largest chain
    prot_chain = max(chains.items(), key=lambda x: x[1])[0]

    # Ligand chain = second-largest (not the protein), pick largest non-protein
    non_prot = [(ch, n) for ch, n in chains.items() if ch != prot_chain and n > 0]
    if not non_prot:
        return prot_chain, "B"

    # Pick the largest non-protein chain (exclude ions: n==1)
    non_ion = [(ch, n) for ch, n in non_prot if n > 1]
    if non_ion:
        lig_chain = max(non_ion, key=lambda x: x[1])[0]
    else:
        lig_chain = max(non_prot, key=lambda x: x[1])[0]

    return prot_chain, lig_chain


def cif_to_pdb_sdf_simple(cif_path, out_dir, name, smiles=None):
    """Convert a single CIF to protein PDB + ligand SDF.

    Uses PyMOL + ProDy (same as ensemble_generation.py cif_to_pdb_sdf).
    Auto-detects protein and ligand chains to exclude ZN/NAG.
    """
    from pymol import cmd as pymol_cmd
    from casp17_ligand.models.ensemble_generation import cif_to_pdb_sdf

    prot_chain, lig_chain = detect_protein_and_ligand_chains(cif_path)
    log.debug(f"[{name}] auto-detected chains: prot={prot_chain}, lig={lig_chain}")

    return cif_to_pdb_sdf(
        cif_path, out_dir, name,
        smiles=smiles,
        prot_chain=prot_chain,
        lig_chain=lig_chain,
    )


# ─────────────────────────────────────────────────────────────
# Confidence score loading
# ─────────────────────────────────────────────────────────────

def load_ranking_score(summary_json_path):
    """Load ranking_score from Protenix summary_confidence JSON."""
    try:
        with open(summary_json_path) as f:
            d = json.load(f)
        # Try different key names
        score = d.get("ranking_score") or d.get("ranking_score_v") or d.get("iptm")
        if score is None:
            # Fallback: compute from iptm
            score = d.get("iptm", 0.0)
        return float(score)
    except Exception:
        return -1.0


def discover_models(protenix_target_dir, target):
    """Return list of (cif_path, json_path, ranking_score) sorted by score desc."""
    cif_paths = sorted(glob.glob(
        os.path.join(protenix_target_dir, "seed_*", "predictions", f"{target}_sample_*.cif")
    ))
    results = []
    for cif in cif_paths:
        # Find matching summary JSON
        json_path = cif.replace(".cif", "").replace(
            f"{target}_sample_", f"{target}_summary_confidence_sample_"
        ) + ".json"
        # Alternative naming: same dir, summary_confidence
        if not os.path.exists(json_path):
            parts = os.path.basename(cif).split("_")
            sample_num = parts[-1].replace(".cif", "")
            json_path = os.path.join(
                os.path.dirname(cif),
                f"{target}_summary_confidence_sample_{sample_num}.json"
            )
        score = load_ranking_score(json_path) if os.path.exists(json_path) else -1.0
        results.append((cif, json_path, score))

    results.sort(key=lambda x: -x[2])  # descending score
    return results


# ─────────────────────────────────────────────────────────────
# OST Docker evaluation (same as evaluate_ensemble.py)
# ─────────────────────────────────────────────────────────────

def run_ost_compare(model_pdb, model_sdfs, ref_pdb, ref_sdfs, output_json):
    import subprocess
    if isinstance(model_sdfs, str):
        model_sdfs = [model_sdfs]
    if isinstance(ref_sdfs, str):
        ref_sdfs = [ref_sdfs]

    cwd = os.path.abspath(os.getcwd())
    cmd = [
        "docker", "run", "--rm",
        "-v", "/bmlfast:/bmlfast",
        "-w", cwd,
        "registry.scicore.unibas.ch/schwede/openstructure:latest",
        "compare-ligand-structures",
        "-m", os.path.abspath(model_pdb),
        "-ml"] + [os.path.abspath(s) for s in model_sdfs] + [
        "-r", os.path.abspath(ref_pdb),
        "-rl"] + [os.path.abspath(s) for s in ref_sdfs] + [
        "-o", os.path.abspath(output_json),
        "--lddt-pli", "--rmsd"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"OST failed: {result.stderr[:500]}")
    return output_json


def extract_lddt(out_json):
    """Extract best lddt-pli from OST JSON output."""
    with open(out_json) as f:
        metrics = json.load(f)
    lddt_pli = None
    if "lddt_pli" in metrics:
        assigned = metrics["lddt_pli"].get("assigned_scores", [])
        if assigned:
            scores = [s["score"] for s in assigned if s.get("score") is not None]
            if scores:
                lddt_pli = max(scores)
    return lddt_pli


def read_smiles_from_tsv(tsv_path):
    """Read the first SMILES from TSV."""
    try:
        with open(tsv_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    return parts[2]
    except Exception:
        pass
    return None


def build_ref_sdfs(ref_target_dir, smiles_dir, target, temp_dir):
    """Build reference SDF files from PDB + SMILES."""
    from casp17_ligand.utils.data_utils import extract_protein_and_ligands_with_prody

    ref_lig_pdbs = sorted(glob.glob(os.path.join(ref_target_dir, "ligand_*.pdb")))
    smiles_file = os.path.join(smiles_dir, f"{target}.tsv")
    smiles = read_smiles_from_tsv(smiles_file)

    ref_sdf_paths = []
    for ref_lig_pdb in ref_lig_pdbs:
        # Only include the main ligand (not ZN/NAG) by filtering on SMILES match
        basename = os.path.basename(ref_lig_pdb)
        parts = basename.replace(".pdb", "").split("_")
        resname = parts[1] if len(parts) >= 3 else ""
        # Skip known cofactors
        if resname.upper() in {"ZN", "NAG", "MG", "CA", "FE", "MN", "ZN2", "CU", "CO"}:
            log.debug(f"Skipping cofactor ref ligand: {basename}")
            continue

        ref_sdf = os.path.join(temp_dir, basename.replace(".pdb", ".sdf"))
        try:
            extract_protein_and_ligands_with_prody(
                input_pdb_file=ref_lig_pdb,
                protein_output_pdb_file=None,
                ligands_output_sdf_file=ref_sdf,
                ligand_smiles=smiles,
                load_hetatms_as_ligands=True,
                generify_resnames=False,
                write_output_files=True
            )
            if os.path.exists(ref_sdf):
                ref_sdf_paths.append(ref_sdf)
        except Exception as e:
            log.warning(f"[{target}] Failed ref SDF from {basename}: {e}")

    return ref_sdf_paths


# ─────────────────────────────────────────────────────────────
# Per-target evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_target(target, protenix_dir, reference_dir, smiles_dir,
                    output_dir, top5_n=5):
    """Evaluate top-1 and top-5 Protenix models for one target."""
    protenix_target_dir = os.path.join(protenix_dir, target)
    ref_target_dir = os.path.join(reference_dir, target)
    ref_pdb = os.path.join(ref_target_dir, "protein_aligned.pdb")

    if not os.path.isdir(protenix_target_dir):
        log.warning(f"[{target}] Protenix dir missing: {protenix_target_dir}")
        return None
    if not os.path.isdir(ref_target_dir):
        log.warning(f"[{target}] Reference dir missing: {ref_target_dir}")
        return None
    if not os.path.exists(ref_pdb):
        log.warning(f"[{target}] Reference PDB missing: {ref_pdb}")
        return None

    # Sort models by ranking_score
    models = discover_models(protenix_target_dir, target)
    if not models:
        log.warning(f"[{target}] No Protenix CIF files found")
        return None

    log.info(f"[{target}] {len(models)} models found, top score={models[0][2]:.4f}")

    # Temp working dir
    work_dir = os.path.join(output_dir, "eval_work", target)
    os.makedirs(work_dir, exist_ok=True)

    # Build reference SDFs (filters out ZN/NAG)
    ref_sdfs = build_ref_sdfs(ref_target_dir, smiles_dir, target, work_dir)
    if not ref_sdfs:
        log.warning(f"[{target}] No reference SDF built")
        return None

    smiles_file = os.path.join(smiles_dir, f"{target}.tsv")
    smiles = read_smiles_from_tsv(smiles_file)

    # Get top N models (for top-1 and top-5)
    top_models = models[:top5_n]

    # Score cache per target
    cache_file = os.path.join(output_dir, "eval_work", target, "score_cache.json")
    score_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                score_cache = json.load(f)
        except Exception:
            pass
    cache_lock = threading.Lock()

    def score_model(i, cif_path, conf_score):
        model_name = f"{target}_protenix_rank{i+1}"
        cache_key = os.path.basename(cif_path)

        with cache_lock:
            if cache_key in score_cache:
                return i, score_cache[cache_key]

        # Convert CIF → PDB + SDF
        try:
            pdb_path, sdf_path = cif_to_pdb_sdf_simple(
                cif_path, work_dir, model_name, smiles=smiles
            )
        except Exception as e:
            log.warning(f"[{target}] CIF conversion failed for rank {i+1}: {e}")
            return i, None

        if not pdb_path or not sdf_path:
            log.warning(f"[{target}] CIF conversion returned None for rank {i+1}")
            return i, None

        # Run OST
        out_json = os.path.join(work_dir, f"{model_name}_ost.json")
        try:
            run_ost_compare(pdb_path, sdf_path, ref_pdb, ref_sdfs, out_json)
            lddt = extract_lddt(out_json)
        except Exception as e:
            log.warning(f"[{target}] OST failed for rank {i+1}: {e}")
            lddt = None

        result = {"lddt_pli": lddt, "conf_score": conf_score,
                  "cif": os.path.basename(cif_path)}

        with cache_lock:
            score_cache[cache_key] = result
            try:
                with open(cache_file, "w") as f:
                    json.dump(score_cache, f)
            except Exception:
                pass

        return i, result

    # Run in parallel (Docker IO bound)
    lddt_scores = [None] * len(top_models)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(score_model, i, cif, score): i
            for i, (cif, _, score) in enumerate(top_models)
        }
        for fut in concurrent.futures.as_completed(futs):
            i, res = fut.result()
            lddt_scores[i] = res

    # Compute ORACLE metrics: sort all evaluated models by lddt-pli
    valid_scores = sorted(
        [r["lddt_pli"] for r in lddt_scores if r and r["lddt_pli"] is not None],
        reverse=True
    )

    oracle_top1 = valid_scores[0] if valid_scores else None
    oracle_top5_avg = float(np.mean(valid_scores[:5])) if len(valid_scores) >= 1 else None

    t1_s = f"{oracle_top1:.4f}" if oracle_top1 is not None else "N/A"
    t5a_s = f"{oracle_top5_avg:.4f}" if oracle_top5_avg is not None else "N/A"
    log.info(f"[{target}] oracle_top1={t1_s}, oracle_top5_avg={t5a_s} "
             f"(evaluated {len(valid_scores)}/{len(top_models)} models)")

    return {
        "target": target,
        "n_models_total": len(models),
        "n_evaluated": len(valid_scores),
        "oracle_top1_lddt": oracle_top1,
        "oracle_top5_avg_lddt": oracle_top5_avg,
    }


def _worker(args):
    (target, protenix_dir, reference_dir, smiles_dir, output_dir, top5_n) = args
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    try:
        return evaluate_target(target, protenix_dir, reference_dir,
                                smiles_dir, output_dir, top5_n)
    except Exception as e:
        log.error(f"Worker failed for {target}: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Direct lddt-pli eval for Protenix outputs")
    parser.add_argument("--protenix_dir", default="outputs/protenix/casp16_l3000_struct_zn",
                        help="Root dir containing {TARGET}/seed_*/predictions/*.cif")
    parser.add_argument("--reference_dir", default="data/casp16_data/struct/L3000_prepared",
                        help="Root dir containing {TARGET}/protein_aligned.pdb + ligand_*.pdb")
    parser.add_argument("--smiles_dir", default="data/casp16_data/smiles/L3000",
                        help="Dir containing {TARGET}.tsv SMILES files")
    parser.add_argument("--output_csv", default=None,
                        help="Output CSV path (default: outputs/ensemble/casp16_l3000_struct_zn/eval_protenix_zn.csv)")
    parser.add_argument("--output_dir", default="outputs/ensemble/casp16_l3000_struct_zn",
                        help="Output root dir for working files and CSV")
    parser.add_argument("--n_eval", type=int, default=50,
                        help="Number of models to evaluate per target (all by default=50)")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of parallel target workers")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Specific targets to evaluate (default: all)")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.output_dir, "eval_protenix_zn_oracle.csv")

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover targets
    if args.targets:
        targets = args.targets
    else:
        targets = sorted([
            d for d in os.listdir(args.protenix_dir)
            if os.path.isdir(os.path.join(args.protenix_dir, d))
            and d.startswith("L")
        ])

    log.info(f"Evaluating {len(targets)} targets with {args.num_workers} workers")

    import multiprocessing
    worker_args = [
        (t, args.protenix_dir, args.reference_dir, args.smiles_dir,
         args.output_dir, args.n_eval)
        for t in targets
    ]

    all_results = []
    if args.num_workers <= 1:
        for wa in worker_args:
            r = _worker(wa)
            if r:
                all_results.append(r)
    else:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for r in pool.imap_unordered(_worker, worker_args):
                if r:
                    all_results.append(r)
                    if len(all_results) % 20 == 0:
                        log.info(f"Progress: {len(all_results)}/{len(targets)} targets done")

    all_results.sort(key=lambda x: x["target"])

    if not all_results:
        log.error("No results produced!")
        return

    # Write CSV
    fieldnames = [
        "target", "n_models_total", "n_evaluated",
        "oracle_top1_lddt", "oracle_top5_avg_lddt",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

        # Summary row
        valid_top1 = [r["oracle_top1_lddt"] for r in all_results if r["oracle_top1_lddt"] is not None]
        valid_top5_avg = [r["oracle_top5_avg_lddt"] for r in all_results if r["oracle_top5_avg_lddt"] is not None]

        avg_row = {
            "target": "AVERAGE",
            "n_models_total": "",
            "n_evaluated": f"n={len(valid_top1)}/{len(targets)} targets",
            "oracle_top1_lddt": f"{np.mean(valid_top1):.4f}" if valid_top1 else "",
            "oracle_top5_avg_lddt": f"{np.mean(valid_top5_avg):.4f}" if valid_top5_avg else "",
        }
        writer.writerow(avg_row)

    print(f"\nResults saved to {args.output_csv}")
    print(f"Targets evaluated: {len(all_results)}/{len(targets)}")
    if valid_top1:
        print(f"Oracle Top-1  lddt-pli (best of all models):      {np.mean(valid_top1):.4f} (n={len(valid_top1)})")
    if valid_top5_avg:
        print(f"Oracle Top-5 avg lddt-pli (avg of best 5 by lddt): {np.mean(valid_top5_avg):.4f} (n={len(valid_top5_avg)})")


if __name__ == "__main__":
    main()
