#!/usr/bin/env python3
"""Evaluate lddt-pli for a single method's CIF outputs against reference structures.

Standalone script for quick evaluation of any method/dataset combination.
Reuses project utilities (cif_to_pdb_sdf, run_ost_compare) without the full ensemble pipeline.

Usage:
    # Boltz2 L1000 first wave (50 models)
    python scripts/eval_lddt_method.py \
        --method boltz2 \
        --output_dir outputs/boltz2/casp16_l1000 \
        --series L1000 \
        --workers 4

    # Boltz2 L1000 R2 (50 models, seed-based layout)
    python scripts/eval_lddt_method.py \
        --method boltz2 \
        --output_dir outputs/boltz2/casp16_l1000_r2 \
        --series L1000 \
        --seed_layout \
        --workers 4

    # AF3 L1000 R2
    python scripts/eval_lddt_method.py \
        --method af3 \
        --output_dir outputs/alphafold3/casp16_l1000_r2 \
        --series L1000 \
        --workers 4

    # Specific targets only
    python scripts/eval_lddt_method.py \
        --method boltz2 \
        --output_dir outputs/boltz2/casp16_l1000 \
        --series L1000 \
        --targets L1001,L1002
"""

import argparse
import csv
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rootutils

root = rootutils.find_root(search_from=__file__, indicator=".project-root")
sys.path.insert(0, str(root))

from rdkit import Chem
from rdkit.Chem import AllChem

from casp17_ligand.utils.data_utils import extract_protein_and_ligands_with_prody
from casp17_ligand.models.ensemble_generation import cif_to_pdb_sdf, _detect_ligand_chains

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Reference data paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(root, "data", "casp16_data")


def get_reference_paths(series: str):
    """Return (struct_dir, smiles_dir) for a series like L1000."""
    struct_dir = os.path.join(DATA_DIR, "struct", f"{series}_prepared")
    smiles_dir = os.path.join(DATA_DIR, "smiles", series)
    return struct_dir, smiles_dir


def read_smiles_from_tsv(tsv_path):
    ligands = []
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            ligands.append((parts[1], parts[2]))  # (name, smiles)
    return ligands


# ── OST Docker evaluation ─────────────────────────────────────────────────────
def run_ost_compare(model_pdb, model_sdfs, ref_pdb, ref_sdfs, output_json):
    if isinstance(model_sdfs, str):
        model_sdfs = [model_sdfs]
    if isinstance(ref_sdfs, str):
        ref_sdfs = [ref_sdfs]

    # Collect all unique parent directories that need mounting
    all_files = [model_pdb] + model_sdfs + [ref_pdb] + ref_sdfs + [output_json]
    mount_dirs = set()
    for f in all_files:
        mount_dirs.add(os.path.dirname(os.path.abspath(f)))

    # Build volume mounts: each unique dir gets mounted at the same path inside Docker
    volumes = []
    for d in mount_dirs:
        volumes.extend(["-v", f"{d}:{d}"])

    cmd = [
        "docker", "run", "--rm",
    ] + volumes + [
        "registry.scicore.unibas.ch/schwede/openstructure:latest",
        "compare-ligand-structures",
        "-m", os.path.abspath(model_pdb),
        "-ml"] + [os.path.abspath(s) for s in model_sdfs] + [
        "-r", os.path.abspath(ref_pdb),
        "-rl"] + [os.path.abspath(s) for s in ref_sdfs] + [
        "-o", os.path.abspath(output_json),
        "--lddt-pli",
        "--rmsd"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"OST failed: {result.stderr[:500]}")
    return output_json


def extract_best_scores(metrics):
    lddt_pli = None
    rmsd = None
    if "lddt_pli" in metrics:
        assigned = metrics["lddt_pli"].get("assigned_scores", [])
        if assigned:
            scores = [s["score"] for s in assigned if s.get("score") is not None]
            if scores:
                lddt_pli = max(scores)
    if "rmsd" in metrics:
        assigned = metrics["rmsd"].get("assigned_scores", [])
        if assigned:
            scores = [s["score"] for s in assigned if s.get("score") is not None]
            if scores:
                rmsd = min(scores)
    return lddt_pli, rmsd


# ── CIF discovery ─────────────────────────────────────────────────────────────
def discover_cifs_boltz2(output_dir, target, seed_layout=False):
    """Find all CIF files for a boltz2 target."""
    cifs = []
    if seed_layout:
        # R2 layout: output_dir/seed_*/boltz_results_{target}_input/predictions/*/*.cif
        for seed_dir in sorted(glob.glob(os.path.join(output_dir, "seed_*"))):
            pred_dir = os.path.join(seed_dir, f"boltz_results_{target}_input", "predictions")
            if not os.path.isdir(pred_dir):
                continue
            for subdir in os.listdir(pred_dir):
                subpath = os.path.join(pred_dir, subdir)
                if os.path.isdir(subpath):
                    cifs.extend(sorted(glob.glob(os.path.join(subpath, "*.cif"))))
    else:
        # Standard layout: output_dir/boltz_results_{target}_input/predictions/*/*.cif
        pred_dir = os.path.join(output_dir, f"boltz_results_{target}_input", "predictions")
        if os.path.isdir(pred_dir):
            for subdir in os.listdir(pred_dir):
                subpath = os.path.join(pred_dir, subdir)
                if os.path.isdir(subpath):
                    cifs.extend(sorted(glob.glob(os.path.join(subpath, "*.cif"))))
    return cifs


def discover_cifs_af3(output_dir, target):
    """Find all CIF files for an AF3 target."""
    cifs = []
    # AF3 layout: output_dir/{target}/seed-*_sample-*/*_model.cif
    #          or output_dir/{TARGET}/seed-*_sample-*/*_model.cif
    for variant in [target, target.upper(), target.lower()]:
        target_dir = os.path.join(output_dir, variant)
        if os.path.isdir(target_dir):
            cifs = sorted(glob.glob(os.path.join(target_dir, "seed-*_sample-*", "*_model.cif")))
            if cifs:
                break
            # Nested: {TARGET}/{target}/seed-*...
            for sub in os.listdir(target_dir):
                nested = os.path.join(target_dir, sub)
                if os.path.isdir(nested):
                    nested_cifs = sorted(glob.glob(os.path.join(nested, "seed-*_sample-*", "*_model.cif")))
                    cifs.extend(nested_cifs)
            if cifs:
                break
    return cifs


def discover_cifs(method, output_dir, target, seed_layout=False):
    if method == "boltz2":
        return discover_cifs_boltz2(output_dir, target, seed_layout)
    elif method == "af3":
        return discover_cifs_af3(output_dir, target)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Prepare reference SDFs ────────────────────────────────────────────────────
def prepare_ref_sdfs(target, struct_dir, smiles_dir, temp_dir):
    ref_target_dir = os.path.join(struct_dir, target)
    ref_pdb = os.path.join(ref_target_dir, "protein_aligned.pdb")
    if not os.path.exists(ref_pdb):
        return None, None

    ref_lig_pdbs = sorted(glob.glob(os.path.join(ref_target_dir, "ligand_*.pdb")))
    if not ref_lig_pdbs:
        return ref_pdb, None

    smiles_file = os.path.join(smiles_dir, f"{target}.tsv")
    if not os.path.exists(smiles_file):
        return ref_pdb, None

    ligands = read_smiles_from_tsv(smiles_file)
    smiles_by_name = {name: smi for name, smi in ligands}

    ref_sdfs = []
    for ref_lig_pdb in ref_lig_pdbs:
        basename = os.path.basename(ref_lig_pdb)
        parts = basename.replace(".pdb", "").split("_")
        resname = parts[1] if len(parts) >= 3 else ""
        lig_smiles = smiles_by_name.get(resname)
        if lig_smiles is None and len(ligands) == 1:
            lig_smiles = ligands[0][1]
        if lig_smiles is None:
            continue

        ref_sdf = os.path.join(temp_dir, basename.replace(".pdb", ".sdf"))
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
                ref_sdfs.append(ref_sdf)
        except Exception as e:
            log.warning(f"[{target}] Ref SDF failed for {basename}: {e}")

    return ref_pdb, ref_sdfs if ref_sdfs else None


# ── Get SMILES for a target ───────────────────────────────────────────────────
def get_target_smiles(target, smiles_dir):
    """Get the primary ligand SMILES for a target."""
    tsv = os.path.join(smiles_dir, f"{target}.tsv")
    if not os.path.exists(tsv):
        return None
    ligands = read_smiles_from_tsv(tsv)
    return ligands[0][1] if ligands else None


# ── Detect ligand chain from CIF ──────────────────────────────────────────────
def detect_lig_chain(cif_path):
    """Auto-detect the main ligand chain in a CIF file.
    _detect_ligand_chains returns [(chain_id, n_atoms, formula), ...]
    """
    try:
        lig_info = _detect_ligand_chains(cif_path)
        if lig_info:
            max_n = max(n for _, n, _ in lig_info)
            return [ch for ch, n, _ in lig_info if n == max_n][0]
    except Exception as e:
        log.warning(f"detect_lig_chain failed for {cif_path}: {e}")
    return "B"  # default


# ── Main evaluation ──────────────────────────────────────────────────────────
def evaluate_method(method, output_dir, series, targets=None, seed_layout=False,
                    workers=4, max_models=None):
    struct_dir, smiles_dir = get_reference_paths(series)
    if not os.path.isdir(struct_dir):
        log.error(f"Reference dir not found: {struct_dir}")
        return []

    # Discover all targets
    if targets is None:
        targets = sorted([d for d in os.listdir(struct_dir)
                          if os.path.isdir(os.path.join(struct_dir, d))])

    import threading
    eval_dir = os.path.join(output_dir, "_eval_lddt")
    os.makedirs(eval_dir, exist_ok=True)
    cache_file = os.path.join(eval_dir, "score_cache.json")
    score_cache = {}
    if os.path.exists(cache_file):
        try:
            score_cache = json.load(open(cache_file))
        except json.JSONDecodeError:
            log.warning(f"Cache file corrupted, starting fresh: {cache_file}")
            score_cache = {}
    cache_lock = threading.Lock()

    def save_cache():
        with cache_lock:
            tmp = cache_file + ".tmp"
            json.dump(dict(score_cache), open(tmp, "w"))
            os.replace(tmp, cache_file)

    all_results = []

    for target in targets:
        log.info(f"[{target}] Discovering CIFs...")
        cifs = discover_cifs(method, output_dir, target, seed_layout)
        if not cifs:
            log.warning(f"[{target}] No CIFs found, skipping")
            continue
        if max_models:
            cifs = cifs[:max_models]

        log.info(f"[{target}] Found {len(cifs)} CIFs")

        # Prepare reference (must be under /bmlfast so Docker can see them)
        ref_dir = os.path.join(eval_dir, target, "ref")
        os.makedirs(ref_dir, exist_ok=True)
        ref_pdb, ref_sdfs = prepare_ref_sdfs(target, struct_dir, smiles_dir, ref_dir)
        temp_dir = os.path.join(eval_dir, target, "ost_out")
        os.makedirs(temp_dir, exist_ok=True)
        if ref_pdb is None or ref_sdfs is None:
            log.warning(f"[{target}] Reference preparation failed, skipping")
            continue

        smiles = get_target_smiles(target, smiles_dir)
        convert_dir = os.path.join(eval_dir, target, "converted")
        os.makedirs(convert_dir, exist_ok=True)

        # Detect ligand chain from first CIF
        lig_chain = detect_lig_chain(cifs[0])
        log.info(f"[{target}] Ligand chain: {lig_chain}")

        # Step 1: Serial CIF → PDB + SDF conversion (PyMOL is not thread-safe)
        converted = []  # list of (cif_name, cache_key, pdb, sdf)
        for i, cif_path in enumerate(cifs):
            cif_name = os.path.basename(cif_path).replace(".cif", "")
            cache_key = f"{target}_{cif_name}"
            if cache_key in score_cache:
                converted.append((cif_name, cache_key, None, None))  # will use cache
                continue
            pdb, sdf = cif_to_pdb_sdf(cif_path, convert_dir, f"{target}_{cif_name}",
                                        smiles=smiles, lig_chain=lig_chain)
            converted.append((cif_name, cache_key, pdb, sdf))
            if (i + 1) % 10 == 0:
                log.info(f"[{target}] {i+1}/{len(cifs)} converted")

        # Step 2: Parallel OST Docker evaluation
        def eval_single(item):
            cif_name, cache_key, pdb, sdf = item
            with cache_lock:
                if cache_key in score_cache:
                    return score_cache[cache_key]
            if pdb is None or sdf is None:
                res = {"cif": cif_name, "lddt_pli": None, "rmsd": None, "error": "conversion_failed"}
                with cache_lock:
                    score_cache[cache_key] = res
                return res
            out_json = os.path.join(temp_dir, f"ost_{cif_name}.json")
            try:
                run_ost_compare(pdb, sdf, ref_pdb, ref_sdfs, out_json)
                metrics = json.load(open(out_json))
                lddt_pli, rmsd = extract_best_scores(metrics)
                res = {"cif": cif_name, "lddt_pli": lddt_pli, "rmsd": rmsd, "error": None}
            except Exception as e:
                res = {"cif": cif_name, "lddt_pli": None, "rmsd": None, "error": str(e)[:200]}
            with cache_lock:
                score_cache[cache_key] = res
            return res

        target_scores = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(eval_single, item): item for item in converted}
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                target_scores.append(res)
                if (i + 1) % 10 == 0:
                    log.info(f"[{target}] {i+1}/{len(cifs)} evaluated")
                    save_cache()

        save_cache()

        # Compute stats
        valid_lddts = [s["lddt_pli"] for s in target_scores if s["lddt_pli"] is not None]
        valid_rmsds = [s["rmsd"] for s in target_scores if s["rmsd"] is not None]
        n_total = len(cifs)
        n_valid = len(valid_lddts)
        n_failed = n_total - n_valid

        result = {
            "target": target,
            "n_models": n_total,
            "n_valid": n_valid,
            "n_failed": n_failed,
            "best_lddt_pli": max(valid_lddts) if valid_lddts else None,
            "avg_lddt_pli": sum(valid_lddts) / len(valid_lddts) if valid_lddts else None,
            "best_rmsd": min(valid_rmsds) if valid_rmsds else None,
            "avg_rmsd": sum(valid_rmsds) / len(valid_rmsds) if valid_rmsds else None,
        }
        all_results.append(result)
        log.info(f"[{target}] best_lddt={result['best_lddt_pli']:.4f}, "
                 f"avg_lddt={result['avg_lddt_pli']:.4f}, "
                 f"n={n_valid}/{n_total}" if result['best_lddt_pli'] else
                 f"[{target}] ALL FAILED ({n_total} models)")

    return all_results


def save_results(results, output_path):
    if not results:
        return
    fields = ["target", "n_models", "n_valid", "n_failed",
              "best_lddt_pli", "avg_lddt_pli", "best_rmsd", "avg_rmsd"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

        # Average row
        numeric = {k: [] for k in ["best_lddt_pli", "avg_lddt_pli", "best_rmsd", "avg_rmsd"]}
        for r in results:
            for k in numeric:
                if r[k] is not None:
                    numeric[k].append(r[k])
        avg_row = {"target": "AVERAGE"}
        for k, vals in numeric.items():
            avg_row[k] = sum(vals) / len(vals) if vals else None
        avg_row["n_models"] = sum(r["n_models"] for r in results)
        avg_row["n_valid"] = sum(r["n_valid"] for r in results)
        avg_row["n_failed"] = sum(r["n_failed"] for r in results)
        w.writerow(avg_row)

    log.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate lddt-pli for a method's CIF outputs")
    parser.add_argument("--method", required=True, choices=["boltz2", "af3"],
                        help="Method name (boltz2, af3)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing method outputs (relative to project root or absolute)")
    parser.add_argument("--series", required=True,
                        help="Series name for reference data (L1000, L2000, L3000, L4000)")
    parser.add_argument("--targets", default=None,
                        help="Comma-separated target IDs (default: all targets in reference dir)")
    parser.add_argument("--seed_layout", action="store_true",
                        help="Boltz2 R2 seed-based directory layout (seed_*/boltz_results_...)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel Docker workers (default: 4)")
    parser.add_argument("--max_models", type=int, default=None,
                        help="Max models to evaluate per target (default: all)")
    parser.add_argument("--output_csv", default=None,
                        help="Output CSV path (default: {output_dir}/eval_lddt_{method}_{series}.csv)")

    args = parser.parse_args()

    # Resolve output_dir
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(root, args.output_dir)

    targets = args.targets.split(",") if args.targets else None

    results = evaluate_method(
        method=args.method,
        output_dir=args.output_dir,
        series=args.series,
        targets=targets,
        seed_layout=args.seed_layout,
        workers=args.workers,
        max_models=args.max_models,
    )

    output_csv = args.output_csv or os.path.join(
        args.output_dir, f"eval_lddt_{args.method}_{args.series}.csv"
    )
    save_results(results, output_csv)

    # Print summary
    if results:
        avg_best = [r["best_lddt_pli"] for r in results if r["best_lddt_pli"] is not None]
        avg_avg = [r["avg_lddt_pli"] for r in results if r["avg_lddt_pli"] is not None]
        print(f"\n{'='*60}")
        print(f"Method: {args.method} | Series: {args.series}")
        print(f"Targets: {len(results)} | Models evaluated: {sum(r['n_valid'] for r in results)}")
        print(f"Oracle Best LDDT-PLI (avg over targets): {sum(avg_best)/len(avg_best):.4f}" if avg_best else "")
        print(f"Average LDDT-PLI (avg over targets):     {sum(avg_avg)/len(avg_avg):.4f}" if avg_avg else "")
        print(f"Results: {output_csv}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
