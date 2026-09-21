from pathlib import Path
import os
import sys
import glob
import json
import argparse
import logging
import multiprocessing
import subprocess
import tempfile
import csv
import numpy as np

log = logging.getLogger(__name__)

# Targets to skip due to sequence mismatch (Empty now since Docker handles it)
SKIP_TARGETS = set()

from rdkit import Chem
from rdkit.Chem import AllChem

from casp17_ligand.utils.data_utils import extract_protein_and_ligands_with_prody

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_smiles_from_tsv(tsv_path):
    """Read ligand SMILES from TSV file. Returns list of (name, smiles) tuples."""
    ligands = []
    with open(tsv_path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            # ID, Name, SMILES, Task
            ligands.append((parts[1], parts[2]))
    return ligands


def run_ost_compare(model_pdb, model_sdfs, ref_pdb, ref_sdfs, output_json):
    """Run ost compare-ligand-structures CLI via Docker.

    :param model_sdfs: single path (str) or list of model ligand SDF paths.
    :param ref_sdfs: single path (str) or list of reference ligand SDF paths.
    """
    if isinstance(model_sdfs, str):
        model_sdfs = [model_sdfs]
    if isinstance(ref_sdfs, str):
        ref_sdfs = [ref_sdfs]

    cwd = os.path.abspath(os.getcwd())
    cmd = [
        "docker", "run", "--rm",
        # Mount the project root, not a site-specific filesystem root: a
        # `docker -v` on a path that does not exist creates it as an empty
        # root-owned directory instead of failing.
        "-v", f"{_PROJECT_ROOT}:{_PROJECT_ROOT}",
        "-w", cwd,
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"OST Docker failed: {result.stderr}\nSTDOUT: {result.stdout}")
        return output_json
    except subprocess.TimeoutExpired:
        raise RuntimeError("OST Docker command timed out after 120 seconds")

def _extract_best_scores(metrics):
    """Extract best lDDT-PLI and best RMSD from OST JSON output.

    For multi-ligand targets, OST returns multiple entries in assigned_scores.
    We take the max lDDT-PLI (higher is better) and min RMSD (lower is better).
    """
    lddt_pli = None
    rmsd = None
    valid = False

    if "lddt_pli" in metrics:
        assigned = metrics["lddt_pli"].get("assigned_scores", [])
        if assigned:
            scores = [s["score"] for s in assigned if s.get("score") is not None]
            if scores:
                lddt_pli = max(scores)
                valid = True
        else:
            # Legacy format fallback
            for lig_key, lig_data in metrics["lddt_pli"].items():
                if isinstance(lig_data, dict) and "lddt_pli" in lig_data:
                    val = lig_data["lddt_pli"]
                    if val is not None and (lddt_pli is None or val > lddt_pli):
                        lddt_pli = val
                        valid = True

    if "rmsd" in metrics:
        assigned = metrics["rmsd"].get("assigned_scores", [])
        if assigned:
            scores = [s["score"] for s in assigned if s.get("score") is not None]
            if scores:
                rmsd = min(scores)
                valid = True
        else:
            for lig_key, lig_data in metrics["rmsd"].items():
                if isinstance(lig_data, dict) and "rmsd" in lig_data:
                    val = lig_data["rmsd"]
                    if val is not None and (rmsd is None or val < rmsd):
                        rmsd = val
                        valid = True

    return lddt_pli, rmsd, valid


def evaluate_target(target, ensemble_dir, reference_dir, smiles_dir, ranks=(1,2,3,4,5), docker_threads=4):
    target_results = []

    # Paths for ground truth
    ref_target_dir = os.path.join(reference_dir, target)
    if not os.path.isdir(ref_target_dir):
        log.warning(f"Skipping {target}: Reference dir missing {ref_target_dir}")
        return target_results

    ref_pdb_path = os.path.join(ref_target_dir, "protein_aligned.pdb")

    # Find ALL reference ligand PDBs
    ref_lig_pdbs = sorted(glob.glob(os.path.join(ref_target_dir, "ligand_*.pdb")))
    if not ref_lig_pdbs:
        log.warning(f"Skipping {target}: Reference ligand missing in {ref_target_dir}")
        return target_results

    # Get SMILES for all ligands
    smiles_file = os.path.join(smiles_dir, f"{target}.tsv")
    if not os.path.exists(smiles_file):
        log.warning(f"Skipping {target}: SMILES TSV missing {smiles_file}")
        return target_results

    ligands = read_smiles_from_tsv(smiles_file)
    if not ligands:
        log.warning(f"Skipping {target}: No valid SMILES found in {smiles_file}")
        return target_results

    # Build name→smiles lookup from TSV
    smiles_by_name = {name: smi for name, smi in ligands}

    ost_eval_dir = os.path.join(ensemble_dir, "ost_eval")
    os.makedirs(ost_eval_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"ost_eval_{target}_", dir=ost_eval_dir)

    # Convert ALL reference ligand PDBs to SDFs
    ref_sdf_paths = []
    for ref_lig_pdb in ref_lig_pdbs:
        basename = os.path.basename(ref_lig_pdb)  # e.g. ligand_L0R_E_1.pdb
        # Extract residue name from filename: ligand_{RESNAME}_{CHAIN}_{...}.pdb
        parts = basename.replace(".pdb", "").split("_")
        resname = parts[1] if len(parts) >= 3 else ""
        lig_smiles = smiles_by_name.get(resname)
        if lig_smiles is None:
            # Fallback: use first SMILES if only one ligand type
            if len(ligands) == 1:
                lig_smiles = ligands[0][1]
            else:
                log.warning(f"[{target}] No SMILES for residue '{resname}', skipping ref ligand {basename}")
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
                write_output_files=True
            )
            if os.path.exists(ref_sdf):
                ref_sdf_paths.append(ref_sdf)
        except Exception as e:
            log.warning(f"[{target}] Failed to extract ref SDF from {basename}: {e}")

    if not ref_sdf_paths:
        log.warning(f"Skipping {target}: Could not produce any reference SDF")
        return target_results

    log.info(f"[{target}] {len(ref_sdf_paths)} reference ligand SDFs prepared")

    # Score cache
    cache_file = os.path.join(ensemble_dir, "targets", target, "score_cache_docker.json")
    score_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            score_cache = json.load(f)

    import concurrent.futures
    import threading
    cache_lock = threading.Lock()

    def get_ost_score(model_pdb, model_sdf):
        base_name = os.path.basename(model_sdf)
        cache_key = base_name.split("_rank")[0] if "_rank" in base_name else base_name

        with cache_lock:
            if cache_key in score_cache:
                return score_cache[cache_key]

        out_json = os.path.join(temp_dir, f"out_{cache_key}.json")
        try:
            run_ost_compare(model_pdb, model_sdf, ref_pdb_path, ref_sdf_paths, out_json)
            with open(out_json, "r") as f:
                metrics = json.load(f)

            lddt_pli, rmsd, valid = _extract_best_scores(metrics)
            res = {"lddt_pli": lddt_pli, "rmsd": rmsd, "valid": valid, "error": None}
        except Exception as e:
            res = {"lddt_pli": None, "rmsd": None, "valid": False, "error": str(e)}
            err_log = os.path.join(ensemble_dir, "err.log")
            try:
                with open(err_log, "a") as f:
                    f.write(f"[{target}] LDDT calculation failed for {cache_key}: {str(e)[:500]}...\n")
            except Exception:
                pass

        with cache_lock:
            score_cache[cache_key] = res
            with open(cache_file, "w") as f:
                json.dump(score_cache, f)
        return res

    # Collect tasks
    score_tasks = []
    for method in ["ranking_rmsd", "ranking_sucos", "ranking_rmsd_6a", "ranking_rmsd_8a", "ranking_rmsd_10a"]:
        method_dir = os.path.join(ensemble_dir, "targets", target, method)
        if not os.path.isdir(method_dir):
            continue

        all_model_sdfs = glob.glob(os.path.join(method_dir, "*_rank*.sdf"))
        for model_sdf in all_model_sdfs:
            model_pdb = model_sdf.replace("_pb=True", "").replace("_pb=False", "").replace(".sdf", ".pdb")
            if not os.path.exists(model_pdb):
                continue

            base_name = os.path.basename(model_sdf)
            try:
                rank_str = base_name.split("_rank")[1].split("_")[0]
                rank = int(rank_str)
            except Exception:
                continue

            score_tasks.append((method, model_pdb, model_sdf, base_name, rank))

    # Run in parallel (threads for Docker I/O)
    log.info(f"[{target}] Submitting {len(score_tasks)} evaluations...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=docker_threads) as executor:
        future_to_info = {
            executor.submit(get_ost_score, t[1], t[2]): t
            for t in score_tasks
        }
        for future in concurrent.futures.as_completed(future_to_info):
            t = future_to_info[future]
            method, model_pdb, model_sdf, base_name, rank = t
            score_res = future.result()
            target_results.append({
                "target": target,
                "method": method,
                "model_name": base_name,
                "rank": rank,
                "lddt_pli": score_res["lddt_pli"],
                "rmsd": score_res["rmsd"],
                "valid": score_res["valid"],
                "error": score_res["error"]
            })

    log.info(f"[{target}] Finished: {len(target_results)} evaluations")
    return target_results

def _evaluate_target_worker(args_tuple):
    """Worker for multiprocessing. Unpacks args and calls evaluate_target."""
    target, ensemble_dir, reference_dir, smiles_dir = args_tuple
    # Configure logging in child process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return evaluate_target(target, ensemble_dir, reference_dir, smiles_dir, ranks=None)
    except Exception as e:
        log.error(f"Worker failed for {target}: {e}")
        return []


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate ensemble rankings with OST")
    parser.add_argument("--ensemble_dir", type=str, default="outputs/ensemble/casp16_l3000_struct")
    parser.add_argument("--reference_dir", type=str, default="data/casp16_data/struct/L3000_prepared")
    parser.add_argument("--smiles_dir", type=str, default="data/casp16_data/smiles/L3000")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=14,
                        help="Number of parallel target workers (1=serial)")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.ensemble_dir, "evaluation_summary.csv")

    targets_dir = os.path.join(args.ensemble_dir, "targets")
    if not os.path.isdir(targets_dir):
        print(f"Targets dir not found: {targets_dir}")
        sys.exit(1)

    targets = sorted([d for d in os.listdir(targets_dir)
                       if os.path.isdir(os.path.join(targets_dir, d))
                       and d not in SKIP_TARGETS])

    # Check which targets already have cached scores (for skip/resume)
    pending = []
    cached_results = []
    for t in targets:
        cache_file = os.path.join(args.ensemble_dir, "targets", t, "score_cache_docker.json")
        if os.path.exists(cache_file):
            # Still need to re-evaluate to populate results, but cache makes it fast
            pass
        pending.append(t)

    log.info(f"Evaluating {len(pending)} targets with {args.num_workers} workers")

    worker_args = [(t, args.ensemble_dir, args.reference_dir, args.smiles_dir) for t in pending]

    all_results = []
    if args.num_workers <= 1:
        for wa in worker_args:
            results = _evaluate_target_worker(wa)
            all_results.extend(results)
            log.info(f"Done {wa[0]} ({len(all_results)} total results)")
    else:
        completed = 0
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for results in pool.imap_unordered(_evaluate_target_worker, worker_args):
                all_results.extend(results)
                completed += 1
                if completed % 10 == 0:
                    log.info(f"Progress: {completed}/{len(pending)} targets")
        
    if all_results:
        # Write CSV
        keys = ["target", "method", "model_name", "rank", "lddt_pli", "rmsd", "valid", "error"]
        with open(args.output_csv, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"Results saved to {args.output_csv}")
        
        # Calculate Top-1 and Top-5 success per method
        print("\n=== Summary ===")
        
        # Prepare data for detailed top1_top5_scores.csv
        csv_data = []
        unique_targets = sorted(list(set(r["target"] for r in all_results if r["valid"])))
        for t in unique_targets:
            t_row = {"Target": t}
            all_t_results = [r for r in all_results if r["valid"] and r["target"] == t and r["lddt_pli"] is not None]
            
            # Find all independent generation methods (e.g. boltz2, af3) and figure out their max bounds natively
            # Split model names format: method_modelID to get method
            all_methods_used = set()
            for r in all_t_results:
                method_used = r["model_name"].split("_model")[0].split("_rank")[0]
                # Sometimes its method_model_0 but the above split normally works (e.g. boltz2_model6 -> boltz2)
                if method_used:
                    all_methods_used.add(method_used)
            
            # Identify absolute Oracle Best
            if all_t_results:
                best_model = max(all_t_results, key=lambda x: x["lddt_pli"])
                oracle_best = best_model["lddt_pli"]
                best_model_name = best_model["model_name"]
            else:
                oracle_best = None
                best_model_name = ""
                
            t_row["Overall Best LDDT-PLI"] = f"{round(oracle_best, 3):.3f}" if oracle_best is not None else ""
            t_row["Overall Best Source Model"] = best_model_name
            
            # Identify individual method specific Oracles
            for gen_method in sorted(list(all_methods_used)):
                method_specific_results = [r for r in all_t_results if r["model_name"].startswith(f"{gen_method}_")]
                if method_specific_results:
                    method_best = max(method_specific_results, key=lambda x: x["lddt_pli"])["lddt_pli"]
                    t_row[f"{gen_method} Best LDDT-PLI"] = f"{round(method_best, 3):.3f}"
                else:
                    t_row[f"{gen_method} Best LDDT-PLI"] = ""
            
            for method in ["ranking_rmsd", "ranking_sucos", "ranking_rmsd_6a", "ranking_rmsd_8a", "ranking_rmsd_10a"]:
                m_t_results = [r for r in all_t_results if r["method"] == method]
                if not m_t_results:
                    t_row[f"{method} (Top-1 LDDT)"] = ""
                    t_row[f"{method} (Top-5 Best)"] = ""
                    t_row[f"{method} (Top-1 Rank)"] = ""
                    t_row[f"{method} (Top-5 Rank)"] = ""
                    continue
                
                # Sorted model structures by actual LDDT-PLI (descending) to find empirical rank among all models
                sorted_by_lddt = sorted(m_t_results, key=lambda x: x["lddt_pli"], reverse=True)
                # Map true model name/rank to its Oracle Rank position (1-indexed)
                lddt_rank_map = {r["rank"]: idx + 1 for idx, r in enumerate(sorted_by_lddt)}
                
                rank1 = [r for r in m_t_results if r["rank"] == 1]
                t1_val = rank1[0]["lddt_pli"] if rank1 else None
                top5_vals = [r["lddt_pli"] for r in m_t_results if r["rank"] <= 5]
                t5_best = max(top5_vals, default=None)
                
                # Ranks
                t1_oracle_rank = lddt_rank_map.get(1, "")
                
                # Top 5 Oracle Rank
                best_t5_pred_rank = None
                best_t5_lddt = -1
                for r in m_t_results:
                    if r["rank"] <= 5 and r["lddt_pli"] > best_t5_lddt:
                        best_t5_lddt = r["lddt_pli"]
                        best_t5_pred_rank = r["rank"]
                t5_oracle_rank = lddt_rank_map.get(best_t5_pred_rank, "") if best_t5_pred_rank is not None else ""
                
                t_row[f"{method} (Top-1 LDDT)"] = f"{round(t1_val, 3):.3f}" if t1_val is not None else ""
                t_row[f"{method} (Top-5 Best)"] = f"{round(t5_best, 3):.3f}" if t5_best is not None else ""
                t_row[f"{method} (Top-1 Rank)"] = t1_oracle_rank
                t_row[f"{method} (Top-5 Rank)"] = t5_oracle_rank
                
            csv_data.append(t_row)
            
        # Load baseline groups for CASP16 if they exist
        baseline_lg207 = {}
        baseline_lg494 = {}
        try:
            import pandas as pd
            if os.path.exists("outputs/Multicom_LG207_1_casp16.csv"):
                df_207 = pd.read_csv("outputs/Multicom_LG207_1_casp16.csv", header=None, names=["target", "group", "model", "model_ligand", "lddt_pli", "lddt_pli_coverage", "lddt_pli_reference_ligand", "lddt_pli_unassigned", "rmsd", "lddt_lp", "bb_rmsd", "rmsd_coverage", "rmsd_reference_ligand", "rmsd_unassigned"])
                # The grep command doesn't have a header.
                # Find Max lddt_pli for LG207_1 for each target
                for t, g in df_207.groupby("target"):
                    baseline_lg207[t] = g["lddt_pli"].max()
            if os.path.exists("outputs/Champion_LG494_1_casp16.csv"):
                df_494 = pd.read_csv("outputs/Champion_LG494_1_casp16.csv", header=None, names=["target", "group", "model", "model_ligand", "lddt_pli", "lddt_pli_coverage", "lddt_pli_reference_ligand", "lddt_pli_unassigned", "rmsd", "lddt_lp", "bb_rmsd", "rmsd_coverage", "rmsd_reference_ligand", "rmsd_unassigned"])
                for t, g in df_494.groupby("target"):
                    baseline_lg494[t] = g["lddt_pli"].max()
        except ImportError:
            pass

        # Determine dataset name to append to CSV (e.g. L1000)
        ds_name = os.path.basename(os.path.normpath(args.ensemble_dir))
        prefix = ds_name.split("_")[-1].upper() if "_" in ds_name else ds_name.upper()
        top_scores_csv = os.path.join(os.path.dirname(args.output_csv), f"top1_top5_scores_{prefix}.csv")
        with open(top_scores_csv, "w", newline='') as f:
            headers = [
                "Target", "Overall Best LDDT-PLI", "Overall Best Source Model",
                "af3 Best LDDT-PLI", "boltz2 Best LDDT-PLI", "protenix Best LDDT-PLI", "rf3 Best LDDT-PLI", "seedfold Best LDDT-PLI",
                "Baseline Multicom LDDT-PLI", "Baseline Champion LDDT-PLI",
                "ranking_rmsd (Top-1 LDDT)", "ranking_rmsd (Top-5 Best)", "ranking_rmsd (Top-1 Rank)", "ranking_rmsd (Top-5 Rank)",
                "ranking_sucos (Top-1 LDDT)", "ranking_sucos (Top-5 Best)", "ranking_sucos (Top-1 Rank)", "ranking_sucos (Top-5 Rank)",
                "ranking_rmsd_6a (Top-1 LDDT)", "ranking_rmsd_6a (Top-5 Best)", "ranking_rmsd_6a (Top-1 Rank)", "ranking_rmsd_6a (Top-5 Rank)",
                "ranking_rmsd_8a (Top-1 LDDT)", "ranking_rmsd_8a (Top-5 Best)", "ranking_rmsd_8a (Top-1 Rank)", "ranking_rmsd_8a (Top-5 Rank)",
                "ranking_rmsd_10a (Top-1 LDDT)", "ranking_rmsd_10a (Top-5 Best)", "ranking_rmsd_10a (Top-1 Rank)", "ranking_rmsd_10a (Top-5 Rank)"
            ]
            
            # Inject baseline scores into rows
            for r in csv_data:
                t = r["Target"]
                r["Baseline Multicom LDDT-PLI"] = ""
                r["Baseline Champion LDDT-PLI"] = ""
                if baseline_lg207 or baseline_lg494:
                    if t in baseline_lg207 and pd.notna(baseline_lg207[t]):
                        r["Baseline Multicom LDDT-PLI"] = f"{baseline_lg207[t]:.3f}"
                    if t in baseline_lg494 and pd.notna(baseline_lg494[t]):
                        r["Baseline Champion LDDT-PLI"] = f"{baseline_lg494[t]:.3f}"
                        
                # Ensure all missing keys in row have empty strings
                for h in headers:
                    if h not in r:
                        r[h] = ""
            
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in csv_data:
                writer.writerow(r)
                
            # Compute and append averages row
            avg_row = {"Target": "AVERAGE"}
            for col in headers:
                if col in ["Target", "Overall Best Source Model"]:
                    continue
                
                vals = []
                for r in csv_data:
                    val = r.get(col, "")
                    if val != "":
                        try:
                            vals.append(float(val))
                        except ValueError:
                            pass
                if vals:
                    avg_row[col] = f"{sum(vals)/len(vals):.3f}"
                else:
                    avg_row[col] = ""
            writer.writerow(avg_row)
        print(f"Detailed Top 1/Top 5/Oracle Scores exported to {top_scores_csv}")
        
        # Method comparison summaries
        print("\n=== Ranking Method Head-to-Head ===")
        rmsd_top1_wins = 0
        sucos_top1_wins = 0
        ties = 0
        rmsd_mean_rank = []
        sucos_mean_rank = []
        
        for r in csv_data:
            r_rmsd = float(r["ranking_rmsd (Top-1 LDDT)"]) if r["ranking_rmsd (Top-1 LDDT)"] else 0
            r_sucos = float(r["ranking_sucos (Top-1 LDDT)"]) if r["ranking_sucos (Top-1 LDDT)"] else 0
            
            if r_rmsd > r_sucos:
                rmsd_top1_wins += 1
            elif r_sucos > r_rmsd:
                sucos_top1_wins += 1
            else:
                ties += 1
                
            if r.get("ranking_rmsd (Top-1 Rank)"): rmsd_mean_rank.append(float(r["ranking_rmsd (Top-1 Rank)"]))
            if r.get("ranking_sucos (Top-1 Rank)"): sucos_mean_rank.append(float(r["ranking_sucos (Top-1 Rank)"]))
                
        n_targets = len(csv_data)
        if n_targets > 0:
            print(f"RMSD Top-1 LDDT > SuCOS Top-1 LDDT: {rmsd_top1_wins}/{n_targets} ({rmsd_top1_wins/n_targets:.1%})")
            print(f"SuCOS Top-1 LDDT > RMSD Top-1 LDDT: {sucos_top1_wins}/{n_targets} ({sucos_top1_wins/n_targets:.1%})")
            print(f"Ties: {ties}/{n_targets}")
            print(f"RMSD Mean Top-1 Rank: {np.mean(rmsd_mean_rank):.1f}")
            print(f"SuCOS Mean Top-1 Rank: {np.mean(sucos_mean_rank):.1f}")
        for method in ["ranking_rmsd", "ranking_sucos", "ranking_rmsd_6a", "ranking_rmsd_8a", "ranking_rmsd_10a"]:
            m_results = [r for r in all_results if r["method"] == method and r["valid"]]
            if not m_results:
                continue
                
            targets_evaluated = set(r["target"] for r in m_results)
            top1_successes = 0
            top5_successes = 0
            n_targets = len(targets_evaluated)
            
            for t in targets_evaluated:
                t_results = [r for r in m_results if r["target"] == t]
                # Check top 1
                rank1 = [r for r in t_results if r["rank"] == 1]
                if rank1 and rank1[0]["lddt_pli"] is not None and rank1[0]["lddt_pli"] > 0.7:
                    top1_successes += 1
                    
                # Check top 5
                max_lddt = max([r["lddt_pli"] for r in t_results if r["lddt_pli"] is not None], default=0.0)
                if max_lddt > 0.7:
                    top5_successes += 1
                    
            print(f"- {method}:")
            print(f"  Targets evaluated: {n_targets}")
            print(f"  Top-1 Success (LDDT-PLI > 0.7): {top1_successes} ({top1_successes/n_targets:.1%})")
            print(f"  Top-5 Success (LDDT-PLI > 0.7): {top5_successes} ({top5_successes/n_targets:.1%})")

if __name__ == "__main__":
    main()
