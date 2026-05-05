"""Evaluate benefits of Top-N confidence prefiltering.

1. Calculates average lDDT of top-N models *per method*.
2. Calculates actual Rank-1 lDDT after running RMSD/SuCOS consensus on the Top-N pool, 
   leveraging existing pairwise caches to avoid recomputation.
"""

import argparse
import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from casp17_ligand.analysis.confidence_metric_analysis import collect_scores, parse_model_source
from casp17_ligand.models.ensemble_generation import rmsd_consensus_rank, sucos_consensus_rank, rmsd_6a_consensus_rank, rmsd_8a_consensus_rank, rmsd_10a_consensus_rank


def _get_target_predictions(target_dir):
    cif_out = os.path.join(target_dir, "cif_converted")
    if not os.path.isdir(cif_out):
        return []

    predictions = []
    pdb_files = sorted([f for f in os.listdir(cif_out) if f.endswith("_protein.pdb")])
    for pdb_f in pdb_files:
        base = pdb_f.replace("_protein.pdb", "")
        sdf_f = base + "_ligand.sdf"
        pdb_path = os.path.join(cif_out, pdb_f)
        sdf_path = os.path.join(cif_out, sdf_f)
        if os.path.exists(sdf_path):
            parts = base.split("_")
            if len(parts) >= 4:
                method_name = f"{parts[1]}_model{parts[3]}"
            else:
                method_name = base
            predictions.append((method_name, pdb_path, sdf_path))
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output_json", default=None,
                        help="Save per-target intermediate results to JSON for incremental reuse")
    args = parser.parse_args()

    dataset = args.dataset
    metric = args.metric
    topn_values = [10, 20, 30, 40]
    
    ensemble_dir = f"outputs/ensemble/{dataset}"
    targets_dir = os.path.join(ensemble_dir, "targets")
    eval_path = os.path.join(ensemble_dir, "evaluation_summary.csv")
    
    if not os.path.exists(eval_path):
        print(f"Missing evaluation summary: {eval_path}")
        return
        
    df_eval = pd.read_csv(eval_path)
    # We only need one row per model to get its lDDT. 
    # The evaluation_summary duplicates models for RMSD/SuCOS ranking.
    # Group by target and model_name to get unique lDDTs.
    lddt_map = {}
    pb_map = {}
    for _, row in df_eval.iterrows():
        t = row["target"]
        name = row["model_name"]
        src_method, m_idx = parse_model_source(name)
        if src_method:
            key = (t, f"{src_method}_model{m_idx}")
            if pd.notna(row.get("lddt_pli")):
                lddt_map[key] = row["lddt_pli"]
            if "valid" in row:
                pb_map[key] = row["valid"]

    methods_out = {
        "af3": f"outputs/alphafold3/{dataset}",
        "boltz2": f"outputs/boltz2/{dataset}",
        "protenix": f"outputs/protenix/{dataset}",
        "rf3": f"outputs/rf3/{dataset}",
        "seedfold": f"outputs/seedfold/{dataset}",
    }
    input_dirs = {
        "af3": f"data/test_cases/{dataset}/af3_inputs",
        "protenix": f"data/test_cases/{dataset}/protenix_inputs",
        "seedfold": f"data/test_cases/{dataset}/seedfold_inputs",
    }

    targets = sorted(df_eval["target"].unique())
    print(f"Processing {dataset} with {metric}...")

    # For Part 1: Average lDDT per method for top-N
    method_lddt_sums = defaultdict(lambda: defaultdict(float))
    method_lddt_counts = defaultdict(lambda: defaultdict(int))
    
    # For Part 2: Rank-1 lDDTs per target for top-N
    consensus_results = defaultdict(lambda: defaultdict(list))
    
    # 1. Collect all confidence scores
    all_scores = {}
    for target in targets:
        for src_method, out_dir in methods_out.items():
            if not os.path.isdir(out_dir):
                continue
            scores = collect_scores(metric, src_method, target, out_dir, input_dirs.get(src_method, ""))
            if scores:
                all_scores[(target, src_method)] = scores

    import concurrent.futures

    def process_target(target):
        target_dir = os.path.join(targets_dir, target)
        predictions = _get_target_predictions(target_dir)
        if not predictions:
            return None
            
        rmsd_cache_path = os.path.join(target_dir, "pairwise_rmsd_cache.json")
        sucos_cache_path = os.path.join(target_dir, "pairwise_sucos_cache.json")
        rmsd_6a_cache_path = os.path.join(target_dir, "pairwise_rmsd_6a_cache.json")
        rmsd_8a_cache_path = os.path.join(target_dir, "pairwise_rmsd_8a_cache.json")
        rmsd_10a_cache_path = os.path.join(target_dir, "pairwise_rmsd_10a_cache.json")
        
        # Build score map for local target predictions
        pred_scores = {}
        for (pred_name, _, _) in predictions:
            # pred_name format: "af3_model0", "boltz2_model5", etc.
            # Append "_" so regex "^(\w+?)_model(\d+)_" can match the trailing underscore
            src_method, m_idx = parse_model_source(pred_name)
            if src_method and (target, src_method) in all_scores:
                pred_scores[pred_name] = all_scores[(target, src_method)].get(m_idx, float("-inf"))
            else:
                pred_scores[pred_name] = float("-inf")
                
        # Group by method for Top-N filtering
        by_method = defaultdict(list)
        for pred in predictions:
            src_method = pred[0].split("_model")[0]
            by_method[src_method].append((pred_scores[pred[0]], pred))
            
        for m in by_method:
            by_method[m].sort(key=lambda x: x[0], reverse=True)
            
        target_res = {
            "method_lddts": {}, # topn -> method -> list of lddts
            "consensus": {}     # topn -> {"rmsd": rank1_lddt, "sucos": rank1_lddt}
        }
        
        target_res["method_lddts"]["all"] = defaultdict(list)
        
        for topn in topn_values + ["all"]:
            subset_preds = []
            target_res["method_lddts"][topn] = defaultdict(list)
            
            for m, items in by_method.items():
                if topn == "all":
                    selected = items
                else:
                    selected = items[:topn]
                
                for score, pred in selected:
                    subset_preds.append(pred)
                    lddt = lddt_map.get((target, pred[0]))
                    if lddt is not None:
                        target_res["method_lddts"][topn][m].append(lddt)
            
            # Run Consensus Fast via cached JSONs
            if len(subset_preds) >= 2:
                try:
                    subset_names = [p[0] for p in subset_preds]
                    
                    def get_topk_from_cache(s_names, cache_dict, pb_map, t, is_sucos, k=5):
                        """Return top-k model names sorted by consensus score + PB tiebreak."""
                        scores = {}
                        for m_i in s_names:
                            row = []
                            for m_j in s_names:
                                if m_i == m_j: continue
                                val = cache_dict.get(m_i, {}).get(m_j)
                                if val is None:
                                    val = float('-inf') if is_sucos else float('inf')
                                row.append(val)
                            mean_val = np.mean(row) if row else (float('-inf') if is_sucos else float('inf'))
                            scores[m_i] = mean_val

                        cands = []
                        for m_i, s in scores.items():
                            pb = pb_map.get((t, m_i), pd.NA)
                            pb_sort = 0 if pb is True else (1 if pd.isna(pb) else 2)
                            comp_s = -s if is_sucos else s
                            cands.append((pb_sort, comp_s, m_i))
                        cands.sort()
                        return [c[2] for c in cands[:k]] if cands else []

                    rmsd_cache = {}
                    if os.path.exists(rmsd_cache_path):
                        with open(rmsd_cache_path) as f:
                            rmsd_cache = json.load(f)

                    sucos_cache = {}
                    if os.path.exists(sucos_cache_path):
                        with open(sucos_cache_path) as f:
                            sucos_cache = json.load(f)

                    rmsd_6a_cache = {}
                    if os.path.exists(rmsd_6a_cache_path):
                        with open(rmsd_6a_cache_path) as f:
                            rmsd_6a_cache = json.load(f)

                    rmsd_8a_cache = {}
                    if os.path.exists(rmsd_8a_cache_path):
                        with open(rmsd_8a_cache_path) as f:
                            rmsd_8a_cache = json.load(f)

                    rmsd_10a_cache = {}
                    if os.path.exists(rmsd_10a_cache_path):
                        with open(rmsd_10a_cache_path) as f:
                            rmsd_10a_cache = json.load(f)

                    def _store_consensus(cons_name, cache_dict, is_sucos):
                        topk = get_topk_from_cache(subset_names, cache_dict, pb_map, target, is_sucos=is_sucos, k=5)
                        if topk:
                            if topn not in target_res["consensus"]:
                                target_res["consensus"][topn] = {}
                            target_res["consensus"][topn][cons_name] = lddt_map.get((target, topk[0]))
                            top5_lddts = [lddt_map.get((target, m)) for m in topk if lddt_map.get((target, m)) is not None]
                            target_res["consensus"][topn][f"{cons_name}_top5"] = max(top5_lddts) if top5_lddts else None

                    _store_consensus("rmsd", rmsd_cache, False)
                    _store_consensus("sucos", sucos_cache, True)
                    _store_consensus("rmsd_6a", rmsd_6a_cache, False)
                    _store_consensus("rmsd_8a", rmsd_8a_cache, False)
                    _store_consensus("rmsd_10a", rmsd_10a_cache, False)
                        
                except Exception as e:
                    print(f"Error consensus {target} top-{topn}: {e}")
                    
        return target, target_res

    # We can process sequentially because with cache it's extremely fast.
    results = {}
    print("Evaluating consensus...")
    for t in targets:
        res = process_target(t)
        if res:
            results[res[0]] = res[1]

    # Aggregate Method Averages
    print("\\n1. Average lDDT of Top-N models per method:")
    methods_found = set()
    for res in results.values():
        for topn, m_dict in res["method_lddts"].items():
            for m, vals in m_dict.items():
                methods_found.add(m)
                method_lddt_sums[topn][m] += sum(vals)
                method_lddt_counts[topn][m] += len(vals)
                
    for m in sorted(list(methods_found)):
        row = f"{m:10s} | "
        for topn in topn_values + ["all"]:
            s = method_lddt_sums[topn][m]
            c = method_lddt_counts[topn][m]
            avg = s / c if c > 0 else 0
            row += f"Top-{str(topn):3s}: {avg:.4f} | "
        print(row)

    # Aggregate Consensus Rank-1 and Top-5 Best
    print("\\n2. Consensus Rank-1 / Top-5 Best lDDT with Prefiltered Pools:")
    cons_methods = ["rmsd", "sucos", "rmsd_6a", "rmsd_8a", "rmsd_10a"]
    rank1_sums = {m: defaultdict(float) for m in cons_methods}
    rank1_counts = {m: defaultdict(int) for m in cons_methods}
    top5_sums = {m: defaultdict(float) for m in cons_methods}
    top5_counts = {m: defaultdict(int) for m in cons_methods}

    for res in results.values():
        for topn, c_dict in res["consensus"].items():
            for cm in cons_methods:
                if cm in c_dict and c_dict[cm] is not None:
                    rank1_sums[cm][topn] += c_dict[cm]
                    rank1_counts[cm][topn] += 1
                t5_key = f"{cm}_top5"
                if t5_key in c_dict and c_dict[t5_key] is not None:
                    top5_sums[cm][topn] += c_dict[t5_key]
                    top5_counts[cm][topn] += 1

    for cm in cons_methods:
        print(f"{cm.upper()} Ranking Strategy:")
        for topn in topn_values + ["all"]:
            s1 = rank1_sums[cm][topn]
            c1 = rank1_counts[cm][topn]
            avg1 = s1 / c1 if c1 > 0 else 0
            s5 = top5_sums[cm][topn]
            c5 = top5_counts[cm][topn]
            avg5 = s5 / c5 if c5 > 0 else 0
            print(f"  Top-{str(topn):3s} subset ({c1} targets): rank1={avg1:.4f}  top5_best={avg5:.4f}")
        print()

    # Save intermediate results to JSON for incremental reuse
    output_json = args.output_json
    if output_json is None:
        output_json = os.path.join(ensemble_dir, f"topn_benefits_{metric}.json")

    # Build serializable output: per-target consensus rank-1 model + lDDT for each topn
    save_data = {
        "dataset": dataset,
        "metric": metric,
        "topn_values": topn_values,
        "per_target": {},
        "summary": {
            "rmsd": {},
            "sucos": {},
            "rmsd_6a": {},
            "rmsd_8a": {},
            "rmsd_10a": {},
        },
    }
    for target, res in results.items():
        t_out = {}
        for topn in topn_values + ["all"]:
            c = res["consensus"].get(topn, {})
            entry = {}
            for cm in cons_methods:
                entry[f"{cm}_rank1_lddt"] = c.get(cm)
                entry[f"{cm}_top5_best_lddt"] = c.get(f"{cm}_top5")
            t_out[str(topn)] = entry
        save_data["per_target"][target] = t_out

    for topn in topn_values + ["all"]:
        k = str(topn)
        for cm in cons_methods:
            rc = rank1_counts[cm][topn]
            t5c = top5_counts[cm][topn]
            save_data["summary"][cm][k] = {
                "mean_lddt": rank1_sums[cm][topn] / rc if rc > 0 else None,
                "mean_top5_best": top5_sums[cm][topn] / t5c if t5c > 0 else None,
                "n_targets": rc,
            }

    with open(output_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nIntermediate results saved to {output_json}")


if __name__ == "__main__":
    main()
