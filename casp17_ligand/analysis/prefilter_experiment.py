"""Prefilter experiment: check if confidence-based top-N filtering preserves oracle quality.

For each target:
  1. Collect all confidence scores per method
  2. Mark which model indices survive the top-N filter
  3. From evaluation_summary, check if the rank-1 (best lDDT) model survives
  4. If not, find the best lDDT among survivors → this is the "filtered oracle"

Usage:
    PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/prefilter_experiment.py \
        --dataset casp16_l1000 --metric pair_iptm --topn 30
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.getcwd())
from casp17_ligand.analysis.confidence_metric_analysis import collect_scores, parse_model_source
from casp17_ligand.analysis.self_ranking_comparison import _get_ensemble_cif_list


def run_experiment(dataset, metric, topn_values, ensemble_dir="outputs/ensemble"):
    ens_base = os.path.join(ensemble_dir, dataset)
    eval_path = os.path.join(ens_base, "evaluation_summary.csv")
    if not os.path.exists(eval_path):
        print(f"Missing: {eval_path}")
        return

    df_eval = pd.read_csv(eval_path)
    
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
    ranking_methods_list = df_eval["method"].unique()  # 'ranking_rmsd', 'ranking_sucos'

    print(f"Dataset: {dataset} ({len(targets)} targets)")
    print(f"Ranking methods: {list(ranking_methods_list)}")
    print(f"Filter metric: {metric}")
    print(f"TopN values: {topn_values}")

    # Pre-collect all confidence scores per target×method
    # {(target, src_method): {model_idx: score}}
    all_scores = {}
    for target in targets:
        for src_method, out_dir in methods_out.items():
            if not os.path.isdir(out_dir):
                continue
            scores = collect_scores(metric, src_method, target, out_dir,
                                    input_dirs.get(src_method, ""))
            if scores:
                all_scores[(target, src_method)] = scores

    for ranking_method in ranking_methods_list:
        print(f"\n{'='*80}")
        print(f"Ranking: {ranking_method}")
        print(f"{'='*80}")

        sub = df_eval[df_eval["method"] == ranking_method].copy()
        
        for topn in topn_values:
            results = []
            for target in targets:
                t_df = sub[sub["target"] == target]
                if t_df.empty or "lddt_pli" not in t_df.columns:
                    continue
                
                t_df = t_df.dropna(subset=["lddt_pli"])
                if t_df.empty:
                    continue

                # Current rank-1 lDDT (from full ensemble)
                rank1_row = t_df[t_df["rank"] == 1]
                if rank1_row.empty:
                    continue
                full_rank1_lddt = rank1_row["lddt_pli"].values[0]

                # Oracle (best lDDT across all models)
                oracle_lddt = t_df["lddt_pli"].max()

                # Build set of surviving model indices per method
                surviving = set()  # set of (src_method, model_idx)
                
                # Parse each model in the evaluation to get its source
                for _, row in t_df.iterrows():
                    src_method, model_idx = parse_model_source(row["model_name"])
                    if src_method is None:
                        surviving.add(row["model_name"])  # can't parse, keep
                        continue
                    
                    key = (target, src_method)
                    if key not in all_scores:
                        # No scores for this method, keep all
                        surviving.add((src_method, model_idx))
                        continue
                    
                    scores = all_scores[key]
                    # Sort scores descending, check if this model_idx is in top-N
                    sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
                    top_indices = set(sorted_indices[:topn])
                    
                    if model_idx in top_indices:
                        surviving.add((src_method, model_idx))

                # Count models that survive  
                n_total = len(t_df)
                n_survive = 0
                filtered_lddts = []
                
                for _, row in t_df.iterrows():
                    src_method, model_idx = parse_model_source(row["model_name"])
                    if src_method is None or (target, src_method) not in all_scores:
                        # Keep all unscored models
                        n_survive += 1
                        filtered_lddts.append(row["lddt_pli"])
                    elif (src_method, model_idx) in surviving:
                        n_survive += 1
                        filtered_lddts.append(row["lddt_pli"])

                filtered_oracle = max(filtered_lddts) if filtered_lddts else 0
                
                results.append({
                    "target": target,
                    "n_total": n_total,
                    "n_survive": n_survive,
                    "full_rank1": full_rank1_lddt,
                    "oracle_full": oracle_lddt,
                    "oracle_filtered": filtered_oracle,
                    "oracle_preserved": "✓" if abs(filtered_oracle - oracle_lddt) < 1e-6 else "✗",
                    "oracle_delta": filtered_oracle - oracle_lddt,
                })

            if not results:
                continue

            df_r = pd.DataFrame(results)
            n_preserved = (df_r["oracle_preserved"] == "✓").sum()
            avg_survive = df_r["n_survive"].mean()
            
            print(f"\n  --- top-{topn} filter ---")
            print(f"  Oracle preserved: {n_preserved}/{len(df_r)} targets "
                  f"({100*n_preserved/len(df_r):.1f}%)")
            print(f"  Models per target: {avg_survive:.1f}/{df_r['n_total'].mean():.1f}")
            print(f"  Full rank-1 avg lDDT:     {df_r['full_rank1'].mean():.4f}")
            print(f"  Full oracle avg lDDT:     {df_r['oracle_full'].mean():.4f}")
            print(f"  Filtered oracle avg lDDT: {df_r['oracle_filtered'].mean():.4f}")
            print(f"  Oracle delta avg:         {df_r['oracle_delta'].mean():.4f}")
            
            # Show targets where oracle was lost
            lost = df_r[df_r["oracle_preserved"] == "✗"]
            if not lost.empty:
                print(f"  Oracle lost in {len(lost)} targets:")
                for _, row in lost.iterrows():
                    print(f"    {row['target']}: oracle {row['oracle_full']:.3f} → {row['oracle_filtered']:.3f} "
                          f"(delta={row['oracle_delta']:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="casp16_l1000")
    parser.add_argument("--metric", default="pair_iptm",
                        choices=["pair_iptm", "ligand_plddt", "ranking_score"])
    parser.add_argument("--topn", type=int, nargs="+", default=[10, 20, 30])
    args = parser.parse_args()
    run_experiment(args.dataset, args.metric, args.topn)


if __name__ == "__main__":
    main()
