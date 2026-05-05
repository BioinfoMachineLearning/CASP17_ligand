"""Compare top-1 / top-5 best lDDT between r1-only and r1+r2 (or any two
cluster_experiment_detail CSVs). Produces per-target delta CSV + aggregate
improved/degraded/tied counts.

Usage:
    python casp17_ligand/analysis/compare_r1_vs_r1r2.py \
        --r1 outputs/clustering/archive/cluster_experiment_detail_20260405_030400_4methods_consensus_pb.csv \
        --r1r2 outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv \
        --output compare_l2000.csv
"""
import argparse
import os
import sys

import pandas as pd


def load_cluster_csv(path: str, config_filter: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config_filter:
        df = df[df["config"] == config_filter]
    if len(df) == 0:
        raise SystemExit(f"No rows in {path} after config filter='{config_filter}'")
    # Keep relevant columns only
    keep = ["target", "config", "top1_lddt", "top5_best_lddt",
            "num_clusters", "largest_size", "num_models"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True, help="r1-only cluster_experiment_detail CSV")
    ap.add_argument("--r1r2", required=True, help="r1+r2 cluster_experiment_detail CSV")
    ap.add_argument("--config", default="自适应: 40HA->0.70",
                    help="Config filter (default: 自适应: 40HA->0.70)")
    ap.add_argument("--output", required=True, help="Per-target delta CSV output path")
    args = ap.parse_args()

    df_r1 = load_cluster_csv(args.r1, args.config).set_index("target")
    df_r2 = load_cluster_csv(args.r1r2, args.config).set_index("target")

    # Find common targets
    common = sorted(set(df_r1.index) & set(df_r2.index))
    only_r1 = sorted(set(df_r1.index) - set(df_r2.index))
    only_r2 = sorted(set(df_r2.index) - set(df_r1.index))

    print(f"=== Coverage ===")
    print(f"  r1-only CSV: {len(df_r1)} targets")
    print(f"  r1+r2 CSV:   {len(df_r2)} targets")
    print(f"  Common:      {len(common)} targets")
    if only_r1:
        print(f"  Only in r1 ({len(only_r1)}): {only_r1[:10]}{'...' if len(only_r1)>10 else ''}")
    if only_r2:
        print(f"  Only in r1+r2 ({len(only_r2)}): {only_r2[:10]}{'...' if len(only_r2)>10 else ''}")

    rows = []
    improved_top1 = degraded_top1 = tied_top1 = 0
    improved_top5 = degraded_top5 = tied_top5 = 0
    for t in common:
        r1 = df_r1.loc[t]
        r2 = df_r2.loc[t]
        top1_r1 = float(r1["top1_lddt"]) if pd.notna(r1["top1_lddt"]) else None
        top1_r2 = float(r2["top1_lddt"]) if pd.notna(r2["top1_lddt"]) else None
        top5_r1 = float(r1["top5_best_lddt"]) if pd.notna(r1["top5_best_lddt"]) else None
        top5_r2 = float(r2["top5_best_lddt"]) if pd.notna(r2["top5_best_lddt"]) else None

        d_top1 = None
        if top1_r1 is not None and top1_r2 is not None:
            d_top1 = top1_r2 - top1_r1
            if d_top1 > 1e-6:
                improved_top1 += 1
            elif d_top1 < -1e-6:
                degraded_top1 += 1
            else:
                tied_top1 += 1

        d_top5 = None
        if top5_r1 is not None and top5_r2 is not None:
            d_top5 = top5_r2 - top5_r1
            if d_top5 > 1e-6:
                improved_top5 += 1
            elif d_top5 < -1e-6:
                degraded_top5 += 1
            else:
                tied_top5 += 1

        rows.append({
            "target": t,
            "top1_r1": top1_r1,
            "top1_r1r2": top1_r2,
            "delta_top1": d_top1,
            "top5_best_r1": top5_r1,
            "top5_best_r1r2": top5_r2,
            "delta_top5": d_top5,
            "num_clusters_r1": r1.get("num_clusters"),
            "num_clusters_r1r2": r2.get("num_clusters"),
            "num_models_r1": r1.get("num_models"),
            "num_models_r1r2": r2.get("num_models"),
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print()
    print(f"=== Top-1 lDDT ===")
    n = len(df_out.dropna(subset=["delta_top1"]))
    if n:
        mean_r1 = df_out["top1_r1"].mean()
        mean_r2 = df_out["top1_r1r2"].mean()
        mean_d = df_out["delta_top1"].mean()
        print(f"  Mean r1     : {mean_r1:.4f}")
        print(f"  Mean r1+r2  : {mean_r2:.4f}")
        print(f"  Mean Δ      : {mean_d:+.4f}")
        print(f"  Improved    : {improved_top1}/{n}")
        print(f"  Degraded    : {degraded_top1}/{n}")
        print(f"  Tied        : {tied_top1}/{n}")

    print()
    print(f"=== Top-5 Best lDDT ===")
    n = len(df_out.dropna(subset=["delta_top5"]))
    if n:
        mean_r1 = df_out["top5_best_r1"].mean()
        mean_r2 = df_out["top5_best_r1r2"].mean()
        mean_d = df_out["delta_top5"].mean()
        print(f"  Mean r1     : {mean_r1:.4f}")
        print(f"  Mean r1+r2  : {mean_r2:.4f}")
        print(f"  Mean Δ      : {mean_d:+.4f}")
        print(f"  Improved    : {improved_top5}/{n}")
        print(f"  Degraded    : {degraded_top5}/{n}")
        print(f"  Tied        : {tied_top5}/{n}")

    print()
    print(f"[+] Per-target delta CSV: {args.output}")


if __name__ == "__main__":
    main()
