"""Aggregate topn_benefits_*.json results into formatted tables.

Reads all topn_benefits_{metric}.json files from the ensemble output dirs
and produces markdown-formatted tables suitable for experiments.md.

Usage:
    python casp17_ligand/analysis/aggregate_topn_results.py \
        --datasets casp16_l1000 casp16_l3000_struct \
        [--ensemble_dir outputs/ensemble]
"""

import argparse
import json
import os
from collections import defaultdict


METRICS = [
    "pair_iptm",
    "ligand_plddt",
    "combined_iptm_plddt",
    "pocket_plddt_4.5",
    "pocket_plddt_6",
    "pocket_plddt_8",
    "pocket_plddt_10",
]

TOPN_LABELS = ["10", "20", "30", "40", "all"]
CONSENSUS_METHODS = ["rmsd", "sucos", "rmsd_6a", "rmsd_8a", "rmsd_10a"]


def load_results(ensemble_dir, dataset, metrics=None):
    """Load all topn_benefits_{metric}.json for a dataset."""
    if metrics is None:
        metrics = METRICS
    results = {}
    base = os.path.join(ensemble_dir, dataset)
    for metric in metrics:
        path = os.path.join(base, f"topn_benefits_{metric}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[metric] = json.load(f)
    return results


def build_table(results, consensus_method, score_key="mean_lddt"):
    """Build a table: rows = metrics, columns = Top-N values.

    Returns list of dicts with keys: metric, Top-10, Top-20, Top-30, Top-40, All.
    score_key: "mean_lddt" for rank-1, "mean_top5_best" for top-5 best.
    """
    rows = []
    for metric in METRICS:
        if metric not in results:
            continue
        data = results[metric]
        summary = data.get("summary", {}).get(consensus_method, {})
        row = {"metric": metric}
        for topn in TOPN_LABELS:
            entry = summary.get(topn, {})
            val = entry.get(score_key)
            row[f"Top-{topn}"] = f"{val:.4f}" if val is not None else "-"
        rows.append(row)
    return rows


def format_markdown_table(rows, title):
    """Format rows into a markdown table string."""
    if not rows:
        return f"### {title}\n\nNo data available.\n"

    cols = ["metric"] + [f"Top-{t}" for t in TOPN_LABELS]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    lines = [f"### {title}", "", header, sep]
    for row in rows:
        vals = [row.get(c, "-") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    # Find and annotate the best value
    best_val = None
    best_metric = None
    best_topn = None
    for row in rows:
        for topn in TOPN_LABELS:
            key = f"Top-{topn}"
            try:
                v = float(row.get(key, "-"))
                if best_val is None or v > best_val:
                    best_val = v
                    best_metric = row["metric"]
                    best_topn = topn
            except ValueError:
                pass

    if best_val is not None:
        lines.append("")
        lines.append(f"**Best: {best_metric}, Top-{best_topn} = {best_val:.4f}**")

    lines.append("")
    return "\n".join(lines)


def build_combined_table(all_results, consensus_method, score_key_suffix="rank1_lddt"):
    """Build a combined table averaging across datasets.

    all_results: {dataset: {metric: json_data}}
    score_key_suffix: "rank1_lddt" or "top5_best_lddt"
    """
    rows = []
    for metric in METRICS:
        row = {"metric": metric}
        for topn in TOPN_LABELS:
            vals = []
            for dataset, results in all_results.items():
                if metric not in results:
                    continue
                data = results[metric]
                per_target = data.get("per_target", {})
                for target, t_data in per_target.items():
                    entry = t_data.get(topn, {})
                    v = entry.get(f"{consensus_method}_{score_key_suffix}")
                    if v is not None:
                        vals.append(v)
            mean = sum(vals) / len(vals) if vals else None
            row[f"Top-{topn}"] = f"{mean:.4f}" if mean is not None else "-"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate topn_benefits results into formatted tables")
    parser.add_argument("--datasets", nargs="+",
                        default=["casp16_l1000", "casp16_l3000_struct"])
    parser.add_argument("--ensemble_dir", default="outputs/ensemble")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    all_results = {}
    for dataset in args.datasets:
        all_results[dataset] = load_results(args.ensemble_dir, dataset)

    output_lines = []

    # Per-dataset tables
    for dataset in args.datasets:
        results = all_results[dataset]
        if not results:
            output_lines.append(f"## {dataset}\n\nNo topn_benefits files found.\n")
            continue

        output_lines.append(f"## {dataset}\n")
        for cm in CONSENSUS_METHODS:
            rows = build_table(results, cm, score_key="mean_lddt")
            title = f"{dataset} — {cm.upper()} Consensus Rank-1 lDDT"
            output_lines.append(format_markdown_table(rows, title))
            rows5 = build_table(results, cm, score_key="mean_top5_best")
            title5 = f"{dataset} — {cm.upper()} Consensus Top-5 Best lDDT"
            output_lines.append(format_markdown_table(rows5, title5))

    # Combined tables (all datasets merged at target level)
    if len(args.datasets) > 1:
        output_lines.append("## Combined (all datasets, target-level average)\n")
        for cm in CONSENSUS_METHODS:
            rows = build_combined_table(all_results, cm, score_key_suffix="rank1_lddt")
            title = f"Combined — {cm.upper()} Consensus Rank-1 lDDT"
            output_lines.append(format_markdown_table(rows, title))
            rows5 = build_combined_table(all_results, cm, score_key_suffix="top5_best_lddt")
            title5 = f"Combined — {cm.upper()} Consensus Top-5 Best lDDT"
            output_lines.append(format_markdown_table(rows5, title5))

    text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Saved to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
