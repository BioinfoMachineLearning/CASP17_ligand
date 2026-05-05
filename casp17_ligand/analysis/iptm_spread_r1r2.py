"""Per-series inter-method pair_iptm spread on the r1+r2 (100 models/method) setup.

Reads `configs/model/ensemble_generation_r1r2_l{X}.yaml` to find each method's
output dirs (r1 + r2), pulls every confidence JSON via the shared helper
`_iter_confidence_jsons`, extracts pair_iptm via `extract_pair_chains_iptm`,
and aggregates:

  • per-model CSV   → outputs/ensemble_r1r2/casp16_l{X}_r1r2/iptm_all_models.csv
  • per-target view → outputs/ensemble_r1r2/casp16_l{X}_r1r2/iptm_per_target.csv
  • cross-series summary → outputs/ensemble_r1r2/iptm_spread_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import rootutils
import yaml

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from casp17_ligand.analysis.confidence_metric_analysis import _iter_confidence_jsons  # noqa: E402
from casp17_ligand.analysis.self_ranking_comparison import extract_pair_chains_iptm  # noqa: E402

SERIES = {
    "L1000": "configs/model/ensemble_generation_r1r2_l1000.yaml",
    "L2000": "configs/model/ensemble_generation_r1r2_l2000.yaml",
    "L3000": "configs/model/ensemble_generation_r1r2_l3000.yaml",
    "L4000": "configs/model/ensemble_generation_r1r2_l4000.yaml",
}


def _input_path_for(method: str, target: str, base_input_csv: str) -> str:
    """Return JSON input file path used by extract_pair_chains_iptm to
    identify protein/ligand chain indices. Boltz2 inputs are YAML and not
    JSON-loadable by _parse_input_entities → return "" so the chain-0=protein
    fallback kicks in (single-protein/single-ligand assumption)."""
    test_root = Path(base_input_csv).parent
    candidates = {
        "af3": test_root / "af3_inputs" / f"{target}.json",
        "protenix": test_root / "protenix_inputs" / f"{target}.json",
        "seedfold": test_root / "seedfold_inputs" / f"{target}.json",
    }
    p = candidates.get(method)
    return str(p) if p and p.exists() else ""


def _collect_one(args):
    method, target, dirs, input_path = args
    rows = []
    for jf, _cif in _iter_confidence_jsons(method, target, dirs):
        try:
            with open(jf) as fh:
                data = json.load(fh)
            score = extract_pair_chains_iptm(data, method, input_path)
        except Exception:
            continue
        if score is None:
            continue
        rows.append({
            "target": target,
            "method": method,
            "json": jf,
            "pair_iptm": float(score),
        })
    return rows


def _targets_for(series_dir: Path) -> list[str]:
    """Use evaluation_summary.csv (ranking_rmsd subset) to get the canonical
    target list for this r1+r2 series."""
    eval_csv = series_dir / "evaluation_summary.csv"
    df = pd.read_csv(eval_csv)
    return sorted(df.loc[df["method"] == "ranking_rmsd", "target"].unique())


def analyze_series(series: str, cfg_path: str, n_workers: int = 16) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dataset = cfg["dataset"]
    base_csv = cfg["input_csv"]
    methods = cfg["methods"]

    series_dir = ROOT / "outputs" / "ensemble_r1r2" / dataset
    targets = _targets_for(series_dir)
    print(f"\n[{series}] dataset={dataset} targets={len(targets)} methods={list(methods)}")

    work = []
    for m, mcfg in methods.items():
        dirs = list(mcfg["output_dirs"])
        for t in targets:
            ip = _input_path_for(m, t, base_csv)
            work.append((m, t, dirs, ip))

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_collect_one, w) for w in work]
        for fut in as_completed(futs):
            rows.extend(fut.result())

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  WARN: no records for {series}")
        return df

    out_all = series_dir / "iptm_all_models.csv"
    df.to_csv(out_all, index=False)
    counts = df.groupby(["target", "method"]).size().unstack("method", fill_value=0)
    print(f"  per-(target,method) count summary:")
    print(counts.describe().loc[["min", "50%", "max"]].astype(int).to_string())
    print(f"  saved {len(df)} records → {out_all}")

    pertarget = df.groupby(["target", "method"])["pair_iptm"].agg(["mean", "max"]).unstack("method")
    pertarget.columns = ["_".join(c) for c in pertarget.columns]
    pertarget.reset_index().to_csv(series_dir / "iptm_per_target.csv", index=False)

    means = df.groupby(["target", "method"])["pair_iptm"].mean().unstack("method")
    means["spread_max_min"] = means.max(axis=1) - means.min(axis=1)
    means["std_across_methods"] = means.iloc[:, :-1].std(axis=1)
    means["series"] = series
    return means


def cross_series_summary(per_series_means: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for series, df in per_series_means.items():
        if df.empty:
            continue
        method_cols = [c for c in df.columns if c not in ("series", "spread_max_min", "std_across_methods")]
        row = {
            "series": series,
            "n_targets": len(df),
            "spread_mean": df["spread_max_min"].mean(),
            "spread_median": df["spread_max_min"].median(),
            "spread_std": df["spread_max_min"].std(),
            "std_mean": df["std_across_methods"].mean(),
        }
        for m in sorted(method_cols):
            row[f"{m}_iptm_mean"] = df[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--series", nargs="*", default=list(SERIES.keys()),
                    help="Subset of series to run (default: all)")
    args = ap.parse_args()

    per_series = {}
    for s in args.series:
        cfg = SERIES[s]
        per_series[s] = analyze_series(s, cfg, n_workers=args.workers)

    summary = cross_series_summary(per_series)
    out_summary = ROOT / "outputs" / "ensemble_r1r2" / "iptm_spread_summary.csv"
    summary.to_csv(out_summary, index=False)
    print("\n" + "=" * 80)
    print("Cross-series summary (sorted by spread_mean):")
    print("=" * 80)
    print(summary.sort_values("spread_mean", ascending=False).round(3).to_string(index=False))
    print(f"\nsaved → {out_summary}")


if __name__ == "__main__":
    main()
