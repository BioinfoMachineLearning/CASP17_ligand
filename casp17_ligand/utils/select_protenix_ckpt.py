"""Pick the winning Protenix ckpt for a CASP17 target by mean protein-ligand iptm.

Reads `*_summary_confidence_sample_*.json` from two r1 output dirs (one per
ckpt) and prints the winner. Fail-fast: errors on missing files, mismatched
sample counts, or empty dirs.

**Metric: cross-chain protein-ligand iptm, NOT the global `iptm` field.**
The global value averages over every chain pair, so on a multimer it is
dominated by the protein-protein interface and barely notices the ligand.
T2451 (BifA homodimer, 2x255 aa + 2 ligand copies) is where this first bit:

    ckpt      global iptm   cross-chain pair_iptm   ligand-ligand clashes
    default      0.7922            0.8406                   0/50
    applied      0.8391            0.7166                  27/50  (min 0.48 A)

The global metric picked `applied` — the checkpoint that stacks both ligand
copies on top of each other. The cross-chain metric picks `default`, which
agrees with the geometry. On single-chain targets the two are IDENTICAL (a
2-chain system's global iptm *is* the protein-ligand value), verified on
T2409-T2414: same winner, same delta to 4 decimals. So this is a correctness
fix for multimers and a no-op for everything shipped so far.

Audit row appended to `--audit-csv` (header auto-created), recording BOTH
metrics so a disagreement is visible after the fact.

Usage (from project root):
    python casp17_ligand/utils/select_protenix_ckpt.py \
        --target R2317 \
        --default-dir outputs/protenix/casp17_R_default_r1 \
        --applied-dir outputs/protenix/casp17_R_applied_r1 \
        --audit-csv casp17/audit/protenix_ckpt_selection.csv \
        --expected-n 50 \
        --input-json data/test_cases/casp17_R/protenix_inputs/R2317.json
"""

import argparse
import csv
import datetime as _dt
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from casp17_ligand.analysis.self_ranking_comparison import (  # noqa: E402
    extract_pair_chains_iptm,
)


def _collect_iptm(dir_path: str, target: str, input_json: str,
                  n_protein_chains: int) -> tuple[list[float], list[float]]:
    """Return (cross_chain_pair_iptm, global_iptm) per sample, in file order."""
    pat = os.path.join(
        dir_path, target, "seed_*", "predictions",
        f"{target}_summary_confidence_sample_*.json",
    )
    files = sorted(glob.glob(pat))
    pair: list[float] = []
    glob_: list[float] = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        g = data.get("iptm")
        if g is None:
            raise SystemExit(f"FAIL: {f} has no `iptm` key")
        glob_.append(float(g))
        v = extract_pair_chains_iptm(
            data, "protenix",
            input_path=input_json or None,
            n_protein_chains=n_protein_chains,
        )
        if v is None:
            raise SystemExit(
                f"FAIL: cannot extract cross-chain protein-ligand iptm from {f}. "
                f"Pass --input-json (for exact chain typing) or --n-protein-chains."
            )
        pair.append(float(v))
    return pair, glob_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--default-dir", required=True,
                    help="Output dir of `protenix_base_default_v1.0.0` r1")
    ap.add_argument("--applied-dir", required=True,
                    help="Output dir of `protenix_base_20250630_v1.0.0` r1")
    ap.add_argument("--audit-csv", required=True,
                    help="CSV file to append the decision row")
    ap.add_argument("--expected-n", type=int, default=50,
                    help="Required samples per ckpt; fail-fast if short")
    ap.add_argument("--input-json", default="",
                    help="Protenix input JSON, used to type protein vs ligand chains "
                         "exactly. Strongly recommended for multimers.")
    ap.add_argument("--n-protein-chains", type=int, default=0,
                    help="Fallback protein chain count when --input-json is absent "
                         "or disagrees with the confidence matrix (0 -> 1).")
    args = ap.parse_args()

    d_iptm, d_glob = _collect_iptm(args.default_dir, args.target,
                                   args.input_json, args.n_protein_chains)
    a_iptm, a_glob = _collect_iptm(args.applied_dir, args.target,
                                   args.input_json, args.n_protein_chains)

    if len(d_iptm) < args.expected_n:
        raise SystemExit(
            f"FAIL: default has only {len(d_iptm)} samples (<{args.expected_n}) for {args.target}"
        )
    if len(a_iptm) < args.expected_n:
        raise SystemExit(
            f"FAIL: applied has only {len(a_iptm)} samples (<{args.expected_n}) for {args.target}"
        )
    if len(d_iptm) != len(a_iptm):
        raise SystemExit(
            f"FAIL: sample count mismatch for {args.target}: "
            f"default={len(d_iptm)} applied={len(a_iptm)}"
        )

    d_mean = statistics.mean(d_iptm)
    d_std = statistics.stdev(d_iptm)
    a_mean = statistics.mean(a_iptm)
    a_std = statistics.stdev(a_iptm)

    # Locked: mean cross-chain protein-ligand iptm, no fallback.
    # Tied -> applied wins (stable, matches the historical tie-break).
    if d_mean > a_mean:
        winner, winner_dir = "default", args.default_dir
    else:
        winner, winner_dir = "applied", args.applied_dir
    delta = d_mean - a_mean

    # Surface a global-vs-cross-chain disagreement loudly: it means the global
    # metric is being driven by a non-ligand interface (the homodimer case).
    dg_mean, ag_mean = statistics.mean(d_glob), statistics.mean(a_glob)
    global_winner = "default" if dg_mean > ag_mean else "applied"
    if global_winner != winner:
        print(
            f"# WARNING {args.target}: global iptm would pick '{global_winner}' "
            f"(default {dg_mean:.4f} vs applied {ag_mean:.4f}) but cross-chain "
            f"protein-ligand iptm picks '{winner}' (default {d_mean:.4f} vs "
            f"applied {a_mean:.4f}). Trusting the cross-chain metric — the global "
            f"one averages in the protein-protein interface. Verify ligand "
            f"geometry before submitting.",
            file=sys.stderr,
        )

    # Audit row
    os.makedirs(os.path.dirname(args.audit_csv), exist_ok=True)
    new_file = not os.path.exists(args.audit_csv)
    with open(args.audit_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow([
                "timestamp", "target", "n_default", "mean_iptm_default",
                "std_iptm_default", "n_applied", "mean_iptm_applied",
                "std_iptm_applied", "winner", "delta_default_minus_applied",
                "mean_global_iptm_default", "mean_global_iptm_applied",
                "global_winner",
            ])
        w.writerow([
            _dt.datetime.now().isoformat(timespec="seconds"),
            args.target,
            len(d_iptm), f"{d_mean:.6f}", f"{d_std:.6f}",
            len(a_iptm), f"{a_mean:.6f}", f"{a_std:.6f}",
            winner, f"{delta:+.6f}",
            f"{dg_mean:.6f}", f"{ag_mean:.6f}", global_winner,
        ])

    # Stdout (for shell capture): two lines, key=value
    print(f"WINNER={winner}", flush=True)
    print(f"WINNER_DIR={winner_dir}", flush=True)
    print(
        f"# {args.target}: default mean_iptm={d_mean:.4f}±{d_std:.4f} "
        f"vs applied mean_iptm={a_mean:.4f}±{a_std:.4f} "
        f"-> winner={winner} (Δ={delta:+.4f})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
