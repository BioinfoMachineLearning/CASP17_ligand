"""Run only the RNA MSA search step from a Protenix input JSON, then patch
the JSON in-place with absolute `unpairedMsaPath`.

Why a custom wrapper instead of `protenix prep`:
  `protenix prep` hard-codes use_msa=True + use_template=True, which means
  RNA-only targets (no proteinChain) waste time / hit network on empty
  protein MSA + template stages. This wrapper imports the official
  `update_rna_msa_info` from `runner.rna_msa_search` (same nhmmer × 3-DB
  search Protenix runs internally) and runs only that step.

Output a3m:
    {out_dir}/{task_name}/rna_msa/{idx}/rna_msa.a3m

The input JSON is patched in-place (unless --no-patch) so r1/r2/etc. all
reuse the same a3m without per-seed re-search.

Usage:
    PROTENIX_ROOT_DIR={PROJECT_ROOT}/weights/protenix \
        conda run -n protenix python casp17/scripts/run_rna_msa_only.py \
        --input data/test_cases/casp17_R/protenix_inputs/R2314.json \
        --out_dir data/test_cases/casp17_R/protenix_msa
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "forks" / "Protenix"))

from runner.rna_msa_search import update_rna_msa_info  # noqa: E402


def _abs(p: str) -> str:
    return str(Path(p).resolve())


def _patch_json_in_place(json_path: str, json_data: list[dict]) -> int:
    """Rewrite each rnaSequence's unpairedMsaPath to an absolute path
    (update_rna_msa_info writes a relative path). Returns the number of
    paths absolutised. The on-disk file is overwritten."""
    n = 0
    for task in json_data:
        for seq in task.get("sequences", []):
            rna = seq.get("rnaSequence")
            if rna and "unpairedMsaPath" in rna:
                rna["unpairedMsaPath"] = _abs(rna["unpairedMsaPath"])
                n += 1
    if n:
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Protenix input JSON")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write {task}/rna_msa/{idx}/rna_msa.a3m",
    )
    parser.add_argument("--nhmmer_n_cpu", type=int, default=16)
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Do NOT overwrite the input JSON with absolute unpairedMsaPath",
    )
    args = parser.parse_args()

    if not os.environ.get("PROTENIX_ROOT_DIR"):
        raise SystemExit(
            "PROTENIX_ROOT_DIR must be set so the 90 GB RNA databases land in "
            "weights/protenix/search_database/ (one-time download)."
        )

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    updated = update_rna_msa_info(
        data, out_dir=args.out_dir, nhmmer_n_cpu=args.nhmmer_n_cpu
    )
    print(f"update_rna_msa_info: updated={updated}")

    if updated and not args.no_patch:
        n = _patch_json_in_place(args.input, data)
        print(f"patched {args.input}: {n} rnaSequence(s) → absolute unpairedMsaPath")
    elif updated and args.no_patch:
        print("--no-patch set → JSON not modified")
        print(json.dumps(data, indent=2))
    else:
        print("Nothing to patch (a3m path already present in JSON).")


if __name__ == "__main__":
    main()
