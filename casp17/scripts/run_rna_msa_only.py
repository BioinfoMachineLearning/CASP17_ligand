"""Run only the RNA MSA search step from a Protenix input JSON.

Avoids `protenix prep`'s hard-coded protein MSA + template steps, which are
no-ops (and can stall on network) for RNA-only targets.

Usage:
    PROTENIX_ROOT_DIR=weights/protenix \
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Protenix input JSON")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write {task}/rna_msa/{idx}/rna_msa.a3m",
    )
    parser.add_argument("--nhmmer_n_cpu", type=int, default=16)
    args = parser.parse_args()

    if not os.environ.get("PROTENIX_ROOT_DIR"):
        raise SystemExit(
            "PROTENIX_ROOT_DIR must be set so DBs are written to "
            "weights/protenix/search_database/"
        )

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    updated = update_rna_msa_info(
        data, out_dir=args.out_dir, nhmmer_n_cpu=args.nhmmer_n_cpu
    )
    print(f"updated={updated}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
