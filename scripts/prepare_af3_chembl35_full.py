#!/usr/bin/env python3
"""Generate AF3 input JSONs from chembl35_full_input.csv.

CSV schema (differs from chembl35_dataset_A.csv):
    compound_id,target_sequence,compound_iso_smiles,affinity,status

Target ID is the prefix of compound_id before "_CHEMBL".

Usage:
    python scripts/prepare_af3_chembl35_full.py \
        --csv data/chembl35/chembl35_full_input.csv \
        --targets_file data/test_cases/chembl35_full/targets_short.txt \
        --output_dir data/test_cases/chembl35_full/af3_inputs \
        --manifest_out data/test_cases/chembl35_full/af3_manifest_short.json
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--targets_file", required=True,
                    help="TSV with target_id\\tseq_len\\tnum_compounds per line")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--manifest_out", required=True)
    args = ap.parse_args()

    with open(args.targets_file) as f:
        target_ids = [ln.split("\t")[0] for ln in f if ln.strip()]
    target_set = set(target_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_target = {}
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row["compound_id"]
            if "_CHEMBL" not in cid:
                continue
            tid = cid.split("_CHEMBL")[0]
            if tid not in target_set:
                continue
            by_target.setdefault(tid, []).append(row)

    manifest = {}
    total = 0
    for tid in target_ids:  # preserve ordering from the input list
        rows = by_target.get(tid, [])
        names = []
        for i, row in enumerate(rows):
            name = f"{tid}_{i:03d}"
            names.append(name)
            out_path = out_dir / f"{name}.json"
            if out_path.exists():
                total += 1
                continue
            input_json = {
                "name": name,
                "sequences": [
                    {"protein": {"id": "A", "sequence": row["target_sequence"]}},
                    {"ligand":  {"id": "B", "smiles":  row["compound_iso_smiles"]}},
                ],
                "dialect": "alphafold3",
                "version": 1,
            }
            out_path.write_text(json.dumps(input_json, indent=2))
            total += 1
        manifest[tid] = names
        print(f"  {tid}: {len(rows)} compounds")

    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))
    print(f"\nTotal JSONs for {len(manifest)} targets: {total}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
