#!/usr/bin/env python3
"""Generate AF3 input JSONs from chembl35 CSV.

Usage:
    python scripts/prepare_af3_chembl35.py \
        --csv data/chembl35/chembl35_dataset_A.csv \
        --targets Q12809,P08183,P51449,P24941,P31153,Q03111,Q06124,Q99538,P10275,P42866 \
        --output_dir data/test_cases/chembl35_A/af3_inputs
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--targets", required=True, help="Comma-separated Target IDs")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    target_ids = set(args.targets.split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group compounds by Target ID
    series = {}
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["Target ID"]
            if tid not in target_ids:
                continue
            series.setdefault(tid, []).append(row)

    # Generate JSONs
    total = 0
    manifest = {}  # series -> [compound_names]
    for tid in sorted(series.keys()):
        compounds = series[tid]
        names = []
        for i, row in enumerate(compounds):
            name = f"{tid}_{i:03d}"
            names.append(name)

            smiles = row["compound_iso_smiles"]
            sequence = row["target_sequence"]

            input_json = {
                "name": name,
                "sequences": [
                    {"protein": {"id": "A", "sequence": sequence}},
                    {"ligand": {"id": "B", "smiles": smiles}},
                ],
                "dialect": "alphafold3",
                "version": 1,
            }

            out_path = output_dir / f"{name}.json"
            out_path.write_text(json.dumps(input_json, indent=2))
            total += 1

        manifest[tid] = names
        print(f"  {tid}: {len(compounds)} compounds")

    # Write manifest for orchestration script
    manifest_path = output_dir.parent / "af3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nTotal: {total} JSONs written to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
