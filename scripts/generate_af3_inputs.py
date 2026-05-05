#!/usr/bin/env python3
"""Generate AlphaFold3 input JSONs for L1000/L3000 struct targets."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/bmlfast/Lyuwei/0.Projects/CASP17_ligand")

DATASETS = {
    "L1000": {
        "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L1000_prepared",
        "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L1000",
        "seq_file":   PROJECT_ROOT / "data/casp16_data/sequences/L1000.txt",
        "out_dir":    PROJECT_ROOT / "data/test_cases/casp16_l1000/af3_inputs",
    },
    "L3000": {
        "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L3000_prepared",
        "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L3000",
        "seq_file":   PROJECT_ROOT / "data/casp16_data/sequences/L3000.txt",
        "out_dir":    PROJECT_ROOT / "data/test_cases/casp16_l3000_struct/af3_inputs",
    },
}


def read_fasta_sequence(fasta_path):
    lines = fasta_path.read_text().strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">")).strip()


def read_smiles(tsv_path):
    lines = tsv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"No data in {tsv_path}")
    return lines[1].split("\t")[2]


def generate(dataset_name, cfg, num_seeds=5, overwrite=True):
    sequence = read_fasta_sequence(cfg["seq_file"])
    targets = sorted(os.listdir(cfg["struct_dir"]))
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    ok, skip, err = 0, 0, 0
    for target in targets:
        out_file = cfg["out_dir"] / f"{target}.json"
        if out_file.exists() and not overwrite:
            skip += 1
            continue
        smiles_file = cfg["smiles_dir"] / f"{target}.tsv"
        if not smiles_file.exists():
            print(f"  [WARN] SMILES not found: {smiles_file}")
            err += 1
            continue
        try:
            smiles = read_smiles(smiles_file)
        except Exception as e:
            print(f"  [WARN] {target}: {e}")
            err += 1
            continue

        input_json = {
            "name": target,
            "modelSeeds": list(range(1, num_seeds + 1)),
            "sequences": [
                {"protein": {"id": "A", "sequence": sequence}},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "dialect": "alphafold3",
            "version": 1,
        }
        out_file.write_text(json.dumps(input_json, indent=2))
        ok += 1

    print(f"{dataset_name}: generated={ok}, skipped={skip}, errors={err}, total={len(targets)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["L1000", "L3000", "all"])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, cfg in datasets.items():
        generate(name, cfg, num_seeds=args.num_seeds, overwrite=args.overwrite)
