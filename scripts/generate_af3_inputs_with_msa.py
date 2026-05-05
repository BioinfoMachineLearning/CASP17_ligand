#!/usr/bin/env python3
"""Generate AF3 input JSONs with pre-computed MSA from a reference run.

All L1000 targets share the same protein, so MSA only needs to be computed once.
This script extracts MSA/templates from a completed run and injects them into
new target JSONs, allowing --norun_data_pipeline for subsequent runs.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/bmlfast/Lyuwei/0.Projects/CASP17_ligand")


def read_fasta_sequence(fasta_path):
    lines = fasta_path.read_text().strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">")).strip()


def read_smiles(tsv_path):
    lines = tsv_path.read_text().strip().splitlines()
    return lines[1].split("\t")[2]


def extract_protein_msa(data_json_path):
    """Extract protein MSA fields from a completed AF3 data JSON."""
    with open(data_json_path) as f:
        data = json.load(f)
    for seq in data["sequences"]:
        if "protein" in seq:
            p = seq["protein"]
            return {
                "unpairedMsa": p.get("unpairedMsa"),
                "pairedMsa": p.get("pairedMsa"),
                "templates": p.get("templates"),
            }
    raise ValueError("No protein entity found")


def generate_with_msa(dataset_cfg, msa_fields, num_seeds=10):
    """Generate AF3 JSONs with pre-computed MSA injected."""
    sequence = read_fasta_sequence(dataset_cfg["seq_file"])
    targets = sorted(os.listdir(dataset_cfg["struct_dir"]))
    out_dir = dataset_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, err = 0, 0
    for target in targets:
        smiles_file = dataset_cfg["smiles_dir"] / f"{target}.tsv"
        if not smiles_file.exists():
            print(f"  [WARN] SMILES not found: {smiles_file}")
            err += 1
            continue
        smiles = read_smiles(smiles_file)

        protein_entry = {"id": "A", "sequence": sequence}
        protein_entry.update(msa_fields)

        input_json = {
            "name": target,
            "modelSeeds": [42],
            "sequences": [
                {"protein": protein_entry},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "dialect": "alphafold3",
            "version": 2,
        }
        (out_dir / f"{target}.json").write_text(json.dumps(input_json))
        ok += 1

    print(f"Generated {ok} JSONs, {err} errors, total {len(targets)} targets")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_data_json", required=True,
                        help="Path to a completed AF3 *_data.json with MSA")
    parser.add_argument("--dataset", default="L1000", choices=["L1000", "L3000"])
    args = parser.parse_args()

    DATASETS = {
        "L1000": {
            "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L1000_prepared",
            "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L1000",
            "seq_file": PROJECT_ROOT / "data/casp16_data/sequences/L1000.txt",
            "out_dir": PROJECT_ROOT / "data/test_cases/casp16_l1000/af3_inputs_msa",
        },
        "L3000": {
            "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L3000_prepared",
            "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L3000",
            "seq_file": PROJECT_ROOT / "data/casp16_data/sequences/L3000.txt",
            "out_dir": PROJECT_ROOT / "data/test_cases/casp16_l3000_struct/af3_inputs_msa",
        },
    }

    print("Extracting MSA from reference run...")
    msa_fields = extract_protein_msa(args.ref_data_json)
    print(f"  unpairedMsa: {len(msa_fields['unpairedMsa'])} chars")
    print(f"  pairedMsa: {len(msa_fields['pairedMsa'])} chars")
    print(f"  templates: {len(msa_fields['templates'])} entries")

    generate_with_msa(DATASETS[args.dataset], msa_fields)
