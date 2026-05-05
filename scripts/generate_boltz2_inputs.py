#!/usr/bin/env python3
"""Generate boltz2 input YAMLs for L1000 and L3000 struct targets."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/bmlfast/Lyuwei/0.Projects/CASP17_ligand")

DATASETS = {
    "L1000": {
        "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L1000_prepared",
        "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L1000",
        "seq_file":   PROJECT_ROOT / "data/casp16_data/sequences/L1000.txt",
        "msa_path":   "data/test_cases/casp16_l1000/msa/L1000_msa.csv",
        "out_dir":    PROJECT_ROOT / "data/test_cases/casp16_l1000/boltz2_inputs",
    },
    "L3000": {
        "struct_dir": PROJECT_ROOT / "data/casp16_data/struct/L3000_prepared",
        "smiles_dir": PROJECT_ROOT / "data/casp16_data/smiles/L3000",
        "seq_file":   PROJECT_ROOT / "data/casp16_data/sequences/L3000.txt",
        "msa_path":   "data/test_cases/casp16_l3000/msa/L3000_msa.csv",
        "out_dir":    PROJECT_ROOT / "data/test_cases/casp16_l3000_struct/boltz2_inputs",
    },
}

YAML_TEMPLATE = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
      msa: {msa_path}
  - ligand:
      id: B
      smiles: '{smiles}'
properties:
  - affinity:
      binder: B
"""

def read_smiles(tsv_path):
    """Read SMILES from tsv file (skip header, return SMILES column)."""
    with open(tsv_path) as f:
        lines = f.read().strip().splitlines()
    # lines[0] is header: ID\tName\tSMILES\tTask
    # lines[1] is data
    if len(lines) < 2:
        raise ValueError(f"No data in {tsv_path}")
    parts = lines[1].split("\t")
    return parts[2]  # SMILES column

def read_fasta_sequence(fasta_path):
    """Read sequence from FASTA file, skipping header lines."""
    lines = fasta_path.read_text().strip().splitlines()
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines).strip()

def generate(dataset_name, cfg, overwrite=True):
    sequence = read_fasta_sequence(cfg["seq_file"])
    targets = sorted(os.listdir(cfg["struct_dir"]))
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    ok, skip, err = 0, 0, 0
    for target in targets:
        out_file = cfg["out_dir"] / f"{target}_input.yaml"
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
        yaml_content = YAML_TEMPLATE.format(
            sequence=sequence,
            msa_path=cfg["msa_path"],
            smiles=smiles,
        )
        out_file.write_text(yaml_content)
        ok += 1

    print(f"{dataset_name}: generated={ok}, skipped={skip}, errors={err}, total_targets={len(targets)}")

if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        generate(name, cfg, overwrite=True)
