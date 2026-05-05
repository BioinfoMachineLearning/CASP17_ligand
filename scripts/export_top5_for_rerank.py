"""
Export top-5 cluster representatives (PDB + SDF) into a portable directory tree
for handing off to a new scoring/re-ranking method on a different machine.

Layout:
    top5_export/
    ├── l1000/L{ID}/{protein.pdb, ligand.sdf}*5
    ├── l2000/...
    ├── l3000/...
    ├── l4000/...
    └── input_meta.csv     # one row per (target, model)

Usage:
    python scripts/export_top5_for_rerank.py            # full export (227 targets)
    python scripts/export_top5_for_rerank.py --targets L1001    # one target
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RERANK_CSV = PROJECT_ROOT / "outputs/ensemble/rtmscore_rerank_top5.csv"
ENSEMBLE_DIR = PROJECT_ROOT / "outputs/ensemble"
EXPORT_DIR = PROJECT_ROOT / "top5_export"

DATASET_DIRS = {
    "l1000": "casp16_l1000",
    "l2000": "casp16_l2000_struct",
    "l3000": "casp16_l3000_struct",
    "l4000": "casp16_l4000",
}


def dataset_for_target(target_id: str) -> str:
    """Map L1001 -> l1000, L2002 -> l2000, etc."""
    series = target_id[:2]  # 'L1', 'L2', ...
    return f"l{series[1]}000"


def find_target_dir(target_id: str) -> Path | None:
    short = dataset_for_target(target_id)
    long_name = DATASET_DIRS[short]
    candidate = ENSEMBLE_DIR / long_name / "targets" / target_id
    return candidate if candidate.is_dir() else None


def find_protein_pdb(target_dir: Path, target_id: str, model_id: str, sdf_path: Path | None = None) -> Path | None:
    """Locate protein PDB. Tries:
      1. cif_converted/{target}_{source}_model_{N}_protein.pdb (single-ligand layout)
      2. cif_converted/lig_*/{target}_{source}_model_{N}_lig*_protein.pdb (multi-ligand layout)
      3. SDF-paired PDB in ranking_sucos/ (strip _pb=True.sdf suffix, append .pdb)
    """
    m = re.match(r"^(.+)_model(\d+)$", model_id)
    if not m:
        return None
    source, num = m.group(1), m.group(2)
    cif_dir = target_dir / "cif_converted"

    pdb = cif_dir / f"{target_id}_{source}_model_{num}_protein.pdb"
    if pdb.exists():
        return pdb

    multi = sorted(cif_dir.glob(f"lig_*/{target_id}_{source}_model_{num}_lig*_protein.pdb"))
    if multi:
        return multi[0]

    if sdf_path is not None and sdf_path.name.endswith("_pb=True.sdf"):
        paired = sdf_path.parent / sdf_path.name.replace("_pb=True.sdf", ".pdb")
        if paired.exists():
            return paired
    return None


def find_ligand_sdf(target_dir: Path, model_id: str) -> Path | None:
    """Strict *_pb=True.sdf only — pose must have passed PoseBusters validity.
    Returns None if no PB-valid pose exists for this model."""
    ranking_dir = target_dir / "ranking_sucos"
    pb_true = sorted(ranking_dir.glob(f"{model_id}_rank*_pb=True.sdf"))
    return pb_true[0] if pb_true else None


def export(targets_filter: set[str] | None = None) -> None:
    EXPORT_DIR.mkdir(exist_ok=True)
    meta_rows = []
    missing = []

    with open(RERANK_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target"]
            if targets_filter and target not in targets_filter:
                continue
            if row.get("error"):
                missing.append((target, "csv-error", row["error"]))
                continue

            top5_reps = row["top5_reps"].split(";")
            top5_lddts = row["top5_lddts"].split(";")
            if len(top5_reps) != len(top5_lddts):
                missing.append((target, "len-mismatch", row["top5_reps"]))
                continue

            target_dir = find_target_dir(target)
            if target_dir is None:
                missing.append((target, "target-dir-missing", str(target)))
                continue

            # Pre-pass: only export this target if all 5 models have PB-valid SDFs.
            # If any model lacks a *_pb=True.sdf, drop the entire target — the
            # downstream re-ranker (e.g. PBCNet2.0) needs all 5 to compute Top-1.
            short_ds = dataset_for_target(target)
            target_rows = []
            target_files = []  # (src_path, dst_path) pairs to copy if all 5 pass
            target_failed = False
            out_dir = EXPORT_DIR / short_ds / target

            for pos, (model_id, lddt) in enumerate(zip(top5_reps, top5_lddts), start=1):
                model_id = model_id.strip()
                if not model_id:
                    continue
                source_method = model_id.rsplit("_model", 1)[0]

                sdf_src = find_ligand_sdf(target_dir, model_id)
                if sdf_src is None:
                    missing.append((target, "target-dropped-sdf-missing", model_id))
                    target_failed = True
                    break
                pdb_src = find_protein_pdb(target_dir, target, model_id, sdf_src)
                if pdb_src is None:
                    missing.append((target, "target-dropped-pdb-missing", model_id))
                    target_failed = True
                    break

                pdb_canonical = f"{target}_{source_method}_model_{model_id.rsplit('_model', 1)[1]}_protein.pdb"
                target_files.append((pdb_src, out_dir / pdb_canonical))
                target_files.append((sdf_src, out_dir / sdf_src.name))
                target_rows.append({
                    "dataset": short_ds,
                    "target": target,
                    "model_id": model_id,
                    "source_method": source_method,
                    "protein_pdb": f"{short_ds}/{target}/{pdb_canonical}",
                    "ligand_sdf": f"{short_ds}/{target}/{sdf_src.name}",
                    "predicted_top5_position": pos,
                    "lddt_pli": float(lddt),
                })

            if target_failed or not target_rows:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            for src, dst in target_files:
                if not dst.exists():
                    shutil.copy2(src, dst)
            meta_rows.extend(target_rows)

    meta_path = EXPORT_DIR / "input_meta.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "target", "model_id", "source_method",
            "protein_pdb", "ligand_sdf", "predicted_top5_position", "lddt_pli",
        ])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"Wrote {len(meta_rows)} rows to {meta_path}")
    if missing:
        print(f"\n{len(missing)} missing entries:")
        for t, kind, info in missing[:30]:
            print(f"  {t}\t{kind}\t{info}")
        if len(missing) > 30:
            print(f"  ... ({len(missing) - 30} more)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", help="Only export these target IDs")
    args = p.parse_args()
    filt = set(args.targets) if args.targets else None
    export(filt)


if __name__ == "__main__":
    main()
