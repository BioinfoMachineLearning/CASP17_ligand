"""Package top-5 cluster representatives per target for local PyMOL visualization.

Reads the canonical cluster CSV (cluster_experiment_detail_latest_<TGT>.csv) to get
MODEL 1-5 in submission order (largest cluster first; same as `top5_sdf_paths`).
For each model, copies the receptor PDB and ligand SDF out of the submission
directory into tests/casp17_R_our_alignments/<TGT>/, and emits a PyMOL .pml that
loads + aligns all 5 models to MODEL 1.

Usage:
    conda run -n casp17_ligand python casp17_ligand/analysis/generate_alignment_zips.py \\
        --targets R2326 R2327 R2328

If --targets is omitted, processes every <TGT>_files dir under
casp17/submissions/casp17_R/ that has a matching cluster CSV.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_CSV_DIR = PROJECT_ROOT / "outputs" / "ensemble_r1r2"
# These get re-assigned in main() based on --series flag.
SUBMISSIONS_DIR = PROJECT_ROOT / "casp17" / "submissions" / "casp17_R"
OUT_ROOT = PROJECT_ROOT / "tests" / "casp17_R_our_alignments"
ZIP_PATH = PROJECT_ROOT / "tests" / "casp17_R_our_alignments.zip"


def read_top5_order(target: str) -> list[str]:
    """Return ordered SDF basenames for MODEL 1..5 from cluster CSV.

    `top5_sdf_paths` in the CSV is `;`-joined and already in submission order
    (largest cluster first). We only need the basenames to find local copies in
    the submission dir.
    """
    csv_path = CLUSTER_CSV_DIR / f"cluster_experiment_detail_latest_{target}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"cluster CSV missing: {csv_path}")
    import csv as _csv
    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty cluster CSV: {csv_path}")
    row = rows[0]
    paths = [p.strip() for p in row["top5_sdf_paths"].split(";") if p.strip()]
    return [os.path.basename(p) for p in paths]


def sdf_to_pdb_basename(sdf_basename: str) -> str:
    """`xxx_sucos0.441_pb=True.sdf` → `xxx_sucos0.441` (the PDB stem)."""
    stem = sdf_basename
    if stem.endswith(".sdf"):
        stem = stem[:-4]
    for suffix in ("_pb=True", "_pb=False"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def emit_target(target: str) -> bool:
    src_dir = SUBMISSIONS_DIR / f"{target}_files"
    if not src_dir.exists():
        print(f"[{target}] WARN submission dir not found: {src_dir} — skipping")
        return False

    try:
        sdf_order = read_top5_order(target)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[{target}] WARN {e} — skipping")
        return False

    target_dir = OUT_ROOT / target
    target_dir.mkdir(parents=True, exist_ok=True)
    pml_path = target_dir / f"align_{target}.pml"

    print(f"[{target}] packaging {len(sdf_order)} models (in submission MODEL order)")
    obj_names: list[tuple[str, str, str]] = []  # (rec_obj, lig_obj, base)
    ref_rec_obj = None

    with open(pml_path, "w") as f:
        f.write(f"# PyMOL alignment script for {target} (Top-5 cluster reps, MODEL 1..5)\n")
        f.write(f"# Reference = MODEL 1 (largest cluster)\n")
        f.write("bg_color white\n\n")

        for i, sdf_basename in enumerate(sdf_order, start=1):
            pdb_stem = sdf_to_pdb_basename(sdf_basename)
            pdb_src = src_dir / f"{pdb_stem}.pdb"
            sdf_src = src_dir / sdf_basename

            if not pdb_src.exists():
                print(f"  MODEL {i}: PDB missing {pdb_src.name} — skip")
                continue
            if not sdf_src.exists():
                # Try alternate _pb= flag
                alt = pdb_stem + ("_pb=False.sdf" if "_pb=True" in sdf_basename else "_pb=True.sdf")
                if (src_dir / alt).exists():
                    sdf_src = src_dir / alt
                else:
                    cands = list(src_dir.glob(f"{pdb_stem}*.sdf"))
                    if cands:
                        sdf_src = cands[0]
                    else:
                        print(f"  MODEL {i}: SDF missing {sdf_basename} — skip")
                        continue

            rec_dest = f"model{i}_{pdb_stem}_rec.pdb"
            lig_dest = f"model{i}_{pdb_stem}_lig.sdf"
            shutil.copy2(pdb_src, target_dir / rec_dest)
            shutil.copy2(sdf_src, target_dir / lig_dest)

            # PyMOL-safe object name: short prefix + i to keep grouping clear
            clean = pdb_stem[:24].replace("=", "_").replace(".", "_").replace("-", "_")
            rec_obj = f"m{i}_{clean}_rec"
            lig_obj = f"m{i}_{clean}_lig"

            f.write(f"load {rec_dest}, {rec_obj}\n")
            f.write(f"load {lig_dest}, {lig_obj}\n")
            f.write(f"group model{i}, {rec_obj} {lig_obj}\n")
            obj_names.append((rec_obj, lig_obj, pdb_stem))
            if i == 1:
                ref_rec_obj = rec_obj

        if ref_rec_obj is None and obj_names:
            ref_rec_obj = obj_names[0][0]

        f.write("\n# Alignment to MODEL 1\n")
        if ref_rec_obj:
            f.write(f"# Reference: {ref_rec_obj}\n")
            for rec, lig, _ in obj_names:
                if rec == ref_rec_obj:
                    continue
                f.write(f"align {rec}, {ref_rec_obj}\n")
                f.write(f"matrix_copy {rec}, {lig}\n")

        f.write("\n# Visualization\n")
        f.write("as cartoon, polymer\n")
        f.write("show sticks, not polymer\n")
        f.write("util.cbc\n")
        f.write("center\n")
        f.write("zoom\n")

    print(f"[{target}] wrote {pml_path.relative_to(PROJECT_ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=[],
                    help="Target IDs (e.g. R2326 R2327 / T2383). Default: all under <series> submissions.")
    ap.add_argument("--series", default=None,
                    help="Submission series (casp17_R or casp17_T). Auto-detected from "
                         "first --targets prefix (R*→casp17_R, T*→casp17_T). Default: casp17_R.")
    ap.add_argument("--no-zip", action="store_true", help="Skip creating the .zip bundle.")
    args = ap.parse_args()

    # Resolve series: explicit --series wins, else infer from first target prefix
    global SUBMISSIONS_DIR, OUT_ROOT, ZIP_PATH
    if args.series:
        series = args.series
    elif args.targets and args.targets[0].startswith("T"):
        series = "casp17_T"
    elif args.targets and args.targets[0].startswith("M"):
        series = "casp17_M"
    else:
        series = "casp17_R"
    SUBMISSIONS_DIR = PROJECT_ROOT / "casp17" / "submissions" / series
    OUT_ROOT = PROJECT_ROOT / "tests" / f"{series}_our_alignments"
    ZIP_PATH = PROJECT_ROOT / "tests" / f"{series}_our_alignments.zip"

    if not args.targets:
        if not SUBMISSIONS_DIR.exists():
            print(f"submissions dir missing: {SUBMISSIONS_DIR}", file=sys.stderr)
            return 2
        args.targets = sorted(
            d.name.removesuffix("_files")
            for d in SUBMISSIONS_DIR.iterdir()
            if d.is_dir() and d.name.endswith("_files")
        )
        if not args.targets:
            print("no <TGT>_files dirs found", file=sys.stderr)
            return 2

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"output root: {OUT_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"targets: {', '.join(args.targets)}")

    ok = sum(emit_target(t) for t in args.targets)
    print(f"\nemit_target succeeded for {ok}/{len(args.targets)} targets")

    if not args.no_zip and ok > 0:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        rc = subprocess.run(
            ["zip", "-r", ZIP_PATH.name, OUT_ROOT.name],
            cwd=ZIP_PATH.parent,
        ).returncode
        if rc != 0:
            print(f"zip command failed (rc={rc})", file=sys.stderr)
            return 3
        print(f"zip: {ZIP_PATH.relative_to(PROJECT_ROOT)}")

    return 0 if ok == len(args.targets) else 1


if __name__ == "__main__":
    sys.exit(main())
