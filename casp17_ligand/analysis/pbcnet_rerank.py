"""
PBCNet2.0 Re-ranking of Top-5 Cluster Representatives

Validate whether PBCNet2.0 (pairwise relative-binding-affinity model) can
re-rank the 5 cluster representatives per target to give a higher Top-1
lDDT-PLI than the upstream consensus baseline.

PBCNet2.0 predicts pIC50(lig_i) - pIC50(lig_j) for a pair. To get an absolute
score per pose we take the mean over all j != i of pre(i, j) — i.e. how much
better lig_i looks than the other 4 poses on average.

Must be run inside the `pbcnet` conda env (RDKit + dgl + torch + pbcnet code):
    conda run --no-capture-output -n pbcnet \\
        python casp17_ligand/analysis/pbcnet_rerank.py \\
        --meta top5_export/input_meta.csv \\
        --output outputs/ensemble/pbcnet_rerank.csv

Use --targets L1001 L1002 to sanity-check on a small subset first.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PBCNET_DIR = PROJECT_ROOT / "forks/PBCNet2.0"
sys.path.insert(0, str(PBCNET_DIR))
sys.path.insert(0, str(PBCNET_DIR / "model_code"))


def extract_pocket(ligand_sdf: Path, protein_pdb: Path, out_pocket_pdb: Path) -> None:
    """Save residues within 8 A of any ligand atom into out_pocket_pdb (PBCNet2.0 convention)."""
    from rdkit import Chem
    from scipy.spatial import distance_matrix
    from Bio.PDB import PDBParser, PDBIO, Select

    mol = Chem.MolFromMolFile(str(ligand_sdf))
    if mol is None:
        raise ValueError(f"RDKit could not read ligand SDF: {ligand_sdf}")
    lig_pos = mol.GetConformer().GetPositions()

    structure = PDBParser(QUIET=True).get_structure("p", str(protein_pdb))

    class PocketSelector(Select):
        def accept_residue(self, residue):
            heavy = np.array([
                list(a.get_vector()) for a in residue.get_atoms()
                if "H" not in a.get_id()
            ])
            if heavy.ndim < 2 or heavy.size == 0:
                return 0
            return int(np.min(distance_matrix(heavy, lig_pos)) < 8.0)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pocket_pdb), PocketSelector())


def build_graph_pkl(ligand_sdf: Path, pocket_pdb: Path, out_pkl: Path) -> None:
    """Build the PBCNet2.0 heterograph and pickle it.

    Monkey-patches Chem.MolFromMolFile during the call so AF3/Protenix SDFs
    that fail strict valence checks (`Explicit valence for atom # N C, K, is
    greater than permitted`) can still be loaded — we fall back to
    sanitize=False then run partial sanitize that skips property checks.
    """
    from rdkit import Chem
    from Graph2pickle import Graph_Information

    PARTIAL = (Chem.SanitizeFlags.SANITIZE_ALL
               ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)

    def make_safe(orig_loader):
        def safe(path, *a, **kw):
            kw.pop("sanitize", None)
            m = orig_loader(path, sanitize=True)
            if m is not None:
                return m
            m = orig_loader(path, sanitize=False)
            if m is None:
                return None
            try:
                Chem.SanitizeMol(m, sanitizeOps=PARTIAL)
            except Exception:
                pass
            try:
                Chem.GetSSSR(m)  # init ring info, needed by featurizers
            except Exception:
                pass
            return m
        return safe

    orig_sdf = Chem.MolFromMolFile
    orig_pdb = Chem.MolFromPDBFile
    orig_remove = Chem.RemoveAllHs

    def safe_remove(mol, *a, **kw):
        if mol is None:
            return None
        kw.pop("sanitize", None)
        try:
            return orig_remove(mol, sanitize=True)
        except Exception:
            pass
        try:
            m = orig_remove(mol, sanitize=False)
            try:
                Chem.GetSSSR(m)
            except Exception:
                pass
            return m
        except Exception:
            return mol

    Chem.MolFromMolFile = make_safe(orig_sdf)
    Chem.MolFromPDBFile = make_safe(orig_pdb)
    Chem.RemoveAllHs = safe_remove
    try:
        g = Graph_Information(str(ligand_sdf), str(pocket_pdb))
    finally:
        Chem.MolFromMolFile = orig_sdf
        Chem.MolFromPDBFile = orig_pdb
        Chem.RemoveAllHs = orig_remove

    with open(out_pkl, "wb") as f:
        pickle.dump(g, f)


def predict_pairs(pair_csv: Path, model, device: str, batch_size: int) -> np.ndarray:
    """Run PBCNet2.0 on every row of pair_csv, return the prediction array."""
    import torch
    from torch.utils.data import DataLoader
    from Dataloader.dataloader import LeadOptDataset, collate_fn
    from predict.predict import predict

    ds = LeadOptDataset(str(pair_csv))
    dl = DataLoader(
        ds, collate_fn=collate_fn, batch_size=batch_size,
        drop_last=False, shuffle=False, pin_memory=False,
    )
    out = predict(model, dl, device)
    pre = out[4]  # (mae, rmse, mae_g, rmse_g, valid_prediction, ...)
    return np.asarray(pre)


def phase1_build_graphs(meta: pd.DataFrame, export_root: Path, cache_dir: Path) -> None:
    """For every (target, model), extract pocket + build graph pkl. Skips existing."""
    print(f"[Phase 1/3] Build graph pickles for {len(meta)} poses ...")
    skipped = 0
    failed = []
    for i, row in meta.iterrows():
        target, model_id = row["target"], row["model_id"]
        sdf = export_root / row["ligand_sdf"]
        pdb = export_root / row["protein_pdb"]
        td = cache_dir / target
        td.mkdir(parents=True, exist_ok=True)
        pocket_pdb = td / f"{model_id}_pocket.pdb"
        pkl = td / f"{model_id}.pkl"
        if pkl.exists():
            skipped += 1
            continue
        try:
            if not pocket_pdb.exists():
                extract_pocket(sdf, pdb, pocket_pdb)
            build_graph_pkl(sdf, pocket_pdb, pkl)
        except Exception as e:
            failed.append((target, model_id, str(e)[:200]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(meta)} processed (skipped={skipped}, failed={len(failed)})")
    print(f"  done: skipped={skipped}, built={len(meta) - skipped - len(failed)}, failed={len(failed)}")
    for t, m, e in failed[:10]:
        print(f"    [fail] {t}/{m}: {e}")
    if len(failed) > 10:
        print(f"    ... ({len(failed) - 10} more)")


def phase2_predict(meta: pd.DataFrame, cache_dir: Path, model_path: Path,
                   device: str, batch_size: int) -> pd.DataFrame:
    """Build a single pair csv across all targets, run model once, return per-pair predictions."""
    import torch

    print("[Phase 2/3] Building pair csv ...")
    pair_rows = []
    skipped_targets = 0
    for target, group in meta.groupby("target"):
        group = group.sort_values("predicted_top5_position")
        td = cache_dir / target
        pkls = []
        for _, row in group.iterrows():
            pkl = td / f"{row['model_id']}.pkl"
            if not pkl.exists():
                pkls = None
                break
            pkls.append((row["model_id"], pkl))
        if pkls is None or len(pkls) != 5:
            skipped_targets += 1
            continue
        for mi, pi in pkls:
            for mj, pj in pkls:
                pair_rows.append({
                    "lig1": pi.name, "lig2": pj.name,
                    "Label": 0.0, "Label1": 0.0, "Label2": 0.0,
                    "dir_1": str(pi), "dir_2": str(pj),
                    "target": target, "model_i": mi, "model_j": mj,
                })

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        print(f"  0 pairs (skipped {skipped_targets} incomplete targets)")
        return pair_df
    print(f"  {len(pair_df)} pairs over {pair_df['target'].nunique()} targets "
          f"(skipped {skipped_targets} incomplete)")

    pair_csv = cache_dir / "_all_pairs.csv"
    pbc_cols = ["lig1", "lig2", "Label", "Label1", "Label2", "dir_1", "dir_2"]
    pair_df[pbc_cols].to_csv(pair_csv, index=False)

    print(f"  loading model from {model_path} ...")
    model = torch.load(str(model_path), map_location=torch.device(device), weights_only=False)
    model.to(device).eval()

    print(f"  predicting ({device}, batch_size={batch_size}) ...")
    pair_df["pre"] = predict_pairs(pair_csv, model, device, batch_size)
    return pair_df


def phase3_aggregate(meta: pd.DataFrame, pair_df: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Aggregate per-pose score, rerank within target, write csv, print summary table."""
    print("[Phase 3/3] Aggregate + rerank ...")
    self_pairs = pair_df["model_i"] != pair_df["model_j"]
    agg = (pair_df[self_pairs]
           .groupby(["target", "model_i"])["pre"].mean()
           .reset_index()
           .rename(columns={"model_i": "model_id", "pre": "pbcnet_score"}))

    out = meta.merge(agg, on=["target", "model_id"], how="left")
    out["pbcnet_rank"] = out.groupby("target")["pbcnet_score"].rank(
        ascending=False, method="first")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"  wrote {output}")

    print_summary(out)
    return out


def print_summary(out: pd.DataFrame) -> None:
    valid = out.dropna(subset=["pbcnet_score"])
    if valid.empty:
        print("\n(no targets scored)")
        return
    n_targets = valid["target"].nunique()

    baseline = (valid[valid["predicted_top5_position"] == 1]
                .groupby("target")["lddt_pli"].first())
    pbc_top1 = (valid[valid["pbcnet_rank"] == 1]
                .groupby("target")["lddt_pli"].first())
    oracle = valid.groupby("target")["lddt_pli"].max()
    avg5 = valid.groupby("target")["lddt_pli"].mean()

    print(f"\n=== Mean Top-1 lDDT-PLI ({n_targets} targets) ===")
    print(f"  Baseline (consensus Top-1): {baseline.mean():.4f}")
    print(f"  PBCNet2.0 reranked Top-1:   {pbc_top1.mean():.4f}  "
          f"(delta vs baseline: {pbc_top1.mean() - baseline.mean():+.4f})")
    print(f"  Random pick (mean of 5):    {avg5.mean():.4f}")
    print(f"  Oracle (best of 5):         {oracle.mean():.4f}")

    # per-dataset breakdown
    print("\n=== Per-dataset Top-1 lDDT-PLI ===")
    for ds, gv in valid.groupby("dataset"):
        n = gv["target"].nunique()
        b = gv[gv["predicted_top5_position"] == 1].groupby("target")["lddt_pli"].first().mean()
        p = gv[gv["pbcnet_rank"] == 1].groupby("target")["lddt_pli"].first().mean()
        o = gv.groupby("target")["lddt_pli"].max().mean()
        print(f"  {ds} (n={n:3d}): baseline={b:.4f}  pbcnet={p:.4f}  "
              f"delta={p - b:+.4f}  oracle={o:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", default="top5_export/input_meta.csv",
                   help="Path to input_meta.csv from export_top5_for_rerank.py")
    p.add_argument("--export-root", default="top5_export",
                   help="Root the meta paths are relative to")
    p.add_argument("--output", default="outputs/ensemble/pbcnet_rerank.csv")
    p.add_argument("--cache-dir", default="outputs/ensemble/pbcnet_cache",
                   help="Where pocket PDBs and graph pickles are kept")
    p.add_argument("--model", default=str(PBCNET_DIR / "PBCNet2.pth"))
    p.add_argument("--device", default=None,
                   help="cuda or cpu; auto-detect if omitted")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--targets", nargs="*",
                   help="Only run these target IDs (sanity-check on subset)")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip Phase 1, only do Phase 2+3 (graphs already cached)")
    args = p.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_path = Path(args.meta)
    if not meta_path.is_absolute():
        meta_path = PROJECT_ROOT / meta_path
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = PROJECT_ROOT / export_root
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    meta = pd.read_csv(meta_path)
    if args.targets:
        meta = meta[meta["target"].isin(set(args.targets))].reset_index(drop=True)
        print(f"Filtered to {meta['target'].nunique()} targets / {len(meta)} rows")

    if not args.skip_build:
        phase1_build_graphs(meta, export_root, cache_dir)

    pair_df = phase2_predict(meta, cache_dir, Path(args.model), args.device, args.batch_size)
    if pair_df.empty:
        print("No complete (5-pose) targets — nothing to score.")
        return

    phase3_aggregate(meta, pair_df, output)


if __name__ == "__main__":
    main()
