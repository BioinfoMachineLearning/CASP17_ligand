#!/usr/bin/env python3
"""Build a candidate table for a two-slate submission and propose a selection.

CASP17 T2451 asks for models 1-5 in "conformation 1" and 6-10 in "conformation 2".
Our 800-model pool contains a single receptor conformation (verified: 793/800
within 0.644 A at the 45 interface residues; four independent sampling
strategies failed to produce a second arrangement), so the second slate cannot
be a different receptor state. The instruction we work to instead is:

    models 1-5   the top ligand pose with the dominant protein conformation
    models 6-10  OTHER ligand poses, with protein conformations that are either
                 very similar OR less similar to the dominant one — both should
                 be represented.

The stock Stage 3 output cannot express that. It reports one representative per
cluster and only the top five, which hard-codes "cluster i -> model i". Two
things this case needs are therefore missing from it: reaching past cluster 5,
and picking SEVERAL members of the SAME ligand-pose cluster that differ in
receptor conformation.

So this tool re-derives the clustering from Stage 1's cached pairwise SuCOS
matrix using the very same `butina_cluster` and threshold schedule as Stage 3
(cluster ids therefore agree with the standard pipeline), and adds a second,
orthogonal axis: how far each model's receptor sits from the reference. The two
axes together are what the instruction above is phrased in.

Nothing is auto-submitted. The table is the deliverable; `--propose` prints one
defensible selection so there is a starting point to argue with.

Usage:
    python casp17_ligand/analysis/select_two_slate.py \\
        --target-dir outputs/ensemble_r1r2/casp17_T2451_pool/targets/T2451 \\
        --maxclust 0.80:60:0.05:0.30 --n-res 255 \\
        --out-dir outputs/conformation/T2451/slate_selection --propose
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from casp17_ligand.analysis.conformation_diagnostic import (  # noqa: E402
    apply_rt, kabsch, read_dimer_ca_any,
)
from casp17_ligand.analysis.evaluate_topn_clusters import (  # noqa: E402
    build_geom_ok, butina_cluster, parse_ranking_sucos,
)

RANK_RE = re.compile(r"^(.+?)_rank(\d+)_orig(\d+)_sucos([\d.]+)_pb=(True|False)\.sdf$")


def sdf_for_probe(td: Path):
    """Any one ranked SDF, just to inspect the ligand's chemistry."""
    return sorted(glob.glob(str(td / "ranking_sucos" / "*.sdf")))


def _count_checkable_sp3(sdf_path):
    """sp3 atoms with >=2 heavy neighbours, i.e. the ones that CAN form an angle.

    Zero means the sp3 filter is structurally unable to reject anything for this
    ligand, which is very different from "every pose passed". T2451's ligand is
    15/16 sp2 (fluorophenyl-azo-pyrazole); its only sp3 atom is the terminal F.
    """
    if not sdf_path:
        return 0
    try:
        from rdkit import Chem
    except ImportError:
        return 0
    mol = Chem.MolFromMolFile(sdf_path)
    if mol is None:
        return 0
    n = 0
    for a in mol.GetAtoms():
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        if sum(1 for x in a.GetNeighbors() if x.GetAtomicNum() > 1) >= 2:
            n += 1
    return n


def cluster_poses(cache: dict, spec: str):
    """Reproduce Stage 3's maxclust schedule so cluster ids line up with it."""
    start, max_clusters, step, floor = (float(x) for x in spec.split(":"))
    max_clusters = int(max_clusters)
    models = list(cache.keys())
    thr = start
    clusters = butina_cluster(models, cache, thr, mode="similarity")
    while len(clusters) > max_clusters and thr > floor:
        thr = max(thr - step, floor)
        clusters = butina_cluster(models, cache, thr, mode="similarity")
    while len(clusters) < 5 and thr < 0.99:
        thr = min(thr + 0.05, 0.90) if thr < 0.90 else min(thr + 0.01, 0.99)
        clusters = butina_cluster(models, cache, thr, mode="similarity")
    return clusters, thr


def receptor_deviation(pdb_by_model: dict, n_res: int, ref_model: str):
    """Interface-frame displacement of protomer 1 after fitting protomer 0.

    Interface positions only: across the full 255 residues the flexible
    non-interface loops contribute ~1.7 A of noise, three times the signal the
    interface itself carries (pool medians 1.663 A vs 0.643 A).
    """
    coords, names = [], []
    for m, p in pdb_by_model.items():
        r = read_dimer_ca_any(p, n_res)
        if r is None:
            continue
        coords.append(r[0])
        names.append(m)
    if not coords:
        return {}, None
    X = np.array(coords, dtype=np.float64)

    hits = np.zeros(n_res)
    step = max(1, len(X) // 60)
    used = 0
    for i in range(0, len(X), step):
        d = np.linalg.norm(X[i, 0][:, None, :] - X[i, 1][None, :, :], axis=-1)
        hits += (d < 12).any(1)
        used += 1
    mask = hits > 0.5 * used

    ridx = names.index(ref_model) if ref_model in names else 0
    ref0 = X[ridx, 0][mask]
    Y = np.empty((len(X), int(mask.sum()), 3))
    for i in range(len(X)):
        R, cP, cQ = kabsch(X[i, 0][mask], ref0)
        Y[i] = apply_rt(X[i, 1][mask], R, cP, cQ)
    dev = np.sqrt(((Y - Y[ridx]) ** 2).sum(-1).mean(-1))
    return dict(zip(names, dev.tolist())), int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--maxclust", default="0.80:60:0.05:0.30")
    ap.add_argument("--n-res", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--slate1-sucos-drop", type=float, default=0.06,
                    help="how far below the best SuCOS a cluster-1 member may fall "
                         "and still be eligible for MODEL 1-5 diversity picking")
    ap.add_argument("--slate2-priority", choices=("pose", "receptor"), default="pose",
                    help="what MODEL 6-10 leads with. 'pose' fills from other ligand-pose "
                         "clusters; 'receptor' puts the most RECEPTOR-dissimilar models "
                         "first, which is what a target asking for a second protein "
                         "conformation actually wants.")
    args = ap.parse_args()

    td = Path(args.target_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = json.loads((td / "pairwise_sucos_cache.json").read_text())
    # takes the TARGET dir and appends ranking_sucos/ itself
    ranking = parse_ranking_sucos(str(td))
    # Stage 3 requires a representative to clear the sp3-geometry check ON TOP of
    # PoseBusters, because PB's `bond_angles` test is really a 1-3 DISTANCE test:
    # when a bond collapses the 1-3 distance shrinks with it, so an sp3 centre can
    # open to 180 deg and still pass. Reading only the `_pb=` flag off the filename
    # would let this tool propose poses Stage 3 has already disqualified.
    geom_ok = build_geom_ok(str(td), ranking)
    n_geom_bad = sum(1 for v in geom_ok.values() if not v)
    print(f"  pairwise cache: {len(cache)} models   ranking entries: {len(ranking)}")
    n_sp3 = _count_checkable_sp3(next(iter(sdf_for_probe(td)), None))
    note = ("  (ligand has no sp3 centre with 2+ heavy neighbours — "
            "this filter cannot fire for this target)") if n_sp3 == 0 else ""
    print(f"  sp3-geometry: {n_geom_bad}/{len(geom_ok)} rejected, "
          f"{n_sp3} checkable sp3 centre(s){note}")

    clusters, thr = cluster_poses(cache, args.maxclust)
    print(f"  {len(clusters)} ligand-pose clusters at SuCOS threshold {thr:.2f}")

    # size first, then best in-cluster SuCOS — same ordering Stage 3 uses
    def best_sucos(c):
        return max((ranking[m][0] for m in c["members"] if m in ranking), default=-1.0)

    clusters.sort(key=lambda c: (len(c["members"]), best_sucos(c)), reverse=True)

    sdf_by_model, pdb_by_model = {}, {}
    for f in glob.glob(str(td / "ranking_sucos" / "*.sdf")):
        m = RANK_RE.match(os.path.basename(f))
        if not m:
            continue
        sdf_by_model[m.group(1)] = f
        pdb = f[: -len(f"_pb={m.group(5)}.sdf")] + ".pdb"
        if os.path.isfile(pdb):
            pdb_by_model[m.group(1)] = pdb

    # reference = best PB-valid member of the largest cluster == prospective MODEL 1
    top = clusters[0]["members"]
    cand = [(ranking[m][0], m) for m in top if m in ranking and ranking[m][1]]
    if not cand:
        cand = [(ranking[m][0], m) for m in top if m in ranking]
    ref_model = max(cand)[1]
    print(f"  reference (prospective MODEL 1): {ref_model}")

    dev, n_iface = receptor_deviation(pdb_by_model, args.n_res, ref_model)
    print(f"  receptor deviation computed for {len(dev)} models "
          f"over {n_iface} interface positions")

    rows = []
    for ci, c in enumerate(clusters, 1):
        for m in c["members"]:
            su, pb = ranking.get(m, (float("nan"), False))
            rows.append({
                "model": m, "method": m.split("_model")[0],
                "pose_cluster": ci, "cluster_size": len(c["members"]),
                "sucos": round(su, 4) if su == su else "",
                "pb_valid": pb,
                "sp3_ok": geom_ok.get(m, True),
                "valid": bool(pb and geom_ok.get(m, True)),
                "receptor_dev_A": round(dev.get(m, float("nan")), 3) if m in dev else "",
                "sdf": sdf_by_model.get(m, ""), "pdb": pdb_by_model.get(m, ""),
            })
    rows.sort(key=lambda r: (r["pose_cluster"], -(r["sucos"] or 0)))
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n  {'clu':>4s} {'size':>5s} {'PBok':>5s} {'bestSuCOS':>10s} "
          f"{'recept.dev min/med/max':>24s}  methods")
    for ci, c in enumerate(clusters[:15], 1):
        mem = [r for r in rows if r["pose_cluster"] == ci]
        d = [r["receptor_dev_A"] for r in mem if r["receptor_dev_A"] != ""]
        pbok = sum(1 for r in mem if r["valid"])
        bs = max((r["sucos"] for r in mem if r["sucos"] != ""), default=0)
        ds = (f"{min(d):5.2f}/{np.median(d):5.2f}/{max(d):5.2f}" if d else "  -  ")
        print(f"  {ci:4d} {len(mem):5d} {pbok:5d} {bs:10.3f} {ds:>24s}  "
              f"{dict(Counter(r['method'] for r in mem))}")

    if args.propose:
        print("\n  === proposed slates ===")
        print(f"  MODEL 1-5  (cluster 1, dominant receptor conformation)")
        pool1 = sorted((r for r in rows if r["pose_cluster"] == 1 and r["valid"]),
                       key=lambda r: -(r["sucos"] or 0))
        # Taking the top five by SuCOS wastes slots: the highest-scoring members of
        # a cluster are by construction the ones most similar to each other. On
        # T2451 that gave 3 near-identical SeedFold models (mutual ligand RMSD
        # 0.26-0.39 A) plus 2 near-identical Protenix models -- 5 slots carrying 2
        # distinct poses. Cluster 1 actually spans 4.1 A internally, because SuCOS
        # scores shape overlap and a pose can flip in place while still overlapping
        # well. So seed with the best-scoring member and then repeatedly take the
        # one FURTHEST from everything already chosen, keeping only members whose
        # SuCOS is within `--slate1-sucos-drop` of the best.
        if pool1:
            cut = (pool1[0]["sucos"] or 0) - args.slate1_sucos_drop
            elig = [r for r in pool1 if (r["sucos"] or 0) >= cut]
            c1 = [pool1[0]]
            while len(c1) < 5 and len(c1) < len(elig):
                far, fard = None, -1.0
                for r in elig:
                    if r in c1:
                        continue
                    d = min(cache.get(r["model"], {}).get(q["model"], 1.0) for q in c1)
                    # cache holds SuCOS similarity; least similar == most distant
                    if -d > fard:
                        far, fard = r, -d
                if far is None:
                    break
                c1.append(far)
        else:
            c1 = []
        for i, r in enumerate(c1[:5], 1):
            print(f"    MODEL {i}: {r['model']:22s} sucos {r['sucos']} "
                  f"recept.dev {r['receptor_dev_A']}")
        print(f"  MODEL 6-10 (priority={args.slate2_priority})")
        slate2, used_clu = [], set()
        if args.slate2_priority == "receptor":
            # Lead with the largest receptor deviations available anywhere in the
            # pool, then fill the rest from other ligand-pose clusters.
            far = sorted((r for r in rows if r["valid"] and r["receptor_dev_A"] != ""
                          and r["model"] not in {x["model"] for x in c1[:5]}),
                         key=lambda r: -r["receptor_dev_A"])
            for r in far[:2]:
                slate2.append((int(r["pose_cluster"]), r, f"receptor {r['receptor_dev_A']}A"))
                used_clu.add(int(r["pose_cluster"]))
        for ci in range(2, len(clusters) + 1):
            mem = [r for r in rows if r["pose_cluster"] == ci and r["valid"]
                   and r["receptor_dev_A"] != ""]
            if not mem:
                continue
            mem.sort(key=lambda r: r["receptor_dev_A"])
            want_far = len(slate2) % 2 == 1        # alternate similar / dissimilar
            pick = mem[-1] if want_far else mem[0]
            slate2.append((ci, pick, "far" if want_far else "near"))
            used_clu.add(ci)
            if len(slate2) == 5:
                break
        for i, (ci, r, kind) in enumerate(slate2, 6):
            print(f"    MODEL {i}: {r['model']:22s} cluster {ci:3d} "
                  f"sucos {r['sucos']} recept.dev {r['receptor_dev_A']} ({kind})")
        if len(slate2) < 5:
            print(f"    ! only {len(slate2)} usable clusters beyond #1 — "
                  f"fall back to same-cluster members with the largest receptor spread")

    print(f"\n  wrote {out / 'candidates.csv'}")


if __name__ == "__main__":
    main()
