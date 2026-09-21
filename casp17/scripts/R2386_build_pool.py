#!/usr/bin/env python
"""R2386 step 1-3: build the solvent candidate pool and score it with two
independent evidence streams.

Design notes
------------
Everything is expressed in the coordinate frame of PDB 9C6I, the 2.56 A cryo-EM
structure of this exact 417 nt construct (100% sequence identity, 393/417 nt
modelled). Every donor structure and every predicted model is superposed onto
9C6I's RNA before its solvent is harvested.

Two evidence streams, kept separate on purpose because they fail differently:

  template   36 independent experimental structures of the same intron
             (92.3-98.6% identity, superposition RMSD <= 3.5 A). Blind to the
             target in the sense that none of them IS the target map. Misses
             sites that no crystal ever resolved, and covers nothing in the
             C-terminal 24 nt.

  cofolding  400 models = 4 methods x 4 ion-stoichiometry points x 25 models.
             Independent of crystallography, so it finds sites the template
             stream systematically cannot -- measured earlier: 5 of 9C6I's 21
             experimental Mg are found ONLY by co-folding, 3 of them missed by
             templates by 9.7-13.2 A.

Per-ion physical filtering (applied to the co-folding stream only, before
pooling): Protenix places 15-18% of its Mg at physically impossible Mg-Mg
distances at the higher stoichiometry points, versus 1-3% for AF3 and Boltz-2.
Rather than down-weighting whole runs, drop the individual offending ions --
that keeps the ions Protenix placed correctly. Two Mg2+ cannot approach closer
than ~3.5 A even sharing a bridging ligand; <3.0 A is impossible.

9C6I's 21 modelled Mg are held out as a blind test set for AF3, Boltz-2 and
protenix_default (training cutoffs predate the 2024-06-07 deposition), but NOT
for protenix_applied (2025-06-30 cutoff). Contaminated numbers are labelled.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import R2386_paths as paths                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REF_CIF = str(paths.ref_cif())
CIF_CACHE = str(paths.cif_cache())

# 36 donors: identity >= 92.3% to R2386 and superposition RMSD <= 3.5 A.
# 9C6I is the reference frame and the held-out test set; 9C6J is its twin
# deposition (same 21 Mg) so including it would be circular.
DONORS = [
    "3G78", "5J01", "5J02", "4FAQ", "4FAU", "4FAW", "4E8Q", "4E8N", "4E8P", "4E8R",
    "8RUM", "8RUJ", "8RUI", "8RUL", "8RUN", "4FAR", "8OLY", "8OLZ", "8OLS", "8OLV",
    "4FAX", "4FB0", "6T3K", "6T3N", "6T3R", "6T3S", "4E8K", "4E8T", "4E8V", "8RUH",
    "4DS6", "8OM0", "8RUK", "8OLW", "9QTJ", "4E8M",
]

METHODS = {
    "af3":         os.environ.get("AF3_RESULTS_DIR", "/bml/Lyuwei/alphafold_results") + "/casp17_R_R2386/{pt}/seed-*/*model.cif",
    "boltz2":      str(ROOT / "outputs/boltz2/casp17_R2386/boltz_results_R2386_{pt}_input/predictions/*/*_model_*.cif"),
    "ptx_default": str(ROOT / "outputs/protenix/casp17_R2386_default/R2386_{pt}/seed_*/predictions/*_sample_*.cif"),
    "ptx_applied": str(ROOT / "outputs/protenix/casp17_R2386_applied/R2386_{pt}/seed_*/predictions/*_sample_*.cif"),
}
POINTS = ["S1", "S2", "S3", "S4"]
BLIND = {"af3", "boltz2", "ptx_default"}          # training cutoff predates 9C6I
IONS = ("MG", "K", "NA")
MIN_ION_ION = 3.0                                  # A; below this is physically impossible


def cif_path(pdb: str) -> str | None:
    for c in (pdb.upper(), pdb.lower()):
        f = os.path.join(CIF_CACHE, f"{c}.cif")
        if os.path.exists(f):
            return f
    return None


def longest_rna(st):
    best = None
    for ch in st[0]:
        pl = ch.get_polymer()
        if pl.check_polymer_type() == gemmi.PolymerType.Rna and (best is None or len(pl) > len(best)):
            best = pl
    return best


def load(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    return st


def harvest(st, names):
    out = []
    for ch in st[0]:
        for r in ch:
            if r.name in names:
                a = r[0]
                out.append((r.name, np.array([a.pos.x, a.pos.y, a.pos.z])))
    return out


# --------------------------------------------------------------------------
# local re-alignment
#
# One rigid fit over 417 residues is a compromise: it minimises RMSD everywhere
# and is therefore wrong by 1-3 A in most single places. Every solvent atom it
# carries inherits that local error, ions no less than water -- a magnesium
# perfectly coordinated in its own structure arrives here 2 A from anything it
# could bind, and then looks like a modelling mistake that is really a transfer
# mistake.
#
# The fix is to stop asking one transform to be right everywhere. For each
# solvent atom, refit using only the residues that surround it, and move that
# atom alone. Measured in R2386_transfer_error.py, in each donor's own frame
# against its own untransferred solvent -- the only setting where the answer key
# carries no transfer error of its own -- this raises recall@1.0 for water from
# 21.6% to 23.9% and for Mg from 45.1% to 54.9% on 3G78, and from 19.4/35.7% to
# 23.6/45.2% on 5J01, where the global fit is worse (3.0 A) and there is more to
# repair. 12 A is the best or joint-best radius on four of five measures.
# --------------------------------------------------------------------------
BB_ATOMS = ("P", "C4'", "C1'", "O5'", "C3'")
LOCAL_RADIUS = 12.0
LOCAL_MIN_PAIRS = 12


def backbone_map(st):
    """{atom name: (positions, kd-tree)} over the RNA backbone."""
    from collections import defaultdict
    by = defaultdict(list)
    for ch in st[0]:
        for r in ch:
            if r.name in ("A", "C", "G", "U"):
                for a in r:
                    if a.name in BB_ATOMS:
                        by[a.name].append([a.pos.x, a.pos.y, a.pos.z])
    return {n: (np.array(v), cKDTree(np.array(v))) for n, v in by.items() if v}


def local_pairs(st_moved, ref_bb):
    """Matched (moving, fixed) backbone pairs after the global fit."""
    mv, fx = [], []
    for ch in st_moved[0]:
        for r in ch:
            if r.name not in ("A", "C", "G", "U"):
                continue
            for a in r:
                if a.name not in ref_bb:
                    continue
                P, t = ref_bb[a.name]
                p = [a.pos.x, a.pos.y, a.pos.z]
                d, j = t.query(p)
                if d < 4.0:
                    mv.append(p)
                    fx.append(P[j])
    return np.array(mv), np.array(fx)


def local_refine(points, mv, fx, radius=LOCAL_RADIUS):
    """Re-place each point using only the backbone around it."""
    if len(mv) < 50:
        return points
    tfx = cKDTree(fx)
    out = []
    for x in points:
        idx = tfx.query_ball_point(x, radius)
        if len(idx) < LOCAL_MIN_PAIRS:
            out.append(x)                      # under-determined: keep global
            continue
        r = gemmi.superpose_positions([gemmi.Position(*fx[j]) for j in idx],
                                      [gemmi.Position(*mv[j]) for j in idx])
        q = r.transform.apply(gemmi.Position(*x))
        out.append(np.array([q.x, q.y, q.z]))
    return out


def physical_filter(items):
    """Drop ions closer than MIN_ION_ION to another ion in the SAME model."""
    if len(items) < 2:
        return items, 0
    P = np.array([p for _, p in items])
    d = np.linalg.norm(P[:, None] - P[None, :], axis=2)
    np.fill_diagonal(d, 9e9)
    keep = d.min(1) >= MIN_ION_ION
    return [it for it, k in zip(items, keep) if k], int((~keep).sum())


def cluster(X, labels, cut):
    """Greedy single-pass clustering; returns centroids and member index lists."""
    cents, members = [], []
    for i in range(len(X)):
        placed = False
        for ci, c in enumerate(cents):
            if np.linalg.norm(X[i] - c) <= cut:
                members[ci].append(i)
                cents[ci] = X[members[ci]].mean(0)
                placed = True
                break
        if not placed:
            cents.append(X[i].copy())
            members.append([i])
    return np.array(cents), members


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/R2386_pool")
    ap.add_argument("--cut-ion", type=float, default=1.5)
    ap.add_argument("--cut-wat", type=float, default=1.0)
    ap.add_argument("--local-radius", type=float, default=LOCAL_RADIUS,
                    help="Re-fit the superposition locally around each solvent "
                         "atom using the backbone within this radius, instead of "
                         "carrying it on the one global fit. 0 disables.")
    ap.add_argument("--include-ref", action="store_true",
                    help="Add 9C6I and its twin 9C6J to the donor pool. Correct for "
                         "building the SUBMISSION -- 9C6I is experimental solvent for "
                         "this exact construct and the organizers explicitly offer it "
                         "as a reference. Must stay OFF when measuring blind recall, "
                         "since 9C6I's 21 Mg are the held-out test set.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    donors = list(DONORS) + (["9C6I", "9C6J"] if args.include_ref else [])

    ref = load(REF_CIF)
    rpoly = longest_rna(ref)
    ref_bb = backbone_map(ref)
    local_shift = []
    ref_mg = np.array([p for n, p in harvest(ref, ("MG",))])
    print(f"reference 9C6I: {len(rpoly)} nt modelled, {len(ref_mg)} Mg (held-out test set)\n")

    rows = []          # (stream, source, point, species, xyz)

    # ---------------- template stream ----------------
    print("=== template stream ===")
    t_drop = 0
    for pdb in donors:
        f = cif_path(pdb)
        if not f:
            print(f"  {pdb}: cif missing, skipped")
            continue
        st = load(f)
        pl = longest_rna(st)
        if pl is None:
            continue
        try:
            sup = gemmi.calculate_superposition(rpoly, pl, gemmi.PolymerType.Rna, gemmi.SupSelect.CaP)
        except Exception:
            continue
        if sup.rmsd > 3.5:
            print(f"  {pdb}: superposition RMSD {sup.rmsd:.2f} > 3.5, skipped")
            continue
        st[0].transform_pos_and_adp(sup.transform)
        items = harvest(st, IONS + ("HOH",))
        if args.local_radius > 0 and items:
            mv, fx = local_pairs(st, ref_bb)
            moved = local_refine([p for _, p in items], mv, fx, args.local_radius)
            local_shift += [float(np.linalg.norm(a - b))
                            for (_, a), b in zip(items, moved)]
            items = [(nm, q) for (nm, _), q in zip(items, moved)]
        for nm, p in items:
            rows.append(("template", pdb, "-", nm, p))
    n_t = sum(1 for r in rows if r[0] == "template")
    print(f"  harvested {n_t} solvent positions from {len({r[1] for r in rows})} donors")
    if local_shift:
        a = np.array(local_shift)
        print(f"  local re-alignment (r={args.local_radius:.0f} A) moved them "
              f"{a.mean():.2f} A on average, median {np.median(a):.2f}, "
              f"{100*(a > 1.0).mean():.0f}% by more than 1 A")

    # ---------------- co-folding stream ----------------
    print("\n=== co-folding stream (per-ion physical filter, min ion-ion "
          f"{MIN_ION_ION} A) ===")
    print(f"  {'method':13s}{'pt':4s}{'models':>7s}{'ions kept':>11s}{'dropped':>9s}{'drop%':>7s}")
    for meth, pat in METHODS.items():
        for pt in POINTS:
            files = sorted(glob.glob(pat.format(pt=pt)))
            if not files:
                continue
            kept = dropped = 0
            for f in files:
                st = load(f)
                pl = longest_rna(st)
                if pl is None:
                    continue
                try:
                    sup = gemmi.calculate_superposition(rpoly, pl, gemmi.PolymerType.Rna, gemmi.SupSelect.CaP)
                except Exception:
                    continue
                st[0].transform_pos_and_adp(sup.transform)
                items = harvest(st, IONS)
                if args.local_radius > 0 and items:
                    mv, fx = local_pairs(st, ref_bb)
                    moved = local_refine([q for _, q in items], mv, fx,
                                         args.local_radius)
                    items = [(nm, q) for (nm, _), q in zip(items, moved)]
                items, nd = physical_filter(items)
                dropped += nd
                kept += len(items)
                for nm, p in items:
                    rows.append(("cofolding", meth, pt, nm, p))
            tot = kept + dropped
            print(f"  {meth:13s}{pt:4s}{len(files):7d}{kept:11d}{dropped:9d}"
                  f"{100*dropped/max(tot,1):6.0f}%")

    # ---------------- cluster + score ----------------
    print("\n=== clustering into distinct sites ===")
    sites = {}
    for species, cut in (("MG", args.cut_ion), ("K", args.cut_ion),
                         ("NA", args.cut_ion), ("HOH", args.cut_wat)):
        sub = [r for r in rows if r[3] == species]
        if not sub:
            continue
        X = np.array([r[4] for r in sub])
        cents, members = cluster(X, None, cut)
        recs = []
        for ci, mem in enumerate(members):
            tmpl_src = {sub[j][1] for j in mem if sub[j][0] == "template"}
            cof = [(sub[j][1], sub[j][2]) for j in mem if sub[j][0] == "cofolding"]
            cof_methods = {m for m, _ in cof}
            cof_blind = {m for m in cof_methods if m in BLIND}
            pts = [p for _, p in cof]
            recs.append(dict(
                xyz=cents[ci].tolist(),
                n_raw=len(mem),
                t_support=len(tmpl_src),
                t_sources=sorted(tmpl_src),
                c_hits=len(cof),
                c_methods=sorted(cof_methods),
                c_blind_methods=sorted(cof_blind),
                first_point=min(pts) if pts else None,
            ))
        sites[species] = recs
        print(f"  {species:4s}: {len(sub):6d} raw -> {len(recs):5d} distinct sites (cut {cut} A)")

    with open(args.out / "sites.json", "w") as fh:
        json.dump(sites, fh)
    np.save(args.out / "ref_mg.npy", ref_mg)
    print(f"\nwrote {args.out/'sites.json'}")


if __name__ == "__main__":
    main()
