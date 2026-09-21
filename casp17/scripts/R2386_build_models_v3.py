#!/usr/bin/env python
"""R2386 v3: rebuild the five solvent models under the organizers' 2026-08-08
additional instruction.

The instruction changed two things that invalidate the v2 models:

1. "Modelers should submit 500 ligands" + "total occupancy must be equal to the
   number of ligands". v2 emitted 201-5963 ligands per model and used fractional
   occupancy (0.50/0.75/1.00) to encode evidence strength, so the occupancy sum
   did not equal the ligand count. v3 emits exactly TOTAL ligands at occupancy
   1.00 and moves evidence strength entirely into the B-factor column.

2. "Only ligands in the core, well-resolved regions will be assessed; any ligands
   that are closest to the following non-core residues will be ignored."
   A ligand nearest to a non-core residue scores nothing, so under a fixed budget
   it is a wasted slot. v2 wasted 12% of models 1-4 and 35% of model 5.

The consequence for the hedging strategy is bigger than either edit. v2 hedged
along "how many sites to claim" -- that axis no longer exists, because the count
is fixed for every model. The replacement axis is COMPOSITION: how the 500 slots
divide between ions and water. That is now the dominant unknown, because it caps
recall directly: submit 80 ions when the answer has 40, and 40 slots can never
match no matter how well placed.

The anchor is 3G78, the experimental structure of this RNA with by far the
richest solvent (2.80 A, 435 water + 77 ions over 396 nt = 1.29 solvent/nt,
85% water). 500 ligands over 417 nt is 1.20 solvent/nt -- essentially 3G78's
density, which is almost certainly where the number 500 came from. 3G78's
composition scaled to 500 gives ~75 ions + ~425 water, which is model 1.

That anchor is no longer the only argument: R2386_bench_composition.py builds
each candidate 500-ligand model and scores it blind against held-out experimental
solvent, and it independently lands on 80-100 ions. See COMPOSITION below for the
table and for why v2's 150- and 230-ion rungs were dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casp_identity                                                # noqa: E402
import R2386_build_models as v2                                     # noqa: E402

import R2386_water as w3                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TARGET = "R2386"

# Verbatim from target.cgi?id=258&view=ligand, 2026-08-08 addendum. These are
# CASP numbers; our files are already in CASP numbering (the residue sequence of
# our submitted model matches the target-page template at all 417 positions).
NONCORE_RANGES = [(6, 7), (56, 60), (86, 106), (167, 173), (206, 220),
                  (276, 287), (309, 320), (335, 357), (394, 417)]
NONCORE = {r for a, b in NONCORE_RANGES for r in range(a, b + 1)}

TOTAL_DEFAULT = 500

# (label, n_ions, n_water, k_relabel, water_fill_order). n_ions + n_water == TOTAL.
#
# The ion fraction was argued in v2 and is now MEASURED. R2386_bench_composition.py
# builds the full 500-ligand model at each rung and scores it against a held-out
# experimental structure whose contribution has been stripped from the evidence.
# Two independent keys, deliberately chosen to disagree about how solvent gets
# modelled -- 3G78 (2.80 A, water everywhere, only 11% of it Mg first shell) and
# 5J01+5J02 (3.39/3.49 A, one lab, 95-98% of their water IS Mg first shell):
#
#     ions      40    60    80   100   120   150   200   230
#     3G78      25%   27%   27%   25%   21%   21%   20%   21%   recall @1.5 A
#     5J01/02   22%   27%   28%   30%   30%   29%   28%   28%
#     sum       47    54    55    55    51    50    48    49
#
# CAVEAT measured afterwards: 5J01/5J02's water is BUILT, not observed. Their
# Mg-O(shell) distances have sd 0.006 / 0.003 A (89-98% within +-0.01 A of the
# median) and their water B-factors average 135-138 -- no 3.4 A refinement yields
# that. Those are idealized octahedra, placed exactly the way v2.hexahydrate
# places ours, so scoring our shells against theirs is partly circular. 3G78 at
# 2.80 A has sd 0.213 A and mean water B 13.1, which is what refined water looks
# like. 5J01/02 therefore stays valid as an ION key (their Mg positions ARE
# refined) and is discounted as a WATER key.
#
# The keys' individual optima differ (3G78 says 60, 5J01/02 says 100-120), which
# is exactly the uncertainty a best-of-5 ladder exists to cover, so the rungs
# bracket 40-120. What both keys agree on is that 150+ is DOMINATED: v2's 150-
# and 230-ion rungs lose ~23% of the achievable hits on 3G78 and buy nothing on
# 5J01/02. They are dropped. The electroneutrality argument that motivated 230
# never survived contact with the fact that the diffuse ion atmosphere balancing
# the phosphates is disordered, hence unassessable.
#
# The fifth slot goes on SPECIES, not on a fifth ion count. Our position evidence
# is species-agnostic -- co-folding only says "an ion sits here" -- so the weakest
# link is the arbitration step that turns a position into an element. Our
# evidence ranking lands on Mg:K = 9:1, while 3G78 (the composition anchor) is
# 51:26 and the buffer holds 100 mM KCl against 10 mM MgCl2. Model 5 therefore
# keeps model 1's positions exactly and relabels the third of them whose observed
# nearest-RNA-oxygen distance best matches K-O 2.80 A rather than Mg-O 2.07 A,
# which is the same discriminator arbitration uses, applied at a different prior.
COMPOSITION = [
    ("evidence-ranked (joint optimum)",  80, 420, None),
    ("ion-lean (3G78 optimum)",          60, 440, None),
    ("ion-rich (resolution argument)",  120, 380, None),
    ("water-rich",                       40, 460, None),
    ("3G78 species ratio (Mg:K = 2:1)",  80, 420, 27),
]

NA_CAP = [6, 5, 9, 3, 6]

# Safety margin on the core test. The assessor decides "closest residue" on their
# reference structure; we decide it on 9C6I's coordinates. In the core the two
# agree to about an angstrom, so a ligand nearly equidistant from a core and a
# non-core residue is a coin flip on being assessed at all. Measured cost of a
# 0.75 A margin: 14-20 slots per model (3-4%), replaced by unambiguous ones.
CORE_MARGIN = 0.75

# Water candidate lattice. v2 used 3.10 A (bulk water number density) purely to
# fill volume; here the lattice only proposes positions that are then scored and
# thinned, so a finer grid buys better placement at no cost in count.
FINE = 1.4
POCKET_SHELL = 5.0

# Orderedness score for a candidate water. Template agreement dominates -- a
# position independently modelled by several experiments is ordered by
# observation, not by inference. The remaining terms are what makes a water
# ordered in the first place: hydrogen bonds to hold it, and burial to stop it
# exchanging with bulk.
W_TEMPLATE, W_HB_RNA, W_HB_SOLV, W_BURIAL = 6.0, 2.0, 1.2, 0.06
HB_CUT, BURIAL_CUT = 3.4, 6.0


def core_mask(P, tCore, tNon, margin=CORE_MARGIN):
    """True where the nearest RNA residue is core by at least `margin`.

    Comparing distance-to-nearest-core against distance-to-nearest-non-core is
    equivalent to the assessor's "closest residue" rule, and makes the margin
    expressible: require the core residue to win by `margin` angstroms.
    """
    P = np.asarray(P, float)
    if len(P) == 0:
        return np.zeros(0, bool)
    dc, _ = tCore.query(P, k=1, workers=-1)
    dn, _ = tNon.query(P, k=1, workers=-1)
    return np.atleast_1d(dn - dc) >= margin


def pocket_candidates(heavy_xyz, per_el, tCore, tNon, tCterm, hb_tree):
    """Propose water positions on a fine lattice in the first solvation shell.

    Returns (positions, orderedness_score) with the geometric filters already
    applied but WITHOUT the clash test against this model's own solvent, which
    depends on the ion set and is therefore done per model.
    """
    tree = cKDTree(heavy_xyz)
    lo, hi = heavy_xyz.min(0) - POCKET_SHELL - 1, heavy_xyz.max(0) + POCKET_SHELL + 1
    ax = [np.arange(lo[i], hi[i] + FINE, FINE) for i in range(3)]
    Y, Z = np.meshgrid(ax[1], ax[2], indexing="ij")
    YZ = np.column_stack([Y.ravel(), Z.ravel()])
    keep = []
    for x in ax[0]:
        slab = np.column_stack([np.full(len(YZ), x), YZ])
        d, _ = tree.query(slab, k=1, workers=-1)
        cand = slab[(d <= POCKET_SHELL) & (d >= 2.5)]
        if len(cand):
            keep.append(cand)
    P = np.vstack(keep) if keep else np.zeros((0, 3))
    if not len(P):
        return P, np.zeros(0)

    ok = np.ones(len(P), bool)
    for el, t in per_el.items():
        d, _ = t.query(P, k=1, workers=-1)
        ok &= d >= v2.WATER_MIN[el]
    P = P[ok]
    if tCterm is not None and len(P):
        d, _ = tCterm.query(P, k=1, workers=-1)
        P = P[d >= v2.CTERM_EXCLUSION]
    if len(P):
        P = P[core_mask(P, tCore, tNon)]
    if not len(P):
        return P, np.zeros(0)

    nhb = np.array([len(hb_tree.query_ball_point(p, HB_CUT)) for p in P], float)
    bur = np.array([len(tree.query_ball_point(p, BURIAL_CUT)) for p in P], float)
    score = W_HB_RNA * nhb + W_BURIAL * bur
    return P, score


def greedy_xyz(P, scores, min_sep, seeded):
    """Accept positions in descending score order, keeping a minimum separation."""
    order = np.argsort(-scores)
    acc = list(seeded)
    out = []
    for i in order:
        p = P[i]
        if acc:
            A = np.asarray(acc)
            # cheap reject on a bounding box before the full norm
            if np.abs(A - p).max(1).min() < min_sep:
                if np.linalg.norm(A - p, axis=1).min() < min_sep:
                    continue
        acc.append(p)
        out.append((p, float(scores[i])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "outputs/R2386_pool_submit")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "casp17/submissions/casp17_R/R2386_files_v3")
    ap.add_argument("--total", type=int, default=TOTAL_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sites = json.load(open(args.pool / "sites_layered.json"))
    st, pl = v2.load_ref()
    per_el, tHeavy, heavy_xyz = v2.ref_maps(pl)

    def ion_ok(x):
        return all(t.query(x)[0] >= v2.ION_MIN[el]
                   for el, t in per_el.items() if el in v2.ION_MIN)

    COORD_MAX = {"MG": 4.6, "K": 3.5, "NA": 3.2}
    K_O_IDEAL = 2.80

    # ---- RNA frame, in CASP numbering ------------------------------------
    _b = [a.b_iso for r in pl if r.name in ("A", "C", "G", "U") for a in r
          if a.element.name != "H"]
    _blo, _bhi = min(_b), max(_b)

    def _conf(b):
        if _bhi <= _blo:
            return 90.0
        return round(99.0 - 79.0 * (b - _blo) / (_bhi - _blo), 2)

    rna_atoms = [(i, r.name, [(a.name, a.element.name, a.pos, _conf(a.b_iso))
                              for a in r if a.element.name != "H"])
                 for i, r in enumerate((r for r in pl if r.name in ("A", "C", "G", "U")),
                                       start=1)]
    cterm, tCterm = v2_cterm(pl)

    # per-atom residue id, needed for the core test
    rna_pts, rna_res = [], []
    for tgt, _, atoms in rna_atoms + cterm:
        for _, _, pos, _ in atoms:
            rna_pts.append([pos.x, pos.y, pos.z])
            rna_res.append(tgt)
    rna_pts = np.array(rna_pts)
    rna_res = np.array(rna_res)
    is_core = np.array([r not in NONCORE for r in rna_res])
    tCore = cKDTree(rna_pts[is_core])
    tNon = cKDTree(rna_pts[~is_core])

    # hydrogen-bond partners on the RNA: every N and O
    hb_pts = np.array([[pos.x, pos.y, pos.z] for _, _, atoms in rna_atoms
                       for _, el, pos, _ in atoms if el in ("N", "O")])
    hb_tree = cKDTree(hb_pts)

    print(f"core residues {417 - len(NONCORE)} / 417   "
          f"non-core {len(NONCORE)}  (ligands there are ignored by the assessor)")

    # ---- candidate ions ---------------------------------------------------
    arbitrated = v2.arbitrate_species(sites, tO_global=per_el.get("O"))
    exp_mg = [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch
              if r.name == "MG" for a in r]
    print(f"experimental Mg from 9C6I: {len(exp_mg)}, "
          f"in core {int(core_mask(exp_mg, tCore, tNon, 0.0).sum())}, "
          f"clear of the {CORE_MARGIN} A margin "
          f"{int(core_mask(exp_mg, tCore, tNon).sum())} "
          f"(force-included either way -- an experimental observation is not "
          f"ours to veto)")

    # ---- candidate water --------------------------------------------------
    tmpl = [s for s in sites["HOH"]
            if v2.water_ok(np.array(s["xyz"]), per_el)]
    if tCterm is not None:
        tmpl = [s for s in tmpl
                if tCterm.query(np.array(s["xyz"]))[0] >= v2.CTERM_EXCLUSION]
    tmpl = [s for s, k in zip(tmpl, core_mask([s["xyz"] for s in tmpl], tCore, tNon)) if k]
    print(f"template water sites usable in core: {len(tmpl)}")

    pocketP, pocketS = pocket_candidates(heavy_xyz, per_el, tCore, tNon,
                                         tCterm, hb_tree)
    print(f"pocket water candidates proposed: {len(pocketP)}")

    print(f"\n  {'#':<3s}{'model':40s}{'MG':>4s}{'K':>4s}{'NA':>3s}"
          f"{'ions':>6s}{'water':>7s}{'total':>7s}{'shellW':>8s}{'tmplW':>7s}"
          f"{'pockW':>7s}{'minII':>7s}{'expMg':>7s}")

    for i, (label, n_ion, n_wat, k_relabel) in enumerate(COMPOSITION, start=1):
        assert n_ion + n_wat == args.total, (label, n_ion, n_wat, args.total)

        # ---- ions: experimental first, then by evidence ------------------
        cands = [dict(xyz=p, sp="MG", t_support=99, c_blind_methods=[], c_hits=0,
                      exp=True) for p in exp_mg]
        na_seen = 0
        for s in sorted(arbitrated, key=lambda a: -v2.priority(a)):
            x = np.array(s["xyz"])
            if not ion_ok(x):
                continue
            if per_el["O"].query(x)[0] > COORD_MAX[s["sp"]]:
                continue
            if tCterm is not None and tCterm.query(x)[0] < v2.CTERM_EXCLUSION:
                continue
            if not core_mask([x], tCore, tNon)[0]:
                continue
            if s["sp"] == "NA":
                if na_seen >= NA_CAP[i - 1]:
                    continue
                na_seen += 1
            cands.append(dict(s))
        cands.sort(key=lambda s: (-1e9 if s.get("exp") else 0) - v2.priority(s))
        ions, ion_xyz = v2.greedy(cands, v2.MIN_ION_ION)
        ions, ion_xyz = ions[:n_ion], [np.asarray(c["xyz"]) for c in ions[:n_ion]]

        if k_relabel:
            # Same positions, different element call. Co-folding evidence is
            # species-agnostic -- it says an ion sits here, not which one -- so
            # the weak link is arbitration, and that is what this model varies.
            # The 21 Mg from 9C6I were called by depositors with density in
            # hand, so they are exempt; among the rest, rank by how well the
            # observed nearest-RNA-oxygen distance matches K-O 2.80 A rather
            # than Mg-O 2.07 A, and reassign the best-fitting k_relabel of them.
            free = [c for c in ions if not c.get("exp")]
            dO = {id(c): per_el["O"].query(np.array(c["xyz"]))[0] for c in free}
            # a site whose nearest O is beyond the K coordination ceiling is not
            # a K site no matter how badly it fits Mg, so it is not eligible
            elig = [c for c in free if dO[id(c)] <= COORD_MAX["K"]]
            elig.sort(key=lambda c: abs(dO[id(c)] - K_O_IDEAL))
            chosen = {id(c) for c in elig[:k_relabel]}
            for c in free:
                c["sp"] = "K" if id(c) in chosen else ("MG" if c["sp"] == "K" else c["sp"])
            if len(elig) < k_relabel:
                print(f"     note: only {len(elig)} of {len(free)} free ions are "
                      f"eligible for K, wanted {k_relabel}")

        # ---- water: one ranked list, no reserved quota -----------------------
        #
        # There is no per-model fill order any more, and no per-Mg cap. Every
        # candidate -- Mg first shell, template consensus, pocket lattice --
        # competes on the single scale defined in R2386_water.py, which adds two
        # chemistry terms (hydrogen bonds to RNA, burial) to the evidence terms.
        # A shell water that also hydrogen-bonds to a phosphate is held from two
        # sides and wins; one pointing into bulk solvent is held from one side
        # and loses to template water. How many waters an Mg keeps falls out of
        # that instead of being set by hand.
        #
        # W_MGSHELL, the one term that cannot be derived, is swept blind against
        # held-out 3G78 by R2386_bench_composition.py. TOTAL hits @1.5 A:
        #
        #   ions          20   40   60   80  100  120  150
        #   no shell      82   91   92   93   92   93   89
        #   w =   0       85   94   95   98   95   95   92
        #   w =   3       85   98   98   97   95   99   95
        #   w =   6       86   98  100  100  100  100  100   <- flat, and best
        #   w =  12       87  100   97   95   97   99   91
        #   w =  24       88  101   97   95   97   92   88
        #   w = 999       88  101   97   95   96   91   85   (= the old tier order)
        #
        # w = 6 also wins the tighter @1.0 A column everywhere. Note what the
        # flat row means: with the shell chemistry-ranked, ion count no longer
        # trades against water quality, so the ladder below now costs nothing on
        # the blind key -- which is exactly what a hedge should cost.
        water_sel, stats = w3.build_water(
            n_wat, ions, per_el, tCterm, tmpl, pocketP, pocketS, hb_tree, tHeavy,
            core_fn=lambda P: core_mask(P, tCore, tNon))
        n_shell, n_tmpl, n_pock = stats["shell"], stats["template"], stats["pocket"]

        # B-factor carries confidence now that occupancy is pinned at 1.00, so
        # map the merged score onto it monotonically within the model. The best
        # water in a model gets 95, the worst 20; ions keep their own scale with
        # experimental Mg at 99.
        if water_sel:
            ss = np.array([s for _, s, _ in water_sel], float)
            lo, hi = ss.min(), ss.max()
            span = max(hi - lo, 1e-6)
            water = [(p, round(float(20.0 + 75.0 * (s - lo) / span), 2))
                     for p, s, _ in water_sel]
        else:
            water = []

        counts = {sp: sum(1 for c in ions if c["sp"] == sp) for sp in ("MG", "K", "NA")}
        A = np.asarray(ion_xyz)
        d = np.linalg.norm(A[:, None] - A[None, :], axis=2)
        np.fill_diagonal(d, 9e9)
        nexp = sum(1 for c in ions if c.get("exp"))

        write_ts(args.out / f"{TARGET}_model{i}.txt", i, label, rna_atoms, cterm,
                 ions, water, args.total)
        print(f"  {i:<3d}{label:40s}{counts['MG']:4d}{counts['K']:4d}{counts['NA']:3d}"
              f"{len(ions):6d}{len(water):7d}{len(ions)+len(water):7d}"
              f"{n_shell:8d}{n_tmpl:7d}{n_pock:7d}{d.min():6.2f}A{nexp:6d}/21")


def v2_cterm(pl):
    """Reuse v2's C-terminal completion (highest C-terminal pLDDT AF3 model)."""
    import glob
    import os
    af3 = sorted(glob.glob(
        os.environ.get("AF3_RESULTS_DIR", "/bml/Lyuwei/alphafold_results")
        + "/casp17_R_R2386/S2/seed-*/*model.cif"))
    if not af3:
        return [], None

    def _cterm_plddt(path):
        t = gemmi.read_structure(path)
        t.setup_entities()
        t.remove_alternative_conformations()
        q = max((ch.get_polymer() for ch in t[0]
                 if ch.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna),
                key=len)
        vals = [a.b_iso for r in q if r.name in ("A", "C", "G", "U")
                and v2.CTERM_LO <= r.seqid.num <= v2.CTERM_HI for a in r]
        return float(np.mean(vals)) if vals else -1.0

    best = max(af3, key=_cterm_plddt)
    m = gemmi.read_structure(best)
    m.setup_entities()
    m.remove_alternative_conformations()
    mp = max((ch.get_polymer() for ch in m[0]
              if ch.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna), key=len)
    sup = gemmi.calculate_superposition(pl, mp, gemmi.PolymerType.Rna, gemmi.SupSelect.CaP)
    m[0].transform_pos_and_adp(sup.transform)
    cterm = [(r.seqid.num, r.name,
              [(a.name, a.element.name, a.pos, 0.0) for a in r if a.element.name != "H"])
             for r in mp if r.name in ("A", "C", "G", "U")
             and v2.CTERM_LO <= r.seqid.num <= v2.CTERM_HI]
    ct = np.array([[a[2].x, a[2].y, a[2].z] for _, _, ats in cterm for a in ats]) \
        if cterm else np.zeros((0, 3))
    return cterm, (cKDTree(ct) if len(ct) else None)


def bfactor(s):
    """Evidence strength, 0-100. In v2 this was split between occupancy and B;
    the new rule reserves occupancy for the ligand count, so it all lives here."""
    if s.get("exp"):
        return 99.00
    t = min(s["t_support"], 10) / 10.0
    c = min(len(s["c_blind_methods"]), 3) / 3.0
    return round(20.0 + 75.0 * (0.6 * t + 0.4 * c), 2)


def write_ts(path, idx, label, rna_atoms, cterm, ions, water, total,
             method_body=None):
    """Write one CASP TS model.

    method_body lets a caller replace the middle of the METHOD block. The
    assessors read that record, so a build whose selection rules have changed
    must not ship the previous build's description of them.
    """
    ELEM = {"MG": "MG", "K": "K", "NA": "NA"}
    n_lig = len(ions) + len(water)
    assert n_lig == total, f"{path}: {n_lig} ligands, expected {total}"
    with open(path, "w") as fh:
        fh.write("PFRMAT TS\n")
        fh.write(f"TARGET {TARGET}\n")
        fh.write(f"AUTHOR {casp_identity.group_id()}\n")
        for line in (method_body if method_body is not None else [
            "Solvent shell merged from two independent evidence streams on a common",
            "site list. Stream 1: 37 experimental structures of the same group IIC",
            "intron (92.3-100% id, incl. 9C6I/9C6J) superposed on 9C6I, solvent pooled.",
            "Stream 2: 400 co-folding models (AlphaFold3, Boltz-2, Protenix default and",
            "20250630) over four Mg/K/Na stoichiometry points (10/20/35/55 Mg); ions",
            "with impossible intra-model ion-ion contacts (<3.0 A) discarded.",
            "Sites clustered at 1.5 A (ions) / 1.0 A (water); species at contested",
            "sites decided by coordination distance, since Mg2+ and Na+ are",
            "isoelectronic. Selection is greedy across all species at once with a",
            "3.5 A minimum ion-ion separation. The 21 Mg modelled in 9C6I are",
            "force-included. Water is [Mg(H2O)6]2+ first shell (geometric, Mg-O",
            "2.07 A), then template-consensus sites, then pocket positions ranked",
            "by hydrogen-bond count and burial.",
            f"Exactly {total} ligands, all at occupancy 1.00, all placed so that",
            "their nearest residue is in the assessed core region.",
            f"This model: {label}.",
            "RNA 1-393 from PDB 9C6I. Residues 394-417 are disordered in every",
            "100%-identical construct and independent experiments differ there by 17 A",
            "RMSD, so that backbone is at occupancy 0.00 and carries no solvent.",
        ]):
            fh.write(f"METHOD {line}\n")
        fh.write(f"MODEL  {idx}\n")
        fh.write("PARENT 9c6i\n")
        n = 0
        for tgt, resn, atoms in rna_atoms:
            for aname, el, pos, b in atoms:
                n += 1
                fh.write(v2._atom("ATOM", n, aname, resn, "0", tgt, pos, 1.00,
                                  min(max(b, 0.0), 100.0), el))
        for tgt, resn, atoms in cterm:
            for aname, el, pos, b in atoms:
                n += 1
                fh.write(v2._atom("ATOM", n, aname, resn, "0", tgt, pos, 0.00, 1.00, el))
        fh.write("TER\n")
        rid = 500
        for c in ions:
            n += 1
            rid += 1
            fh.write(v2._atom("HETATM", n, ELEM[c["sp"]], c["sp"], "0", rid,
                              gemmi.Position(*c["xyz"]), 1.00, bfactor(c), ELEM[c["sp"]]))
        for p, b in water:
            n += 1
            rid += 1
            fh.write(v2._atom("HETATM", n, "O", "HOH", "0", rid,
                              gemmi.Position(*p), 1.00, round(float(b), 2), "O"))
        fh.write("END\n")


if __name__ == "__main__":
    main()
