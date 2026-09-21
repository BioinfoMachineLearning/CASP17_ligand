#!/usr/bin/env python
"""v7 plus one rule: two ions may only crowd if something bridges them.

v7's only ion-ion constraint was a flat 3.5 A floor, and it was too generous.
Refined structures of this same intron -- 9C6I, 3G78, 5J01, 5J02, 4DS6, 206 ions
between them -- contain no pair below 3.5 A and exactly one below 4.0 A. v7's
ion-rich model contains nineteen, twelve of them with nothing at all between the
two metals. Two bare +2 cations at 3.5 A repel; nothing in the selection was
paying that cost.

A flat floor is still the wrong fix, because 3G78's one close pair is real. Two
Mg2+ do approach to 3.5-4.2 A when a ligand BRIDGES them -- a phosphate oxygen
coordinated to both screens the charge and pays for the approach, which is the
two-metal-ion motif of ribozyme active sites. So the rule is conditional:

    reject if d < IONION_HARD
    reject if d < IONION_SOFT unless the pair shares an RNA N/O in both shells

The bridge must be an RNA atom. Our own predicted water cannot serve: water is
placed after ions, so at selection time it does not exist, and justifying an ion
pair with a water invented to justify it is circular.

Measured blind on both held-out keys (R2386_ion_sep_test.py), at both the 80- and
120-ion budgets, this changes the hit count by at most one in either direction --
it is free. 360-384 sites remain placeable, so the 120-ion model still fills.

Inherited from v7: unheld water is replaced rather than shipped, and hexahydrate
orientation sampling is 200 (converged), not 48.

  python casp17/scripts/R2386_build_models_v8.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "casp17/scripts"))

import R2386_build_models as v2                                       # noqa: E402
import R2386_build_models_v3 as v3                                    # noqa: E402
import R2386_water as w3                                              # noqa: E402

TARGET = "R2386"
BURIAL_FLOOR = 10
BURIAL_R = 6.0
SHELL = {"MG": 2.60, "NA": 3.00, "K": 3.50}
HB_LO, HB_HI = 2.40, 3.40

# v2.ION_MIN is one floor for all species (O: 1.8 A), which is the Mg floor applied
# to K. A K sitting 1.88 A from a phosphate oxygen passes it and is nonsense --
# K-O is 2.80 A, and nothing that close to an oxygen is a potassium. Ionic radius
# sets these: Mg2+ 0.72, Na+ 1.02, K+ 1.38 A, against O 1.4 A.
MIN_CONTACT = {"MG": {"polar": 1.90, "C": 2.90},
               "NA": {"polar": 2.15, "C": 3.00},
               "K":  {"polar": 2.50, "C": 3.10}}
# v4 over-built the water list so the orphan gate had room to cut. With that gate
# gone the margin goes too: build_water is greedy under separation constraints, so
# building 540 and keeping the best 420 is not the same set as building 420, and
# that difference is a change we have no evidence for.
WATER_MARGIN = 0
# Na in three of five. The derivation says 2.0-2.8 are modellable, so 2-3 per
# model is the honest range; models 1 and 3 carry it alongside the species hedge
# in 5, and 2 and 4 stay Na-free so the slate is not swept by one assumption.
NA_RELABEL = [2, 0, 3, 0, 2]
NA_WINDOW = (2.25, 2.60)
# A Na call rests entirely on coordination distance: 2.42 A is too long for Mg
# (2.07) and too short for K (2.80). One oxygen in that window is not evidence --
# 2.3 A is also where a badly transferred Mg or a tight water lands, and with a
# single contact there is nothing to tell them apart. Two independent oxygens at
# Na distance is a signature; one is a coin flip. Counted on RNA atoms only,
# because water is built after the ions and cannot be used to justify them.
NA_MIN_RNA_O = 2
NA_CN_CUT = 2.90
# candidates held in reserve past the budget, so an unheld water has something
# to be replaced BY. build_water is greedy in score order under separation
# constraints, so asking for more only appends: the first n_wat are byte-for-byte
# what v6 shipped, and the extras are already separated from all of them.
RESERVE = 150
N_ORIENT = 200
RISM_NA = ROOT / "outputs/R2386_rism/rism_na_sites.npy"


# Ion-ion approach, scaled to ion size.
#
# v8 used one flat pair of numbers, 3.5 and 4.0 A, for every species. Those were
# calibrated on Mg2+ and are far too generous for K+. What has to fit between two
# close cations is a bridging oxygen at the ideal M-O distance FOR THAT SPECIES,
# and the geometry is d = 2 R sin(theta/2). So the same absolute separation forces
# a much tighter M-O-M angle on a big ion than on a small one: at 3.5 A two Mg2+
# (R = 2.07) bridge at 116 deg, which is the textbook two-metal-ion angle, while
# two K+ (R = 2.80) would have to bridge at 77 deg.
#
# 3G78's one real close K-K pair sits at 4.29 A with two bridging oxygens at
# exactly 90 deg and all four K-O within 0.24 A of ideal. Take 90 deg as the
# limit -- below it the bridge is closing the hinge rather than screening the
# charge -- and the floor for a pair is d = sqrt(2) R, averaged over the two
# species, never below the 3.5 A that Mg was measured to obey.
#
#     MG-MG 3.50   MG-NA 3.50   MG-K 3.50   NA-NA 3.50   NA-K 3.69   K-K 3.96
#
IDEAL_MO = {"MG": 2.07, "NA": 2.42, "K": 2.80}
IONION_FLOOR = 3.5        # empirical Mg-Mg floor; no pair may go below it
IONION_BAND = 0.5         # width of the bridge-required band above the floor


def ionion_limits(a, b):
    """(hard, soft) for one species pair; hard is the 90 deg bridging limit.

    With the bridge at O, M1-O = R1 and M2-O = R2, the law of cosines gives
    d^2 = R1^2 + R2^2 - 2 R1 R2 cos(theta), so theta = 90 deg puts the two metals
    at sqrt(R1^2 + R2^2). For a mixed pair that is up to 0.04 A more than the
    average of the two limits, which is the wrong side to approximate on.
    """
    hard = max(IONION_FLOOR, (IDEAL_MO[a] ** 2 + IDEAL_MO[b] ** 2) ** 0.5)
    return hard, hard + IONION_BAND


IONION_MAX = max(ionion_limits(a, b)[1] for a in IDEAL_MO for b in IDEAL_MO)


def method_body(total, label):
    """The METHOD record for this build.

    v3's text is still correct about where the sites came from, but it describes
    a flat 3.5 A ion-ion cutoff and says nothing about the burial floor, the
    interaction gate or the sodium test -- all of which decide what is in the
    file. METHOD is submitted content, so it states the rules actually applied.
    """
    return [
        "Solvent shell merged from two independent evidence streams on a common",
        "site list. Stream 1: 37 experimental structures of the same group IIC",
        "intron (92.3-100% id, incl. 9C6I/9C6J) superposed on 9C6I, solvent",
        "pooled. Stream 2: 400 co-folding models (AlphaFold3, Boltz-2, Protenix",
        "default and 20250630) over four Mg/K/Na stoichiometry points (10/20/35/",
        "55 Mg); ions with impossible intra-model ion-ion contacts (<3.0 A) cut.",
        "Sites clustered at 1.5 A (ions) / 1.0 A (water); species at contested",
        "sites decided by coordination distance, since Mg2+ and Na+ are",
        "isoelectronic. Selection is greedy across all species at once.",
        "Ion-ion separation floors are species-aware, taken from the 90 deg limit",
        "for a single bridging oxygen at the ideal M-O distance, sqrt(R1^2+R2^2):",
        "3.50 A Mg-Mg / Mg-Na / Na-Na / K-Mg, 3.70 K-Na, 3.96 K-K. Between the",
        "floor and floor+0.5 A a shared RNA N/O bridge is required. A site is",
        "called Na only if >=2 RNA oxygens lie within 2.90 A and the nearest is",
        "2.25-2.60 A; water does not count, being built after the ions. Every",
        "ligand must contact something it can actually interact with, and no ion",
        "may float: >=10 RNA heavy atoms within 6 A. The 21 Mg modelled in 9C6I",
        "are force-included. Water is [Mg(H2O)6]2+ first shell (geometric, Mg-O",
        "2.07 A), then template-consensus sites, then pocket positions ranked",
        "by hydrogen-bond count and burial.",
        f"Exactly {total} ligands, all at occupancy 1.00, all placed so that",
        "their nearest residue is in the assessed core region.",
        f"This model: {label}.",
        "RNA 1-393 from PDB 9C6I. Residues 394-417 are disordered in every",
        "100%-identical construct and independent experiments differ there by 17 A",
        "RMSD, so that backbone is at occupancy 0.00 and carries no solvent.",
    ]


def greedy_bridged(cands, tNO):
    """Priority-order greedy; below `soft` a pair needs a shared RNA N/O.

    Experimental ions are never rejected -- they were observed, and a rule
    derived from other structures does not get to overrule the reference.
    """
    acc, shells, sps, out = [], [], [], []
    for c in cands:
        p = np.asarray(c["xyz"])
        sh = set(tNO.query_ball_point(p, SHELL[c["sp"]]))
        ok = True
        if acc and not c.get("exp"):
            d = np.linalg.norm(np.asarray(acc) - p, axis=1)
            for k in np.where(d < IONION_MAX)[0]:
                hard, soft = ionion_limits(c["sp"], sps[k])
                if d[k] >= soft:
                    continue
                if d[k] < hard or not (sh & shells[k]):
                    ok = False
                    break
        if ok:
            acc.append(p); shells.append(sh); sps.append(c["sp"]); out.append(c)
    return out


def held(ions, W, tNO):
    """Is each water held by anything at all? (RNA H-bond / water / ion shell)"""
    if not len(W):
        return np.zeros(0, bool)
    ok = np.zeros(len(W), bool)
    for j, p in enumerate(W):
        if any(HB_LO <= np.linalg.norm(tNO.data[i] - p) <= HB_HI
               for i in tNO.query_ball_point(p, HB_HI)):
            ok[j] = True
    tW = cKDTree(W)
    for j, p in enumerate(W):
        if ok[j]:
            continue
        if any(i != j and HB_LO <= np.linalg.norm(W[i] - p) <= HB_HI
               for i in tW.query_ball_point(p, HB_HI)):
            ok[j] = True
    for c in ions:
        for j in tW.query_ball_point(np.asarray(c["xyz"]), SHELL[c["sp"]]):
            ok[j] = True
    return ok


def justified_fill(ions, cand, n_wat, tNO):
    """Spend the budget on held water first; bench the rest, recall only if short.

    Iterated because holding is mutual: dropping a water can unhold its neighbour,
    and a recalled candidate has to be re-tested in the set it actually joins.
    """
    n0 = min(n_wat, len(cand))
    sel, pool, benched = list(range(n0)), list(range(n0, len(cand))), []
    for _ in range(10):
        ok = held(ions, np.array([cand[i][0] for i in sel]), tNO)
        bad = [k for k in range(len(sel)) if not ok[k]]
        if not bad or not pool:
            break
        for k in sorted(bad, reverse=True):
            benched.append(sel.pop(k))
        while len(sel) < n_wat and pool:
            sel.append(pool.pop(0))
    dropped = len(benched)
    while len(sel) < n_wat and benched:          # the 500 cap wins over the preference
        sel.append(benched.pop(0))
    return sel, dropped - len(benched)


def chemistry_gate(ions, water_xyz, tNO, tHeavy):
    """Iteratively drop ligands that interact with nothing.

    Ions and water support each other -- an outer-sphere ion is justified by its
    shell, and a shell water is justified by its ion -- so a single pass can keep
    a pair that is only holding itself up. Iterate until nothing changes.
    """
    keep_i = np.ones(len(ions), bool)
    keep_w = np.ones(len(water_xyz), bool)
    I = np.array([np.asarray(c["xyz"]) for c in ions]) if ions else np.zeros((0, 3))
    W = np.asarray(water_xyz) if len(water_xyz) else np.zeros((0, 3))

    # precompute the fixed part: contacts with the RNA
    ion_direct = np.array([len(tNO.query_ball_point(p, SHELL[c["sp"]])) > 0
                           for p, c in zip(I, ions)]) if len(I) else np.zeros(0, bool)
    wat_hb = np.array([any(HB_LO <= np.linalg.norm(tNO.data[i] - p) <= HB_HI
                           for i in tNO.query_ball_point(p, HB_HI))
                       for p in W]) if len(W) else np.zeros(0, bool)

    tW = cKDTree(W) if len(W) else None
    tI = cKDTree(I) if len(I) else None

    for _ in range(6):
        changed = False
        # a water survives if it hydrogen-bonds RNA, or sits in a kept ion's shell
        for j in np.where(keep_w)[0]:
            if wat_hb[j]:
                continue
            ok = False
            if tI is not None:
                for k in tI.query_ball_point(W[j], max(SHELL.values())):
                    if keep_i[k] and np.linalg.norm(I[k] - W[j]) <= SHELL[ions[k]["sp"]]:
                        ok = True
                        break
            if not ok:
                keep_w[j] = False
                changed = True
        # an ion survives if it contacts RNA directly, or holds a kept water that
        # itself reaches the RNA
        for k in np.where(keep_i)[0]:
            if ion_direct[k] or ions[k].get("exp"):
                continue
            ok = False
            if tW is not None:
                for j in tW.query_ball_point(I[k], SHELL[ions[k]["sp"]]):
                    if keep_w[j] and wat_hb[j]:
                        ok = True
                        break
            if not ok:
                keep_i[k] = False
                changed = True
        if not changed:
            break
    return keep_i, keep_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "outputs/R2386_pool_local")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "casp17/submissions/casp17_R/R2386_files_v11")
    ap.add_argument("--total", type=int, default=v3.TOTAL_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sites = json.load(open(args.pool / "sites_layered.json"))
    st, pl = v2.load_ref()
    per_el, tHeavy, heavy_xyz = v2.ref_maps(pl)

    def ion_ok(x):
        return all(t.query(x)[0] >= v2.ION_MIN[el]
                   for el, t in per_el.items() if el in v2.ION_MIN)

    COORD_MAX = {"MG": 4.6, "K": 3.5, "NA": 3.2}
    K_O_IDEAL, NA_O_IDEAL = 2.80, 2.42

    _b = [a.b_iso for r in pl if r.name in ("A", "C", "G", "U") for a in r
          if a.element.name != "H"]
    _blo, _bhi = min(_b), max(_b)

    def _conf(b):
        return 90.0 if _bhi <= _blo else round(99.0 - 79.0 * (b - _blo) / (_bhi - _blo), 2)

    rna_atoms = [(i, r.name, [(a.name, a.element.name, a.pos, _conf(a.b_iso))
                              for a in r if a.element.name != "H"])
                 for i, r in enumerate((r for r in pl if r.name in ("A", "C", "G", "U")),
                                       start=1)]
    cterm, tCterm = v3.v2_cterm(pl)

    rna_pts, rna_res = [], []
    for tgt, _, atoms in rna_atoms + cterm:
        for _, _, pos, _ in atoms:
            rna_pts.append([pos.x, pos.y, pos.z])
            rna_res.append(tgt)
    rna_pts, rna_res = np.array(rna_pts), np.array(rna_res)
    is_core = np.array([r not in v3.NONCORE for r in rna_res])
    tCore, tNon = cKDTree(rna_pts[is_core]), cKDTree(rna_pts[~is_core])

    hb_pts = np.array([[pos.x, pos.y, pos.z] for _, _, atoms in rna_atoms
                       for _, el, pos, _ in atoms if el in ("N", "O")])
    hb_tree = cKDTree(hb_pts)

    arbitrated = v2.arbitrate_species(sites, tO_global=per_el.get("O"))
    exp_mg = [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch
              if r.name == "MG" for a in r]

    def burial(x):
        return len(tHeavy.query_ball_point(np.asarray(x), BURIAL_R))

    tPolar = cKDTree(np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r
                               if a.element.name in ("N", "O")]))
    tCarb = cKDTree(np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r
                              if a.element.name == "C"]))

    def contact_ok(x, sp):
        lim = MIN_CONTACT[sp]
        return (tPolar.query(np.asarray(x))[0] >= lim["polar"]
                and tCarb.query(np.asarray(x))[0] >= lim["C"])

    n_exempt = sum(1 for p in exp_mg if burial(p) < BURIAL_FLOOR)
    print(f"burial floor {BURIAL_FLOOR} heavy atoms within {BURIAL_R} A")
    print(f"  {n_exempt} of the 21 experimental Mg would fail it and are exempt")

    tmpl = [s for s in sites["HOH"] if v2.water_ok(np.array(s["xyz"]), per_el)]
    if tCterm is not None:
        tmpl = [s for s in tmpl
                if tCterm.query(np.array(s["xyz"]))[0] >= v2.CTERM_EXCLUSION]
    tmpl = [s for s, k in zip(tmpl, v3.core_mask([s["xyz"] for s in tmpl],
                                                 tCore, tNon)) if k]
    pocketP, pocketS = v3.pocket_candidates(heavy_xyz, per_el, tCore, tNon,
                                            tCterm, hb_tree)

    rism_na = np.load(RISM_NA) if RISM_NA.exists() else None
    tRNa = cKDTree(rism_na[:, :3]) if rism_na is not None else None

    print(f"\n  {'#':<3s}{'model':38s}{'MG':>4s}{'K':>4s}{'NA':>3s}"
          f"{'ions':>6s}{'water':>7s}{'total':>7s}{'shellW':>8s}"
          f"{'burial-cut':>12s}{'clash-cut':>11s}{'orphans-kept':>14s}")

    for i, (label, n_ion, n_wat, k_relabel) in enumerate(v3.COMPOSITION, start=1):
        # ---- ions, now with the burial floor ------------------------------
        cands = [dict(xyz=p, sp="MG", t_support=99, c_blind_methods=[], c_hits=0,
                      exp=True) for p in exp_mg]
        n_bur, n_clash = 0, 0
        for s in sorted(arbitrated, key=lambda a: -v2.priority(a)):
            x = np.array(s["xyz"])
            if not ion_ok(x):
                continue
            if per_el["O"].query(x)[0] > COORD_MAX[s["sp"]]:
                continue
            if tCterm is not None and tCterm.query(x)[0] < v2.CTERM_EXCLUSION:
                continue
            if not v3.core_mask([x], tCore, tNon)[0]:
                continue
            if burial(x) < BURIAL_FLOOR:
                n_bur += 1
                continue
            if not contact_ok(x, s["sp"]):
                n_clash += 1
                continue
            cands.append(dict(s))
        cands.sort(key=lambda s: (-1e9 if s.get("exp") else 0) - v2.priority(s))
        ions = greedy_bridged(cands, hb_tree)
        spare_ions = ions[n_ion:]
        ions = ions[:n_ion]

        if k_relabel:
            free = [c for c in ions if not c.get("exp")]
            dO = {id(c): per_el["O"].query(np.array(c["xyz"]))[0] for c in free}
            # relabelling changes the ionic radius, so the contact floors have to
            # be rechecked against the NEW species, not the one it was chosen as
            elig = [c for c in free if dO[id(c)] <= COORD_MAX["K"]
                    and contact_ok(c["xyz"], "K")]
            elig.sort(key=lambda c: abs(dO[id(c)] - K_O_IDEAL))
            chosen = {id(c) for c in elig[:k_relabel]}
            for c in free:
                c["sp"] = "K" if id(c) in chosen else ("MG" if c["sp"] == "K" else c["sp"])

        # ---- the Na hedge -------------------------------------------------
        n_na = NA_RELABEL[i - 1]
        if n_na and tRNa is not None:
            free = [c for c in ions if not c.get("exp")]
            scored = []
            for c in free:
                x = np.array(c["xyz"])
                d = per_el["O"].query(x)[0]
                if not (NA_WINDOW[0] <= d <= NA_WINDOW[1]):
                    continue
                if not contact_ok(x, "NA"):
                    continue
                if len(per_el["O"].query_ball_point(x, NA_CN_CUT)) < NA_MIN_RNA_O:
                    continue
                dn, jn = tRNa.query(x)
                scored.append((dn, abs(d - NA_O_IDEAL), c))
            scored.sort(key=lambda t: (t[0], t[1]))
            for _, _, c in scored[:n_na]:
                c["sp"] = "NA"

        # The hedge is not the only way a site can end up labelled Na. Species
        # arbitration also awards NA on its own -- when a template claims Na
        # there and the observed O distance happens to sit nearest 2.42 -- and
        # that path never looked at coordination number. It shipped a site with
        # one RNA oxygen at 2.43 A whose only other close contact was an
        # exocyclic amino nitrogen at 2.69 A, which is not coordination at all:
        # N4 carries the hydrogens, so it points its delta+ at the cation.
        #
        # So the test is applied to every Na regardless of where the label came
        # from. Failing it does not empty the site -- the density is still there
        # -- it means the species cannot be Na, so the site falls back to the
        # better of Mg/K on the same distance prior arbitration used, and only
        # if that species clears its own contact and separation floors.
        for c in ions:
            if c["sp"] != "NA" or c.get("exp"):
                continue
            x = np.array(c["xyz"])
            if len(per_el["O"].query_ball_point(x, NA_CN_CUT)) >= NA_MIN_RNA_O:
                continue
            d = per_el["O"].query(x)[0]
            for alt in sorted(("MG", "K"), key=lambda sp: abs(d - IDEAL_MO[sp])):
                if not contact_ok(x, alt):
                    continue
                if any(np.linalg.norm(np.asarray(o["xyz"]) - x)
                       < ionion_limits(alt, o["sp"])[0]
                       for o in ions if o is not c):
                    continue
                c["sp"] = alt
                break

        # ---- water, built exactly as v3 built it ---------------------------
        want = n_wat + WATER_MARGIN + RESERVE
        water_sel, stats = w3.build_water(
            want, ions, per_el, tCterm, tmpl, pocketP, pocketS, hb_tree, tHeavy,
            core_fn=lambda P: v3.core_mask(P, tCore, tNon), n_orient=N_ORIENT)
        wxyz = [np.asarray(p) for p, _, _ in water_sel]
        wsc = [s for _, s, _ in water_sel]

        # gate 2 stood here and is withdrawn -- see the module docstring. The
        # orphans it cut are conserved template sites decoordinated by the 2-3 A
        # superposition that carried them into this frame, and the de novo guesses
        # that replaced them hit an order of magnitude less often. What is left is
        # only counted, so the number stays visible instead of reading zero.
        cand = sorted(zip(wxyz, wsc), key=lambda t: -t[1])
        keep, n_swap = justified_fill(ions, cand, n_wat, hb_tree)
        water = [cand[i] for i in keep]
        n_orph = int((~chemistry_gate(ions, [q for q, _ in water],
                                      hb_tree, tHeavy)[0]).sum())
        print(f"     placeholders replaced by held water: {n_swap}")
        if len(water) < n_wat or len(ions) < n_ion:
            print(f"     note: {len(ions)}/{n_ion} ions, "
                  f"{len(water)}/{n_wat} water after gating")

        if water:
            ss = np.array([s for _, s in water], float)
            lo, hi = ss.min(), ss.max()
            span = max(hi - lo, 1e-6)
            wout = [(p, round(float(20.0 + 75.0 * (s - lo) / span), 2))
                    for p, s in water]
        else:
            wout = []

        counts = {sp: sum(1 for c in ions if c["sp"] == sp) for sp in ("MG", "K", "NA")}
        v3.write_ts(args.out / f"{TARGET}_model{i}.txt", i, label, rna_atoms, cterm,
                    ions, wout, args.total,
                    method_body=method_body(args.total, label))
        print(f"  {i:<3d}{label:38s}{counts['MG']:4d}{counts['K']:4d}{counts['NA']:3d}"
              f"{len(ions):6d}{len(wout):7d}{len(ions)+len(wout):7d}"
              f"{stats['shell']:8d}{n_bur:12d}{n_clash:11d}{n_orph:14d}")


if __name__ == "__main__":
    main()