#!/usr/bin/env python
"""R2386 step 6 (v2): assemble five solvent models and write PFRMAT TS files.

Three corrections over v1, all found by visual inspection of the v1 output:

1. EXPERIMENTAL IONS WERE BEING DROPPED. 9C6I's 21 modelled Mg entered the pool
   as ordinary template donors, giving them t_support=2 (9C6I + its twin 9C6J).
   A site with no co-folding support then failed layer A (needs nb>=1) and
   layer B (needs t>=3) and fell to the discard layer -- so 2-5 experimentally
   observed ions were missing from every model. An experimental observation of
   the target molecule must never be outvoted by an evidence heuristic, so they
   are now force-included at top priority.

2. NO CROSS-SPECIES DE-CLASHING. Mg, K and Na were clustered independently, so
   one real site claimed by different methods as different species produced two
   or three superimposed ions. Measured on v1: minimum ion-ion distance 0.18 A,
   with 54-95% of ions closer than 3.0 A to another ion. Selection now runs a
   single greedy pass over ALL species at once, ordered by evidence strength,
   enforcing a minimum separation.

3. FILL-SHELL HEDGE. R2386 scores the best of five models but does not publish
   its metric. Models 1-3 assume the task rewards picking ordered sites; models
   4-5 assume it rewards filling the shell, generating a bulk-density water
   lattice in the manner of the batch-2 W2386 solvation script, with the
   high-evidence ions kept on top. Hydrogens are omitted (that script emits
   them; this target explicitly forbids them).

Minimum separations, calibrated on 9C6I's own solvent:
    ion-ion    3.5 A   (real Mg-Mg never approach closer, even bridged)
    water-ion  2.0 A   (allows a first-shell water at 2.07 A)
    water-water 2.4 A
"""

from __future__ import annotations

import argparse
import glob
import os
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casp_identity                                                  # noqa: E402
import R2386_paths as paths                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REF_CIF = str(paths.ref_cif())
TARGET = "R2386"
CTERM_LO, CTERM_HI = 394, 417
# Per-element minimum approach for an ION, calibrated on 9C6I's own 21 Mg:
# Mg-O min 1.89 (inner-sphere coordination), Mg-N 3.03, Mg-C 3.13, Mg-P 3.22.
# v2 checked carbon only, which let ions sit 0.69 A from an oxygen.
ION_MIN = {"O": 1.8, "N": 2.6, "C": 2.8, "P": 2.9, "S": 2.9}
CTERM_EXCLUSION = 6.0
MIN_ION_ION = 3.5
MIN_WAT_ION = 2.0
MIN_WAT_WAT = 2.4

# Per-element minimum approach for a water oxygen, from the batch-2 W2386
# solvation script. A single cutoff is wrong: water may sit 2.55 A from an RNA
# oxygen (hydrogen bond) but must stay 3.00 A from carbon (van der Waals only).
WATER_MIN = {"C": 3.00, "N": 2.75, "O": 2.55, "P": 3.15, "S": 3.15}
LATTICE = 3.10          # bulk water number density
SHELL_5, SHELL_65 = 5.0, 6.5

# Na needs its own, looser rule. Arbitration never hands Na a template-supported
# site -- wherever the crystallographers modelled Na, this intron's other
# structures modelled Mg or K at the same place and outvote it. Na only wins at
# co-folding-only positions, so requiring template support zeroes it out. Given
# the buffer is 5 mM Na-cacodylate against 100 mM KCl and 10 mM MgCl2, Na is a
# genuinely minor species; cap it rather than let 166 sites through.
NA_CAP = [8, 4, 15, 20, 8]

# Ion-count ceiling, from the observed density of this very molecule. The richest
# experimental structure here is 3G78 at 2.80 A with 77 ions over 389 nt =
# 0.198 ions/nt, i.e. 83 ions scaled to 417 nt. Going from 2.80 A to the target's
# 1.95 A might optimistically resolve 2-3x that, so 165-248. A tier claiming 553
# ordered ions (1.33 ions/nt, 6.7x the best experiment ever managed on this RNA,
# and more than the 453 ions in the batch-2 W2386 models -- which are a charge-
# neutralisation construct, not a claim about ordered sites) is not defensible,
# so the loosest tier is capped by count instead of by threshold.
# Stopping rule for the densest tier. Two independent physical arguments land on
# essentially the same number, so use the more self-explanatory one:
#   (a) density ceiling: 0.198 ions/nt (3G78, richest ever on this RNA) x 3 x 393
#       modelled residues = 233 ions
#   (b) charge neutrality: the RNA carries 417 phosphates, i.e. -417; adding ions
#       in evidence order until the cation charge reaches +417 also stops near 233
# (b) is used. It gives a tier that is simultaneously at the density ceiling AND
# electroneutral, which no other tier is (models 1/2/5 sit at -312 to -342).
MAX_IONS = [None, None, None, None, None]
NEUTRALISE = [False, False, False, True, False]

MODELS = [
    # (label, mode, mg_t, mg_nb, k_t, k_nb, na_t, na_nb, water_t, shell)
    ("balanced evidence",     "evidence", 2, 1, 2, 1, 0, 1, 3, None),
    ("conservative evidence", "evidence", 3, 2, 3, 2, 0, 2, 4, None),
    ("aggressive evidence",   "evidence", 0, 2, 0, 2, 0, 1, 2, None),
    ("electroneutral",        "evidence", 0, 1, 0, 1, 0, 1, 2, None),
    ("filled 6.5 A shell",    "fill",     2, 1, 2, 1, 0, 1, 0, SHELL_65),
]


def load_ref():
    st = gemmi.read_structure(REF_CIF)
    st.setup_entities()
    st.remove_alternative_conformations()
    pl = max((ch.get_polymer() for ch in st[0]
              if ch.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna), key=len)
    return st, pl


def ref_maps(pl):
    per_el = {}
    for el in WATER_MIN:
        pts = [[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r if a.element.name == el]
        if pts:
            per_el[el] = cKDTree(np.array(pts))
    heavy = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r if a.element.name != "H"])
    return per_el, cKDTree(heavy), heavy


def water_ok(x, per_el) -> bool:
    return all(t.query(x)[0] >= WATER_MIN[el] for el, t in per_el.items())


def priority(s) -> float:
    """Rank a candidate. Template support dominates: it tracked accuracy
    monotonically in validation, whereas co-folding cross-seed agreement did not."""
    return 10.0 * s["t_support"] + 3.0 * len(s["c_blind_methods"]) + min(s["c_hits"], 30) / 30.0


def greedy(cands, min_sep, seeded=None):
    """Accept candidates in priority order, keeping a minimum separation."""
    acc_xyz = list(seeded) if seeded is not None else []
    out = []
    for c in cands:
        p = np.asarray(c["xyz"])
        if acc_xyz:
            d = np.linalg.norm(np.asarray(acc_xyz) - p, axis=1).min()
            if d < min_sep:
                continue
        acc_xyz.append(p)
        out.append(c)
    return out, acc_xyz


def lattice_waters(heavy_xyz, per_el, shell, ion_xyz, cterm_tree, shell_waters=()):
    """Bulk-density water lattice filling the requested shell."""
    tree = cKDTree(heavy_xyz)
    lo, hi = heavy_xyz.min(0) - shell - 2, heavy_xyz.max(0) + shell + 2
    ax = [np.arange(lo[i], hi[i] + LATTICE, LATTICE) for i in range(3)]
    keep = []
    Y, Z = np.meshgrid(ax[1], ax[2], indexing="ij")
    YZ = np.column_stack([Y.ravel(), Z.ravel()])
    for x in ax[0]:
        slab = np.column_stack([np.full(len(YZ), x), YZ])
        d, _ = tree.query(slab, k=1, workers=-1)
        cand = slab[d <= shell]
        for p in cand:
            if not water_ok(p, per_el):
                continue
            if cterm_tree is not None and cterm_tree.query(p)[0] < CTERM_EXCLUSION:
                continue
            keep.append(p)
    keep = np.array(keep) if keep else np.zeros((0, 3))
    if len(keep) and len(ion_xyz):
        d = np.linalg.norm(keep[:, None] - np.asarray(ion_xyz)[None, :], axis=2).min(1)
        keep = keep[d >= MIN_WAT_ION]
    # First-shell waters are WATER, so the lattice must clear them by the
    # water-water minimum, not the (smaller) water-ion one. Passing them in with
    # the ions let lattice points sit 2.0 A from a shell water.
    if len(keep) and len(shell_waters):
        d = np.linalg.norm(keep[:, None] - np.asarray(shell_waters)[None, :], axis=2).min(1)
        keep = keep[d >= MIN_WAT_WAT]
    # thin to respect water-water spacing
    out = []
    for p in keep:
        if out and np.linalg.norm(np.array(out) - p, axis=1).min() < MIN_WAT_WAT:
            continue
        out.append(p)
    return np.array(out) if out else np.zeros((0, 3))



def arbitrate_species(sites, cut=1.5, tO_global=None):
    """Decide ONE species per physical site.

    Mg, K and Na were clustered independently, so a single real site can carry a
    claim from each. Measured on this target: every Na candidate at every
    evidence tier sits within 3.5 A of both a Mg and a K candidate -- there are
    no Na-specific positions. Emitting all three claims produced superimposed
    ions; dropping the loser by raw priority erased Na entirely, because Mg
    simply has more raw data behind it.

    So arbitrate instead: whichever species the EXPERIMENTAL structures called
    most often at that location wins, since the depositors had density and
    coordination geometry to judge chemistry with. Co-folding support breaks
    ties, then Mg > K > Na (Mg is the species actually resolved as ordered in
    9C6I). This lets Na win the sites where Na evidence is genuinely strongest.
    """
    merged = []
    for sp in ("MG", "K", "NA"):
        for s in sites.get(sp, []):
            merged.append((np.asarray(s["xyz"]), sp, s))
    merged.sort(key=lambda m: -(10 * m[2]["t_support"] + len(m[2]["c_blind_methods"])))
    cents, groups = [], []
    for xyz, sp, s in merged:
        placed = False
        for gi, c in enumerate(cents):
            if np.linalg.norm(xyz - c) <= cut:
                groups[gi].append((sp, s))
                placed = True
                break
        if not placed:
            cents.append(xyz.copy())
            groups.append([(sp, s)])
    # Chemistry-based prior. Mg2+ and Na+ are ISOELECTRONIC (10 e- each) so a
    # density peak cannot tell them apart; the discriminator is the coordination
    # distance to the nearest RNA oxygen. Measured on the donors that model Na:
    #   Mg 2.34-2.47 A median,  Na 2.74-3.14 A,  K 3.00-3.02 A
    # (textbook values 2.07 / 2.4 / 2.8; the donors sit at 2.7-4.8 A resolution
    # so the distributions overlap, which is why this is a prior and not a gate.)
    RANK = {"MG": 2, "K": 1, "NA": 0}
    out = []
    for c, g in zip(cents, groups):
        best = {}
        for sp, s in g:
            b = best.get(sp)
            if b is None or (s["t_support"], len(s["c_blind_methods"])) > (b["t_support"], len(b["c_blind_methods"])):
                best[sp] = s
        dO = tO_global.query(c)[0] if tO_global is not None else None
        def chem_bonus(sp):
            if dO is None:
                return 0.0
            ideal = {"MG": 2.07, "NA": 2.40, "K": 2.80}[sp]
            return -abs(dO - ideal)          # closer to the ideal distance is better
        win = max(best, key=lambda sp: (best[sp]["t_support"] + 2.0 * chem_bonus(sp),
                                        len(best[sp]["c_blind_methods"]),
                                        RANK[sp]))
        rec = dict(best[win])
        rec["sp"] = win
        rec["xyz"] = c.tolist()
        rec["exp"] = False
        rec["contested"] = sorted(best)
        out.append(rec)
    return out



def hexahydrate(ion_xyz, per_el, tCterm, other_ions, other_waters, n_orient=48):
    """Generate the [Mg(H2O)6]2+ first shell around each placed Mg.

    Mg2+ is essentially always hexahydrated with Mg-O(water) = 2.07 A, and that
    shell is the most rigidly held water in the structure -- 5J01/5J02 model
    almost nothing else (95-98% of their waters are Mg first-shell). Those
    waters were being filtered out of our models because 5J01 and 5J02 are one
    lab's sister depositions, so a shell water can never exceed template support
    2, below the threshold used for the ordered-water tier.

    The shell is geometry, not a vote, so generate it directly: try several
    octahedral orientations per ion and keep the one placing the most waters
    that clear the RNA and the already-placed solvent.
    """
    AX = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    rng = np.random.default_rng(2386)
    out = []
    # Ions and waters need DIFFERENT minima. A first-shell water sits 2.07 A from
    # its own Mg, so testing it against the 2.4 A water-water floor rejects every
    # shell water via its own centre -- the bug this signature change fixes.
    ions_arr = np.asarray([np.asarray(x) for x in other_ions]) if len(other_ions) else np.zeros((0, 3))
    waters = [np.asarray(x) for x in other_waters]
    for c, sp in ion_xyz:
        if sp != "MG":
            continue
        best = []
        for _ in range(n_orient):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            cand = np.asarray(c) + (R @ AX.T).T * 2.07
            keep = []
            for p in cand:
                if not water_ok(p, per_el):
                    continue
                if tCterm is not None and tCterm.query(p)[0] < CTERM_EXCLUSION:
                    continue
                if len(ions_arr) and np.linalg.norm(ions_arr - p, axis=1).min() < MIN_WAT_ION:
                    continue
                pool = waters + keep
                if pool and np.linalg.norm(np.asarray(pool) - p, axis=1).min() < MIN_WAT_WAT:
                    continue
                keep.append(p)
            if len(keep) > len(best):
                best = keep
        waters.extend(best)
        out.extend(best)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "outputs/R2386_pool_submit")
    ap.add_argument("--out", type=Path, default=ROOT / "casp17/submissions/casp17_R/R2386_files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sites = json.load(open(args.pool / "sites_layered.json"))
    st, pl = load_ref()
    per_el, tHeavy, heavy_xyz = ref_maps(pl)
    def ion_ok(x):
        return all(t.query(x)[0] >= ION_MIN[el] for el, t in per_el.items() if el in ION_MIN)

    # Upper bound on the ion-to-RNA-oxygen distance, i.e. does the site actually
    # coordinate anything. Mg2+ is allowed to be far because outer-sphere Mg keeps
    # its [Mg(H2O)6]2+ shell and sits 4.0-4.5 A out (9 of 9C6I's 21 are like this).
    # K+ and Na+ have labile shells that are not resolvable, so a K or Na with no
    # RNA oxygen in reach is an unsupportable placement rather than an outer-sphere
    # one -- validation caught 4 such Na floating at 3.90 A.
    COORD_MAX = {"MG": 4.6, "K": 3.5, "NA": 3.2}

    def coordinates_something(x, sp):
        return per_el["O"].query(x)[0] <= COORD_MAX[sp]

    arbitrated = arbitrate_species(sites, tO_global=per_el.get('O'))
    from collections import Counter
    print("species arbitration ->", dict(Counter(a["sp"] for a in arbitrated)))

    exp_mg = [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch if r.name == "MG" for a in r]
    print(f"force-included experimental ions: {len(exp_mg)} Mg from 9C6I")

    # 9C6I's experimental B-factors run to 251, but the CASP TS spec defines the
    # B column as a 0-100 confidence (100 = most confident) and rejects models
    # whose residues all share one value. Map experimental B monotonically onto
    # 20-99 so ordering and variation survive while staying in spec.
    _b = [a.b_iso for r in pl if r.name in ("A", "C", "G", "U") for a in r
          if a.element.name != "H"]
    _blo, _bhi = min(_b), max(_b)
    def _conf(b):
        if _bhi <= _blo:
            return 90.0
        return round(99.0 - 79.0 * (b - _blo) / (_bhi - _blo), 2)
    rna_atoms = [(i, r.name, [(a.name, a.element.name, a.pos, _conf(a.b_iso)) for a in r
                              if a.element.name != "H"])
                 for i, r in enumerate((r for r in pl if r.name in ("A", "C", "G", "U")), start=1)]

    cterm = []
    # Pick the AF3 model whose C-terminal pLDDT is highest, rather than whichever
    # filename sorts first. The difference is marginal here (46.7 vs 49.8 out of
    # 100) because NO model is confident about this region -- C-terminal pLDDT
    # runs 37-50 against 91 for the core -- but the code should do what it says.
    af3 = sorted(glob.glob(os.environ.get("AF3_RESULTS_DIR", "/bml/Lyuwei/alphafold_results")
                             + "/casp17_R_R2386/S2/seed-*/*model.cif"))
    if af3:
        def _cterm_plddt(path):
            t = gemmi.read_structure(path); t.setup_entities(); t.remove_alternative_conformations()
            q = max((ch.get_polymer() for ch in t[0]
                     if ch.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna), key=len)
            vals = [a.b_iso for r in q if r.name in ("A", "C", "G", "U")
                    and CTERM_LO <= r.seqid.num <= CTERM_HI for a in r]
            return float(np.mean(vals)) if vals else -1.0
        af3 = [max(af3, key=_cterm_plddt)]
        m = gemmi.read_structure(af3[0]); m.setup_entities(); m.remove_alternative_conformations()
        mp = max((ch.get_polymer() for ch in m[0]
                  if ch.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna), key=len)
        sup = gemmi.calculate_superposition(pl, mp, gemmi.PolymerType.Rna, gemmi.SupSelect.CaP)
        m[0].transform_pos_and_adp(sup.transform)
        cterm = [(r.seqid.num, r.name,
                  [(a.name, a.element.name, a.pos, 0.0) for a in r if a.element.name != "H"])
                 for r in mp if r.name in ("A", "C", "G", "U") and CTERM_LO <= r.seqid.num <= CTERM_HI]
    ct_xyz = np.array([[a[2].x, a[2].y, a[2].z] for _, _, ats in cterm for a in ats]) \
        if cterm else np.zeros((0, 3))
    tCterm = cKDTree(ct_xyz) if len(ct_xyz) else None

    print(f"\n  {'model':6s}{'tier':24s}{'MG':>5s}{'K':>5s}{'NA':>4s}{'HOH':>7s}"
          f"{'minIon':>8s}{'expMg':>7s}")
    for i, (label, mode, mt, mnb, kt, knb, nt, nnb, wt, shell) in enumerate(MODELS, start=1):
        # experimental ions first, unconditionally
        cands = [dict(xyz=p, sp="MG", t_support=99, c_blind_methods=[], c_hits=0, exp=True)
                 for p in exp_mg]
        thr = {"MG": (mt, mnb), "K": (kt, knb), "NA": (nt, nnb)}
        na_seen = 0
        for s in sorted(arbitrated, key=lambda a: -priority(a)):
            tmin, nbmin = thr[s["sp"]]
            if s["t_support"] < tmin or len(s["c_blind_methods"]) < nbmin:
                continue
            x = np.array(s["xyz"])
            if not ion_ok(x):
                continue
            if not coordinates_something(x, s["sp"]):
                continue
            if tCterm is not None and tCterm.query(x)[0] < CTERM_EXCLUSION:
                continue
            if s["sp"] == "NA":
                if na_seen >= NA_CAP[i - 1]:
                    continue
                na_seen += 1
            cands.append(dict(s))
        cands.sort(key=lambda s: (-1e9 if s.get("exp") else 0) - priority(s))
        ions, ion_xyz = greedy(cands, MIN_ION_ION)
        cap = MAX_IONS[i - 1]
        if cap is not None and len(ions) > cap:
            ions = ions[:cap]                      # already priority-ordered
            ion_xyz = [np.asarray(c["xyz"]) for c in ions]
        if NEUTRALISE[i - 1]:
            n_phos = sum(1 for _, _, ats in rna_atoms for a in ats if a[1] == "P") \
                     + sum(1 for _, _, ats in cterm for a in ats if a[1] == "P")
            q, cut = 0, len(ions)
            for j, c in enumerate(ions):
                dq = 2 if c["sp"] == "MG" else 1
                if q + dq > n_phos:
                    cut = j
                    break
                q += dq
            ions = ions[:cut]
            ion_xyz = [np.asarray(c["xyz"]) for c in ions]

        if mode == "evidence":
            wat = [s for s in sites["HOH"] if s["t_support"] >= wt]
            wat = [s for s in wat if water_ok(np.array(s["xyz"]), per_el)]
            if tCterm is not None:
                wat = [s for s in wat if tCterm.query(np.array(s["xyz"]))[0] >= CTERM_EXCLUSION]
            wat.sort(key=lambda s: -priority(s))
            wsel, _ = greedy(wat, MIN_WAT_WAT, seeded=[np.asarray(x) for x in ion_xyz])
            water_xyz = [np.asarray(s["xyz"]) for s in wsel]
            water_b = [round(20 + 60 * min(s["t_support"], 8) / 8, 2) for s in wsel]
            hexw = hexahydrate([(c["xyz"], c["sp"]) for c in ions], per_el, tCterm,
                               list(ion_xyz), water_xyz)
            water_xyz += hexw
            # geometrically determined, so scored high but below the ions
            water_b += [85.00 - (k % 11) * 1.0 for k in range(len(hexw))]
        else:
            # The lattice sits on a 3.10 A grid, so it essentially never lands at
            # the 2.07 A Mg-O(water) distance and the first hydration shell was
            # being missed entirely (18-23 shell waters out of ~3800). Place the
            # shell explicitly first, then let the lattice fill around it.
            hexw = hexahydrate([(c["xyz"], c["sp"]) for c in ions], per_el, tCterm,
                               list(ion_xyz), [])
            bulk = list(lattice_waters(heavy_xyz, per_el, shell,
                                       list(ion_xyz), tCterm, shell_waters=hexw))
            water_xyz = hexw + bulk
            water_b = ([85.00 - (k % 11) * 1.0 for k in range(len(hexw))]
                       + [50.00 + (k % 17) * 1.5 for k in range(len(bulk))])

        counts = {sp: sum(1 for c in ions if c["sp"] == sp) for sp in ("MG", "K", "NA")}
        allion = np.asarray(ion_xyz)
        d = np.linalg.norm(allion[:, None] - allion[None, :], axis=2)
        np.fill_diagonal(d, 9e9)
        nexp = sum(1 for c in ions if c.get("exp"))
        write_ts(args.out / f"{TARGET}_model{i}.txt", i, label, rna_atoms, cterm,
                 ions, water_xyz, water_b)
        print(f"  {i:<6d}{label:24s}{counts['MG']:5d}{counts['K']:5d}{counts['NA']:4d}"
              f"{len(water_xyz):7d}{d.min():7.2f}A{nexp:6d}/21")


def bfactor(s):
    if s.get("exp"):
        return 99.00
    t = min(s["t_support"], 10) / 10.0
    c = min(len(s["c_blind_methods"]), 3) / 3.0
    return round(20.0 + 75.0 * (0.6 * t + 0.4 * c), 2)


def occ_of(s):
    if s.get("exp"):
        return 1.00
    nb = len(s["c_blind_methods"])
    if s["t_support"] >= 3 and nb >= 2:
        return 1.00
    if s["t_support"] >= 2 or nb >= 2:
        return 0.75
    return 0.50


def write_ts(path, idx, label, rna_atoms, cterm, ions, water_xyz, water_b):
    ELEM = {"MG": "MG", "K": "K", "NA": "NA"}
    with open(path, "w") as fh:
        fh.write("PFRMAT TS\n")
        fh.write(f"TARGET {TARGET}\n")
        fh.write(f"AUTHOR {casp_identity.group_id()}\n")
        for line in [
            "Solvent shell merged from two independent evidence streams on a common",
            "site list. Stream 1: 37 experimental structures of the same group IIC",
            "intron (92.3-100% id, incl. 9C6I/9C6J) superposed on 9C6I, solvent pooled.",
            "Stream 2: 400 co-folding models (AlphaFold3, Boltz-2, Protenix default and",
            "20250630) over four Mg/K/Na stoichiometry points (10/20/35/55 Mg); ions",
            "with impossible intra-model ion-ion contacts (<3.0 A) discarded.",
            "Sites clustered at 1.5 A (ions) / 1.0 A (water), then selected greedily",
            "across all species at once with a 3.5 A minimum ion-ion separation.",
            "The 21 Mg modelled in 9C6I are force-included in every model.",
            f"This model: {label}.",
            "RNA 1-393 from PDB 9C6I. Residues 394-417 are disordered in every",
            "100%-identical construct and independent experiments differ there by 17 A",
            "RMSD, so that backbone is at occupancy 0.00 and carries no solvent.",
        ]:
            fh.write(f"METHOD {line}\n")
        fh.write(f"MODEL  {idx}\n")
        fh.write("PARENT 9c6i\n")
        n = 0
        for tgt, resn, atoms in rna_atoms:
            for aname, el, pos, b in atoms:
                n += 1
                fh.write(_atom("ATOM", n, aname, resn, "0", tgt, pos, 1.00,
                               min(max(b, 0.0), 100.0), el))
        for tgt, resn, atoms in cterm:
            for aname, el, pos, b in atoms:
                n += 1
                fh.write(_atom("ATOM", n, aname, resn, "0", tgt, pos, 0.00, 1.00, el))
        fh.write("TER\n")
        rid = 500
        for c in ions:
            n += 1; rid += 1
            p = gemmi.Position(*c["xyz"])
            fh.write(_atom("HETATM", n, ELEM[c["sp"]], c["sp"], "0", rid, p,
                           occ_of(c), bfactor(c), ELEM[c["sp"]]))
        for k, (p, b) in enumerate(zip(water_xyz, water_b)):
            n += 1; rid += 1
            fh.write(_atom("HETATM", n, "O", "HOH", "0", rid, gemmi.Position(*p),
                           1.00, round(float(b), 2), "O"))
        fh.write("END\n")


def _atom(rec, serial, aname, resn, chain, resi, pos, occ, b, elem):
    # Columns 13-16 hold the atom name, and the element symbol is RIGHT-justified
    # within 13-14: a one-letter element gets a leading space (" K  ", " O  "),
    # a two-letter one does not ("MG  ", "NA  "). Padding everything by one space
    # pushes NA's N into column 14, so any parser that falls back to columns
    # 13-14 -- the legacy rule, still live in plenty of code -- reads " N" and
    # calls our sodium a nitrogen, and " M" for magnesium. Species identity is
    # the scored quantity on this target, so this is not cosmetic.
    if len(aname) >= 4:
        an = aname[:4]
    elif len(elem.strip()) == 2:
        an = f"{aname:<4s}"
    else:
        an = f" {aname:<3s}"
    return (f"{rec:<6s}{serial:5d} {an}{'':1s}{resn:>3s} {chain:1s}{resi:4d}{'':4s}"
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{occ:6.2f}{b:6.2f}{'':10s}{elem:>2s}{'':2s}\n")


if __name__ == "__main__":
    main()
