#!/usr/bin/env python
"""One ranked water list for R2386 -- no reserved quotas, no tier order.

WHY THIS REPLACES THE TIER SCHEME
---------------------------------
The previous code filled water in tiers: all Mg first-shell water, then all
template-consensus water, then a pocket lattice. The order was a free parameter,
and it had to be tuned per model because at 120 ions the shell alone (~420
waters) ate the entire 380-water budget and left ZERO template water -- our
highest-precision source. The obvious patch, "cap each Mg at 2 waters", was
rejected for the right reason: 2 is a number chosen to make the accounting come
out, not a statement about which water is actually ordered.

So there is no quota here. Every candidate -- shell, template, pocket -- gets a
score on ONE scale and a single greedy pass spends the budget in score order.
The number of waters an Mg keeps is an OUTPUT of the ranking.

WHAT THE SCALE MEANS
--------------------
    score = W_TEMPLATE * t_support      evidence: depositors saw water here
          + W_MGSHELL  * first_shell    chemistry: held by a 2+ cation at 2.07 A
          + W_HB_RNA   * n_hbond_rna    chemistry: RNA N/O partners within 3.4 A
          + W_BURIAL   * n_heavy_6A     chemistry: enclosed, so it cannot diffuse

The two chemistry terms are what make this answer the real objection: a first-
shell water that ALSO donates a hydrogen bond to a phosphate oxygen is held from
two sides and is genuinely ordered, while a first-shell water pointing into bulk
solvent is held from one side and is not. The old code could not tell them apart
because it counted shell waters instead of scoring them. Now they compete, and
the bulk-facing ones lose to template water -- which is the behaviour we wanted
from the tier order, obtained from chemistry instead of from a hand-set order.

The same reasoning is pushed one level down into the octahedron: hexahydrate()
used to pick the orientation that placed the MOST waters. It now picks the
orientation with the highest total chemistry score, i.e. the one that points its
waters at RNA hydrogen-bond partners. Same six positions, better rotation.

W_MGSHELL is the one number that cannot be derived, so it is not guessed: it is
swept against held-out experimental solvent by R2386_bench_composition.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import R2386_build_models as v2

W_TEMPLATE = 6.0
W_HB_RNA = 2.0
W_BURIAL = 0.06
HB_CUT = 3.4
BURIAL_CUT = 6.0

# Set by the blind sweep in R2386_bench_composition.py. Reported there as
# "mgshell w=..."; the winning row came from the water-source benchmark.
W_MGSHELL_DEFAULT = 6.0

MG_O = 2.07
_AX = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def chem_score(P, hb_tree, tHeavy):
    """Hydrogen-bond count to RNA N/O plus burial, for arbitrary positions."""
    P = np.atleast_2d(np.asarray(P, float))
    if not len(P):
        return np.zeros(0)
    nhb = np.array([len(hb_tree.query_ball_point(p, HB_CUT)) for p in P], float)
    bur = np.array([len(tHeavy.query_ball_point(p, BURIAL_CUT)) for p in P], float)
    return W_HB_RNA * nhb + W_BURIAL * bur


def hexahydrate_scored(ions, per_el, tCterm, hb_tree, tHeavy, n_orient=48, seed=2386):
    """[Mg(H2O)6]2+ first shells, orientation chosen by chemistry not by count.

    Returns [(xyz, chem_score)] for every placeable shell water. Waters generated
    for earlier ions block later ones (they are physically there), but nothing is
    reserved: the caller ranks these against every other candidate.
    """
    rng = np.random.default_rng(seed)
    ions_arr = np.asarray([np.asarray(c) for c, _ in ions]) if ions else np.zeros((0, 3))
    placed, out = [], []
    for c, sp in ions:
        if sp != "MG":
            continue                       # K+/Na+ shells are not ordered here
        best, best_score = [], -np.inf
        for _ in range(n_orient):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            keep = []
            for p in np.asarray(c) + (R @ _AX.T).T * MG_O:
                if not v2.water_ok(p, per_el):
                    continue
                if tCterm is not None and tCterm.query(p)[0] < v2.CTERM_EXCLUSION:
                    continue
                if len(ions_arr) and np.linalg.norm(ions_arr - p, axis=1).min() < v2.MIN_WAT_ION:
                    continue
                pool = placed + keep
                if pool and np.linalg.norm(np.asarray(pool) - p, axis=1).min() < v2.MIN_WAT_WAT:
                    continue
                keep.append(p)
            if not keep:
                continue
            # THE CHANGE: rank orientations by total chemistry, not by len(keep).
            # A 4-water orientation whose waters all touch RNA beats a 6-water
            # orientation pointing into bulk solvent.
            s = float(chem_score(np.asarray(keep), hb_tree, tHeavy).sum())
            if s > best_score:
                best, best_score = keep, s
        if best:
            placed.extend(best)
            out.extend((p, float(chem_score(p, hb_tree, tHeavy)[0])) for p in best)
    return out


def build_water(n_wat, ions, per_el, tCterm, tmpl, pocketP, pocketS,
                hb_tree, tHeavy, core_fn, w_mgshell=W_MGSHELL_DEFAULT,
                n_orient=48):
    """Rank every water candidate on one scale and greedily spend the budget.

    ions      [{"xyz": [x,y,z], "sp": "MG"|"K"|"NA"}, ...]  already placed
    tmpl      template-consensus water sites with a "t_support" field
    pocketP/S pocket lattice positions and their chemistry scores
    core_fn   callable(P) -> bool mask, the assessor's nearest-residue rule

    Returns (waters, stats) where waters is [(xyz, score, kind)].
    """
    ion_xyz = [np.asarray(s["xyz"]) for s in ions]
    I = np.asarray(ion_xyz) if ion_xyz else np.zeros((0, 3))

    cand = []                                   # (score, xyz, kind)

    shell = hexahydrate_scored([(s["xyz"], s["sp"]) for s in ions], per_el,
                               tCterm, hb_tree, tHeavy, n_orient=n_orient) if ions else []
    if shell:
        sp = np.array([p for p, _ in shell])
        keep = core_fn(sp)
        for (p, cs), k in zip(shell, keep):
            if k:
                cand.append((w_mgshell + cs, np.asarray(p), "shell"))

    if tmpl:
        tp = np.array([s["xyz"] for s in tmpl], float)
        cs = chem_score(tp, hb_tree, tHeavy)
        for s, p, c in zip(tmpl, tp, cs):
            cand.append((W_TEMPLATE * float(s["t_support"]) + float(c), p, "template"))

    if len(pocketP):
        for p, c in zip(pocketP, pocketS):
            cand.append((float(c), np.asarray(p), "pocket"))

    cand.sort(key=lambda t: -t[0])

    # Water-vs-ion uses MIN_WAT_ION (2.0 A): below that is a clash for any
    # cation, above it the water is simply coordinated. Water-vs-water uses the
    # stricter MIN_WAT_WAT (2.4 A). Using the water-water floor against ions is
    # the trap that once rejected every shell water via its own Mg.
    acc, out = [], []
    stats = {"shell": 0, "template": 0, "pocket": 0}
    for sc, p, kind in cand:
        if len(out) >= n_wat:
            break
        if len(I) and np.linalg.norm(I - p, axis=1).min() < v2.MIN_WAT_ION:
            continue
        if acc:
            A = np.asarray(acc)
            if np.abs(A - p).max(1).min() < v2.MIN_WAT_WAT:
                if np.linalg.norm(A - p, axis=1).min() < v2.MIN_WAT_WAT:
                    continue
        acc.append(p)
        out.append((p, sc, kind))
        stats[kind] += 1
    return out, stats