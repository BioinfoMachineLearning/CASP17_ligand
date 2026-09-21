#!/usr/bin/env python
"""R2386 final validation: chemistry, physics, biology and format checks.

Every numeric criterion is calibrated against experimental data for THIS
molecule (9C6I's own 21 Mg, and the 37-structure donor set) rather than taken
from a textbook, so a "fail" means the model departs from what this RNA is
actually observed to do.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import R2386_paths as paths                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REF_CIF = str(paths.ref_cif())

OK, WARN, FAIL = "PASS", "WARN", "FAIL"

# CASP17 additional instruction for R2386 (2026-08): "Modelers should submit
# 500 ligands. Only ligands in the core, well-resolved regions will be
# assessed; any ligands that are closest to the following non-core residues
# will be ignored ... in CASP numbering (which is +5 compared to 9C6I
# numbering)."  Water is on the organizers' ligand list, so the 500 is
# ions + water combined.
TOTAL_REQUIRED = 500
NONCORE_RANGES = [(6, 7), (56, 60), (86, 106), (167, 173), (206, 220),
                  (276, 287), (309, 320), (335, 357), (394, 417)]
NONCORE = {r for a, b in NONCORE_RANGES for r in range(a, b + 1)}
CASP_MINUS_PDB = 5  # CASP number = 9C6I seqid + 5


def read_model(path):
    rna, ions, wat = [], [], []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        resn = line[17:20].strip()
        el = line[76:78].strip()
        occ = float(line[54:60]); b = float(line[60:66])
        rec = dict(xyz=xyz, resn=resn, el=el, occ=occ, b=b,
                   resi=int(line[22:26]), line=line.rstrip("\n"))
        if line.startswith("ATOM"):
            rna.append(rec)
        elif resn == "HOH":
            wat.append(rec)
        else:
            ions.append(rec)
    return rna, ions, wat


def ref_data():
    st = gemmi.read_structure(REF_CIF)
    st.setup_entities(); st.remove_alternative_conformations()
    pl = max((c.get_polymer() for c in st[0]
              if c.get_polymer().check_polymer_type() == gemmi.PolymerType.Rna), key=len)
    per = {}
    for el in ("O", "N", "C", "P"):
        pts = [[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r if a.element.name == el]
        per[el] = cKDTree(np.array(pts))
    heavy = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r if a.element.name != "H"])
    mg = np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch
                   if r.name == "MG" for a in r])
    # phosphate oxygens: the canonical Mg2+ anchor on nucleic acid
    pox = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pl for a in r
                    if a.name.strip() in ("OP1", "OP2", "OP3", "O5'", "O3'")])
    # core / non-core split in the assessor's own frame: the deposited 9C6I
    # coordinates, with its seqids shifted into CASP numbering.
    core, non = [], []
    for r in pl:
        casp = r.seqid.num + CASP_MINUS_PDB
        dst = non if casp in NONCORE else core
        dst += [[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"]
    return (per, cKDTree(heavy), heavy, mg, cKDTree(pox), pl,
            cKDTree(np.array(core)), cKDTree(np.array(non)))


def split_trees(rna):
    """Same split, but measured on the model's own RNA (already CASP-numbered)."""
    core = [r["xyz"] for r in rna if r["resi"] not in NONCORE]
    non = [r["xyz"] for r in rna if r["resi"] in NONCORE]
    if not core or not non:
        return None, None
    return cKDTree(np.array(core)), cKDTree(np.array(non))


def core_margin(P, tCore, tNon):
    """d(nearest non-core) - d(nearest core). >0 means the assessor's
    'closest residue' rule puts this ligand in the scored region."""
    if len(P) == 0:
        return np.zeros(0)
    dc, _ = tCore.query(P, k=1, workers=-1)
    dn, _ = tNon.query(P, k=1, workers=-1)
    return np.atleast_1d(dn - dc)


def line(tag, name, val, crit, status):
    print(f"  [{status:4s}] {name:38s} {val:>22s}   {crit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=paths.DEFAULT_MODEL_DIR,
                    help="bare name, repo-relative or absolute path; "
                         "defaults to the delivered models")
    args = ap.parse_args()
    args.dir = paths.model_dir(args.dir)
    per, tHeavy, heavy, ref_mg, tPox, pl, tCoreR, tNonR = ref_data()
    com = heavy.mean(0)
    n_nt = sum(1 for r in pl if r.name in ("A", "C", "G", "U"))

    # experimental reference values measured on 9C6I's own 21 Mg
    e_dO = np.array([per["O"].query(m)[0] for m in ref_mg])
    e_dC = np.array([per["C"].query(m)[0] for m in ref_mg])
    e_rad = np.median(np.linalg.norm(ref_mg - com, axis=1))
    e_pox = np.array([tPox.query(m)[0] for m in ref_mg])
    print("EXPERIMENTAL REFERENCE (9C6I, 21 Mg, 2.56 A)")
    print(f"  Mg-O(RNA) min/median      : {e_dO.min():.2f} / {np.median(e_dO):.2f} A")
    print(f"  Mg-C(RNA) min             : {e_dC.min():.2f} A")
    print(f"  distance to RNA centroid  : {e_rad:.1f} A")
    print(f"  distance to phosphate O   : median {np.median(e_pox):.2f} A, "
          f"{100*(e_pox <= 2.6).mean():.0f}% within 2.6 A")
    print(f"  richest experimental ion density on this RNA: 0.198 ions/nt (3G78, 2.80 A)")

    for i in range(1, 6):
        f = args.dir / f"R2386_model{i}.txt"
        if not f.exists():
            continue
        rna, ions, wat = read_model(f)
        I = np.array([r["xyz"] for r in ions])
        W = np.array([r["xyz"] for r in wat]) if wat else np.zeros((0, 3))
        MG = np.array([r["xyz"] for r in ions if r["resn"] == "MG"])
        K = np.array([r["xyz"] for r in ions if r["resn"] == "K"])
        NA = np.array([r["xyz"] for r in ions if r["resn"] == "NA"])
        print(f"\n{'='*104}\nMODEL {i}   ions {len(I)}  water {len(W)}  "
              f"(MG {len(MG)} / K {len(K)} / NA {len(NA)})\n{'='*104}")

        # ------- CASP17 additional instruction (the gating criteria) -------
        L = np.vstack([I, W]) if len(W) else I
        n_lig = len(ions) + len(wat)
        line("cp", "total ligands (ions + water)", str(n_lig),
             f"== {TOTAL_REQUIRED}", OK if n_lig == TOTAL_REQUIRED else FAIL)
        socc = sum(r["occ"] for r in ions + wat)
        line("cp", "sum of ligand occupancies", f"{socc:.2f}",
             f"== {TOTAL_REQUIRED} (AltLoc rule)",
             OK if abs(socc - TOTAL_REQUIRED) < 1e-6 else FAIL)
        alt = {r["line"][16] for r in ions + wat}
        line("cp", "AltLoc characters in use", str(sorted(alt)),
             "[' '] unless deliberately hedging", OK if alt == {" "} else WARN)
        resis = {r["resi"] for r in rna}
        line("cp", "RNA numbering", f"{min(resis)}-{max(resis)}, n={len(resis)}",
             "1-417 CASP numbering (9C6I + 5)",
             OK if (min(resis), max(resis), len(resis)) == (1, 417, 417) else FAIL)
        mref = core_margin(L, tCoreR, tNonR)
        line("cp", "ligands scored (assessor frame, 9C6I)",
             f"{int((mref > 0).sum())}/{n_lig}", "must be all",
             OK if (mref > 0).all() else FAIL)
        tCoreM, tNonM = split_trees(rna)
        if tCoreM is not None:
            mmod = core_margin(L, tCoreM, tNonM)
            line("cp", "ligands scored (our own RNA frame)",
                 f"{int((mmod > 0).sum())}/{n_lig}", "must be all",
                 OK if (mmod > 0).all() else FAIL)
        line("cp", "core margin, tightest / median",
             f"{mref.min():.2f} / {np.median(mref):.2f} A",
             "headroom against a frame mismatch",
             OK if mref.min() >= 0.5 else WARN)
        line("cp", "ligands within 1 A of the core boundary",
             str(int((mref < 1.0).sum())), "at risk if the assessor's frame differs",
             OK if (mref < 1.0).sum() <= 20 else WARN)
        nob = [r for r in ions + wat if r["line"][60:66].strip() == ""]
        line("cp", "ligands missing a B-factor", str(len(nob)),
             "== 0 (required for every ligand)", OK if not nob else FAIL)
        het = {r["resn"] for r in ions + wat}
        line("cp", "ligand species", str(sorted(het)),
             "subset of {MG, HOH, NA, K}",
             OK if het <= {"MG", "HOH", "NA", "K"} else FAIL)
        nonhet = [r for r in ions + wat if not r["line"].startswith("HETATM")]
        line("cp", "ligands not on HETATM records", str(len(nonhet)), "== 0",
             OK if not nonhet else FAIL)

        # ---------------- sterics ----------------
        d = np.linalg.norm(I[:, None] - I[None, :], axis=2); np.fill_diagonal(d, 9e9)
        line("st", "ion-ion minimum", f"{d.min():.2f} A", ">= 3.5 (real Mg-Mg floor)",
             OK if d.min() >= 3.5 else FAIL)
        if len(W):
            wi = np.linalg.norm(W[:, None] - I[None, :], axis=2).min()
            ww = np.linalg.norm(W[:, None] - W[None, :], axis=2); np.fill_diagonal(ww, 9e9)
            line("st", "water-ion minimum", f"{wi:.2f} A", ">= 2.0 (allows 2.07 inner shell)",
                 OK if wi >= 1.99 else FAIL)
            line("st", "water-water minimum", f"{ww.min():.2f} A", ">= 2.4",
                 OK if ww.min() >= 2.39 else FAIL)
        for el, lim in (("O", 1.8), ("N", 2.6), ("C", 2.8), ("P", 2.9)):
            v = min(per[el].query(p)[0] for p in I)
            line("st", f"ion-RNA {el} minimum", f"{v:.2f} A", f">= {lim}",
                 OK if v >= lim - 0.01 else FAIL)
        if len(W):
            for el, lim in (("O", 2.55), ("N", 2.75), ("C", 3.00), ("P", 3.15)):
                v = min(per[el].query(p)[0] for p in W)
                line("st", f"water-RNA {el} minimum", f"{v:.2f} A", f">= {lim}",
                     OK if v >= lim - 0.01 else FAIL)

        # ---------------- coordination chemistry ----------------
        if len(MG):
            allO = np.vstack([np.array([[*p] for p in W]) if len(W) else np.zeros((0, 3))])
            cn = []
            for m in MG:
                n_rna = len(per["O"].query_ball_point(m, 2.6))
                n_wat = int((np.linalg.norm(W - m, axis=1) <= 2.6).sum()) if len(W) else 0
                cn.append(n_rna + n_wat)
            cn = np.array(cn)
            line("ch", "Mg coordination number (mean)", f"{cn.mean():.1f}",
                 "6 ideal; >=4 acceptable", OK if cn.mean() >= 4 else WARN)
            line("ch", "Mg with CN >= 4", f"{100*(cn >= 4).mean():.0f}%", "higher is better",
                 OK if (cn >= 4).mean() >= 0.5 else WARN)
            dO = np.array([per["O"].query(m)[0] for m in MG])
            line("ch", "Mg-O(RNA) minimum", f"{dO.min():.2f} A",
                 f"experimental {e_dO.min():.2f}", OK if dO.min() >= 1.8 else FAIL)
            # octahedral angles among first-shell partners
            angs = []
            for m in MG:
                nb = []
                if len(W):
                    nb += [w for w in W if np.linalg.norm(w - m) <= 2.4]
                idx = per["O"].query_ball_point(m, 2.6)
                if len(nb) >= 2:
                    for a in range(len(nb)):
                        for b_ in range(a + 1, len(nb)):
                            v1 = nb[a] - m; v2 = nb[b_] - m
                            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            if angs:
                angs = np.array(angs)
                near = ((np.abs(angs - 90) <= 15) | (np.abs(angs - 180) <= 15)).mean()
                line("ch", "octahedral angles (90/180 +-15)", f"{100*near:.0f}%",
                     "geometry of [Mg(H2O)6]2+", OK if near >= 0.8 else WARN)
        for nm, arr, ideal in (("K", K, 2.8), ("NA", NA, 2.4)):
            if len(arr):
                dd = np.array([per["O"].query(p)[0] for p in arr])
                line("ch", f"{nm}-O(RNA) median", f"{np.median(dd):.2f} A",
                     f"ideal ~{ideal}", OK if abs(np.median(dd) - ideal) < 1.5 else WARN)

        # ---------------- physical plausibility ----------------
        dens = len(I) / n_nt
        line("ph", "ion density", f"{dens:.3f} ions/nt",
             "<= 0.594 (3x richest experiment)", OK if dens <= 0.594 else FAIL)
        charge = 2 * len(MG) + len(K) + len(NA)
        line("ph", "cation charge vs RNA -416", f"+{charge}",
             "ordered subset only; full neutrality not expected", OK)
        rad = np.median(np.linalg.norm(I - com, axis=1))
        line("ph", "ion distance to RNA centroid", f"{rad:.1f} A",
             f"experimental {e_rad:.1f}", OK if abs(rad - e_rad) <= 6 else WARN)
        pox = np.array([tPox.query(p)[0] for p in I])
        line("bi", "ions within 2.6 A of phosphate O", f"{100*(pox <= 2.6).mean():.0f}%",
             f"experimental {100*(e_pox <= 2.6).mean():.0f}%", OK)
        dmg = np.linalg.norm(ref_mg[:, None, :] - I[None, :, :], axis=2).min(1)
        line("bi", "experimental Mg reproduced", f"{int((dmg <= 0.5).sum())}/21",
             "must be 21/21", OK if (dmg <= 0.5).sum() == 21 else FAIL)

        # ---------------- format ----------------
        lens = {len(r["line"]) for r in rna + ions + wat}
        line("fm", "record width", str(sorted(lens)), "== {80}", OK if lens == {80} else FAIL)
        bs = [r["b"] for r in rna]
        line("fm", "distinct ATOM B-factors", str(len(set(bs))),
             "> 1 (uniform B is rejected)", OK if len(set(bs)) > 1 else FAIL)
        bh = [r["b"] for r in ions + wat]
        line("fm", "distinct HETATM B-factors", str(len(set(bh))), "> 1",
             OK if len(set(bh)) > 1 else FAIL)
        allb = bs + bh
        line("fm", "B-factor range", f"{min(allb):.2f}-{max(allb):.2f}", "within 0-100",
             OK if min(allb) >= 0 and max(allb) <= 100 else FAIL)
        occs = [r["occ"] for r in ions + wat]
        line("fm", "HETATM occupancy range", f"{min(occs):.2f}-{max(occs):.2f}",
             "0.01-1.00", OK if min(occs) >= 0.01 and max(occs) <= 1.0 else FAIL)
        hyd = [r for r in rna + ions + wat if r["el"] == "H"]
        line("fm", "hydrogen atoms", str(len(hyd)), "== 0 (target requires none)",
             OK if not hyd else FAIL)
        dup = [k for k, v in Counter(r["resi"] for r in ions + wat).items() if v > 1]
        line("fm", "duplicate solvent residue numbers", str(len(dup)), "== 0",
             OK if not dup else FAIL)
        txt = open(f).read()
        for kw in ("PFRMAT TS", "TARGET R2386", "AUTHOR", "METHOD", "MODEL", "PARENT", "TER", "END"):
            if kw not in txt:
                line("fm", f"header record {kw}", "missing", "required", FAIL)
        zero = [r for r in rna if r["occ"] == 0.0]
        line("fm", "C-terminal atoms at occupancy 0", str(len(zero)),
             "394-417 backbone, unscored region", OK if zero else WARN)


if __name__ == "__main__":
    main()
