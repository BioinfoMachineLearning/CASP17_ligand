#!/usr/bin/env python3
"""Check that every method's input for a target agrees, and that MSA wiring survived.

Run this after ANY `*_input_preparation.py` invocation. Prep regenerates inputs
from the SMILES/sequence files, which silently drops hand-injected MSA paths
(`msa:` in the Boltz-2 YAML, `unpairedMsaPath` in the Protenix JSON). Without
those the methods fall back to searching their own MSA — the run still succeeds,
just slower and with a different MSA than intended, so nothing else catches it.

Checks per target:
  - sequence + SMILES identical (canonical form) across TSV, ensemble_inputs.csv,
    and all four method inputs
  - Boltz-2 `msa:` and Protenix `unpairedMsaPath` exist on disk, and their query
    record matches the target sequence
  - AF3 staged data.json (if present) has the right name/seeds and a non-trivial MSA
  - get_max_heavy_atoms resolves, so clustering uses SuCOS and not the
    single-atom RMSD branch

Usage:
    python casp17/scripts/verify_target_inputs.py T2413v1 T2414v1
    python casp17/scripts/verify_target_inputs.py --series casp17_T --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SERIES_DIR = {"casp17_T": "CASP17_T", "casp17_R": "CASP17_R",
              "casp17_M": "CASP17_M", "casp17_L": "CASP17_L"}


def a3m_query(path: Path) -> str:
    """First record of an a3m, with a3m insertion columns and gaps stripped."""
    buf, lines = [], path.read_text().splitlines()
    for line in lines[1:]:
        if line.startswith(">"):
            break
        buf.append(line.strip())
    return "".join(c for c in "".join(buf) if not c.islower() and c != "-")


def verify(target: str, series: str, require_af3: bool) -> list[str]:
    R = PROJECT_ROOT
    sdir = SERIES_DIR.get(series, series.upper())
    fails: list[str] = []

    def chk(cond, msg):
        print(("  OK   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(f"{target}: {msg}")

    print(f"\n===== {target} =====")
    seq = (R / f"data/casp17_data/sequences/{sdir}/{target}.txt").read_text().splitlines()[1].strip()
    tsv = list(csv.reader((R / f"data/casp17_data/smiles/{sdir}/{target}.tsv").open(), delimiter="\t"))
    smi = tsv[1][2].strip()
    ref = Chem.CanonSmiles(smi)

    csv_path = R / f"data/test_cases/{series}/ensemble_inputs.csv"
    rows = [l for l in csv_path.read_text().splitlines() if l.startswith(target + ",")]
    chk(len(rows) == 1, f"exactly one ensemble_inputs.csv row (found {len(rows)})")
    if rows:
        f = rows[0].split(",")
        chk(len(f) == 7, f"ensemble_inputs.csv has 7 fields (found {len(f)}) — no commas in notes")
        chk(f[2] == seq, f"ensemble_inputs.csv sequence ({len(f[2])} aa)")
        chk(Chem.CanonSmiles(f[4]) == ref, "ensemble_inputs.csv SMILES")

    raw = R / f"data/casp17_data/raw/{target}.smiles.txt"
    chk(raw.is_file() and raw.read_text() ==
        (R / f"data/casp17_data/smiles/{sdir}/{target}.tsv").read_text(),
        "raw/ copy present for LG_validation")

    # ---- Boltz-2 ----
    y = (R / f"data/test_cases/{series}/boltz2_inputs/{target}_input.yaml").read_text()
    chk(seq in y, "boltz2 sequence")
    chk(Chem.CanonSmiles([l.split("'")[1] for l in y.splitlines() if "smiles:" in l][0]) == ref,
        "boltz2 SMILES")
    msa_lines = [l.split("msa:")[1].strip() for l in y.splitlines() if "msa:" in l]
    if msa_lines:
        p = Path(msa_lines[0])
        chk(p.is_file(), f"boltz2 msa file exists ({p.name})")
        if p.is_file():
            with p.open() as fh:
                rr = list(csv.reader(fh))
            chk(rr[1][1] == seq, f"boltz2 msa query == target sequence (depth {len(rr)-1})")
    else:
        print("  NOTE  boltz2 has no `msa:` — will search via colabfold at runtime")

    # ---- Protenix ----
    pj = json.loads((R / f"data/test_cases/{series}/protenix_inputs/{target}.json").read_text())[0]
    pc = pj["sequences"][0]["proteinChain"]
    chk(pj["name"] == target, "protenix name")
    chk(pc["sequence"] == seq, "protenix sequence")
    lig = pj["sequences"][1]["ligand"]["ligand"]
    chk(Chem.CanonSmiles(lig) == ref, "protenix SMILES")
    if "unpairedMsaPath" in pc:
        p = Path(pc["unpairedMsaPath"])
        chk(p.is_file(), f"protenix unpairedMsaPath exists ({p.name})")
        if p.is_file():
            chk(a3m_query(p) == seq, "protenix a3m query == target sequence")
    else:
        print("  NOTE  protenix has no `unpairedMsaPath` — will run its own MSA search")

    # ---- AF3 ----
    for rnd in ("r1", "r2"):
        dj = R / f"outputs/alphafold3/{series}_{rnd}/{target}/{target}_{rnd}_data.json"
        if not dj.is_file():
            if require_af3:
                chk(False, f"af3 {rnd} data.json missing")
            else:
                print(f"  NOTE  af3 {rnd} data.json not staged — phase1 will run from scratch")
            continue
        d = json.loads(dj.read_text())
        n = int(rnd[1:])
        p = d["sequences"][0]["protein"]
        chk(d["name"] == f"{target}_{rnd}", f"af3 {rnd} name")
        chk(d["modelSeeds"] == list(range((n - 1) * 10 + 1, n * 10 + 1)), f"af3 {rnd} modelSeeds")
        chk(p["sequence"] == seq, f"af3 {rnd} sequence")
        chk(Chem.CanonSmiles(d["sequences"][1]["ligand"]["smiles"]) == ref, f"af3 {rnd} SMILES")
        chk(p.get("unpairedMsa", "").count(">") > 100,
            f"af3 {rnd} MSA unpaired={p.get('unpairedMsa','').count('>')} "
            f"paired={p.get('pairedMsa','').count('>')} templates={len(p.get('templates', []))}")

    # ---- SeedFold ----
    sf = R / f"data/test_cases/{series}/seedfold_inputs/{target}.json"
    if sf.is_file():
        sj = json.loads(sf.read_text())
        chk(len(sj) > 0, f"seedfold job count = {len(sj)}")
        ents = sj[0]["entities"]
        chk(ents[0]["sequence"] == seq, "seedfold sequence")
        chk(Chem.CanonSmiles(ents[1]["sequence"]) == ref, "seedfold SMILES")

    from casp17_ligand.analysis.evaluate_topn_clusters import get_max_heavy_atoms
    ha = get_max_heavy_atoms(target)
    chk(ha == Chem.MolFromSmiles(smi).GetNumHeavyAtoms(),
        f"get_max_heavy_atoms = {ha} (>2 routes to SuCOS clustering)")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="target ids to check")
    ap.add_argument("--series", default="casp17_T")
    ap.add_argument("--all", action="store_true", help="check every target in the series")
    ap.add_argument("--require-af3", action="store_true",
                    help="treat a missing AF3 data.json as a failure (use when MSA was staged)")
    args = ap.parse_args()

    targets = args.targets
    if args.all:
        sdir = SERIES_DIR.get(args.series, args.series.upper())
        targets = sorted(p.stem for p in
                         (PROJECT_ROOT / f"data/casp17_data/smiles/{sdir}").glob("*.tsv"))
    if not targets:
        ap.error("pass target ids or --all")

    fails: list[str] = []
    for t in targets:
        fails += verify(t, args.series, args.require_af3)

    print()
    if fails:
        print(f"{len(fails)} FAILURE(S):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print(f"ALL CHECKS PASSED ({len(targets)} target(s))")


if __name__ == "__main__":
    main()
