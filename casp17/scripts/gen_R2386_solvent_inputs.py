#!/usr/bin/env python
"""Generate co-folding inputs for R2386 (group IIC intron solvent-model target).

R2386 is not an LG target: the scored quantity is the *solvent shell*
(Mg2+ / K+ / Na+ / H2O) around a 417 nt RNA, submitted as PFRMAT TS.
None of AF3 / Boltz-2 / Protenix predicts water, but all three accept ion
entities, so co-folding is used here as an *ion-site candidate generator*
(and as the only source of coordinates for the C-terminal 24 nt, which every
homologous structure leaves disordered).

Rather than guessing one ion stoichiometry, we titrate: four points from
"only the strongest sites" to "beyond what a 1.95 A map resolves". A site's
rank is the lowest stoichiometry at which it first appears, so no post-hoc
model selection between the points is needed.

Per point: 25 models / method (same total budget as the usual r1+r2 = 100).

Ion entity syntax verified against each codebase 2026-08-01:
  AF3       {"ligand": {"id": [...], "ccdCodes": ["MG"]}}  <- one id per ion.
            Putting ["MG","MG"] in one ccdCodes list makes ONE branched chain.
  Boltz-2   - ligand: {id: [M1, M2, ...], ccd: MG}         <- chain ids <= 5 chars
  Protenix  {"ion": {"ion": "MG", "count": N}}             <- bare CCD, no CCD_ prefix

MSA handling (RNA has no template support in any of the three):
  AF3       run the data pipeline once on R2386_msa.json, then inject the
            embedded MSA into the four S-point JSONs (inject_af3_msa()).
  Protenix  run_rna_msa_only.py once -> patch unpairedMsaPath into all four.
  Boltz-2   does not consume RNA MSA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET = "R2386"
SERIES = "casp17_R"

# (label, n_Mg, n_K, n_Na, af3_seeds, protenix_seeds, boltz_seed)
# K:Mg ~ 2:5, mirroring the template pool (59 K sites vs 219 Mg sites).
# Na stays small: only 5 mM in the W2386 buffer, and Na+ is isoelectronic with
# water (both 10 e-) so it is the least identifiable species in the map.
POINTS = [
    ("S1", 10, 4, 0, [1, 2, 3, 4, 5], "101,102,103,104,105", 3861),
    ("S2", 20, 8, 2, [6, 7, 8, 9, 10], "106,107,108,109,110", 3862),
    ("S3", 35, 14, 4, [11, 12, 13, 14, 15], "111,112,113,114,115", 3863),
    ("S4", 55, 22, 6, [16, 17, 18, 19, 20], "116,117,118,119,120", 3864),
]

N_MODELS_PER_POINT = 25
AF3_DIFFUSION_SAMPLES = 5  # x 5 seeds = 25


def read_sequence(root: Path) -> str:
    fasta = root / "data/casp17_data/sequences/CASP17_R" / f"{TARGET}.txt"
    lines = [ln.strip() for ln in fasta.read_text().splitlines() if ln.strip()]
    seq = "".join(ln for ln in lines if not ln.startswith(">"))
    if len(seq) != 417 or set(seq) - set("ACGU"):
        raise SystemExit(f"unexpected sequence: len={len(seq)} alphabet={sorted(set(seq))}")
    return seq


def af3_chain_ids(n: int, skip: set[str]) -> list[str]:
    """Unique uppercase-alpha chain ids. AF3 requires isalpha() and not islower()."""
    out, i = [], 0
    letters = [chr(ord("A") + k) for k in range(26)]
    pool = letters + [a + b for a in letters for b in letters]
    while len(out) < n:
        cid = pool[i]
        i += 1
        if cid not in skip:
            out.append(cid)
    return out


def build_af3(seq: str, label: str, n_mg: int, n_k: int, n_na: int, seeds: list[int]) -> dict:
    sequences: list[dict] = [{"rna": {"id": "A", "sequence": seq}}]
    used = {"A"}
    for ccd, n in (("MG", n_mg), ("K", n_k), ("NA", n_na)):
        if n <= 0:
            continue
        ids = af3_chain_ids(n, used)
        used.update(ids)
        sequences.append({"ligand": {"id": ids, "ccdCodes": [ccd]}})
    return {
        "name": f"{TARGET}_{label}",
        "modelSeeds": seeds,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }


def build_boltz(seq: str, n_mg: int, n_k: int, n_na: int) -> str:
    lines = ["version: 1", "sequences:", "  - rna:", "      id: A", f"      sequence: {seq}"]
    for prefix, ccd, n in (("M", "MG", n_mg), ("K", "K", n_k), ("N", "NA", n_na)):
        if n <= 0:
            continue
        ids = [f"{prefix}{i}" for i in range(1, n + 1)]
        too_long = [x for x in ids if len(x) > 5]
        if too_long:
            raise SystemExit(f"boltz chain id >5 chars: {too_long[:3]}")
        lines += ["  - ligand:", f"      id: [{', '.join(ids)}]", f"      ccd: {ccd}"]
    return "\n".join(lines) + "\n"


def build_protenix(seq: str, label: str, n_mg: int, n_k: int, n_na: int) -> list[dict]:
    sequences: list[dict] = [{"rnaSequence": {"sequence": seq, "count": 1}}]
    for ccd, n in (("MG", n_mg), ("K", n_k), ("NA", n_na)):
        if n > 0:
            sequences.append({"ion": {"ion": ccd, "count": n}})
    return [{"name": f"{TARGET}_{label}", "sequences": sequences}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    root = args.root
    seq = read_sequence(root)

    af3_dir = root / f"data/test_cases/{SERIES}/af3_inputs"
    boltz_dir = root / f"data/test_cases/{SERIES}/boltz2_inputs"
    ptx_dir = root / f"data/test_cases/{SERIES}/protenix_inputs"
    for d in (af3_dir, boltz_dir, ptx_dir):
        d.mkdir(parents=True, exist_ok=True)

    # RNA-only seed JSONs: AF3 phase-1 MSA search, and the Protenix MSA search.
    (af3_dir / f"{TARGET}_msa.json").write_text(
        json.dumps(
            {
                "name": f"{TARGET}_msa",
                "modelSeeds": [1],
                "sequences": [{"rna": {"id": "A", "sequence": seq}}],
                "dialect": "alphafold3",
                "version": 1,
            },
            indent=2,
        )
        + "\n"
    )
    (ptx_dir / f"{TARGET}_msa.json").write_text(
        json.dumps([{"name": TARGET, "sequences": [{"rnaSequence": {"sequence": seq, "count": 1}}]}], indent=2) + "\n"
    )

    print(f"{TARGET}: {len(seq)} nt RNA")
    print(f"{'point':6s} {'Mg':>4s} {'K':>4s} {'Na':>4s} {'ions':>5s} {'tokens':>7s} {'models':>7s}")
    for label, n_mg, n_k, n_na, af3_seeds, ptx_seeds, boltz_seed in POINTS:
        (af3_dir / f"{TARGET}_{label}.json").write_text(
            json.dumps(build_af3(seq, label, n_mg, n_k, n_na, af3_seeds), indent=2) + "\n"
        )
        (boltz_dir / f"{TARGET}_{label}_input.yaml").write_text(build_boltz(seq, n_mg, n_k, n_na))
        (ptx_dir / f"{TARGET}_{label}.json").write_text(
            json.dumps(build_protenix(seq, label, n_mg, n_k, n_na), indent=2) + "\n"
        )
        n_ion = n_mg + n_k + n_na
        print(
            f"{label:6s} {n_mg:4d} {n_k:4d} {n_na:4d} {n_ion:5d} {len(seq) + n_ion:7d} "
            f"{N_MODELS_PER_POINT:7d}"
        )
    print(f"\nwrote 4 x 3 inputs + 2 MSA seed files under data/test_cases/{SERIES}/")


if __name__ == "__main__":
    main()
