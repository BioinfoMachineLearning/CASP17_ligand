#!/usr/bin/env python
"""Propagate the R2386 RNA MSA a3m path into every stoichiometry-point JSON.

`run_rna_msa_only.py` searches once and patches `unpairedMsaPath` into the JSON
it was given (R2386_msa.json). The four sweep points S1..S4 need the same path;
without it Protenix silently runs the RNA single-sequence, which would quietly
make all 100 models worse than they should be.

Idempotent. Re-run after any regeneration of the inputs, since
gen_R2386_solvent_inputs.py rewrites the JSONs and drops the path.

    conda run -n casp17_ligand python casp17/scripts/inject_rna_msa_R2386.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_a3m(msa_root: Path, src_json: Path) -> str:
    """Prefer the path already patched into the MSA seed JSON; else search."""
    if src_json.exists():
        try:
            spec = json.load(open(src_json))
            job = spec[0] if isinstance(spec, list) else spec
            for ent in job.get("sequences", []):
                p = ent.get("rnaSequence", {}).get("unpairedMsaPath")
                if p and Path(p).exists():
                    return str(Path(p).resolve())
        except Exception:
            pass
    hits = sorted(msa_root.glob("**/rna_msa/**/rna_msa.a3m"))
    hits = [h for h in hits if "R2386" in str(h)]
    if not hits:
        raise SystemExit(
            f"no R2386 rna_msa.a3m under {msa_root}; run run_rna_msa_only.py first"
        )
    return str(hits[0].resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--points", default="S1,S2,S3,S4")
    args = ap.parse_args()

    ptx_dir = args.root / "data/test_cases/casp17_R/protenix_inputs"
    a3m = find_a3m(args.root / "data/test_cases/casp17_R/protenix_msa", ptx_dir / "R2386_msa.json")
    n_seq = sum(1 for line in open(a3m) if line.startswith(">"))
    print(f"a3m: {a3m}\n     {n_seq} sequences")

    for point in args.points.split(","):
        jf = ptx_dir / f"R2386_{point.strip()}.json"
        if not jf.exists():
            print(f"  {jf.name}: MISSING, skipped")
            continue
        spec = json.load(open(jf))
        job = spec[0] if isinstance(spec, list) else spec
        patched = 0
        for ent in job.get("sequences", []):
            if "rnaSequence" in ent:
                ent["rnaSequence"]["unpairedMsaPath"] = a3m
                patched += 1
        json.dump(spec, open(jf, "w"), indent=2)
        print(f"  {jf.name}: patched {patched} rnaSequence entry(ies)")


if __name__ == "__main__":
    main()
