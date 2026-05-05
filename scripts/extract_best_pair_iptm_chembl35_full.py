#!/usr/bin/env python3
"""For every compound in chembl35_full, find the seed-N_sample-M with the
highest protein-ligand pair_iptm (chain_pair_iptm[0][1]) and emit a CSV.

Handles both output layouts that ended up in the shared dir:
  aster (flat):   chembl35_full/seed-{N}_sample-{M}/<lower>_seed-...{model.cif,summary_confidences.json}
  lily  (nested): chembl35_full/<UPPER>[_<TIMESTAMP>]/seed-{N}_sample-{M}/<UPPER>_seed-...{...}

For lily, when multiple timestamped dirs exist for one compound we pick the
one with the most valid (json + cif) sample pairs; ties broken by latest
timestamp.

Outputs:
  data/AF3_structures/chembl35_full/best_pair_iptm.csv
  data/AF3_structures/chembl35_full/missing.csv  (rows that produced no candidate)
"""

from __future__ import annotations
import csv
import json
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT      = Path("/bmlfast/Lyuwei/0.Projects/CASP17_ligand")
OUT_DIR   = ROOT / "outputs/alphafold3/chembl35_full"
MAP_CSV   = ROOT / "data/test_cases/chembl35_full/compound_id_map.csv"
DST_DIR   = ROOT / "data/AF3_structures/chembl35_full"
DST_CSV   = DST_DIR / "best_pair_iptm.csv"
MISS_CSV  = DST_DIR / "missing.csv"

SEEDS   = (1, 2)
SAMPLES = (0, 1, 2, 3, 4)
TS_RE   = re.compile(r"_(\d{8}_\d{6})$")


def _summary_to_iptm(p: Path) -> float | None:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    m = d.get("chain_pair_iptm")
    if not m or len(m) < 2 or len(m[0]) < 2:
        return None
    # 1-protein + 1-ligand → [0][1] is the cross-chain value
    return m[0][1]


def _scan_layout_root(af3_name: str) -> tuple[str, Path] | None:
    """Pick a base dir that contains the seed-N_sample-M tree for this compound.

    Returns (layout, base_dir):
      - ('aster', OUT_DIR)              if any flat sample json exists at root/seed-*/
      - ('lily',  OUT_DIR/<UPPER>[_<ts>]) if no flat hit; pick best nested dir.
      - None if nothing usable.
    """
    lower = af3_name.lower()
    upper = af3_name.upper()

    # 1. aster flat — sample if any seed-N_sample-M/<lower>_..._summary_confidences.json exists
    for s in SEEDS:
        for k in SAMPLES:
            j = OUT_DIR / f"seed-{s}_sample-{k}" / f"{lower}_seed-{s}_sample-{k}_summary_confidences.json"
            if j.is_file():
                return ("aster", OUT_DIR)

    # 2. lily nested — list all matching dirs at root, pick best
    candidates = []
    for d in OUT_DIR.iterdir():
        if not d.is_dir():
            continue
        nm = d.name
        m = TS_RE.search(nm)
        ts  = m.group(1) if m else ""
        bare = nm[: -len(m.group(0))] if m else nm
        if bare != upper:
            continue
        # Count valid sample pairs (json + cif)
        valid = 0
        for s in SEEDS:
            for k in SAMPLES:
                j = d / f"seed-{s}_sample-{k}" / f"{upper}_seed-{s}_sample-{k}_summary_confidences.json"
                c = d / f"seed-{s}_sample-{k}" / f"{upper}_seed-{s}_sample-{k}_model.cif"
                if j.is_file() and c.is_file():
                    valid += 1
        if valid:
            candidates.append((valid, ts, d))
    if not candidates:
        return None
    # Sort: most-valid first; tiebreak: latest timestamp
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ("lily", candidates[0][2])


def find_best(af3_name: str) -> dict | None:
    """Return best sample row dict, or None if nothing found."""
    pick = _scan_layout_root(af3_name)
    if pick is None:
        return None
    layout, base = pick

    name = af3_name.lower() if layout == "aster" else af3_name.upper()
    best = None
    for s in SEEDS:
        for k in SAMPLES:
            j = base / f"seed-{s}_sample-{k}" / f"{name}_seed-{s}_sample-{k}_summary_confidences.json"
            if not j.is_file():
                continue
            iptm = _summary_to_iptm(j)
            if iptm is None:
                continue
            cif = j.with_name(j.name.replace("_summary_confidences.json", "_model.cif"))
            if not cif.is_file():
                continue
            if best is None or iptm > best["pair_iptm"]:
                best = {
                    "layout": layout,
                    "best_seed": s,
                    "best_sample": k,
                    "pair_iptm": iptm,
                    "src_cif_abspath": str(cif),
                    "src_summary_json": str(j),
                }
    return best


def main():
    # Load mapping (af3_name → compound_id, target_id)
    with open(MAP_CSV) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} af3_name → compound_id rows")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    found_rows = []
    missing_rows = []

    with ProcessPoolExecutor(max_workers=32) as pool:
        futs = {pool.submit(find_best, r["af3_name"]): r for r in rows}
        for i, fut in enumerate(as_completed(futs)):
            r = futs[fut]
            try:
                best = fut.result()
            except Exception as e:
                best = None
                print(f"  ERROR on {r['af3_name']}: {e}")
            if best:
                merged = {**r, **best}
                found_rows.append(merged)
            else:
                missing_rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1}/{len(rows)}  (found {len(found_rows)}, missing {len(missing_rows)})")

    # Order found_rows by index_within_target to get deterministic CSV
    found_rows.sort(key=lambda x: (x["target_id"], int(x["index_within_target"])))

    fields = ["af3_name", "compound_id", "target_id", "index_within_target",
              "layout", "best_seed", "best_sample", "pair_iptm",
              "src_cif_abspath", "src_summary_json"]
    with open(DST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(found_rows)

    if missing_rows:
        with open(MISS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["af3_name", "compound_id", "target_id", "index_within_target"])
            w.writeheader()
            w.writerows(missing_rows)
    elif MISS_CSV.exists():
        MISS_CSV.unlink()

    print(f"\nWrote {DST_CSV}  ({len(found_rows)} rows)")
    if missing_rows:
        print(f"Wrote {MISS_CSV}  ({len(missing_rows)} missing)")
    else:
        print("No missing compounds.")


if __name__ == "__main__":
    main()
