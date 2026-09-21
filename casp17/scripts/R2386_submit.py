#!/usr/bin/env python
"""Upload the R2386 models to the CASP17 prediction server.

The 2026-08-03 submission was done by hand and left nothing behind but a log,
so the second one is scripted: the same five files, the same endpoint, and a
gate in front of it that refuses to send a file that would be rejected or
silently scored short.

The gate is not a substitute for R2386_validate.py -- it re-checks only the
things the server itself checks, plus the two target-specific rules that cost
score rather than causing a rejection: exactly 500 ligands (fewer caps the
maximum score pro rata) and occupancies summing to the ligand count.

  python R2386_submit.py --dry-run     # gate only, sends nothing
  python R2386_submit.py               # gate, then upload all five
  python R2386_submit.py --only 3      # one model
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casp_identity                                                  # noqa: E402
import R2386_paths as paths                                           # noqa: E402

ROOT = paths.ROOT
ENDPOINT = "https://predictioncenter.org/casp17/predictions_submission.cgi"
TARGET = "R2386"
N_LIGANDS = 500
LIGAND_RESN = {"MG", "K", "NA", "HOH"}


def gate(path, idx):
    """Everything that must hold before a file is allowed onto the wire."""
    L = path.read_text().splitlines()
    problems = []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    need(L[0] == "PFRMAT TS", f"line 1 is {L[0]!r}, expected 'PFRMAT TS'")
    need(L[1] == f"TARGET {TARGET}", f"line 2 is {L[1]!r}")
    need(L[2] == f"AUTHOR {casp_identity.group_id()}", f"line 3 is {L[2]!r}")
    need(any(l == f"MODEL  {idx}" for l in L), f"no 'MODEL  {idx}' record")
    need(sum(l.startswith("MODEL") for l in L) == 1, "more than one MODEL record")
    need(L[-1] == "END", f"last line is {L[-1]!r}, expected 'END'")
    need(any(l.startswith("TER") for l in L), "no TER record")

    # A truncated upload is the failure mode that would not announce itself, so
    # the record geometry is checked rather than assumed.
    het = [l for l in L if l.startswith("HETATM")]
    atom = [l for l in L if l.startswith("ATOM")]
    bad = [i + 1 for i, l in enumerate(L)
           if l.startswith(("ATOM", "HETATM")) and len(l) != 80]
    need(not bad, f"{len(bad)} coordinate records are not 80 columns wide")
    need(atom, "no ATOM records")

    resn = Counter(l[17:20].strip() for l in het)
    need(set(resn) <= LIGAND_RESN, f"unexpected HETATM residues {set(resn) - LIGAND_RESN}")
    need(len(het) == N_LIGANDS,
         f"{len(het)} ligands, not {N_LIGANDS} -- the maximum score scales with this")
    occ = sum(float(l[54:60]) for l in het)
    need(abs(occ - len(het)) < 1e-6,
         f"occupancies sum to {occ:.2f}, ligand count is {len(het)}")
    need(all(float(l[60:66]) > 0 for l in het), "a ligand carries B = 0")
    need(len(set(l[21:27] for l in het)) == len(het), "duplicate solvent residue ids")
    need(not any(l[76:78].strip() == "H" for l in L if l.startswith(("ATOM", "HETATM"))),
         "hydrogens present; this target takes none")

    return problems, len(het), resn


def upload(path):
    """POST one file; return (ok, message)."""
    out = subprocess.run(
        ["curl", "-s", "-i", "-m", "180",
         "-F", f"email={casp_identity.email()}",
         "-F", f"prediction_file=@{path};type=text/plain",
         "-F", "action=process",
         "-F", "post=Submit",
         ENDPOINT],
        capture_output=True, text=True).stdout

    # The CGI answers in HTML. Strip it and keep the sentences that say whether
    # the prediction was taken -- the accession code on success, the complaint
    # on failure.
    text = re.sub(r"<[^>]+>", " ", out)
    text = re.sub(r"&nbsp;?", " ", text)
    lines = [" ".join(l.split()) for l in text.splitlines()]
    keep = [l for l in lines if l and re.search(
        r"accession|accepted|success|error|reject|invalid|fail|not |wrong|"
        r"missing|deadline|expired|permission|Location:", l, re.I)]
    ok = bool(re.search(r"accession|accepted|success", out, re.I)) and not \
        re.search(r"error|reject|invalid|failed", out, re.I)
    return ok, keep[:12] or lines[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=paths.DEFAULT_MODEL_DIR,
                    help="bare name, repo-relative or absolute path; "
                         "defaults to the delivered models")
    ap.add_argument("--only", help="model number, or a comma-separated list")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.dir = paths.model_dir(a.dir)

    idxs = [int(x) for x in a.only.split(",")] if a.only else [1, 2, 3, 4, 5]
    files = [(i, a.dir / f"{TARGET}_model{i}.txt") for i in idxs]

    print(f"gate -- {paths.label(a.dir)}")
    blocked = False
    summary = []
    for i, f in files:
        if not f.exists():
            print(f"  model{i}  MISSING {f}")
            blocked = True
            continue
        problems, n, resn = gate(f, i)
        comp = " ".join(f"{k}:{resn[k]}" for k in ("MG", "K", "NA", "HOH") if resn[k])
        if problems:
            blocked = True
            print(f"  model{i}  BLOCKED")
            for p in problems:
                print(f"           {p}")
        else:
            print(f"  model{i}  ok   {n} ligands   {comp}   {f.stat().st_size // 1024}K")
        summary.append((i, f, n, comp))

    if blocked:
        sys.exit("\nnothing sent: fix the above first")
    if a.dry_run:
        print("\n--dry-run: nothing sent")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = a.dir / f"submit_{TARGET}_{stamp}.log"
    rec = [f"=== {TARGET} submission {datetime.now():%Y-%m-%d %H:%M:%S} ===",
           f"group  : {casp_identity.group_id()}", f"email  : {casp_identity.email()}", "format : PFRMAT TS",
           f"endpoint: {ENDPOINT}", f"source : {a.dir}", ""]

    print(f"\nuploading to {ENDPOINT}")
    for i, f, n, comp in summary:
        ok, msg = upload(f)
        head = f"--- model{i}  ({n} ligands, {comp}) --- {'OK' if ok else 'CHECK'}"
        print(head)
        for m in msg:
            print(f"    {m}")
        rec.append(head)
        rec += [f"  {m}" for m in msg]
    log.write_text("\n".join(rec) + "\n")
    print(f"\nlog: {log}")


if __name__ == "__main__":
    main()