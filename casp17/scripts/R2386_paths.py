#!/usr/bin/env python
"""Shared path resolution for the R2386 scripts.

Two things used to be wrong here, and both failed silently, which is the worst
way for a path to be wrong.

  the cif cache.  Every builder and every audit reads the 58 experimental
      structures from one directory, and that directory was hardcoded as
      /tmp/t258 in seven separate scripts. /tmp is swept. Losing it does not
      raise anything interesting -- gemmi just cannot open 9c6i.cif -- and the
      pipeline is unreproducible from that moment on. The cache now lives in the
      project, next to the pools and the RISM grids it is used with.

  the model directory.  Four audits take --dir and they did not agree on what
      it means: two joined it onto casp17/submissions/casp17_R, two took it as
      given. Three of the four defaulted to a version that is no longer the one
      we shipped. Point one of them at the wrong base and it prints the
      experimental controls, skips the models, and exits 0.

Both are now resolved through this module, which accepts every spelling that
used to work and fails loudly when a path is genuinely absent.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "casp17/submissions/casp17_R"

#: where the delivered models live; the default for every --dir
DEFAULT_MODEL_DIR = "R2386_files"

# Where to look for the experimental cif cache, best first. /tmp/t258 is kept
# last so a session that predates the move keeps working.
_CIF_CANDIDATES = (
    ROOT / "outputs/R2386_cif_cache",
    Path("/tmp/t258"),
)

_CIF_MISSING = (
    "  it holds 58 structures (78 MB) selected by the rule in\n"
    "  Set $R2386_CIF_CACHE or restore the directory."
)


def cif_cache():
    """Directory holding the 58 experimental cif files.

    $R2386_CIF_CACHE moves the cache without patching anything, which is how a
    rerun on another machine is meant to work. If it is set and does not
    resolve, that is a typo and we say so -- falling back to the built-in
    location would run the whole pipeline against a directory the caller did
    not ask for, which is the failure this module exists to prevent.
    """
    env = os.environ.get("R2386_CIF_CACHE")
    if env:
        if Path(env).is_dir():
            return Path(env)
        raise SystemExit(f"R2386: $R2386_CIF_CACHE={env!r} is not a directory.\n{_CIF_MISSING}")
    for cand in _CIF_CANDIDATES:
        if Path(cand).is_dir():
            return Path(cand)
    raise SystemExit(
        "R2386: the experimental cif cache is missing.\n"
        f"  looked in: $R2386_CIF_CACHE, {ROOT}/outputs/R2386_cif_cache, /tmp/t258\n"
        + _CIF_MISSING
    )


def ref_cif():
    """9C6I -- the reference frame everything is superposed onto."""
    p = cif_cache() / "9c6i.cif"
    if not p.is_file():
        raise SystemExit(f"R2386: {p} is missing; it is the reference structure.")
    return p


def label(p):
    """How a resolved path should be printed.

    Repo-relative if it is inside the project, absolute otherwise. Scripts print
    this rather than the raw --dir, so the same directory reads the same way in
    the log whichever spelling was typed on the command line.
    """
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def model_dir(spec=DEFAULT_MODEL_DIR):
    """Resolve --dir, accepting all three spellings that were previously in use.

    A bare name (R2386_files_v9), a repo-relative path
    (casp17/submissions/casp17_R/R2386_files) and an absolute path all work, so
    a command copied from any of these scripts runs under any of the others.
    """
    p = Path(spec).expanduser()
    for cand in (p, SUB / p, ROOT / p):
        if cand.is_dir():
            return cand
    raise SystemExit(
        f"R2386: no model directory matches {spec!r}.\n"
        f"  tried {p}, {SUB / p}, {ROOT / p}\n"
        f"  the delivered models are in {SUB / DEFAULT_MODEL_DIR}"
    )