"""Batch PB-aware early-stop relax for ensemble Stage 1.5.

Designed to be called from `ensemble_generation.py::process_target` after
`cif_to_pdb_sdf` has populated `cif_converted/`.

Per pose:
  1. Run PoseBusters dock-mode.
  2. If pass → leave SDF unchanged.
  3. If only the cofactor checks fail → leave SDF unchanged and mark the pose
     invalid (see below).
  4. Otherwise → run `relax_iterative` (step=30, k=100, max=300, with max=500 retry).

Cofactors are judged, not fixed.  When a pose overlaps a supplied cofactor
(ZN, SFG) the model has put the fragment where the cofactor lives; that is a
wrong binding mode, not a geometry defect, and nudging it a few tenths of an
angstrom would only hide the error.  Such poses are reported as
``action="cofactor_clash"`` with ``passed=False`` so the downstream filter drops
them, and no simulation time is spent.

Conversely the relax early-stop predicate ignores the four cofactor checks: it
runs against the cofactor-free receptor, so those checks are vacuously true.
Leaving them in would make `pb_check_fn` unsatisfiable for any pose that starts
out overlapping the cofactor, and the loop would burn its full 300+500
iterations before giving up.

To avoid re-deriving AM1-BCC charges for every pose (5+ min each), the runner
precomputes one charged OpenFF Molecule per unique input SMILES (or by SDF
atom-formula if no SMILES known), pickles it once, and shares with workers.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


#: The four dock-mode checks that only fire when the receptor carries a cofactor.
#: Names are PoseBusters' post-rename column labels (config/dock.yml).
COFACTOR_CHECKS = (
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
)


def _pb_pass(sdf: str, pdb: str) -> bool:
    """PoseBusters dock-mode check; all 22 per-check booleans must be True."""
    try:
        from posebusters import PoseBusters
        r = PoseBusters(config="dock", top_n=None).bust(
            [sdf], mol_cond=pdb, full_report=False)
        return bool(r.iloc[0].all())
    except Exception:
        return False


def _pb_failed_checks(sdf: str, pdb: str) -> Optional[List[str]]:
    """Names of the dock-mode checks that failed, or None if PoseBusters errored.

    An empty list means the pose passed all 22 — distinct from None, which means
    no verdict was obtained.  :func:`_pb_pass` collapses both to False, which is
    the safe reading for an early-stop predicate but useless for deciding whether
    a failure is worth relaxing.
    """
    try:
        from posebusters import PoseBusters
        r = PoseBusters(config="dock", top_n=None).bust(
            [sdf], mol_cond=pdb, full_report=False)
        row = r.iloc[0]
        return [c for c in r.columns if not bool(row[c])]
    except Exception:
        return None


def _precompute_off_molecule(sdf_path: str, out_pickle: str) -> bool:
    """Compute AM1-BCC-charged OpenFF Molecule from an SDF, pickle it.
    Returns True on success.
    """
    try:
        from openff.toolkit.topology import Molecule
        from rdkit import Chem
        rd = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=True)
        if rd is None:
            return False
        rd = Chem.AddHs(rd, addCoords=True)
        off = Molecule.from_rdkit(rd, allow_undefined_stereo=True)
        # keep in sync with select_relax_walk._charge_cache: charge the pose's
        # own conformer, never an ETKDG re-embed (fails on bridged cages)
        off.assign_partial_charges(partial_charge_method="am1bcc",
                                   use_conformers=off.conformers or None)
        with open(out_pickle, "wb") as f:
            pickle.dump(off, f)
        return True
    except Exception as e:
        log.warning(f"AM1-BCC precompute failed for {sdf_path}: {e}")
        return False


def _worker(args: Tuple) -> dict:
    """Process one pose.

    Args: (pdb, sdf, cached_pkl|None, step_iter, max_iter, k, retry_max_iter,
    pocket_cutoff, pb_pdb|None). ``pb_pdb`` is the receptor PoseBusters judges
    against; pass the cofactor-bearing sidecar there. ``pdb`` (cofactor-free)
    always drives OpenMM and the early-stop predicate.
    """
    # OpenMM has a C++ ABI dependency on conda-env libstdc++.so.6 — if other
    # deps (e.g., posebusters → numpy) load the system libstdc++ first, the
    # subsequent OpenMM load fails with `GLIBCXX_3.4.29 not found`. Importing
    # openmm first forces the env's libstdc++ into the dynamic linker cache
    # before anything else can grab the system one.
    import openmm  # noqa: F401  (lock libstdc++ before posebusters)
    from casp17_ligand.utils.relax_pose import relax_iterative
    pb_pdb = None
    if len(args) == 9:
        pdb, sdf, cached_pkl, step_iter, max_iter, k, retry_max, pocket_cutoff, pb_pdb = args
    else:
        pdb, sdf, cached_pkl, step_iter, max_iter, k, retry_max, pocket_cutoff = args
    if not pb_pdb or not os.path.exists(pb_pdb):
        pb_pdb = pdb
    name = os.path.basename(sdf).replace("_ligand.sdf", "")
    t0 = time.time()

    fails = _pb_failed_checks(sdf, pb_pdb)
    if fails == []:
        return dict(name=name, action="kept", iters=0, drift=0.0,
                    passed=True, fails=[], cof_fails=[], wall=time.time() - t0)
    # None = PoseBusters itself errored; fall through and let relax try.
    cof_fails = [c for c in (fails or []) if c in COFACTOR_CHECKS]
    other_fails = [c for c in (fails or []) if c not in COFACTOR_CHECKS]
    if fails is not None and cof_fails and not other_fails:
        # Geometry is clean; the fragment is simply sitting on the cofactor.
        # Relax cannot make that the right answer, so don't pay for it.
        return dict(name=name, action="cofactor_clash", iters=0, drift=0.0,
                    passed=False, fails=list(fails), cof_fails=cof_fails,
                    wall=time.time() - t0)

    cached = None
    if cached_pkl and os.path.exists(cached_pkl):
        with open(cached_pkl, "rb") as f:
            cached = pickle.load(f)

    try:
        info = relax_iterative(
            pdb, sdf, sdf,            # overwrite in place
            pb_check_fn=_pb_pass,     # judged vs cofactor-free pdb → cofactor checks vacuous
            platform_name="CPU",
            step_iter=step_iter, max_total_iter=max_iter,
            ligand_restraint_k=k, cached_off_molecule=cached,
            pocket_cutoff=pocket_cutoff,
        )
        # Retry with longer max if still failing
        if not info["passed"] and retry_max > max_iter:
            info = relax_iterative(
                pdb, sdf, sdf,
                pb_check_fn=_pb_pass,
                platform_name="CPU",
                step_iter=step_iter, max_total_iter=retry_max,
                ligand_restraint_k=k, cached_off_molecule=cached,
                pocket_cutoff=pocket_cutoff,
            )
        # Relax optimised against the cofactor-free receptor, so re-judge with the
        # cofactor present: a pose can be nudged onto it while fixing a clash.
        post = _pb_failed_checks(sdf, pb_pdb) if pb_pdb != pdb else (
            [] if info["passed"] else None)
        passed = info["passed"] if post is None else (post == [])
        return dict(name=name, action="relaxed",
                    iters=info["iters_done"], drift=info["ligand_drift_RMSD_A"],
                    passed=passed,
                    fails=list(post or []),
                    cof_fails=[c for c in (post or []) if c in COFACTOR_CHECKS],
                    wall=time.time() - t0)
    except Exception as e:
        return dict(name=name, action="error", error=str(e),
                    passed=False, fails=[], cof_fails=[], wall=time.time() - t0)


def _formula_key(sdf_path: str) -> str:
    """Atomic-formula string used to cache the AM1-BCC molecule per ligand identity."""
    try:
        from collections import Counter
        from rdkit import Chem
        m = Chem.MolFromMolFile(sdf_path, removeHs=True, sanitize=True)
        if m is None:
            return os.path.basename(sdf_path)
        counts = Counter(a.GetSymbol() for a in m.GetAtoms())
        return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}"
                       for e in sorted(counts))
    except Exception:
        return os.path.basename(sdf_path)


def relax_pb_failures(
    pose_list: List[Tuple],
    target: str,
    n_workers: int = 16,
    step_iter: int = 30,
    max_iter: int = 300,
    retry_max_iter: int = 500,
    ligand_restraint_k: float = 100.0,
    cache_dir: Optional[str] = None,
    pocket_cutoff: Optional[float] = None,
) -> dict:
    """Run PB check + early-stop relax on a list of pose tuples.

    Each entry is ``(pdb, sdf)`` or ``(pdb, sdf, pb_pdb)``, where the optional
    third element is a receptor that also contains the supplied cofactor and is
    used for the PoseBusters verdict only.

    Modifies SDFs in place. PB-pass poses and cofactor-only failures are skipped.

    Returns: stats dict with counts + per-pose timing.
    """
    if not pose_list:
        return dict(target=target, total=0)

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix=f"relax_cache_{target}_")

    os.makedirs(cache_dir, exist_ok=True)

    # ── Precompute one charged molecule per unique ligand formula ──
    formula_to_pkl: Dict[str, str] = {}
    formula_examples: Dict[str, str] = {}
    for entry in pose_list:
        sdf = entry[1]
        fkey = _formula_key(sdf)
        if fkey not in formula_examples:
            formula_examples[fkey] = sdf

    log.info(f"[relax_pb_failures] {target}: {len(formula_examples)} unique ligand formula(s) to precompute AM1-BCC")
    for fkey, ex_sdf in formula_examples.items():
        pkl = os.path.join(cache_dir, f"{fkey}.pkl")
        if not os.path.exists(pkl):
            t0 = time.time()
            ok = _precompute_off_molecule(ex_sdf, pkl)
            if ok:
                log.info(f"  [{fkey}] AM1-BCC computed in {time.time()-t0:.0f}s -> {pkl}")
            else:
                log.warning(f"  [{fkey}] AM1-BCC precompute FAILED; per-pose will re-derive (slow)")
                pkl = None
        formula_to_pkl[fkey] = pkl

    # ── Dispatch worker per pose ──
    tasks = []
    n_with_cof = 0
    for entry in pose_list:
        pdb, sdf = entry[0], entry[1]
        pb_pdb = entry[2] if len(entry) > 2 else None
        if pb_pdb and os.path.exists(pb_pdb):
            n_with_cof += 1
        fkey = _formula_key(sdf)
        pkl = formula_to_pkl.get(fkey)
        tasks.append((pdb, sdf, pkl, step_iter, max_iter, ligand_restraint_k,
                      retry_max_iter, pocket_cutoff, pb_pdb))
    if n_with_cof:
        log.info(f"[relax_pb_failures] {target}: {n_with_cof}/{len(tasks)} poses judged "
                 f"against a cofactor-bearing receptor")

    ctx = mp.get_context("spawn")
    results = []
    t_start = time.time()
    log.info(f"[relax_pb_failures] {target}: dispatching {len(tasks)} poses to {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as exe:
        futures = {exe.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                kept = sum(1 for x in results if x.get("action") == "kept")
                relaxed_ok = sum(1 for x in results if x.get("action") == "relaxed" and x.get("passed"))
                relaxed_fail = sum(1 for x in results if x.get("action") == "relaxed" and not x.get("passed"))
                cof = sum(1 for x in results if x.get("action") == "cofactor_clash")
                err = sum(1 for x in results if x.get("action") == "error")
                log.info(f"  {target} relax progress {done}/{len(tasks)}  "
                         f"kept={kept} relaxed_ok={relaxed_ok} relaxed_fail={relaxed_fail} "
                         f"cofactor_clash={cof} err={err}  ({time.time()-t_start:.0f}s)")

    # ── Summary ──
    n = len(results)
    kept = [r for r in results if r["action"] == "kept"]
    cof_clash = [r for r in results if r["action"] == "cofactor_clash"]
    relaxed = [r for r in results if r["action"] == "relaxed"]
    relaxed_ok = [r for r in relaxed if r["passed"]]
    relaxed_fail = [r for r in relaxed if not r["passed"]]
    errs = [r for r in results if r["action"] == "error"]

    import statistics as _stats
    drift_med = (_stats.median(r["drift"] for r in relaxed) if relaxed else 0.0)
    drift_max = (max(r["drift"] for r in relaxed) if relaxed else 0.0)
    iters_med = (_stats.median(r["iters"] for r in relaxed) if relaxed else 0)

    relaxed_onto_cof = [r for r in relaxed if r.get("cof_fails")]

    summary = dict(
        target=target, total=n,
        orig_pb_pass=len(kept),
        cofactor_clash=len(cof_clash),
        relaxed=len(relaxed), relaxed_pass=len(relaxed_ok),
        relaxed_into_cofactor=len(relaxed_onto_cof),
        still_failing=len(relaxed_fail), errors=len(errs),
        drift_median_A=drift_med, drift_max_A=drift_max, iters_median=iters_med,
        wall_s=time.time() - t_start,
    )
    log.info(f"[relax_pb_failures] {target} DONE: "
             f"ORIG-pass {len(kept)}/{n}, relaxed→pass {len(relaxed_ok)}/{len(relaxed)}, "
             f"still-fail {len(relaxed_fail)}, cofactor-clash {len(cof_clash)} (relax skipped), "
             f"drift_med={drift_med:.2f}Å, wall={time.time()-t_start:.0f}s")
    if cof_clash:
        log.warning(f"[relax_pb_failures] {target}: {len(cof_clash)} pose(s) overlap the supplied "
                    f"cofactor with otherwise-clean geometry — kept as-is, marked PB-invalid.")
    if relaxed_onto_cof:
        log.warning(f"[relax_pb_failures] {target}: {len(relaxed_onto_cof)} pose(s) ended up "
                    f"overlapping the cofactor after relax (early-stop ignores those checks).")
    if relaxed_fail:
        log.warning(f"[relax_pb_failures] {target}: {len(relaxed_fail)} pose(s) still PB-invalid after relax; "
                    f"they remain in cif_converted/ but will get _pb=False downstream.")
    return summary


# ──────────────────────────────────────────────────────────────────────
# CLI entry: lets ensemble_generation.py (in casp17_ligand env) call this
# batch runner via `conda run -n PoseBench python -m
# casp17_ligand.utils.relax_pose_batch --manifest <path> --target <T>`
# because OpenMM / openff / pdbfixer live in the PoseBench env, not in
# casp17_ligand (cross-env mamba solve takes 10+h on shared NFS).
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    ap = argparse.ArgumentParser(description="Batch PB-aware early-stop relax (CLI)")
    ap.add_argument("--manifest", required=True,
                    help="JSON file with shape "
                         "{\"target\": str, \"pose_list\": [[pdb, sdf], ...]}. "
                         "Entries may carry a third element, a cofactor-bearing "
                         "receptor used for the PoseBusters verdict only.")
    ap.add_argument("--summary-out", default=None,
                    help="Optional path to write summary JSON. "
                         "If omitted, summary is printed to stdout.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--step-iter", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--retry-max-iter", type=int, default=500)
    ap.add_argument("--restraint-k", type=float, default=100.0)
    ap.add_argument("--pocket-cutoff", type=float, default=None,
                    help="If set (Å), minimize only residues within this "
                         "distance of the ligand (pocket-local relax; fast for "
                         "huge multi-chain receptors). PB check still uses full receptor.")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    target = manifest["target"]
    pose_list = [tuple(p[:3]) for p in manifest["pose_list"]]

    summary = relax_pb_failures(
        pose_list, target,
        n_workers=args.workers,
        step_iter=args.step_iter,
        max_iter=args.max_iter,
        retry_max_iter=args.retry_max_iter,
        ligand_restraint_k=args.restraint_k,
        cache_dir=args.cache_dir,
        pocket_cutoff=args.pocket_cutoff,
    )

    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary, indent=2))
