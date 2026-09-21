"""Phase 2 of L-series pose selection: PB check + early-stop relax, walking the
candidate list produced by casp17/scripts/select_L_candidates.py.

Per target, candidates arrive in descending SuCOS order. For each in turn:

    PoseBusters (dock mode, all 22 checks)
      pass                       -> accept as-is
      fail  -> early-stop relax  -> re-check
                 pass            -> accept the RELAXED pose
                 still failing   -> move to the next candidate

Relax parameters are the competition-locked ones (same defaults
as relax_pose_batch.relax_pb_failures):
    step_iter=30, max_total_iter=300, retry max_total_iter=500,
    ligand_restraint_k=100, platform=CPU, full-receptor (no pocket cutoff)

Nothing under ranking_sucos/ is modified: the accepted pose is copied into
<target>/selected/ first, and relax overwrites only that copy.

Must run in an env with OpenMM + openff + posebusters (PoseBench). Invoke with
`-m casp17_ligand.utils.select_relax_walk`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

log = logging.getLogger(__name__)

STEP_ITER = 30
MAX_ITER = 300
RETRY_MAX_ITER = 500
LIGAND_K = 100.0


def _pb_pass(sdf: str, pdb: str) -> bool:
    """PoseBusters dock-mode check; all 22 per-check booleans must be True.

    Verbatim from relax_pose_batch._pb_pass, including the swallowed exception:
    a pose PoseBusters cannot even parse counts as failing, so it gets a relax
    attempt rather than being accepted on a technicality.
    """
    try:
        from posebusters import PoseBusters
        r = PoseBusters(config="dock", top_n=None).bust(
            [sdf], mol_cond=pdb, full_report=False)
        return bool(r.iloc[0].all())
    except Exception:
        return False


def _mol_key(sdf_path: str) -> str:
    """Cache key = molecular identity AND atom order, not the heavy-atom formula.

    A cached entry is reused by handing its graph to relax_pose._build_simulation,
    which swaps in the current pose's coordinates by array index and later writes
    that graph back out as the relaxed SDF. So a key collision is not a slow path,
    it is a wrong answer. Two things must therefore be in the key:

      * the full graph incl. hydrogens -- a heavy-atom formula lumps together
        constitutional isomers (L010281 c1ccc(COC2CCNCC2)cc1 vs L010695
        NC(c1ccccc1)C1CCOCC1, both "C12NO", both 31 atoms) and molecules that
        differ only in H count (L010214 30 atoms vs L011108 28 atoms, both
        "C11N2" -> "cannot reshape array of size 84 into shape (30,3)").
      * the canonical atom ranking -- same molecule written by two methods can
        carry different atom orders, and an index-wise conformer swap would then
        scramble the geometry silently.

    Different order or stereo simply misses the cache and re-derives charges
    (~10-40 s) against a 28-45 min relax. Cheap insurance.
    """
    import hashlib
    from collections import Counter
    from rdkit import Chem
    m = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=True)
    if m is None:
        return "unparsed_" + os.path.basename(sdf_path)
    mh = Chem.AddHs(m, addCoords=True)          # exactly what gets cached
    sig = (Chem.MolToSmiles(mh) + "|" +
           ",".join(str(r) for r in Chem.CanonicalRankAtoms(mh, breakTies=True)))
    c = Counter(a.GetSymbol() for a in mh.GetAtoms())
    formula = "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in sorted(c))
    return f"{formula}_{hashlib.sha1(sig.encode()).hexdigest()[:12]}"


def _charge_cache(sdf_path: str, cache_dir: str):
    """Return a charged openff Molecule for this pose, or None.

    Returns the object rather than a path on purpose: the previous version wrote
    the pickle and then re-read it by path, so a concurrent worker's os.replace
    landing in between handed back somebody else's molecule.
    """
    key = _mol_key(sdf_path)
    pkl = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"charge cache unreadable ({pkl}): {e}")
    try:
        from openff.toolkit.topology import Molecule
        from rdkit import Chem
        rd = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=True)
        if rd is None:
            return None
        off = Molecule.from_rdkit(Chem.AddHs(rd, addCoords=True),
                                  allow_undefined_stereo=True)
        # Charge the POSE's own geometry. Without use_conformers the toolkit
        # throws the pose away and re-embeds with ETKDG, which fails outright on
        # rigid bridged cages -- L020511's 3-hydroxy-1-adamantyl carboxamide
        # died as ConformerGenerationError -> "No registered toolkits can
        # provide assign_partial_charges" and lost its rank-0 pose. The pose is
        # a real, PB-checked 3D structure; there is nothing to gain by
        # discarding it and everything to lose.
        off.assign_partial_charges(partial_charge_method="am1bcc",
                                   use_conformers=off.conformers or None)
        tmp = pkl + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(off, f)
        os.replace(tmp, pkl)          # atomic: workers share this dir
        return off                    # our own object, never a re-read
    except Exception as e:
        log.warning(f"AM1-BCC precompute failed for {sdf_path}: {e}")
        return None


def _walk_target(args) -> dict:
    import openmm  # noqa: F401  (lock conda libstdc++ before posebusters)
    from casp17_ligand.utils.relax_pose import relax_iterative

    rec, out_root, cache_dir = args
    target = rec["target"]
    t0 = time.time()
    outdir = os.path.join(out_root, target)
    os.makedirs(outdir, exist_ok=True)

    def _finish(res):
        """Checkpoint every target as it lands. A full L01 walk is hours long;
        without this a crash at hour 6 loses all of it, and there is no way to
        watch progress while it runs."""
        with open(os.path.join(outdir, "_walk.json"), "w") as f:
            json.dump(res, f)
        return res

    trail = []
    for rank, cand in enumerate(rec["candidates"]):
        base = os.path.basename(cand["sdf"])
        work = os.path.join(outdir, base)
        shutil.copy2(cand["sdf"], work)          # never touch ranking_sucos
        pdb = cand["pdb"]

        if _pb_pass(work, pdb):
            trail.append(dict(rank=rank, sdf=base, outcome="orig_pass"))
            return _finish(dict(
                target=target, accepted=work, accepted_src=cand["sdf"],
                rank=rank, relaxed=0, drift=0.0, iters=0,
                status="orig_pass", trail=trail, wall=time.time() - t0))

        cached = _charge_cache(work, cache_dir)
        try:
            info = relax_iterative(pdb, work, work, pb_check_fn=_pb_pass,
                                   platform_name="CPU", step_iter=STEP_ITER,
                                   max_total_iter=MAX_ITER,
                                   ligand_restraint_k=LIGAND_K,
                                   cached_off_molecule=cached)
            # Retry exactly as relax_pose_batch does: continue from the pose the
            # first pass left behind (in-place), do NOT restart from the original.
            if not info["passed"]:
                info = relax_iterative(pdb, work, work, pb_check_fn=_pb_pass,
                                       platform_name="CPU", step_iter=STEP_ITER,
                                       max_total_iter=RETRY_MAX_ITER,
                                       ligand_restraint_k=LIGAND_K,
                                       cached_off_molecule=cached)
        except Exception as e:
            trail.append(dict(rank=rank, sdf=base, outcome=f"relax_error:{e}"))
            os.remove(work)
            continue

        if info["passed"]:
            trail.append(dict(rank=rank, sdf=base, outcome="relaxed_pass",
                              drift=info["ligand_drift_RMSD_A"]))
            return _finish(dict(
                target=target, accepted=work, accepted_src=cand["sdf"],
                rank=rank, relaxed=1,
                drift=info["ligand_drift_RMSD_A"], iters=info["iters_done"],
                status="relaxed_pass", trail=trail, wall=time.time() - t0))
        trail.append(dict(rank=rank, sdf=base, outcome="relax_failed",
                          drift=info["ligand_drift_RMSD_A"]))
        os.remove(work)

    # Every candidate exhausted. Keep the top-SuCOS pose unrelaxed and flag it,
    # so the target is still submitted, ranked last rather than dropped.
    top = rec["candidates"][0]
    work = os.path.join(outdir, os.path.basename(top["sdf"]))
    shutil.copy2(top["sdf"], work)
    return _finish(dict(
        target=target, accepted=work, accepted_src=top["sdf"], rank=0,
        relaxed=0, drift=0.0, iters=0, status="exhausted", trail=trail,
        wall=time.time() - t0))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reuse targets that already wrote <out-root>/<t>/_walk.json")
    a = ap.parse_args()

    recs = json.load(open(a.manifest))
    cache_dir = a.cache_dir or tempfile.mkdtemp(prefix="relax_cache_Lsel_")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(a.out_root, exist_ok=True)

    out, t0 = [], time.time()
    if a.resume:
        todo = []
        for r in recs:
            ck = os.path.join(a.out_root, r["target"], "_walk.json")
            try:
                done = json.load(open(ck))
            except Exception:
                todo.append(r)
                continue
            out.append(done)
        log.info(f"resume: {len(out)} already done, {len(todo)} to walk")
        recs = todo

    log.info(f"walking {len(recs)} targets, {a.workers} workers, cache={cache_dir}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_walk_target, (r, a.out_root, cache_dir)): r["target"]
                for r in recs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                out.append(f.result())
            except Exception as e:
                log.error(f"{futs[f]}: walk failed: {e}")
                out.append(dict(target=futs[f], status=f"error:{e}", accepted=None))
            if i % 25 == 0 or i == len(recs):
                c = {}
                for r in out:
                    c[r["status"].split(":")[0]] = c.get(r["status"].split(":")[0], 0) + 1
                log.info(f"  {i}/{len(recs)}  {c}  ({time.time()-t0:.0f}s)")

    json.dump(out, open(a.summary_out, "w"))
    c = {}
    for r in out:
        k = r["status"].split(":")[0]
        c[k] = c.get(k, 0) + 1
    log.info(f"done in {time.time()-t0:.0f}s: {c}")
    log.info(f"accepted beyond rank 0: "
             f"{sum(1 for r in out if r.get('rank', 0) > 0)}/{len(out)}")


if __name__ == "__main__":
    main()
