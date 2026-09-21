"""Unified confidence-metric analysis: pair_iptm, ligand_plddt, ranking_score.

Three confidence metrics for self-ranking evaluation:
  - pair_iptm:      max cross-chain iptm (protein → largest ligand)
  - ligand_plddt:   average B-factor of ligand HETATM atoms from CIF
  - ranking_score:  ranking_score (AF3/RF3) or confidence_score (SeedFold)

Output tables:
  1. all_model_{metric}.csv            — raw score + lDDT-PLI per model
  2. {metric}_per_target_analysis.csv   — per-target: score range, LDDT range,
                                          oracle LDDT, oracle rank
  3. self_ranking_comparison_{metric}.csv — self-ranked top-1/5/10 lDDT-PLI

Usage:
    python casp17_ligand/analysis/confidence_metric_analysis.py \\
        --metric pair_iptm --dataset casp16_l3000_struct --methods af3 seedfold

    python casp17_ligand/analysis/confidence_metric_analysis.py \\
        --metric all --dataset casp16_l3000_struct --methods af3 seedfold
"""

import argparse
import glob
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from casp17_ligand.analysis.self_ranking_comparison import (
    extract_pair_chains_iptm,
    _normalise_pair_matrix,
    _get_ensemble_cif_list,
    _map_cif_to_model_index,
)


ALL_METHODS = ["af3", "boltz2", "protenix", "rf3", "seedfold"]


def parse_model_source(model_name: str):
    """Parse model_name to extract (source_method, model_idx).

    Examples:
        'af3_model34_rank2_rmsd...' -> ('af3', 34)
        'seedfold_model12_rank1...' -> ('seedfold', 12)
        'boltz2_model0'             -> ('boltz2', 0)

    Handles names with or without trailing suffixes after the model index.
    """
    m = re.match(r"^(\w+?)_model(\d+)", model_name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


# ── helpers ────────────────────────────────────────────────────────────────

def _extract_method_and_model_idx(model_name: str) -> Tuple[str, int]:
    m = re.match(r"(\w+?)_model(\d+)_rank", model_name)
    if m:
        return m.group(1), int(m.group(2))
    return "", -1


def _try_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        fv = float(v)
        return fv if not math.isnan(fv) else None
    except (ValueError, TypeError):
        return None


def _resolve_af3_base(method_output_dir: str, target: str) -> str:
    """AF3 layouts in priority order:
      1. nested:        {base}/{TARGET}/{target_lower}/seed-*_sample-*/   (CASP15)
      2. target-direct: {base}/{TARGET}/seed-*_sample-*/                  (CASP17 R-series)
      3. flat:          {base}/{target_lower}/seed-*_sample-*/             (CASP16 single-target)
    """
    nested = os.path.join(method_output_dir, target, target.lower())
    if os.path.isdir(nested):
        return nested
    target_direct = os.path.join(method_output_dir, target)
    if os.path.isdir(target_direct):
        return target_direct
    return os.path.join(method_output_dir, target.lower())


def _higher_is_better(metric: str, method: str = "") -> bool:
    return True  # all three metrics: higher = better


def _iter_confidence_jsons(method: str, target: str, method_output_dir):
    """Yield (json_path, cif_path) pairs for a method × target.

    ``method_output_dir`` may be a single str or a list/tuple of dirs
    (for combining multiple data versions). When a list is provided,
    JSONs from each dir are yielded in the given order so idx matches
    ``_get_ensemble_cif_list`` output.
    """
    # Accept list/tuple/omegaconf ListConfig (anything iterable that is not a str)
    if not isinstance(method_output_dir, (str, bytes)) and method_output_dir is not None:
        try:
            dirs_iter = list(method_output_dir)
        except TypeError:
            dirs_iter = None
        if dirs_iter is not None:
            for d in dirs_iter:
                yield from _iter_confidence_jsons(method, target, d)
            return

    if method == "af3":
        base = _resolve_af3_base(method_output_dir, target)
        any_nested = False
        # Inside target-specific base, drop the filename prefix so the glob
        # tolerates any case / naming convention (R2314 used lowercase `r2314_*`,
        # R2317/R2318 use capital `R2317_*`).
        for jf in sorted(glob.glob(os.path.join(
                base, "seed-*_sample-*",
                "*_seed-*_sample-*_summary_confidences.json"))):
            any_nested = True
            cif = jf.replace("_summary_confidences.json", "_model.cif")
            yield jf, cif
        # Top-level summary_confidences without seed subdirs (rare rank-0-only case)
        flat_json = os.path.join(base, f"{target.lower()}_summary_confidences.json")
        flat_cif = os.path.join(base, f"{target.lower()}_model.cif")
        if os.path.isfile(flat_json) and os.path.isfile(flat_cif):
            yield flat_json, flat_cif
        # Flat dataset layout: all targets share method_output_dir; in this case
        # the filename DOES carry the target prefix to disambiguate co-located
        # JSONs from different targets. Allow either case.
        if not any_nested:
            patterns = [
                f"{target.lower()}*_seed-*_sample-*_summary_confidences.json",
                f"{target}*_seed-*_sample-*_summary_confidences.json",
            ]
            seen = set()
            for pat in patterns:
                for jf in sorted(glob.glob(os.path.join(
                        method_output_dir, "seed-*_sample-*", pat))):
                    if jf in seen:
                        continue
                    seen.add(jf)
                    cif = jf.replace("_summary_confidences.json", "_model.cif")
                    yield jf, cif

    elif method == "boltz2":
        from casp17_ligand.analysis.self_ranking_comparison import _find_boltz2_pred_dir
        base, sub = _find_boltz2_pred_dir(method_output_dir, target)
        if base:
            for jf in sorted(glob.glob(os.path.join(base, f"confidence_{sub}_model_*.json"))):
                midx = os.path.basename(jf).replace(f"confidence_{sub}_model_", "").replace(".json", "")
                cif = os.path.join(base, f"{sub}_model_{midx}.cif")
                yield jf, cif

    elif method == "protenix":
        base = os.path.join(method_output_dir, target)
        for jf in sorted(glob.glob(os.path.join(
                base, "seed_*", "predictions",
                f"{target}_summary_confidence_sample_*.json"))):
            cif = jf.replace("_summary_confidence_sample_", "_sample_").replace(".json", ".cif")
            yield jf, cif

    elif method == "rf3":
        base = method_output_dir
        for seed_dir in sorted(glob.glob(os.path.join(base, f"{target}_seed-*"))):
            csv_paths = sorted(glob.glob(os.path.join(
                seed_dir, "**", f"{target}_ranking_scores.csv"), recursive=True))
            cif_paths = sorted(glob.glob(os.path.join(
                seed_dir, "**", f"{target}_model.cif"), recursive=True))
            if csv_paths and cif_paths:
                yield csv_paths[0], cif_paths[0]

    elif method == "seedfold":
        base = os.path.join(method_output_dir, target)
        for jf in sorted(glob.glob(os.path.join(
                base, f"{target}_*", "confidence_*_model_*.json"))):
            cif = jf.replace("confidence_", "").replace(".json", ".cif")
            yield jf, cif


# ── score extractors ──────────────────────────────────────────────────────

def _extract_ranking_score(data_or_path, method: str, input_path: str = "") -> Optional[float]:
    """Extract ranking_score (AF3/Boltz2/Protenix) or confidence_score (SeedFold).
    For RF3, data_or_path is a CSV path string."""
    if method == "rf3":
        # RF3: ranking_score from CSV
        if isinstance(data_or_path, str) and os.path.isfile(data_or_path):
            try:
                df = pd.read_csv(data_or_path)
                if "ranking_score" in df.columns:
                    return float(df["ranking_score"].iloc[0])
            except Exception:
                pass
        return None

    # JSON-based methods
    data = data_or_path
    score = data.get("ranking_score")
    if score is not None:
        return float(score)
    score = data.get("confidence_score")
    if score is not None:
        return float(score)
    return None


def _extract_mini_ipae(data: dict, method: str, input_path: str = "") -> Optional[float]:
    """Extract inter-chain PAE/PDE metric, unified across methods.

    Returns a score where **higher = better** (for consistent ranking):
      - AF3:      chain_pair_pae_min → protein→ligand cell, negated (lower PAE = better)
      - Boltz2:   complex_ipde → negated scalar (lower iPDE = better)
      - Protenix: chain_pair_gpde → protein→ligand cell (higher gPDE = better, already correct)
      - SeedFold: confidence_score as fallback (no PAE/PDE available)
    """
    if method == "af3":
        raw = data.get("chain_pair_pae_min")
        if raw is None:
            return None
        mat = _normalise_pair_matrix(raw)
        if not mat:
            return None
        # Same logic as extract_pair_chains_iptm: find protein→ligand cell
        # For 2-chain (protein + ligand), it's mat["0"]["1"]
        # Use input_path to identify chains if available
        from casp17_ligand.analysis.self_ranking_comparison import _parse_input_entities
        prot_idx, lig_idx, _ = _parse_input_entities(input_path, method)
        if prot_idx and lig_idx:
            vals = []
            for pi in prot_idx:
                for li in lig_idx:
                    v = mat.get(str(pi), {}).get(str(li))
                    if v is not None:
                        vals.append(v)
            if vals:
                return -min(vals)  # Negate: lower PAE = better → higher score
        # Fallback: assume chain 0=protein, 1=ligand
        v = mat.get("0", {}).get("1")
        if v is not None:
            return -v
        return None

    elif method == "boltz2":
        v = data.get("complex_ipde")
        if v is not None:
            return -float(v)  # Negate: lower iPDE = better
        return None

    elif method == "protenix":
        raw = data.get("chain_pair_gpde")
        if raw is None:
            return None
        mat = _normalise_pair_matrix(raw)
        if not mat:
            return None
        from casp17_ligand.analysis.self_ranking_comparison import _parse_input_entities
        prot_idx, lig_idx, _ = _parse_input_entities(input_path, method)
        if prot_idx and lig_idx:
            vals = []
            for pi in prot_idx:
                for li in lig_idx:
                    v = mat.get(str(pi), {}).get(str(li))
                    if v is not None:
                        vals.append(v)
            if vals:
                return max(vals)  # Higher gPDE = better
        v = mat.get("0", {}).get("1")
        if v is not None:
            return float(v)
        return None

    elif method == "seedfold":
        # SeedFold has no PAE/PDE; use confidence_score as fallback
        v = data.get("confidence_score")
        if v is not None:
            return float(v)
        return None

    return None


def _parse_cif_b_factors(cif_path: str, pocket_radius: float = 0.0) -> Optional[float]:
    """Average B_iso_or_equiv for ligand atoms, optionally including nearby atoms.

    pocket_radius == 0: ligand HETATM atoms only (original ligand_plddt behavior).
    pocket_radius > 0: ligand atoms + all ATOM/HETATM within pocket_radius Å of any
                       ligand atom (pocket pLDDT).
    """
    if not os.path.isfile(cif_path):
        return None
    in_atom_site = False
    headers: List[str] = []
    ligand_coords: List[Tuple[float, float, float]] = []
    ligand_bfactors: List[float] = []
    protein_atoms: List[Tuple[float, float, float, float]] = []  # (x, y, z, b)

    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("loop_"):
                in_atom_site = False
                headers = []
                continue
            if line.startswith("_atom_site."):
                in_atom_site = True
                headers.append(line.split()[0].strip())
                continue
            if in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                if len(parts) < len(headers):
                    continue
                try:
                    idx_group = headers.index("_atom_site.group_PDB")
                    idx_comp = headers.index("_atom_site.label_comp_id")
                    idx_b = -1
                    for name in ["_atom_site.B_iso_or_equiv", "_atom_site.b_iso_or_equiv"]:
                        if name in headers:
                            idx_b = headers.index(name)
                            break
                    if idx_b == -1:
                        return None

                    is_water = parts[idx_comp] in ("HOH", "WAT", "H2O")
                    is_hetatm = parts[idx_group] == "HETATM"
                    b_val = float(parts[idx_b])

                    if is_hetatm and not is_water:
                        ligand_bfactors.append(b_val)
                        if pocket_radius > 0:
                            idx_x = headers.index("_atom_site.Cartn_x")
                            idx_y = headers.index("_atom_site.Cartn_y")
                            idx_z = headers.index("_atom_site.Cartn_z")
                            ligand_coords.append((float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])))
                    elif pocket_radius > 0 and not is_water:
                        idx_x = headers.index("_atom_site.Cartn_x")
                        idx_y = headers.index("_atom_site.Cartn_y")
                        idx_z = headers.index("_atom_site.Cartn_z")
                        protein_atoms.append((float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z]), b_val))
                except (ValueError, IndexError):
                    pass

    if not ligand_bfactors:
        return None

    if pocket_radius <= 0 or not protein_atoms:
        return float(np.mean(ligand_bfactors))

    # Find protein atoms within pocket_radius of any ligand atom
    lig_arr = np.array(ligand_coords)  # (L, 3)
    prot_arr = np.array([(x, y, z) for x, y, z, _ in protein_atoms])  # (P, 3)
    prot_b = np.array([b for _, _, _, b in protein_atoms])

    # Vectorized distance: (L, 1, 3) - (1, P, 3) -> (L, P)
    dists = np.sqrt(np.sum((lig_arr[:, None, :] - prot_arr[None, :, :]) ** 2, axis=-1))
    min_dists = dists.min(axis=0)  # (P,) min distance to any ligand atom
    pocket_mask = min_dists <= pocket_radius

    all_bfactors = ligand_bfactors + prot_b[pocket_mask].tolist()
    return float(np.mean(all_bfactors))


# ── generic score collection ──────────────────────────────────────────────

def collect_scores(
    metric: str,
    method: str,
    target: str,
    method_output_dir,
    input_json_dir: str = "",
    rtmscore_cache_dir: Optional[str] = None,
    n_protein_chains: int = 0,
    n_cofactor_chains: int = 0,
) -> Dict[int, float]:
    """Collect {ensemble_model_index: score} for one metric × method × target.

    ``method_output_dir`` may be a single str or a list/tuple of dirs.
    When a list, CIFs/JSONs from each dir are concatenated in order
    (idx 0..N-1 spans all dirs).

    ``rtmscore_cache_dir`` overrides the RTMScore cache root directory.
    If None, falls back to ``outputs/ensemble/{basename(method_output_dir)}/
    rtmscore_self_ranking`` (legacy behavior — only sensible for single str).

    ``n_protein_chains`` must be set for multimeric targets; it is the fallback
    used by ``extract_pair_chains_iptm`` when the input file is missing (always
    the case for Boltz-2, whose input is YAML) or disagrees with the confidence
    matrix.  Default 0 → 1 preserves the historical single-chain behaviour.

    ``n_cofactor_chains`` must be set for targets shipped with a catalytic
    cofactor (CASP17 L-series: one ZN or SFG), otherwise ``pair_iptm`` reports
    the protein–cofactor confidence instead of the protein–fragment one.
    Default 0 → no chain is dropped.
    """
    input_path = ""
    if input_json_dir:
        candidate = os.path.join(input_json_dir, f"{target}.json")
        if os.path.isfile(candidate):
            input_path = candidate

    ensemble_cifs = _get_ensemble_cif_list(method, target, method_output_dir)
    canon_to_idx = {os.path.realpath(p): i for i, p in enumerate(ensemble_cifs)}

    results: Dict[int, float] = {}

    # CIF-based metrics: ligand_plddt, pocket_plddt_*, combined_iptm_plddt
    import re as _re
    pocket_match = _re.match(r"pocket_plddt_([\d.]+)", metric)

    if metric == "ligand_plddt":
        for cif in ensemble_cifs:
            score = _parse_cif_b_factors(cif, pocket_radius=0.0)
            if score is not None:
                idx = canon_to_idx.get(os.path.realpath(cif))
                if idx is not None:
                    results[idx] = score
    elif pocket_match:
        radius = float(pocket_match.group(1))
        for cif in ensemble_cifs:
            score = _parse_cif_b_factors(cif, pocket_radius=radius)
            if score is not None:
                idx = canon_to_idx.get(os.path.realpath(cif))
                if idx is not None:
                    results[idx] = score
    elif metric == "combined_iptm_plddt":
        # pair_iptm + ligand_plddt/100
        plddt_scores: Dict[int, float] = {}
        for cif in ensemble_cifs:
            score = _parse_cif_b_factors(cif, pocket_radius=0.0)
            if score is not None:
                idx = canon_to_idx.get(os.path.realpath(cif))
                if idx is not None:
                    plddt_scores[idx] = score
        iptm_scores: Dict[int, float] = {}
        for jf_or_csv, cif in _iter_confidence_jsons(method, target, method_output_dir):
            try:
                with open(jf_or_csv) as f:
                    data = json.load(f)
                score = extract_pair_chains_iptm(
                    data, method, input_path,
                    n_protein_chains=n_protein_chains,
                    n_cofactor_chains=n_cofactor_chains)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if score is not None:
                idx = canon_to_idx.get(os.path.realpath(cif))
                if idx is not None:
                    iptm_scores[idx] = score
        # Combine: pair_iptm + ligand_plddt / 100
        all_idxs = set(plddt_scores.keys()) | set(iptm_scores.keys())
        for idx in all_idxs:
            iptm = iptm_scores.get(idx)
            plddt = plddt_scores.get(idx)
            if iptm is not None and plddt is not None:
                results[idx] = iptm + plddt / 100.0
            elif iptm is not None:
                results[idx] = iptm
            elif plddt is not None:
                results[idx] = plddt / 100.0
    elif metric == "rtmscore":
        # RTMScore: load from pre-computed cache produced by rtmscore_self_ranking.py
        # Cache lives in: {rtmscore_cache_dir}/{target}_{method}.json
        if rtmscore_cache_dir:
            cache_dir = rtmscore_cache_dir
        else:
            # Legacy path derivation: outputs/ensemble/{dataset}/rtmscore_self_ranking/
            # Only sensible when method_output_dir is a single str.
            single_dir = method_output_dir
            if isinstance(single_dir, (list, tuple)):
                single_dir = single_dir[0] if single_dir else ""
            dataset = os.path.basename(single_dir)
            cache_dir = os.path.join("outputs/ensemble", dataset, "rtmscore_self_ranking")
        rtm_cache_path = os.path.join(cache_dir, f"{target}_{method}.json")
        if not os.path.isfile(rtm_cache_path):
            return results
        with open(rtm_cache_path) as f:
            rtm_data = json.load(f)
        for idx_str, score in rtm_data.items():
            results[int(idx_str)] = float(score)
        return results
    else:
        # JSON/CSV based: pair_iptm, ranking_score, mini_ipae
        for jf_or_csv, cif in _iter_confidence_jsons(method, target, method_output_dir):
            try:
                if metric == "pair_iptm":
                    with open(jf_or_csv) as f:
                        data = json.load(f)
                    score = extract_pair_chains_iptm(
                        data, method, input_path,
                        n_protein_chains=n_protein_chains,
                        n_cofactor_chains=n_cofactor_chains)
                elif metric == "ranking_score":
                    if method == "rf3":
                        score = _extract_ranking_score(jf_or_csv, method)
                    else:
                        with open(jf_or_csv) as f:
                            data = json.load(f)
                        score = _extract_ranking_score(data, method, input_path)
                elif metric == "mini_ipae":
                    if method == "rf3":
                        score = None
                    else:
                        with open(jf_or_csv) as f:
                            data = json.load(f)
                        score = _extract_mini_ipae(data, method, input_path)
                else:
                    score = None
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

            if score is None:
                continue
            idx = canon_to_idx.get(os.path.realpath(cif))
            if idx is not None:
                results[idx] = score

    return results


# ── per-target worker (for parallel execution) ───────────────────────────

def _collect_target_records(
    target: str,
    metric: str,
    active_methods: List[str],
    methods_config: Dict[str, str],
    input_dirs: Dict[str, str],
    df_eval_target: pd.DataFrame,
) -> List[dict]:
    """Collect all (metric_score, lddt_pli) records for one target. Runs in subprocess."""
    records = []
    for method in active_methods:
        method_dir = methods_config.get(method, "")
        input_dir = input_dirs.get(method, "")
        scores = collect_scores(metric, method, target, method_dir, input_dir)

        for _, row in df_eval_target.iterrows():
            m_name, m_idx = _extract_method_and_model_idx(row["model_name"])
            if m_name != method:
                continue
            score = scores.get(m_idx)
            if score is not None and row["lddt_pli"] is not None:
                records.append({
                    "target": target,
                    "method": method,
                    "model_idx": m_idx,
                    "metric_score": score,
                    "lddt_pli": row["lddt_pli"],
                    "rmsd": row["rmsd"],
                })
    return records


# ── main analysis ─────────────────────────────────────────────────────────

def run_analysis(
    metric: str,
    dataset: str,
    ensemble_dir: str = "outputs/ensemble",
    methods: Optional[List[str]] = None,
    n_workers: int = 8,
):
    ens_base = os.path.join(ensemble_dir, dataset)
    # Dataset tag for CSV filenames: casp16_l3000_struct → L3000_STRUCT
    ds_tag = "_".join(dataset.split("_")[1:]).upper() + "_"
    out_dir = os.path.join(ens_base, "confidence_analysis")
    os.makedirs(out_dir, exist_ok=True)

    methods_config = {
        "af3":       f"outputs/alphafold3/{dataset}",
        "boltz2":    f"outputs/boltz2/{dataset}",
        "protenix":  f"outputs/protenix/{dataset}",
        "rf3":       f"outputs/rf3/{dataset}",
        "seedfold":  f"outputs/seedfold/{dataset}",
    }
    input_dirs = {
        "af3":       f"data/test_cases/{dataset}/af3_inputs",
        "protenix":  f"data/test_cases/{dataset}/protenix_inputs",
        "seedfold":  f"data/test_cases/{dataset}/seedfold_inputs",
    }

    active_methods = methods or ALL_METHODS

    eval_path = os.path.join(ens_base, "evaluation_summary.csv")
    top_path = os.path.join(ens_base, f"top1_top5_scores_{dataset.split('_')[-1].upper()}.csv")
    df_eval = pd.read_csv(eval_path)
    df_top = pd.read_csv(top_path)
    df_eval_rmsd = df_eval[df_eval["method"] == "ranking_rmsd"].copy()
    targets = sorted(df_eval_rmsd["target"].unique())

    print(f"\n{'='*80}")
    print(f"  Metric: {metric}  |  Dataset: {dataset}  |  Methods: {active_methods}")
    print(f"{'='*80}")

    # ── 1. Parallel score collection ──────────────────────────────────────
    print(f"\nCollecting {metric} scores ({n_workers} workers)...")
    all_records = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for t in targets:
            td = df_eval_rmsd[df_eval_rmsd["target"] == t]
            fut = pool.submit(
                _collect_target_records, t, metric, active_methods,
                methods_config, input_dirs, td,
            )
            futures[fut] = t

        for fut in as_completed(futures):
            recs = fut.result()
            all_records.extend(recs)

    df_all = pd.DataFrame(all_records)
    if df_all.empty:
        print("No data found! Check method outputs exist.")
        return
    score_csv = os.path.join(out_dir, f"{ds_tag}all_model_{metric}.csv")
    df_all.to_csv(score_csv, index=False)
    print(f"  Saved {len(df_all)} records to {score_csv}")

    # ── 2. Per-target analysis (merged oracle + variance) ─────────────────
    print(f"\n{'='*80}")
    print(f"Per-Target Analysis: {metric}")
    print(f"{'='*80}")

    analysis_rows = []
    for t in targets:
        rd = {"Target": t}
        for method in active_methods:
            mdf = df_all[(df_all["target"] == t) & (df_all["method"] == method)].copy()
            mdf = mdf.dropna(subset=["lddt_pli"])
            if mdf.empty:
                continue

            n_models = len(mdf)

            # Score range
            s_min, s_max = mdf["metric_score"].min(), mdf["metric_score"].max()
            rd[f"{method} Score Range"] = f"{s_max - s_min:.4f}"
            rd[f"{method} Score [min,max]"] = f"[{s_min:.3f},{s_max:.3f}]"

            # LDDT range
            l_min, l_max = mdf["lddt_pli"].min(), mdf["lddt_pli"].max()
            rd[f"{method} LDDT Range"] = f"{l_max - l_min:.3f}"
            rd[f"{method} LDDT [min,max]"] = f"[{l_min:.3f},{l_max:.3f}]"

            # Spearman ρ (metric_score vs lddt_pli)
            if n_models >= 3 and mdf["metric_score"].nunique() > 1 and mdf["lddt_pli"].nunique() > 1:
                sr_val, _ = stats.spearmanr(mdf["metric_score"], mdf["lddt_pli"])
                rd[f"{method} Spearman ρ"] = f"{sr_val:.4f}"
            else:
                rd[f"{method} Spearman ρ"] = "—"

            # Sort by metric_score descending → assign ranks
            mdf_sorted = mdf.sort_values("metric_score", ascending=False).reset_index(drop=True)
            mdf_sorted["score_rank"] = range(1, n_models + 1)

            # Rank-1 LDDT: lDDT of the model that this metric ranks first
            rd[f"{method} Rank1 LDDT"] = f"{mdf_sorted.loc[0, 'lddt_pli']:.3f}"

            # Oracle: best-lDDT model's rank
            best_lddt_idx = mdf_sorted["lddt_pli"].idxmax()
            rd[f"{method} Oracle LDDT"] = f"{mdf_sorted.loc[best_lddt_idx, 'lddt_pli']:.3f}"
            rd[f"{method} Oracle Rank"] = int(mdf_sorted.loc[best_lddt_idx, "score_rank"])

            # Top-20% Position: earliest metric rank at which a top-20% lDDT model appears
            # e.g., 50 models → top 10 by lDDT → what's the best (earliest) rank (by metric) among those 10?
            top_k = max(1, int(np.ceil(n_models * 0.2)))
            top20_lddts = mdf_sorted.nlargest(top_k, "lddt_pli")
            first_rank_in_top20 = int(top20_lddts["score_rank"].min())
            rd[f"{method} Top20% Pos"] = first_rank_in_top20

            rd[f"{method} #Models"] = n_models

        analysis_rows.append(rd)

    # AVERAGE row
    avg_rd = {"Target": "AVERAGE"}
    all_cols = set()
    for r in analysis_rows:
        all_cols.update(r.keys())
    for col in all_cols:
        if col == "Target":
            continue
        vals = [_try_float(r.get(col, "")) for r in analysis_rows]
        vals = [v for v in vals if v is not None]
        avg_rd[col] = f"{sum(vals)/len(vals):.3f}" if vals else ""
    analysis_rows.append(avg_rd)

    # MAX row (worst-case for rank/position cols, max for others)
    max_rd = {"Target": "MAX"}
    for col in all_cols:
        if col == "Target":
            continue
        vals = [_try_float(r.get(col, "")) for r in analysis_rows[:-1]]  # exclude AVERAGE
        vals = [v for v in vals if v is not None]
        max_rd[col] = f"{max(vals):.3f}" if vals else ""
    analysis_rows.append(max_rd)

    df_analysis = pd.DataFrame(analysis_rows)
    analysis_csv = os.path.join(out_dir, f"{ds_tag}{metric}_per_target_analysis.csv")
    df_analysis.to_csv(analysis_csv, index=False)

    # Print compact per method
    for method in active_methods:
        cols_exist = [c for c in df_analysis.columns if method in c]
        if not cols_exist:
            continue
        print(f"\n  {method}:")
        for _, row in df_analysis.iterrows():
            t = row["Target"]
            lr = row.get(f"{method} LDDT Range", "")
            lm = row.get(f"{method} LDDT [min,max]", "")
            sp = row.get(f"{method} Spearman ρ", "")
            r1 = row.get(f"{method} Rank1 LDDT", "")
            ol = row.get(f"{method} Oracle LDDT", "")
            ork = row.get(f"{method} Oracle Rank", "")
            t20 = row.get(f"{method} Top20% Pos", "")
            n = row.get(f"{method} #Models", "")
            print(f"    {str(t):8s}"
                  f"  LDDT={str(lr):>6s} {str(lm):>20s}"
                  f"  ρ={str(sp):>7s}"
                  f"  Rank1={str(r1):>6s}"
                  f"  Oracle={str(ol):>6s} @{str(ork):>3s}"
                  f"  Top20%@{str(t20):>3s}"
                  f"  /{str(n):>3s}")
    print(f"\nSaved to {analysis_csv}")

    # ── 3. Self-ranking comparison ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Self-Ranking Comparison ({metric})")
    print(f"{'='*80}")

    sr_rows = []
    for t in targets:
        row = {"Target": t}

        t_top = df_top[df_top["Target"] == t]
        if not t_top.empty:
            row["Consensus RMSD Top-1 LDDT"] = t_top.iloc[0].get("ranking_rmsd (Top-1 LDDT)", "")
            row["Consensus SuCOS Top-1 LDDT"] = t_top.iloc[0].get("ranking_sucos (Top-1 LDDT)", "")

        td_eval = df_eval_rmsd[df_eval_rmsd["target"] == t]

        for method in active_methods:
            method_dir = methods_config.get(method, "")
            oracle_col = f"{method} Best LDDT-PLI"
            if not t_top.empty and oracle_col in t_top.columns:
                val = t_top.iloc[0].get(oracle_col, "")
                row[oracle_col] = "" if (isinstance(val, float) and math.isnan(val)) else val

            # Build ranked list
            scores = collect_scores(metric, method, t, method_dir, input_dirs.get(method, ""))
            ensemble_cifs = _get_ensemble_cif_list(method, t, method_dir)

            scored_items = [(s, ensemble_cifs[idx]) for idx, s in scores.items()
                            if idx < len(ensemble_cifs)]
            scored_items.sort(key=lambda x: x[0], reverse=True)
            mapped = _map_cif_to_model_index(scored_items, ensemble_cifs)

            lddts = []
            for score, model_idx in mapped:
                model_prefix = f"{method}_model{model_idx}"
                match = td_eval[td_eval["model_name"].str.startswith(model_prefix + "_rank")]
                if not match.empty and match.iloc[0]["lddt_pli"] is not None:
                    lddts.append(match.iloc[0]["lddt_pli"])

            row[f"{method} Self-Top1 LDDT"] = f"{lddts[0]:.3f}" if len(lddts) >= 1 else ""
            row[f"{method} Self-Top5 Best LDDT"] = (
                f"{max(lddts[:5]):.3f}" if len(lddts) >= 1 else "")
            row[f"{method} Self-Top10 Best LDDT"] = (
                f"{max(lddts[:10]):.3f}" if len(lddts) >= 1 else "")

        sr_rows.append(row)

    # Average
    avg_sr = {"Target": "AVERAGE"}
    sr_cols = set()
    for r in sr_rows:
        sr_cols.update(r.keys())
    for col in sr_cols:
        if col == "Target":
            continue
        vals = [_try_float(r.get(col, "")) for r in sr_rows]
        vals = [v for v in vals if v is not None]
        avg_sr[col] = f"{sum(vals)/len(vals):.3f}" if vals else ""
    sr_rows.append(avg_sr)

    df_sr = pd.DataFrame(sr_rows)
    sr_csv = os.path.join(ens_base, f"{ds_tag}self_ranking_comparison_{metric}.csv")
    df_sr.to_csv(sr_csv, index=False)
    # Print only AVERAGE row
    avg_line = df_sr[df_sr["Target"] == "AVERAGE"]
    print(avg_line.to_string(index=False))
    print(f"\nSaved {len(df_sr)} rows to {sr_csv}")

    # ── Cross-metric comparison ───────────────────────────────────────────
    other_metrics = [m for m in ["pair_iptm", "ligand_plddt", "ranking_score"] if m != metric]
    for other in other_metrics:
        other_csv = os.path.join(ens_base, f"{ds_tag}self_ranking_comparison_{other}.csv")
        if not os.path.isfile(other_csv):
            continue
        df_other = pd.read_csv(other_csv)
        other_avg = df_other[df_other["Target"] == "AVERAGE"]
        if other_avg.empty:
            continue
        other_avg = other_avg.iloc[0]
        cur_avg = df_sr[df_sr["Target"] == "AVERAGE"].iloc[0]
        print(f"\n--- Self-Top1 LDDT: {metric} vs {other} ---")
        for method in active_methods:
            col = f"{method} Self-Top1 LDDT"
            prev_v = other_avg.get(col, "")
            new_v = cur_avg.get(col, "")
            print(f"  {method:10s}: {other}={prev_v}  |  {metric}={new_v}")

    print(f"\n{'='*80}")
    print(f"All {metric} analysis saved to: {out_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified confidence metric analysis (pair_iptm / ligand_plddt / ranking_score)")
    parser.add_argument("--metric", default="pair_iptm",
                        help="Which metric to analyze (default: pair_iptm). "
                             "Supported: pair_iptm, ligand_plddt, ranking_score, "
                             "mini_ipae, rtmscore, pocket_plddt_X (X=4.5/6/8/10), "
                             "combined_iptm_plddt, all")
    parser.add_argument("--dataset", default="casp16_l1000")
    parser.add_argument("--ensemble_dir", default="outputs/ensemble")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to include (default: all). E.g.: --methods af3 seedfold")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of parallel workers for score collection")
    args = parser.parse_args()

    if args.metric == "all":
        for m in ["pair_iptm", "ligand_plddt", "ranking_score", "mini_ipae"]:
            run_analysis(m, args.dataset, args.ensemble_dir, args.methods, args.workers)
    else:
        run_analysis(args.metric, args.dataset, args.ensemble_dir, args.methods, args.workers)


if __name__ == "__main__":
    main()
