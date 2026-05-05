"""Compare each method's self-ranking (native confidence scores) against
consensus ranking (RMSD/SuCOS) and oracle bounds.

Usage:
    python casp17_ligand/analysis/self_ranking_comparison.py [--dataset casp16_l1000]

Output: outputs/ensemble/{dataset}/self_ranking_comparison.csv
"""

import argparse
import glob
import json
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Unified pair_chains_iptm extraction (shared by all methods)
# ---------------------------------------------------------------------------

def _normalise_pair_matrix(raw) -> Optional[Dict[str, Dict[str, float]]]:
    """Convert pair_chains_iptm / chain_pair_iptm to a uniform nested-dict
    representation  {row_str: {col_str: float}}.

    Accepts:
      - nested dict  {"0": {"0": v, "1": v}, ...}    (Boltz2, Seedfold)
      - 2-D list     [[v, v], [v, v]]                (AF3, Protenix)
    Returns None if the field is missing / empty.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        # Already nested dict — just make sure keys are strings
        return {str(r): {str(c): v for c, v in cols.items()}
                for r, cols in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, Dict[str, float]] = {}
        for i, row in enumerate(raw):
            out[str(i)] = {str(j): v for j, v in enumerate(row)}
        return out
    return None


def _parse_input_entities(
    input_path: str,
    method: str,
) -> Tuple[List[int], List[int], Dict[int, str]]:
    """Parse an input file (JSON / YAML) and return:
       (protein_chain_indices, ligand_chain_indices, {chain_idx: smiles}).

    Each method has its own input schema:
      - AF3:      sequences → protein / ligand
      - Protenix: sequences → proteinChain / ligand
      - Seedfold: entities  → Protein / Ligand-Smiles
      - Boltz2:   YAML sequences → protein / smiles / sdf / ccd
    """
    protein_indices: List[int] = []
    ligand_indices: List[int] = []
    ligand_smiles: Dict[int, str] = {}

    if not input_path or not os.path.isfile(input_path):
        return protein_indices, ligand_indices, ligand_smiles

    with open(input_path) as f:
        inp = json.load(f)

    # Handle list-wrapped inputs (all methods may wrap in a list)
    if isinstance(inp, list):
        inp = inp[0]

    # --- Seedfold: entities list ---
    if method == "seedfold" and "entities" in inp:
        for idx, ent in enumerate(inp["entities"]):
            etype = ent.get("entity", "").lower()
            if "protein" in etype:
                protein_indices.append(idx)
            elif "ligand" in etype:
                ligand_indices.append(idx)
                ligand_smiles[idx] = ent.get("sequence", "")
        return protein_indices, ligand_indices, ligand_smiles

    # --- AF3 / Protenix / Boltz2-JSON: sequences list ---
    sequences = inp.get("sequences", [])
    idx = 0
    for seq_entry in sequences:
        if isinstance(seq_entry, dict):
            for key in seq_entry:
                kl = key.lower()
                if "protein" in kl:
                    protein_indices.append(idx)
                elif "ligand" in kl or kl in ("smiles", "sdf", "ccd"):
                    ligand_indices.append(idx)
                    # Try to extract SMILES
                    inner = seq_entry[key]
                    if isinstance(inner, dict):
                        smi = inner.get("smiles", "") or inner.get("ligand", "")
                        ligand_smiles[idx] = smi
                    elif isinstance(inner, str):
                        ligand_smiles[idx] = inner
                # Only count the first key per entry as one chain
                idx += 1
                break

    return protein_indices, ligand_indices, ligand_smiles


def extract_pair_chains_iptm(
    confidence_data: dict,
    method: str,
    input_path: Optional[str] = None,
) -> Optional[float]:
    """Extract the best protein–largest-ligand cross-chain iptm score.

    Generic function shared by all four methods.  Handles:
      - Different field names  (pair_chains_iptm / chain_pair_iptm)
      - Different formats      (nested dict / 2-D list)
      - Multi-chain / multi-ligand scenarios

    Steps:
      1. Read the method input file to identify Protein vs Ligand chains.
      2. Among ligands, find the largest (by heavy-atom count). If tied, keep all.
      3. Return the max cross-chain value among (protein, target_ligand) pairs.
      4. Fallback: max off-diagonal if input file is unavailable.
    """
    # Try both field names
    raw = confidence_data.get("pair_chains_iptm") or confidence_data.get("chain_pair_iptm")
    pci = _normalise_pair_matrix(raw)
    if not pci:
        return None

    chain_ids = sorted(pci.keys(), key=int)
    n_chains = len(chain_ids)

    # --- Identify protein / ligand chain indices from input ---
    protein_indices, ligand_indices, ligand_smiles = _parse_input_entities(
        input_path or "", method
    )

    # Fallback if no input file
    if not protein_indices or not ligand_indices:
        if n_chains >= 2:
            protein_indices = [0]
            ligand_indices = list(range(1, n_chains))
        else:
            return None

    if not protein_indices or not ligand_indices:
        # Last resort: max off-diagonal
        best = None
        for i in chain_ids:
            for j in chain_ids:
                if i != j:
                    v = pci[i].get(j)
                    if v is not None and (best is None or v > best):
                        best = v
        return best

    # --- Find the largest ligand(s) by heavy-atom count ---
    target_ligand_indices = ligand_indices  # default: all
    if len(ligand_indices) > 1 and HAS_RDKIT:
        ha_counts: Dict[int, int] = {}
        for li in ligand_indices:
            smi = ligand_smiles.get(li, "")
            if smi:
                mol = Chem.MolFromSmiles(smi)
                ha_counts[li] = mol.GetNumHeavyAtoms() if mol else 0
            else:
                ha_counts[li] = 0
        max_ha = max(ha_counts.values())
        target_ligand_indices = [li for li, ha in ha_counts.items() if ha == max_ha]
    elif len(ligand_indices) > 1 and not HAS_RDKIT:
        warnings.warn(
            "RDKit not available; using all ligand chains for pair_chains_iptm"
        )

    # --- Find max cross-chain iptm ---
    best = None
    for pi in protein_indices:
        pi_str = str(pi)
        for li in target_ligand_indices:
            li_str = str(li)
            # Check both directions (should be symmetric, but be safe)
            for row, col in [(pi_str, li_str), (li_str, pi_str)]:
                if row in pci and col in pci[row]:
                    v = pci[row][col]
                    if best is None or v > best:
                        best = v
    return best


# Keep the old name as an alias for backward compatibility
extract_seedfold_pair_chains_iptm = extract_pair_chains_iptm


# ---------------------------------------------------------------------------
# Per-method confidence loaders
# Each returns [(score, cif_path), ...] sorted by score descending.
# cif_path is the CIF that was fed to ensemble_generation so we can map to
# the model index (boltz2_model3, af3_model12, ...).
# ---------------------------------------------------------------------------

def _load_af3_scores(
    target: str,
    method_output_dir: str,
    input_json_dir: str = "",
) -> List[Tuple[float, str]]:
    """AF3: chain_pair_iptm (protein-ligand cross-chain iptm)."""
    # Try nested path first: {base_dir}/{TARGET}/{target_lower}/
    # Then flat path: {base_dir}/{target_lower}/
    base = os.path.join(method_output_dir, target, target.lower())
    if not os.path.isdir(base):
        base = os.path.join(method_output_dir, target.lower())
    # `{target_lower}*` matches both `r2314_seed-*` and `r2314_r1_seed-*` styles.
    jsons = sorted(glob.glob(os.path.join(
        base, "seed-*_sample-*",
        f"{target.lower()}*_seed-*_sample-*_summary_confidences.json")))

    # Resolve input json for chain identification
    input_path = ""
    if input_json_dir:
        candidate = os.path.join(input_json_dir, f"{target}.json")
        if os.path.isfile(candidate):
            input_path = candidate

    items = []
    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        score = extract_pair_chains_iptm(data, "af3", input_path)
        if score is None:
            continue
        cif = jf.replace("_summary_confidences.json", "_model.cif")
        items.append((score, cif))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def _find_boltz2_pred_dir(method_output_dir: str, target: str) -> Tuple[str, str]:
    """Find boltz2 prediction subdir and its name. Returns (pred_dir, subdir_name)."""
    pred_parent = os.path.join(method_output_dir,
                               f"boltz_results_{target}_input", "predictions")
    if not os.path.isdir(pred_parent):
        return "", ""
    subdirs = [d for d in os.listdir(pred_parent)
               if os.path.isdir(os.path.join(pred_parent, d))]
    if not subdirs:
        return "", ""
    sub = subdirs[0]
    return os.path.join(pred_parent, sub), sub


def _load_boltz2_scores(
    target: str,
    method_output_dir: str,
    input_json_dir: str = "",
) -> List[Tuple[float, str]]:
    """Boltz2: pair_chains_iptm (protein-ligand cross-chain iptm)."""
    base, sub = _find_boltz2_pred_dir(method_output_dir, target)
    if not base:
        return []
    jsons = sorted(glob.glob(os.path.join(base, f"confidence_{sub}_model_*.json")))

    # Boltz2 input is YAML — no JSON input to parse, use fallback
    # (chain 0 = protein, chain 1 = ligand — standard ordering)

    items = []
    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        score = extract_pair_chains_iptm(data, "boltz2")
        if score is None:
            continue
        # Extract model index from filename
        idx = os.path.basename(jf).replace(f"confidence_{sub}_model_", "").replace(".json", "")
        cif = os.path.join(base, f"{sub}_model_{idx}.cif")
        items.append((score, cif))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def _load_protenix_scores(
    target: str,
    method_output_dir: str,
    input_json_dir: str = "",
) -> List[Tuple[float, str]]:
    """Protenix: chain_pair_iptm (protein-ligand cross-chain iptm)."""
    base = os.path.join(method_output_dir, target)
    jsons = sorted(glob.glob(os.path.join(
        base, "seed_*", "predictions",
        f"{target}_summary_confidence_sample_*.json")))

    # Resolve input json for chain identification
    input_path = ""
    if input_json_dir:
        candidate = os.path.join(input_json_dir, f"{target}.json")
        if os.path.isfile(candidate):
            input_path = candidate

    items = []
    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        score = extract_pair_chains_iptm(data, "protenix", input_path)
        if score is None:
            continue
        # Corresponding CIF
        cif = jf.replace("_summary_confidence_sample_", "_sample_").replace(".json", ".cif")
        items.append((score, cif))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def _load_seedfold_scores(
    target: str,
    method_output_dir: str,
    input_json_dir: str = "",
) -> List[Tuple[float, str]]:
    """Seedfold: pair_chains_iptm (protein-ligand cross-chain iptm).

    Handles two directory naming conventions:
      - L1001_model_* (legacy)  and  L1002_job_* (current)
    """
    base = os.path.join(method_output_dir, target)
    # Generic glob: match any subdirectory starting with target name
    jsons = sorted(glob.glob(os.path.join(
        base, f"{target}_*", "confidence_*_model_*.json")))

    # Resolve input json for chain identification
    input_path = ""
    if input_json_dir:
        candidate = os.path.join(input_json_dir, f"{target}.json")
        if os.path.isfile(candidate):
            input_path = candidate

    items = []
    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        score = extract_pair_chains_iptm(data, "seedfold", input_path)
        if score is None:
            continue
        # CIF path: replace "confidence_" prefix and ".json" -> ".cif"
        cif = jf.replace("confidence_", "").replace(".json", ".cif")
        items.append((score, cif))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


# Registry: method_name -> (loader_func, score_field_name)
METHOD_LOADERS = {
    "af3":       (_load_af3_scores,       "chain_pair_iptm"),
    "boltz2":    (_load_boltz2_scores,     "pair_chains_iptm"),
    "protenix":  (_load_protenix_scores,   "chain_pair_iptm"),
    "seedfold":  (_load_seedfold_scores,   "pair_chains_iptm"),
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _map_cif_to_model_index(
    scored_cifs: List[Tuple[float, str]],
    ensemble_cif_list: List[str],
) -> List[Tuple[float, int]]:
    """Map scored CIF paths to ensemble model indices.

    ensemble_cif_list is the sorted glob of CIFs that ensemble_generation used,
    so index i corresponds to {method}_model_{i}.
    """
    # Build reverse lookup: canonical cif path -> index
    canon = {os.path.realpath(p): i for i, p in enumerate(ensemble_cif_list)}
    result = []
    for score, cif in scored_cifs:
        real = os.path.realpath(cif)
        idx = canon.get(real)
        if idx is not None:
            result.append((score, idx))
    return result


def _get_ensemble_cif_list(method: str, target: str, method_output_dir) -> List[str]:
    """Reproduce the same CIF glob order as ensemble_generation.py.

    ``method_output_dir`` may be a single str (original behavior) or a
    list/tuple of dirs — in the latter case, CIFs from each dir are
    concatenated in the given order. MUST match ``_find_method_cifs`` in
    ensemble_generation.py exactly for canonical idx mapping to work.
    """
    # Accept list/tuple/omegaconf ListConfig (anything iterable that is not a str)
    if not isinstance(method_output_dir, (str, bytes)) and method_output_dir is not None:
        try:
            dirs_iter = list(method_output_dir)
        except TypeError:
            dirs_iter = None
        if dirs_iter is not None:
            combined: List[str] = []
            for d in dirs_iter:
                combined.extend(_get_ensemble_cif_list(method, target, d))
            return combined

    if method == "boltz2":
        # Layout A (single-seed r1)
        base, sub = _find_boltz2_pred_dir(method_output_dir, target)
        if base:
            return sorted(glob.glob(os.path.join(base, f"{sub}_model_*.cif")))
        # Layout B (multi-seed r2): {method_output_dir}/seed_*/boltz_results_{target}_input/predictions/{chain}/
        seed_dirs = sorted(glob.glob(os.path.join(method_output_dir, "seed_*")))
        all_cifs: List[str] = []
        for sd in seed_dirs:
            b, s = _find_boltz2_pred_dir(sd, target)
            if b:
                all_cifs.extend(sorted(glob.glob(os.path.join(b, f"{s}_model_*.cif"))))
        return all_cifs
    elif method == "af3":
        # Try nested layouts first (per-target subdir)
        base = os.path.join(method_output_dir, target, target.lower())
        if not os.path.isdir(base):
            base = os.path.join(method_output_dir, target.lower())
        if not os.path.isdir(base):
            base = os.path.join(method_output_dir, target)
        if os.path.isdir(base):
            cifs = sorted(glob.glob(os.path.join(
                base, "seed-*_sample-*", "*_model.cif")))
            if cifs:
                return cifs
        # Flat layout: targets share method_output_dir, per-target prefix on filename.
        # `{target_lower}*` allows `_r1`/`_r2` round suffix in filenames (CASP17 RNA).
        if os.path.isdir(method_output_dir):
            return sorted(glob.glob(os.path.join(
                method_output_dir, "seed-*_sample-*",
                f"{target.lower()}*_seed-*_sample-*_model.cif")))
        return []
    elif method == "protenix":
        base = os.path.join(method_output_dir, target)
        return sorted(glob.glob(os.path.join(
            base, "seed_*", "predictions", f"{target}_sample_*.cif")))
    elif method == "rf3":
        cifs = sorted(glob.glob(os.path.join(
            method_output_dir, f"{target}_seed-*", "**", "*.cif"), recursive=True))
        return [p for p in cifs if p.endswith("_model.cif")]
    elif method == "seedfold":
        base = os.path.join(method_output_dir, target)
        return sorted(glob.glob(os.path.join(base, "*", "*.cif")))
    return []


def build_comparison_table(
    dataset: str = "casp16_l1000",
    ensemble_dir: str = "outputs/ensemble",
    methods_config: Optional[Dict] = None,
    enabled_methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build the self-ranking comparison table.

    :param dataset: Dataset name (e.g. casp16_l1000).
    :param ensemble_dir: Base ensemble output directory.
    :param methods_config: {method: output_dir} mapping. Auto-detected if None.
    :param enabled_methods: List of methods to include. All if None.
    :return: DataFrame with comparison data.
    """
    ens_base = os.path.join(ensemble_dir, dataset)

    # Default method output dirs
    if methods_config is None:
        methods_config = {
            "af3":       f"outputs/alphafold3/{dataset}",
            "boltz2":    f"outputs/boltz2/{dataset}",
            "protenix":  f"outputs/protenix/{dataset}",
            "seedfold":  f"outputs/seedfold/{dataset}",
        }

    # Input JSON directories for chain identification (per method)
    input_dirs = {
        "af3":       f"data/test_cases/{dataset}/af3_inputs",
        "protenix":  f"data/test_cases/{dataset}/protenix_inputs",
        "seedfold":  f"data/test_cases/{dataset}/seedfold_inputs",
        # Boltz2 uses YAML inputs — handled via fallback (chain 0=protein, 1+=ligand)
    }

    if enabled_methods is None:
        enabled_methods = list(methods_config.keys())

    # Load evaluation and top-scores data
    eval_path = os.path.join(ens_base, "evaluation_summary.csv")
    top_path = os.path.join(ens_base, f"top1_top5_scores_{dataset.split('_')[-1].upper()}.csv")

    df_eval = pd.read_csv(eval_path)
    df_top = pd.read_csv(top_path)

    targets = sorted(df_eval["target"].unique())

    rows = []
    for t in targets:
        row = {"Target": t}

        # Consensus ranking columns (from top scores)
        t_top = df_top[df_top["Target"] == t]
        if not t_top.empty:
            row["Consensus RMSD Top-1 LDDT"] = t_top.iloc[0].get("ranking_rmsd (Top-1 LDDT)", "")
            row["Consensus SuCOS Top-1 LDDT"] = t_top.iloc[0].get("ranking_sucos (Top-1 LDDT)", "")

        # Evaluation data for this target (use ranking_rmsd to avoid duplicates)
        td_eval = df_eval[(df_eval["target"] == t) & (df_eval["method"] == "ranking_rmsd")]

        for method in enabled_methods:
            if method not in METHOD_LOADERS:
                continue
            loader_fn, score_field = METHOD_LOADERS[method]
            method_dir = methods_config.get(method, "")

            # Oracle Best (from top_scores)
            oracle_col = f"{method} Best LDDT-PLI"
            if not t_top.empty and oracle_col in t_top.columns:
                val = t_top.iloc[0].get(oracle_col, "")
                row[oracle_col] = "" if (isinstance(val, float) and math.isnan(val)) else val

            # Load self-ranked scores (all loaders now accept input_json_dir)
            scored_items = loader_fn(t, method_dir,
                                     input_json_dir=input_dirs.get(method, ""))
            ensemble_cifs = _get_ensemble_cif_list(method, t, method_dir)
            mapped = _map_cif_to_model_index(scored_items, ensemble_cifs)

            # Look up lDDT-PLI for each self-ranked model
            lddts = []
            for score, model_idx in mapped:
                model_prefix = f"{method}_model{model_idx}"
                match = td_eval[td_eval["model_name"].str.startswith(model_prefix + "_rank")]
                if not match.empty and match.iloc[0]["lddt_pli"] is not None:
                    lddts.append(match.iloc[0]["lddt_pli"])

            # Self-Top1, Self-Top5 Best, Self-Top10 Best
            row[f"{method} Self-Top1 LDDT"] = f"{lddts[0]:.3f}" if len(lddts) >= 1 else ""
            row[f"{method} Self-Top5 Best LDDT"] = (
                f"{max(lddts[:5]):.3f}" if len(lddts) >= 1 else ""
            )
            row[f"{method} Self-Top10 Best LDDT"] = (
                f"{max(lddts[:10]):.3f}" if len(lddts) >= 1 else ""
            )

        rows.append(row)

    # AVERAGE row
    avg_row = {"Target": "AVERAGE"}
    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    for col in all_cols:
        if col == "Target":
            continue
        vals = []
        for r in rows:
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
                if not math.isnan(fv):
                    vals.append(fv)
            except (ValueError, TypeError):
                pass
        avg_row[col] = f"{sum(vals) / len(vals):.3f}" if vals else ""
    rows.append(avg_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Self-ranking comparison: method confidence vs consensus ranking")
    parser.add_argument("--dataset", default="casp16_l1000",
                        help="Dataset name (default: casp16_l1000)")
    parser.add_argument("--ensemble_dir", default="outputs/ensemble",
                        help="Ensemble output base directory")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to include (default: all available). "
                             "E.g.: --methods af3 boltz2")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: auto)")
    args = parser.parse_args()

    df = build_comparison_table(
        dataset=args.dataset,
        ensemble_dir=args.ensemble_dir,
        enabled_methods=args.methods,
    )

    out_path = args.output or os.path.join(
        args.ensemble_dir, args.dataset, "self_ranking_comparison.csv")
    df.to_csv(out_path, index=False)

    # Pretty print
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
