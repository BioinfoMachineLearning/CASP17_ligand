"""Generate CASP17 LG submission files from r1+r2 ensemble cluster results.

Output: one `.txt` file per target, each containing up to 5 MODEL/END blocks.
The format matches the CASP17 LG spec (https://predictioncenter.org/casp17/index.cgi?page=format,
Example 6.1):

    PFRMAT LG
    TARGET <target>
    AUTHOR <id>
    METHOD <description>
    MODEL  1
    PARENT N/A                       # block 1: receptor coordinates (TS-style)
    ATOM  ...
    ATOM  ...
    TER
    [ATOM ... TER]                   # additional chains
    LIGAND <nnn> <code>              # block 2: ligand coordinates (MDL)
    LSCORE <0..1>
    <RDKit MolBlock with terminating "M  END">
    [LIGAND ... LSCORE ... <MDL>]    # repeat per ligand for multi-ligand targets
    END
    MODEL  2
    ...
    END

Chain renaming:
    Per CASP17 spec, protein chains are labelled alphabetically (A, B, C, ...)
    and nucleic-acid chains numerically (0, 1, 2, ...). Chain types are
    auto-detected from residue names; the original chain id from the
    cif_to_pdb_sdf output is rewritten before emission.

LSCORE source:
    SuCOS consensus extracted from the cluster representative's SDF filename
    (`..._sucos<value>_pb=*.sdf`), which is the mean pairwise SuCOS to all
    other models in the ensemble. Same value used for every LIGAND in a MODEL
    (the SuCOS computation is model-level, not per-ligand).

Usage:
    python casp17/scripts/generate_submission.py \\
        --cluster-csv outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv \\
        --output-dir casp17/submissions/casp17 \\
        --smiles-dir data/casp17_data/smiles

Validate after generation:
    LG_TARGETS_PATH=<smiles dir> python casp17/scripts/LG_validation.py \\
        casp17/submissions/casp17/<target>.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

log = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_AUTHOR = "3282-9868-6371"
DEFAULT_METHOD = (
    "Multi-method co-folding ensemble (AF3 + Boltz-2 + Protenix + SeedFold; "
    "r1+r2 100 models/method, pocket_pLDDT_4.5 top-50 per method, "
    "SuCOS Butina maxclust 60, consensus_pb representative, "
    "PoseBusters + LG chemistry filter)"
)

# Regex to pull SuCOS consensus value out of SDF filename, e.g.
#   af3_model28_rank1_orig1_sucos0.763_pb=True.sdf
SUCOS_FROM_SDF_RE = re.compile(r"_sucos([\d.]+)_pb=")

# Default search dirs for SMILES TSV
SMILES_SEARCH_DIRS = [
    "data/casp17_data/smiles",
    "data/casp16_data/smiles/L1000",
    "data/casp16_data/smiles/L2000",
    "data/casp16_data/smiles/L3000",
    "data/casp16_data/smiles/L4000",
    "data/casp15_data/smiles/CASP15",
]

# Standard 3-letter protein residues (incl. selenomethionine).
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",
}
# Nucleic-acid residue names (RNA single-letter; DNA D-prefixed; RDKit alts).
NUCLEIC_RESIDUES = {
    "A", "C", "G", "U", "T",
    "DA", "DC", "DG", "DT", "DI",
    "RA", "RC", "RG", "RU",
}


# ── Helpers ────────────────────────────────────────────────────────────────
def find_smiles_tsv(target: str, extra_dir: Optional[str] = None) -> Optional[str]:
    dirs = ([extra_dir] if extra_dir else []) + SMILES_SEARCH_DIRS
    for sd in dirs:
        p = os.path.join(sd, f"{target}.tsv")
        if os.path.exists(p):
            return p
    return None


def read_smiles_tsv(tsv_path: str) -> List[Tuple[int, str, str]]:
    """Parse a CASP-style SMILES TSV. Returns [(lig_id, code, smiles), ...].

    File format: tab-separated, header row, columns
    `id<TAB>code<TAB>smiles[<TAB>task]`. The `id` is used verbatim so the
    LIGAND record id matches the SMILES dict key that LG_validation.py reads
    out of the same file.
    """
    ligands: List[Tuple[int, str, str]] = []
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            ligands.append((int(parts[0]), parts[1], parts[2]))
    return ligands


def extract_sucos_from_sdf(sdf_path: str) -> Optional[float]:
    """Pull SuCOS consensus value out of `..._sucos<value>_pb=*.sdf` filename.

    Clamps to [0,1] for LSCORE legality. Returns None if no match.
    """
    m = SUCOS_FROM_SDF_RE.search(os.path.basename(sdf_path))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def derive_pdb_path(sdf_path: str) -> str:
    """sdf path → corresponding pdb path. ensemble_generation writes
    `<base>_pb=True.sdf` next to `<base>.pdb`."""
    pdb = sdf_path
    for sfx in ("_pb=True.sdf", "_pb=False.sdf"):
        if pdb.endswith(sfx):
            return pdb[: -len(sfx)] + ".pdb"
    # fallback: just swap extension
    return os.path.splitext(sdf_path)[0] + ".pdb"


# ── Receptor block (CASP TS-style ATOM/TER) ────────────────────────────────
def detect_chain_type(resnames: set) -> str:
    """'protein', 'nucleic', or 'unknown' from a chain's residue names."""
    for r in resnames:
        r = r.strip().upper()
        if r in PROTEIN_RESIDUES:
            return "protein"
        if r in NUCLEIC_RESIDUES:
            return "nucleic"
    return "unknown"


def parse_pdb_chains(pdb_path: str) -> List[Tuple[str, str, List[str]]]:
    """Read ATOM lines from a PDB, group by original chain id (col 22),
    classify each chain as protein/nucleic/unknown.

    Returns: [(orig_chain_id, chain_type, [atom_lines_no_newline]), ...] in
    file-order.
    """
    seen: List[str] = []
    atoms: Dict[str, List[str]] = {}
    resnames: Dict[str, set] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21:22]
            res = line[17:20]
            if ch not in atoms:
                atoms[ch] = []
                resnames[ch] = set()
                seen.append(ch)
            atoms[ch].append(line.rstrip("\n"))
            resnames[ch].add(res)
    return [(ch, detect_chain_type(resnames[ch]), atoms[ch]) for ch in seen]


def remap_chain_ids(
    chains: List[Tuple[str, str, List[str]]],
) -> List[Tuple[str, List[str]]]:
    """Re-letter protein chains A,B,C... and re-number nucleic chains 0,1,2...
    per CASP17 spec. Unknown chains follow the protein bucket.

    Returns: [(new_chain_id, [rewritten_atom_lines]), ...].
    """
    p_idx = 0
    n_idx = 0
    out = []
    for _, ctype, lines in chains:
        if ctype == "nucleic":
            new_id = str(n_idx)
            n_idx += 1
            if n_idx > 9:
                raise RuntimeError("more than 10 nucleic-acid chains; spec uses single-digit")
        else:
            new_id = chr(ord("A") + p_idx)
            p_idx += 1
            if p_idx > 26:
                raise RuntimeError("more than 26 protein chains; spec uses single letter")
        # column 22 (index 21) holds the chain id in standard PDB
        new_lines = [l[:21] + new_id + l[22:] for l in lines]
        out.append((new_id, new_lines))
    return out


def render_receptor_block(pdb_path: str) -> List[str]:
    """ATOM records (with re-mapped chain ids) + TER terminator per chain.

    Returns a list of lines (no trailing newlines) suitable for joining.
    """
    chains = parse_pdb_chains(pdb_path)
    if not chains:
        log.warning(f"no ATOM records in {pdb_path}")
        return []
    remapped = remap_chain_ids(chains)
    out: List[str] = []
    for new_id, lines in remapped:
        out.extend(lines)
        out.append("TER")
    return out


# ── MODEL block builder ────────────────────────────────────────────────────
def build_model_block(
    model_num: int,
    sdf_path: str,
    pdb_path: str,
    sucos: float,
    ligands_meta: List[Tuple[int, str, str]],
) -> Optional[str]:
    """Render one full MODEL/END block:
        MODEL n
        PARENT N/A
        ATOM ... TER (receptor)
        LIGAND ... LSCORE ... <MDL>  (one block per ligand)
        END
    """
    receptor_lines = render_receptor_block(pdb_path)
    if not receptor_lines:
        log.warning(f"MODEL {model_num}: empty receptor block from {pdb_path}; skipping")
        return None

    # SDMolSupplier defaults (removeHs=True, sanitize=True) match
    # ForwardSDMolSupplier defaults that LG_validation uses.
    suppl = Chem.SDMolSupplier(sdf_path)
    mols = [m for m in suppl if m is not None]
    if not mols:
        log.warning(f"MODEL {model_num}: no valid mols in {sdf_path}; skipping")
        return None

    if len(mols) != len(ligands_meta):
        log.warning(
            f"MODEL {model_num}: {len(mols)} SDF mols vs {len(ligands_meta)} "
            f"SMILES TSV entries in {sdf_path}; pairing by min length"
        )

    out = [f"MODEL  {model_num}", "PARENT N/A"]
    out.extend(receptor_lines)

    pairs = list(zip(ligands_meta, mols))
    if not pairs:
        return None
    for (lig_id, lig_code, _), mol in pairs:
        out.append(f"LIGAND {lig_id:03d} {lig_code}")
        out.append(f"LSCORE {sucos:.4f}")
        # MolToBlock terminates with "M  END" — LG validator's MEND check
        # `line.replace(" ", "") == "MEND"` accepts that.
        out.append(Chem.MolToMolBlock(mol).rstrip())

    out.append("END")
    return "\n".join(out)


# ── Per-target file ────────────────────────────────────────────────────────
def generate_target_file(
    target: str,
    sdf_paths: List[str],
    smiles_tsv: str,
    author: str,
    method_desc: str,
    n_models: int = 5,
) -> Optional[str]:
    """Build the full LG submission text for one target."""
    ligands_meta = read_smiles_tsv(smiles_tsv)
    if not ligands_meta:
        log.warning(f"{target}: empty SMILES TSV {smiles_tsv}; skipping")
        return None

    body_blocks: List[str] = []
    for idx, sdf_path in enumerate(sdf_paths[:n_models], start=1):
        if not sdf_path or not os.path.exists(sdf_path):
            log.warning(f"{target} MODEL {idx}: SDF missing: {sdf_path}")
            continue
        pdb_path = derive_pdb_path(sdf_path)
        if not os.path.exists(pdb_path):
            log.warning(f"{target} MODEL {idx}: PDB missing: {pdb_path}")
            continue
        sucos = extract_sucos_from_sdf(sdf_path)
        if sucos is None:
            log.warning(
                f"{target} MODEL {idx}: cannot parse SuCOS from "
                f"{os.path.basename(sdf_path)}; defaulting LSCORE=0.5"
            )
            sucos = 0.5
        block = build_model_block(idx, sdf_path, pdb_path, sucos, ligands_meta)
        if block:
            body_blocks.append(block)

    if not body_blocks:
        log.warning(f"{target}: no usable MODEL blocks; skipping")
        return None

    header = [
        "PFRMAT LG",
        f"TARGET {target}",
        f"AUTHOR {author}",
        f"METHOD {method_desc}",
    ]
    return "\n".join(header) + "\n" + "\n".join(body_blocks) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate CASP17 LG submission files from cluster results"
    )
    parser.add_argument(
        "--cluster-csv", required=True,
        help="cluster_experiment_detail CSV (must contain target, top5_sdf_paths)",
    )
    parser.add_argument("--output-dir", required=True, help="Submission output dir")
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="CASP registration code")
    parser.add_argument("--method-desc", default=DEFAULT_METHOD)
    parser.add_argument(
        "--smiles-dir", default=None,
        help="Extra dir to search for <target>.tsv (besides defaults)",
    )
    parser.add_argument(
        "--n-models", type=int, default=5, help="Max MODEL blocks per target (cap 5)",
    )
    args = parser.parse_args()
    if args.n_models > 5:
        log.warning("CASP allows at most 5 models per target; capping --n-models at 5")
        args.n_models = 5

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.cluster_csv)
    if "target" not in df.columns or "top5_sdf_paths" not in df.columns:
        log.error(
            f"{args.cluster_csv} missing required columns "
            f"('target', 'top5_sdf_paths'); have: {list(df.columns)}"
        )
        sys.exit(2)

    written = 0
    for _, row in df.iterrows():
        target = row["target"]
        sdf_str = row.get("top5_sdf_paths", "")
        if pd.isna(sdf_str) or not sdf_str:
            log.warning(f"{target}: no top5_sdf_paths; skipping")
            continue
        sdf_paths = [p for p in str(sdf_str).split(";") if p]

        smiles_tsv = find_smiles_tsv(target, args.smiles_dir)
        if not smiles_tsv:
            log.warning(f"{target}: SMILES TSV not found; skipping")
            continue

        text = generate_target_file(
            target, sdf_paths, smiles_tsv,
            args.author, args.method_desc, args.n_models,
        )
        if text is None:
            continue

        out_file = os.path.join(args.output_dir, f"{target}.txt")
        with open(out_file, "w") as f:
            f.write(text)
        written += 1

    log.info(f"Wrote {written} LG submission files to {args.output_dir}")


if __name__ == "__main__":
    main()
