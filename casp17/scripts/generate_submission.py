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
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Sequence

import pandas as pd
from rdkit import Chem

log = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
# Your CASP group code, as issued at registration (format NNNN-NNNN-NNNN).
# Export CASP_GROUP_ID, or pass --author on the command line.
DEFAULT_AUTHOR = os.environ.get("CASP_GROUP_ID", "")

# Method description split by target type. CASP17 R-series (RNA / RNA-ligand)
# uses iptm pre-filter + 3 methods (SeedFold RNA support unconfirmed). Protein
# T/H-series uses the original pocket_pLDDT_4.5 + 4 methods pipeline.
PROTEIN_METHOD_DESC = (
    "4-method co-folding ensemble (AF3 + Boltz-2 + Protenix + SeedFold; "
    "r1+r2 100 models/method, pocket_pLDDT_4.5 top-50 per method, "
    "SuCOS Butina maxclust 60, consensus_pb representative, "
    "PoseBusters + LG chemistry filter)"
)
RNA_METHOD_DESC = (
    "3-method co-folding ensemble (AF3 + Boltz-2 + Protenix; "
    "r1+r2 100 models/method, pair_iptm cross-chain max top-50 per method, "
    "SuCOS Butina maxclust 60, consensus_pb representative, "
    "PoseBusters + LG chemistry filter; receptor PDB RNA-Puzzles-standardized "
    "via rna_pdb_tools.py --get-rnapuzzle-ready)"
)


def is_rna_target(target: str) -> bool:
    """CASP17 R-series target heuristic: target name starts with 'R'."""
    return bool(target) and target[0].upper() == "R"


def default_method_for(target: str) -> str:
    """Pick the method description matching the target's pipeline."""
    return RNA_METHOD_DESC if is_rna_target(target) else PROTEIN_METHOD_DESC

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
    `id<TAB>code<TAB>smiles[<TAB>task]`. The `id` is parsed as int and
    re-emitted as 3-digit zero-padded in the LIGAND record (`LIGAND 000 TRP`),
    matching CASP17 spec example 6.1 (`LIGAND 001 LIG`). LG_validation parses
    via `int(...)` so padding is functionally equivalent to bare integers,
    but the example's 3-digit form is the canonical convention to follow.
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


# ── LG-format sanity check via vendored LG_validation.py ───────────────────
LG_VALIDATION_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "LG_validation.py"
)


def validate_lg_file(lg_file: str, target: str, smiles_tsv: str) -> Tuple[bool, str]:
    """Run vendored LG_validation.py on a single submission .txt.

    LG_validation expects `<TARGET>.smiles.txt` in `LG_TARGETS_PATH`. We feed
    it our `.tsv` by copying into a tmp dir with the right name. Returns
    (passed, stdout). Validator prints `# ERROR! ...` and `sys.exit()` on
    any structural / chemistry failure; otherwise it writes a per-target
    report to `LG_LOG_PATH` containing `ATOMS VALID / BONDS VALID /
    TOPOLOGY VALID` per ligand.
    """
    if not os.path.isfile(LG_VALIDATION_SCRIPT):
        return False, f"LG_validation.py not found at {LG_VALIDATION_SCRIPT}"
    with tempfile.TemporaryDirectory(prefix="lgcheck_") as tmp:
        smi_dst = os.path.join(tmp, f"{target}.smiles.txt")
        shutil.copy2(smiles_tsv, smi_dst)
        log_dir = os.path.join(tmp, "logs")
        os.makedirs(log_dir, exist_ok=True)
        env = os.environ.copy()
        env["LG_TARGETS_PATH"] = tmp
        env["LG_LOG_PATH"] = log_dir
        proc = subprocess.run(
            [sys.executable, LG_VALIDATION_SCRIPT, lg_file],
            env=env, capture_output=True, text=True, timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # LG_validation prints `# ERROR!` and sys.exit() on any failure,
        # which sets returncode=1. No-error runs return 0 with empty stdout
        # (report goes to log file, not stdout).
        passed = proc.returncode == 0 and "# ERROR" not in out
        return passed, out


# ── RNA-Puzzles standardization (CASP17 spec: drop OP3 / non-standard atoms) ─
def rnapuzzle_clean_pdb(pdb_path: str) -> Optional[str]:
    """Run `rna_pdb_tools.py --get-rnapuzzle-ready` on a copy of the receptor PDB.

    The CASP17 LG/TS spec for RNA receptors states: only standard A/C/U/G
    nucleotides, only the whitelisted base + sugar-phosphate atoms; modified
    nucleotides treated as unmodified, all other atoms discarded. The
    rna-tools helper enforces this and adds REMARK 250 marking the file as
    standardized. We copy the input first so the original on disk is left
    untouched, run the tool in-place on the copy, and return the copy path.

    Returns None on tool failure (caller falls back to original PDB).
    """
    if shutil.which("rna_pdb_tools.py") is None:
        log.warning("rna_pdb_tools.py not on PATH; skipping RNA-Puzzles standardization")
        return None
    try:
        # Use a temp copy so the source PDB stays as-is.
        fd, tmp_path = tempfile.mkstemp(suffix=".pdb", prefix="rnapuzzle_")
        os.close(fd)
        shutil.copy2(pdb_path, tmp_path)
        proc = subprocess.run(
            ["rna_pdb_tools.py", "--get-rnapuzzle-ready", tmp_path, "--inplace"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            log.warning(
                f"rna_pdb_tools.py failed (rc={proc.returncode}) on "
                f"{pdb_path}: {proc.stderr.strip()[:200]}"
            )
            os.unlink(tmp_path)
            return None
        return tmp_path
    except Exception as e:
        log.warning(f"rna_pdb_tools.py exception on {pdb_path}: {e}")
        return None


# ── MODEL block builder ────────────────────────────────────────────────────
def build_model_block(
    model_num: int,
    sdf_path: str,
    pdb_path: str,
    sucos: float,
    ligands_meta: List[Tuple[int, str, str]],
    target: str = "",
) -> Optional[str]:
    """Render one full MODEL/END block:
        MODEL n
        PARENT N/A
        ATOM ... TER (receptor)
        LIGAND ... LSCORE ... <MDL>  (one block per ligand)
        END

    For RNA targets (`target` starts with 'R'), the receptor PDB is first
    standardized via rna_pdb_tools.py --get-rnapuzzle-ready before the ATOM
    block is rendered (drops OP3 and non-whitelisted atoms per CASP17 spec).
    """
    pdb_for_render = pdb_path
    cleaned_tmp: Optional[str] = None
    if is_rna_target(target):
        cleaned_tmp = rnapuzzle_clean_pdb(pdb_path)
        if cleaned_tmp is not None:
            pdb_for_render = cleaned_tmp

    try:
        receptor_lines = render_receptor_block(pdb_for_render)
    finally:
        if cleaned_tmp is not None and os.path.exists(cleaned_tmp):
            os.unlink(cleaned_tmp)

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
        # 3-digit zero-padded per CASP17 spec example 6.1 (`LIGAND 001 LIG`).
        out.append(f"LIGAND {lig_id:03d} {lig_code}")
        out.append(f"LSCORE {sucos:.4f}")
        # MDL title line (mol block header line 1) carries the ligand code,
        # matching example 6.1 where `LIG` appears between `LSCORE 0.82` and
        # the program/timestamp line. Default RDKit MolToMolBlock leaves
        # title blank — spec says "can be blank but must exist", but the
        # example shows the code, so we set it explicitly for parity.
        mol.SetProp("_Name", lig_code)
        # MolToBlock terminates with "M  END" — LG validator's MEND check
        # `line.replace(" ", "") == "MEND"` accepts that.
        out.append(Chem.MolToMolBlock(mol).rstrip())

    out.append("END")
    return "\n".join(out)


# ── Per-target / per-model files ───────────────────────────────────────────
def generate_target_files(
    target: str,
    sdf_paths: List[str],
    smiles_tsv: str,
    author: str,
    method_desc: str,
    model_indices: Sequence[int] = (1, 2, 3, 4, 5),
) -> List[Tuple[int, str]]:
    """Build one LG submission text per MODEL for a target.

    Returns a list of (model_index, full_text) tuples — each entry is a
    *complete* self-contained LG file with header (PFRMAT/TARGET/AUTHOR/METHOD)
    + exactly one MODEL/END block. CASP17 servers replace previously accepted
    submissions when (target, format, group, **model_index**) collide, so each
    model needs its own unique index. Splitting into separate files rather than
    concatenating into one file is the safer of the two equally spec-compliant
    options: a malformed receptor block in MODEL 3 won't invalidate MODELs
    1/2/4/5 if they're independent uploads.

    ``model_indices`` sets the MODEL numbers explicitly, which some targets
    require. T2451 (BifA) crystallises as a homodimer in two conformations and
    the organisers ask for "models for conformation 1 as models 1-5, and those
    for conformation 2 as 6,7,8,9,0" — hence the two slates are produced by two
    invocations with disjoint index lists (the second ends in a literal 0, not
    10). ``LG_validation.py`` never parses the MODEL number, so 0 is accepted.
    """
    ligands_meta = read_smiles_tsv(smiles_tsv)
    if not ligands_meta:
        log.warning(f"{target}: empty SMILES TSV {smiles_tsv}; skipping")
        return []

    header = "\n".join([
        "PFRMAT LG",
        f"TARGET {target}",
        f"AUTHOR {author}",
        f"METHOD {method_desc}",
    ])

    results: List[Tuple[int, str]] = []
    for idx, sdf_path in zip(model_indices, sdf_paths[:len(model_indices)]):
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
        block = build_model_block(idx, sdf_path, pdb_path, sucos, ligands_meta, target=target)
        if not block:
            continue
        full_text = header + "\n" + block + "\n"
        results.append((idx, full_text))

    if not results:
        log.warning(f"{target}: no usable MODEL blocks; nothing written")
    return results


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
    parser.add_argument(
        "--method-desc", default=None,
        help="Override METHOD record. If omitted, RNA targets get RNA_METHOD_DESC "
             "and protein targets get PROTEIN_METHOD_DESC (auto-detected by "
             "target prefix: 'R' = RNA).",
    )
    parser.add_argument(
        "--smiles-dir", default=None,
        help="Extra dir to search for <target>.tsv (besides defaults)",
    )
    parser.add_argument(
        "--model-indices", default="1,2,3,4,5",
        help="Comma-separated MODEL indices for this slate (max 5). Default '1,2,3,4,5'. "
             "Targets with two crystallographic conformations ask for a second slate — "
             "T2451 wants '6,7,8,9,0' (a literal 0, not 10) for conformation 2. CASP "
             "replaces a previously accepted submission when (target, format, group, "
             "model_index) collide, so the two slates MUST use disjoint indices.",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip post-write LG_validation.py sanity check (default: run it).",
    )
    args = parser.parse_args()
    if not args.author:
        parser.error(
            "no CASP group code: export CASP_GROUP_ID=NNNN-NNNN-NNNN or pass --author. "
            "It goes into the AUTHOR record and CASP rejects the file without it."
        )
    try:
        model_indices = [int(x) for x in args.model_indices.split(",") if x.strip() != ""]
    except ValueError:
        log.error(f"--model-indices must be comma-separated integers, got {args.model_indices!r}")
        sys.exit(2)
    if not model_indices:
        log.error("--model-indices is empty")
        sys.exit(2)
    if len(model_indices) > 5:
        log.error(f"CASP allows at most 5 MODEL blocks per submission slate; "
                  f"got {len(model_indices)}. Run twice with disjoint index lists instead.")
        sys.exit(2)
    if len(set(model_indices)) != len(model_indices):
        log.error(f"--model-indices has duplicates: {model_indices}")
        sys.exit(2)

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
    validation_failures: List[str] = []
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

        method_desc = args.method_desc if args.method_desc else default_method_for(target)
        per_model = generate_target_files(
            target, sdf_paths, smiles_tsv,
            args.author, method_desc, model_indices,
        )
        if not per_model:
            continue

        # Write into a per-target subdir `<output_dir>/<target>_files/` so all
        # artifacts for one target (5 model .txt + 5 source SDF + 5 source
        # PDB + submission log) live together — matches the layout used for
        # R2314 (the submitted set). Self-contained: deleting
        # outputs/ensemble_r1r2/ won't break re-uploads from this dir.
        target_dir = os.path.join(args.output_dir, f"{target}_files")
        os.makedirs(target_dir, exist_ok=True)

        # Copy source SDF + PDB (the cluster top-N picks) into the target
        # dir so the submission package is self-contained. PDB path is
        # derived from SDF path (same dir, name without `_pb=*` suffix).
        for sdf_path in sdf_paths[:len(model_indices)]:
            if not sdf_path or not os.path.isfile(sdf_path):
                continue
            shutil.copy2(sdf_path, target_dir)
            pdb_path = derive_pdb_path(sdf_path)
            if os.path.isfile(pdb_path):
                shutil.copy2(pdb_path, target_dir)

        for idx, text in per_model:
            out_file = os.path.join(target_dir, f"{target}_model{idx}.txt")
            with open(out_file, "w") as f:
                f.write(text)
            written += 1

            # Inline sanity check via vendored LG_validation.py. Catches
            # format / chemistry mismatches before submission rather than
            # discovering them post-curl from the organizer.
            if not args.no_validate:
                passed, validator_out = validate_lg_file(out_file, target, smiles_tsv)
                if passed:
                    log.info(f"  ✓ LG_validation passed: {os.path.basename(out_file)}")
                else:
                    validation_failures.append(out_file)
                    log.error(
                        f"  ✗ LG_validation FAILED: {os.path.basename(out_file)}\n"
                        f"    output: {validator_out.strip()[:500]}"
                    )

    log.info(f"Wrote {written} LG submission files to {args.output_dir}")
    if validation_failures:
        log.error(
            f"{len(validation_failures)}/{written} files failed LG_validation. "
            f"DO NOT SUBMIT until fixed:\n  " + "\n  ".join(validation_failures)
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
