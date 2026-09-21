"""Unified target data structures and loading functions.

Handles all L-series variations:
  - L1000: 17 targets, 1 ligand each, shared protein sequence
  - L2000: 2 targets, 1 ligand each, shared protein sequence
  - L3000: 189 targets, 1-2 ligands (some with ions like CL), shared protein sequence
  - L4000: 25 targets, 2-6 ligands (dimer A+B chains, some with solvents), shared protein sequence
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Global blacklist: targets excluded from ALL pipeline stages
# (input preparation, inference, ensemble generation, evaluation, clustering).
# Reasons are documented per-target.
BLACKLISTED_TARGETS = {
    # CASP16: data leakage (disqualified by organizers)
    "L4006", "L4007", "L4008", "L4009", "L4010",
    # CASP16: OST graph isomorphism failure (chirality/bond order mismatch)
    "L3103",
    # CASP15: reference ligand PDB atom count mismatch with SMILES (40 vs 92 HA)
    "T1181",
}

# Series whose chains are RNA (not protein). The loader sets
# TargetData.entity_type accordingly; downstream input_preparation scripts
# branch on entity_type to emit `rna` / `rnaSequence` chain specs.
RNA_SERIES = {"CASP17_R"}


@dataclass
class LigandInfo:
    """A single ligand entry from TSV."""
    id: int           # TSV row ID (0-indexed)
    name: str         # Residue name (e.g., "201", "LIG", "CL")
    smiles: str       # SMILES string
    task: str         # "PA" (pose + affinity) or "P" (pose only)


@dataclass
class TargetData:
    """All data for a single prediction target."""
    target_id: str                             # "L1001"
    series: str                                # "L1000"
    protein_sequences: Dict[str, str]          # {chain_id: sequence}, e.g. {"A": "MLLP..."}
                                               # NOTE: for RNA targets (entity_type=="rna")
                                               # the same dict carries the RNA sequence;
                                               # field name retained for backward compat.
    ligands: List[LigandInfo]                  # Ligand entries from TSV
    ref_protein_pdb: Optional[Path] = None     # protein_aligned.pdb
    ref_ligand_pdbs: List[Path] = field(default_factory=list)  # ligand_XXX_C_1.pdb files
    entity_type: str = "protein"               # "protein" or "rna" — drives chain spec
                                               # in *_input_preparation.py

    @property
    def has_struct(self) -> bool:
        """Whether structural reference data exists."""
        return self.ref_protein_pdb is not None and len(self.ref_ligand_pdbs) > 0

    @property
    def is_dimer(self) -> bool:
        """Whether the target protein is a dimer (multiple chains)."""
        return len(self.protein_sequences) > 1

    @property
    def num_ligands(self) -> int:
        return len(self.ligands)

    @property
    def ligand_smiles_combined(self) -> str:
        """Combined SMILES with '.' separator for multi-ligand targets."""
        return ".".join(lig.smiles for lig in self.ligands)


def group_ligands_for_dimer(ligands: List["LigandInfo"]):
    """Group ligands into two pockets (Site A / Site B) for dimer targets.

    Algorithm:
    1. Identify the two main ligands (longest identical SMILES pair)
    2. Even-split identical small ligands between sites
    3. Round-robin remaining unique/odd small ligands

    Returns:
        (group_a, group_b): each is a list of (original_index, LigandInfo)
    """
    if len(ligands) <= 1:
        return [(i, lig) for i, lig in enumerate(ligands)], []

    # Find the two main ligands: longest SMILES that appear at least twice
    smiles_indices = {}
    for i, lig in enumerate(ligands):
        smiles_indices.setdefault(lig.smiles, []).append(i)

    main_smiles = None
    for smi in sorted(smiles_indices.keys(), key=len, reverse=True):
        if len(smiles_indices[smi]) >= 2:
            main_smiles = smi
            break

    if main_smiles is None:
        return [(i, lig) for i, lig in enumerate(ligands)], []

    # Assign one main ligand to each site
    main_indices = smiles_indices[main_smiles]
    group_a = [(main_indices[0], ligands[main_indices[0]])]
    group_b = [(main_indices[1], ligands[main_indices[1]])]
    used = {main_indices[0], main_indices[1]}

    # Collect remaining small ligands
    remaining = [(i, lig) for i, lig in enumerate(ligands) if i not in used]

    # Group remaining by SMILES
    small_by_smiles = {}
    for idx, lig in remaining:
        small_by_smiles.setdefault(lig.smiles, []).append((idx, lig))

    # Phase 1: even-split identical small ligands
    leftovers = []
    for smi, items in small_by_smiles.items():
        half = len(items) // 2
        group_a.extend(items[:half])
        group_b.extend(items[half:2 * half])
        if len(items) % 2 == 1:
            leftovers.append(items[-1])

    # Phase 2: round-robin remaining unique/odd small ligands
    for i, item in enumerate(leftovers):
        if i % 2 == 0:
            group_a.append(item)
        else:
            group_b.append(item)

    return group_a, group_b


def _parse_fasta(filepath: Path) -> Dict[str, str]:
    """Parse a multi-chain FASTA file into {chain_id: sequence}.

    Supports two formats:
      - CASP16: ">L1000 Chymase, ..." header (chain_id extracted as "A")
      - CASP15: ">H1135 Chain A" header (chain_id extracted from header)
    """
    chains: Dict[str, str] = {}
    current_id = None
    current_seq: List[str] = []
    next_auto_chain = 0  # fallback: A, B, C, ...

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None and current_seq:
                    chains[current_id] = "".join(current_seq)
                # Try to extract chain ID from ">... Chain X" pattern
                parts = line.split()
                chain_id = None
                for i, p in enumerate(parts):
                    if p.lower() == "chain" and i + 1 < len(parts):
                        chain_id = parts[i + 1].rstrip(",")
                        break
                if chain_id is None:
                    chain_id = chr(ord("A") + next_auto_chain)
                    next_auto_chain += 1
                current_id = chain_id
                current_seq = []
            else:
                if line:
                    current_seq.append(line)
    if current_id is not None and current_seq:
        chains[current_id] = "".join(current_seq)

    return chains


def read_protein_sequence(
    series: str, data_root: Path, target_id: Optional[str] = None
) -> Dict[str, str]:
    """Read protein sequence for a target.

    Lookup order:
      1. sequences/{series}/{target_id}.txt  (per-target, e.g. CASP15)
      2. sequences/{series}.txt              (shared, e.g. CASP16 L1000)

    Returns dict mapping chain_id -> sequence.
    """
    # 1. Per-target sequence file
    if target_id:
        per_target = data_root / "sequences" / series / f"{target_id}.txt"
        if per_target.exists():
            return _parse_fasta(per_target)
        # Fallback for L-series super-targets (e.g. L010001 -> L01)
        if series == "CASP17_L":
            super_target = target_id[:3]  # "L01" or "L02"
            super_target_file = data_root / "sequences" / series / f"{super_target}.txt"
            if super_target_file.exists():
                return _parse_fasta(super_target_file)

    # 2. Shared series sequence file
    seq_file = data_root / "sequences" / f"{series}.txt"
    if not seq_file.exists():
        logger.warning(f"Sequence file not found: {seq_file}")
        return {}

    with open(seq_file) as f:
        lines = f.readlines()

    if len(lines) < 2:
        logger.warning(f"Sequence file has < 2 lines: {seq_file}")
        return {}

    sequence = lines[1].strip()

    # Known homodimers from CASP competition metadata
    DIMER_SERIES = {"L4000"}
    if series in DIMER_SERIES:
        return {"A": sequence, "B": sequence}

    return {"A": sequence}


def read_protein_chains_from_pdb(pdb_path: Path) -> Dict[str, str]:
    """Extract protein sequences per chain from a PDB file.

    Used for targets where sequence file doesn't indicate multi-chain
    structure (e.g., L4000 dimers).
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    # three_to_one moved in newer BioPython versions
    try:
        from Bio.PDB.Polypeptide import three_to_one
    except ImportError:
        from Bio.Data.IUPACData import protein_letters_3to1
        def three_to_one(res_name):
            return protein_letters_3to1.get(res_name.lower().capitalize(), "X")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    chains = {}
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if is_aa(residue, standard=True):
                    try:
                        seq.append(three_to_one(residue.get_resname()))
                    except KeyError:
                        seq.append("X")
            if seq:
                chains[chain.id] = "".join(seq)
        break  # Only first model
    return chains


def read_ligands_from_tsv(tsv_path: Path) -> List[LigandInfo]:
    """Read ligand info from a target TSV file.

    Supports two formats:
      - CASP16: ID  Name  SMILES  Task      (Task = "P", "PA", etc.)
      - CASP15: ID  Name  SMILES  Relevant  (Relevant = "Yes"/"No")

    For CASP15, Relevant is mapped to task "P" (all included for co-folding;
    Relevant=No ligands are still loaded but can be filtered at evaluation).
    """
    ligands = []
    with open(tsv_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        # Detect format from header
        is_casp15 = len(header) >= 4 and header[3].strip().lower() == "relevant"
        for row in reader:
            if len(row) < 4:
                continue
            col4 = row[3].strip()
            if is_casp15:
                task = "P"  # CASP15 is structure prediction only
            else:
                task = col4
            ligands.append(LigandInfo(
                id=int(row[0]),
                name=row[1].strip(),
                smiles=row[2].strip(),
                task=task,
            ))
    return ligands


def find_ref_ligand_pdbs(struct_dir: Path) -> List[Path]:
    """Find all reference ligand PDB files in a prepared struct directory.

    Matches pattern: ligand_*_*.pdb
    """
    if not struct_dir.exists():
        return []
    return sorted(struct_dir.glob("ligand_*.pdb"))


def load_target(
    target_id: str,
    data_root: Path,
    series: Optional[str] = None,
) -> TargetData:
    """Load all data for a single target.

    Args:
        target_id: Target identifier (e.g., "L1001")
        data_root: Path to data/casp16_data/
        series: Series name (e.g., "L1000"). Auto-detected if None.
    """
    if series is None:
        # Auto-detect: L1001 -> L1000, L3053 -> L3000, L4020 -> L4000
        series = target_id[0] + target_id[1] + "000"

    data_root = Path(data_root)

    # 1. Read SMILES from TSV
    tsv_path = data_root / "smiles" / series / f"{target_id}.tsv"
    if tsv_path.exists():
        ligands = read_ligands_from_tsv(tsv_path)
    else:
        logger.warning(f"TSV not found: {tsv_path}")
        ligands = []

    # 2. Read protein sequence (per-target fallback for CASP15)
    sequences = read_protein_sequence(series, data_root, target_id=target_id)

    # 3. Find reference structure files
    struct_dir = data_root / "struct" / f"{series}_prepared" / target_id
    ref_protein_pdb = struct_dir / "protein_aligned.pdb" if struct_dir.exists() else None
    if ref_protein_pdb and not ref_protein_pdb.exists():
        ref_protein_pdb = None
    ref_ligand_pdbs = find_ref_ligand_pdbs(struct_dir)

    entity_type = "rna" if series in RNA_SERIES else "protein"

    return TargetData(
        target_id=target_id,
        series=series,
        protein_sequences=sequences,
        ligands=ligands,
        ref_protein_pdb=ref_protein_pdb,
        ref_ligand_pdbs=ref_ligand_pdbs,
        entity_type=entity_type,
    )


def parse_target_filter(spec) -> Optional[Set[str]]:
    """Normalise a target allowlist into a set, or None meaning "all targets".

    Accepts a comma-separated string ("T2413v1,T2414v1"), any iterable of ids,
    or None/""/"all". Matches the `targets=` convention already used by
    `boltz2_inference.py`.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.strip().lower() in ("", "all", "null", "none"):
            return None
        ids = [t.strip() for t in spec.split(",")]
    else:
        ids = [str(t).strip() for t in spec]
    ids = [t for t in ids if t]
    return set(ids) or None


def load_all_targets(
    series: str,
    data_root: Path,
    struct_only: bool = True,
    only=None,
) -> List[TargetData]:
    """Load all targets for a given series.

    Args:
        series: Series name (e.g., "L1000")
        data_root: Path to data/casp16_data/
        struct_only: If True, only load targets with structural data.
        only: Optional allowlist of target ids — comma-separated string or
            iterable. None loads the whole series (benchmark/eval mode); an
            allowlist restricts loading to those ids (competition mode, where
            regenerating a whole series would rewrite inputs for targets that
            are already running or already submitted).
    """
    data_root = Path(data_root)
    smiles_dir = data_root / "smiles" / series
    struct_dir = data_root / "struct" / f"{series}_prepared"

    if not smiles_dir.exists():
        logger.error(f"SMILES directory not found: {smiles_dir}")
        return []

    allow = parse_target_filter(only)
    available = {p.stem for p in smiles_dir.glob("*.tsv")}
    if allow is not None:
        # A typo here would otherwise silently prepare nothing and the run would
        # fail much later, on a missing input file.
        missing = sorted(allow - available)
        if missing:
            raise ValueError(
                f"target(s) not found in {smiles_dir}: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(available))}"
            )

    targets = []
    for tsv_file in sorted(smiles_dir.glob("*.tsv")):
        target_id = tsv_file.stem

        if allow is not None and target_id not in allow:
            continue

        # Skip blacklisted targets (data leakage)
        if target_id in BLACKLISTED_TARGETS:
            logger.info(f"Skipping blacklisted target: {target_id}")
            continue

        # Skip if struct_only and no structural data
        if struct_only and not (struct_dir / target_id).exists():
            continue

        target = load_target(target_id, data_root, series=series)
        targets.append(target)

    scope = "all" if allow is None else f"filtered to {len(allow)}"
    logger.info(
        f"Loaded {len(targets)} targets for {series} "
        f"(struct_only={struct_only}, scope={scope})"
    )
    return targets
