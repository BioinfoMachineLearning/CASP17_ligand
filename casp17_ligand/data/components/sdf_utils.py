"""SDF generation utilities: RDKit template matching + optional protonation.

Core workflow:
  Reference ligand PDB + SMILES → AssignBondOrdersFromTemplate → SDF with correct bond orders

Optionally applies Dimorphite-DL protonation at pH 7.4 before template matching.
"""

import logging
import os
import warnings
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def protonate_smiles(smiles: str, min_ph: float = 6.4, max_ph: float = 8.4) -> str:
    """Protonate a SMILES string at physiological pH using Dimorphite-DL.

    Args:
        smiles: Input SMILES string (typically neutral).
        min_ph: Minimum pH for protonation range.
        max_ph: Maximum pH for protonation range.

    Returns:
        Protonated SMILES string.  Returns original SMILES if Dimorphite-DL
        is not available or fails.
    """
    try:
        import dimorphite_dl
    except ImportError:
        logger.warning(
            "dimorphite_dl not installed. Skipping protonation. "
            "Install with: pip install dimorphite-dl"
        )
        return smiles

    try:
        # v2 API: module-level function (ph_min, ph_max)
        if hasattr(dimorphite_dl, 'protonate_smiles'):
            protonated = dimorphite_dl.protonate_smiles(
                smiles, ph_min=min_ph, ph_max=max_ph
            )
        # v1 API: class-based
        elif hasattr(dimorphite_dl, 'DimorphiteDL'):
            d = dimorphite_dl.DimorphiteDL(
                min_ph=min_ph, max_ph=max_ph,
                max_variants=1, label_states=False, pka_precision=1.0,
            )
            protonated = d.protonate(smiles)
        else:
            logger.warning("Unknown Dimorphite-DL API version. Skipping protonation.")
            return smiles

        if protonated and len(protonated) > 0:
            result = protonated[0] if isinstance(protonated, list) else protonated
            if result != smiles:
                logger.info(f"Protonation: {smiles} → {result}")
            return result
        return smiles
    except Exception as e:
        logger.warning(f"Dimorphite-DL failed for '{smiles}': {e}. Using original.")
        return smiles


def process_ligand_pdb_to_mol(
    pdb_path: Path,
    smiles: str,
    sanitize: bool = True,
) -> Optional[Chem.Mol]:
    """Convert a ligand PDB file to an RDKit Mol with correct bond orders.

    Uses RDKit's AssignBondOrdersFromTemplate to assign bond orders from
    SMILES template to PDB coordinates.

    Args:
        pdb_path: Path to ligand PDB file.
        smiles: SMILES string for this ligand.
        sanitize: Whether to attempt sanitization.

    Returns:
        RDKit Mol with correct bond orders and 3D coordinates, or None on failure.
    """
    # 1. Load PDB with RDKit (coordinates but wrong bond orders)
    rd_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=sanitize)

    if rd_mol is None and sanitize:
        logger.warning(
            f"Sanitized PDB loading failed for {pdb_path.name}. "
            f"Retrying without sanitization..."
        )
        rd_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)

    if rd_mol is None:
        logger.error(f"Failed to load PDB: {pdb_path}")
        return None

    # 2. Create template from SMILES
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        logger.error(f"Failed to parse SMILES: {smiles}")
        return rd_mol  # Return mol without bond order correction

    # 3. Assign bond orders from template
    try:
        new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
        return new_mol
    except ValueError as e:
        # ============================================================
        # ⚠️ DOUBLE FAILURE: sanitize failed AND template match failed
        # Bond orders are likely INCORRECT. Flag for manual review.
        # ============================================================
        logger.error(
            f"⚠️ BOND ORDER WARNING for {pdb_path.name}:\n"
            f"  AssignBondOrdersFromTemplate FAILED: {e}\n"
            f"  SMILES: {smiles}\n"
            f"  PDB atoms: {rd_mol.GetNumAtoms()}, Template atoms: {template.GetNumAtoms()}\n"
            f"  The molecule will be used with INCORRECT bond orders.\n"
            f"  This may affect symmetry detection and lDDT-PLI/RMSD scores.\n"
            f"  ➜ Please manually inspect this ligand."
        )
        return rd_mol


def pdb_to_sdf(
    pdb_path: Path,
    smiles: str,
    output_sdf: Path,
    sanitize: bool = True,
) -> Optional[Path]:
    """Convert a single ligand PDB to SDF with correct bond orders.

    Args:
        pdb_path: Path to ligand PDB file.
        smiles: SMILES string for this ligand.
        output_sdf: Path to write output SDF.
        sanitize: Whether to sanitize the molecule.

    Returns:
        Path to output SDF, or None on failure.
    """
    mol = process_ligand_pdb_to_mol(pdb_path, smiles, sanitize=sanitize)
    if mol is None:
        return None

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()

    logger.info(
        f"  SDF written: {output_sdf.name} "
        f"({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)"
    )
    return output_sdf


def generate_ref_sdfs(
    target,  # TargetData
    output_dir: Path,
    protonation: bool = False,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
) -> List[Path]:
    """Generate reference ligand SDF files from PDB + SMILES for a target.

    For each reference ligand PDB, matches it with the corresponding SMILES
    from the target's ligand list and generates an SDF with correct bond orders.

    Args:
        target: TargetData with ligands and ref_ligand_pdbs.
        output_dir: Directory to write SDF files.
        protonation: Whether to apply Dimorphite-DL protonation at physiological pH.
        min_ph: Minimum pH for protonation (only used if protonation=True).
        max_ph: Maximum pH for protonation (only used if protonation=True).

    Returns:
        List of paths to generated SDF files.
    """
    if not target.has_struct:
        logger.warning(f"No structural data for {target.target_id}. Skipping SDF generation.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a mapping from ligand name to SMILES
    # Multiple ligands may share the same name (e.g., L4000 dimers both have "LIG")
    name_to_smiles = {}
    name_to_smiles_idx = {}
    for lig in target.ligands:
        if lig.name not in name_to_smiles:
            name_to_smiles[lig.name] = lig.smiles
            name_to_smiles_idx[lig.name] = 0
        else:
            name_to_smiles_idx[lig.name] += 1

    generated_sdfs = []
    for lig_pdb in target.ref_ligand_pdbs:
        # Extract ligand name from filename: ligand_{name}_{chain}_{resnum}.pdb
        parts = lig_pdb.stem.split("_")
        if len(parts) >= 2:
            lig_name = parts[1]
        else:
            lig_name = lig_pdb.stem

        # Find matching SMILES
        smiles = name_to_smiles.get(lig_name)
        if smiles is None:
            logger.warning(
                f"No SMILES found for ligand '{lig_name}' in {target.target_id}. "
                f"Available: {list(name_to_smiles.keys())}. Skipping."
            )
            continue

        # Optional protonation
        if protonation:
            smiles = protonate_smiles(smiles, min_ph=min_ph, max_ph=max_ph)

        # Generate SDF
        sdf_name = lig_pdb.stem + ".sdf"
        sdf_path = output_dir / sdf_name
        result = pdb_to_sdf(lig_pdb, smiles, sdf_path)
        if result is not None:
            generated_sdfs.append(result)

    if not generated_sdfs:
        logger.warning(f"No SDFs generated for {target.target_id}")
    else:
        logger.info(
            f"Generated {len(generated_sdfs)} SDF(s) for {target.target_id} "
            f"(protonation={'ON' if protonation else 'OFF'})"
        )

    return generated_sdfs


def combine_sdfs(sdf_paths: List[Path], output_sdf: Path) -> Optional[Path]:
    """Combine multiple SDF files into a single multi-entry SDF file.

    Each molecule is written as a separate entry (separated by $$$$),
    NOT combined into one molecule via CombineMols. This is the correct
    format for OpenStructure's compare-ligand-structures -rl parameter.

    Args:
        sdf_paths: List of input SDF file paths.
        output_sdf: Path to write combined SDF.

    Returns:
        Path to output SDF, or None on failure.
    """
    mols = []
    for sdf_path in sdf_paths:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
        for mol in supplier:
            if mol is not None:
                mols.append(mol)

    if not mols:
        logger.warning("No molecules to combine")
        return None

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    for mol in mols:
        writer.write(mol)
    writer.close()

    total_atoms = sum(m.GetNumAtoms() for m in mols)
    logger.info(
        f"Combined {len(mols)} entries into {output_sdf.name} "
        f"({total_atoms} atoms total)"
    )
    return output_sdf


def smiles_to_sdf(
    smiles: str,
    output_sdf: Path,
    protonation: bool = False,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
) -> Optional[Path]:
    """Create an SDF file from a SMILES string (2D, no 3D coordinates).

    Useful for creating model ligand inputs when only SMILES is available.

    Args:
        smiles: SMILES string.
        output_sdf: Path to write output SDF.
        protonation: Whether to apply pH 7.4 protonation.
        min_ph: Minimum pH for protonation.
        max_ph: Maximum pH for protonation.

    Returns:
        Path to output SDF, or None on failure.
    """
    if protonation:
        smiles = protonate_smiles(smiles, min_ph=min_ph, max_ph=max_ph)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        logger.error(f"Failed to parse SMILES: {smiles}")
        return None

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()

    return output_sdf
