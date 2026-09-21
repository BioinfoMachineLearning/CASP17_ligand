"""Ensemble generation and consensus ranking for CASP17 ligand predictions.

Flow: Boltz-2 CIF outputs → PyMOL CIF→PDB+SDF → RMSD consensus ranking → PoseBusters filter
Follows MULTICOM_ligand ranking protocol strictly.
"""

import json
import logging
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import pymol
from omegaconf import DictConfig
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Geometry import Point3D
from posebusters import PoseBusters

from casp17_ligand.utils.lg_chem_validate import validate_sdf_against_smiles_list

log = logging.getLogger(__name__)

# Targets to skip due to sequence mismatch
SKIP_TARGETS = set()

# Type aliases matching MULTICOM conventions
# (method, protein_pdb, ligand_sdf, score)
RankedPredictions = Dict[int, Tuple[str, str, str, float]]


from casp17_ligand.analysis.self_ranking_comparison import _get_ensemble_cif_list
from casp17_ligand.utils.data_utils import extract_protein_and_ligands_with_prody


def _validate_ligand_sdf(sdf_path: str, smiles: str) -> bool:
    """Check that an SDF file has the correct bond count matching the SMILES template.

    :param sdf_path: Path to SDF file to validate.
    :param smiles: SMILES string for the expected ligand.
    :return: True if bond count matches, False otherwise.
    """
    try:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            return False
        template = Chem.RemoveHs(template)
        expected_bonds = template.GetNumBonds()

        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        mol = supplier[0]
        if mol is None:
            return False
        actual_bonds = mol.GetNumBonds()

        if actual_bonds != expected_bonds:
            log.debug(f"Bond count mismatch in {sdf_path}: expected {expected_bonds}, got {actual_bonds}")
            return False
        return True
    except Exception as e:
        log.debug(f"Validation failed for {sdf_path}: {e}")
        return False


def _build_ligand_sdf_via_smiles_mcs(
    cif_path: str, sdf_path: str, smiles: str, lig_chain: str = "B"
) -> bool:
    """Build ligand SDF using MCS mapping: SMILES topology + CIF coordinates.

    PyMOL loads CIF → MOL2 (correct coords, possibly wrong bonds).
    SMILES → RDKit template (correct topology).
    MCS maps atoms between the two, then coordinates are transferred.

    :param cif_path: Path to CIF file containing the ligand.
    :param sdf_path: Output SDF path.
    :param smiles: SMILES string for correct topology.
    :param lig_chain: Chain ID for the ligand (e.g. "B" or "B0").
    :return: True on success, False on failure.
    """
    import tempfile
    mol2_path = None
    try:
        # Step 1: Extract ligand coordinates via PyMOL → MOL2
        obj = f"mcs_{os.path.basename(cif_path).replace('.', '_')}"
        cmd.load(cif_path, obj)
        mol2_fd, mol2_path = tempfile.mkstemp(suffix=".mol2")
        os.close(mol2_fd)
        cmd.save(mol2_path, f'{obj} and chain "{lig_chain}"')
        cmd.delete(obj)

        # Step 2: Load MOL2 into RDKit (coords correct, bonds may be wrong)
        mol2_mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=False)
        if mol2_mol is None:
            log.warning(f"MCS fallback: could not load MOL2 for {cif_path}")
            return False

        # Step 3: Build template from SMILES (correct topology)
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            log.warning(f"MCS fallback: invalid SMILES: {smiles}")
            return False
        template = Chem.RemoveHs(template)

        # Step 4: MCS match with BondCompare.CompareAny to handle bond order mismatches
        mcs = rdFMCS.FindMCS(
            [template, mol2_mol],
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            timeout=10,
        )
        if mcs.numAtoms < template.GetNumAtoms():
            log.warning(
                f"MCS fallback: incomplete match for {cif_path} "
                f"({mcs.numAtoms}/{template.GetNumAtoms()} atoms)"
            )
            return False

        # Step 5: Get atom mapping from MCS
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        match_template = template.GetSubstructMatch(mcs_mol)
        match_mol2 = mol2_mol.GetSubstructMatch(mcs_mol)

        if len(match_template) != template.GetNumAtoms() or len(match_mol2) != len(match_template):
            log.warning(f"MCS fallback: atom mapping incomplete for {cif_path}")
            return False

        # Step 6: Transfer coordinates from MOL2 to template
        # Build mapping: template_atom_idx → mol2_atom_idx
        tmpl_to_mol2 = {}
        for mcs_idx in range(len(match_template)):
            tmpl_to_mol2[match_template[mcs_idx]] = match_mol2[mcs_idx]

        # Generate 3D conformer on template using MOL2 coordinates
        conf_mol2 = mol2_mol.GetConformer()
        AllChem.EmbedMolecule(template, AllChem.ETKDGv3())
        conf = template.GetConformer()
        for tmpl_idx, mol2_idx in tmpl_to_mol2.items():
            pos = conf_mol2.GetAtomPosition(mol2_idx)
            conf.SetAtomPosition(tmpl_idx, pos)

        # Step 7: Write SDF
        writer = Chem.SDWriter(sdf_path)
        writer.write(template)
        writer.close()

        log.info(f"MCS fallback succeeded for {os.path.basename(sdf_path)}")
        return True

    except Exception as e:
        log.warning(f"MCS fallback failed for {cif_path}: {e}")
        try:
            cmd.delete("all")
        except Exception:
            pass
        return False
    finally:
        if mol2_path and os.path.exists(mol2_path):
            os.remove(mol2_path)


def _strip_salt_smiles(smiles: Optional[str]) -> Optional[str]:
    """Return the largest organic fragment, dropping small salt counterions.

    Folding methods (AF3/Boltz-2/Protenix) place a salt's counterion (Cl-, Br-,
    Na+, sulfate, ...) in the SAME ligand residue as the organic fragment, while
    the input SMILES is two disconnected components (e.g. 'Cl.c1ccc...'). The
    1-residue vs 2-component mismatch defeats ProDy/MCS bond assignment and the
    whole target is skipped (CASP17 L-series: 12 salt fragments). Stripping the
    SMILES to its largest organic fragment lets the connected-fragment MCS match
    cleanly (the ion, being disconnected, drops out of the pose SDF).

    Conservative: a no-op unless the SMILES is multi-component AND every
    non-largest component is a small counterion (<=6 heavy atoms) — never strips
    a genuine co-ligand. Returns the input unchanged on any parse failure.
    """
    if not smiles or '.' not in smiles:
        return smiles
    parts = smiles.split('.')
    mols = [(p, Chem.MolFromSmiles(p)) for p in parts]
    if any(m is None for _, m in mols):
        return smiles
    mols.sort(key=lambda pm: pm[1].GetNumHeavyAtoms(), reverse=True)
    if mols[0][1].GetNumHeavyAtoms() > 6 and all(
            m.GetNumHeavyAtoms() <= 6 for _, m in mols[1:]):
        return mols[0][0]
    return smiles


def receptor_with_cofactor_path(protein_pdb: str) -> str:
    """Sidecar receptor that also carries the supplied cofactor, given the protein PDB.

    Derived from the name rather than returned by :func:`cif_to_pdb_sdf` so the
    (pdb, sdf) return contract and every existing caller stay untouched.  Consumers
    must treat the file as optional and fall back to ``protein_pdb`` when absent —
    it is only written for targets that come with a cofactor.
    """
    if protein_pdb.endswith("_protein.pdb"):
        return protein_pdb[: -len("_protein.pdb")] + "_receptor_cofactor.pdb"
    return os.path.splitext(protein_pdb)[0] + "_receptor_cofactor.pdb"


def cif_to_pdb_sdf(
    cif_path: str, out_dir: str, name: str, smiles: Optional[str] = None,
    prot_chain: str = "A", lig_chain: str = "B",
    err_log_path: Optional[str] = None,
    cofactor_chains: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Convert CIF output to PDB (protein) + SDF (ligand).

    Ligand-centric extraction: keeps ALL protein chains (not just prot_chain)
    plus the specified ligand chain. This ensures the full protein environment
    around the ligand is preserved, which is critical for interface ligands
    in dimers/multimers where the binding pocket spans multiple protein chains.

    The prot_chain parameter is kept for backward compatibility but is only used
    as a fallback hint; all large chains (>100 atoms) are automatically included.

    When ``cofactor_chains`` is given, a third file ``{name}_receptor_cofactor.pdb``
    is written alongside as a sidecar (see :func:`receptor_with_cofactor_path`).
    It exists only for PoseBusters, which needs the cofactor present to run its
    four cofactor checks at all.  The cofactor deliberately does NOT go into
    ``{name}_protein.pdb`` or the ProDy complex: the >100-atom rule that builds
    ``prot_chains`` excludes ZN (1 atom) and SFG (27), and ProDy is invoked with
    ``load_hetatms_as_ligands=True``, so a cofactor in the complex would come
    back out as a second ligand and corrupt the SDF.

    :param cif_path: Path to input CIF file.
    :param out_dir: Output directory.
    :param name: Base name for output files (e.g. 'L1001_model_0').
    :param smiles: SMILES string for bond order assignment. If None, raw PyMOL SDF is kept.
    :param prot_chain: Protein chain ID hint (kept for compatibility).
    :param lig_chain: Ligand chain ID to extract (e.g. "B" or "B0").
    :param cofactor_chains: Chain IDs of supplied cofactors (e.g. ["D"] for ZN).
        ``None``/empty reproduces the previous behaviour exactly.
    :return: (protein_pdb_path, ligand_sdf_path) or (None, None) on failure.
    """
    pdb_path = os.path.join(out_dir, f"{name}_protein.pdb")
    sdf_path = os.path.join(out_dir, f"{name}_ligand.sdf")
    cof_rec_path = receptor_with_cofactor_path(pdb_path)
    cofactor_chains = [c for c in (cofactor_chains or []) if c]

    # Normalize SMILES separator: ensemble_inputs.csv uses ':' but RDKit uses '.'
    if smiles:
        smiles = smiles.replace(':', '.')
        # Drop salt counterions so the organic fragment matches the pose (the
        # methods co-locate the ion in the ligand residue → 2-component mismatch).
        smiles = _strip_salt_smiles(smiles)

    # Cache hit: validate existing SDF if smiles provided. A cache written before
    # cofactors were requested has no sidecar, so treat that as a miss too —
    # otherwise PoseBusters silently reverts to a cofactor-free receptor.
    if (os.path.exists(pdb_path) and os.path.exists(sdf_path)
            and (not cofactor_chains or os.path.exists(cof_rec_path))):
        if smiles and not _validate_ligand_sdf(sdf_path, smiles):
            log.info(f"Cached SDF has wrong bonds, regenerating: {name}")
            os.remove(sdf_path)
        else:
            return pdb_path, sdf_path

    complex_pdb = os.path.join(out_dir, f"{name}_complex.pdb")
    try:
        obj = f"obj_{name}"
        cmd.load(cif_path, obj)
        chains = cmd.get_chains(obj)
        if lig_chain not in chains:
            log.warning(f"Expected ligand chain {lig_chain} in {cif_path}, got {chains}")
            cmd.delete(obj)
            return None, None

        # Ligand-centric: include ALL protein chains (>100 atoms) + the target ligand chain
        prot_chains = []
        for ch in chains:
            if ch == lig_chain:
                continue
            n_atoms = cmd.count_atoms(f'{obj} and chain "{ch}"')
            if n_atoms > 100:
                prot_chains.append(ch)
        if not prot_chains:
            # Fallback: use prot_chain hint
            if prot_chain in chains:
                prot_chains = [prot_chain]
            else:
                log.warning(f"No protein chains found in {cif_path}")
                cmd.delete(obj)
                return None, None

        select_chains = prot_chains + [lig_chain]
        sel = " or ".join([f'chain "{ch}"' for ch in select_chains])
        cmd.save(complex_pdb, f'{obj} and ({sel})')
        # RNA-target fallback: ProDy's protein extractor doesn't recognize RNA
        # chains and returns an empty PDB. Save the receptor-only PDB via PyMOL
        # to a sidecar; we swap it in if ProDy's output is empty.
        receptor_sel = " or ".join([f'chain "{ch}"' for ch in prot_chains])
        pymol_receptor_pdb = os.path.join(out_dir, f"{name}_receptor_pymol.pdb")
        cmd.save(pymol_receptor_pdb, f'{obj} and ({receptor_sel})')
        # PoseBusters-only receptor: protein + cofactor. PyMOL writes the cofactor
        # as HETATM, which is what makes PoseBusters classify it as a cofactor
        # instead of protein (ZN also qualifies by element).
        if cofactor_chains:
            present = [ch for ch in cofactor_chains if ch in chains]
            if len(present) < len(cofactor_chains):
                log.warning(f"Cofactor chain(s) {sorted(set(cofactor_chains) - set(present))} "
                            f"absent from {cif_path}; PB receptor built without them")
            if present:
                cof_sel = " or ".join([f'chain "{ch}"' for ch in prot_chains + present])
                cmd.save(cof_rec_path, f'{obj} and ({cof_sel})')
        # Does the receptor contain nucleic-acid chains? ProDy's protein
        # extractor silently drops RNA/DNA, so for mixed (RNA+protein) or
        # RNA-only receptors we must keep the complete PyMOL receptor below.
        has_nucleic = cmd.count_atoms(
            f'{obj} and ({receptor_sel}) and polymer.nucleic') > 0
        cmd.delete(obj)
    except Exception as e:
        log.warning(f"PyMOL conversion failed for {cif_path}: {e}")
        try:
            cmd.delete("all")
        except Exception:
            pass
        return None, None

    prody_ok = False
    try:
        extracted_mol = extract_protein_and_ligands_with_prody(
            input_pdb_file=complex_pdb,
            protein_output_pdb_file=pdb_path,
            ligands_output_sdf_file=sdf_path,
            ligand_smiles=smiles,
            load_hetatms_as_ligands=True,
            generify_resnames=False
        )
        if extracted_mol is not None:
            prody_ok = True
    except Exception as e:
        log.debug(f"ProDy extraction raised for {name}: {e}")

    if os.path.exists(complex_pdb):
        os.remove(complex_pdb)

    # ProDy's protein extractor drops nucleic-acid chains, so prefer the
    # complete PyMOL-saved receptor whenever ProDy came back empty (RNA-only
    # targets) OR the receptor has nucleic chains (mixed RNA+protein, e.g.
    # M-series — otherwise the RNA would be silently lost). Protein-only
    # targets keep ProDy's output unchanged.
    if os.path.exists(pymol_receptor_pdb):
        if has_nucleic or not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
            os.replace(pymol_receptor_pdb, pdb_path)
        else:
            os.remove(pymol_receptor_pdb)

    # Validate the SDF from ProDy and apply MCS fallback if needed
    if prody_ok and os.path.exists(sdf_path) and smiles:
        if not _validate_ligand_sdf(sdf_path, smiles):
            msg = f"[{name}] ProDy SDF has wrong bonds, trying MCS fallback"
            log.warning(msg)
            if err_log_path:
                try:
                    with open(err_log_path, "a") as ef:
                        ef.write(msg + "\n")
                except Exception:
                    pass
            os.remove(sdf_path)
            prody_ok = False

    if not prody_ok or not os.path.exists(sdf_path):
        # ProDy failed to produce a valid ligand SDF — try MCS fallback
        if smiles:
            mcs_ok = _build_ligand_sdf_via_smiles_mcs(cif_path, sdf_path, smiles, lig_chain)
            if not mcs_ok:
                msg = f"[{name}] CRITICAL: Both ProDy and MCS fallback failed — model skipped"
                log.warning(msg)
                if err_log_path:
                    try:
                        with open(err_log_path, "a") as ef:
                            ef.write(msg + "\n")
                    except Exception:
                        pass
                return None, None
        else:
            msg = f"[{name}] ProDy failed and no SMILES for MCS fallback — model skipped"
            log.warning(msg)
            if err_log_path:
                try:
                    with open(err_log_path, "a") as ef:
                        ef.write(msg + "\n")
                except Exception:
                    pass
            return None, None

    # ProDy always writes protein PDB before processing ligand, so it should exist.
    # If it doesn't, something fundamentally went wrong — bail out.
    if not os.path.exists(pdb_path):
        log.warning(f"Protein PDB not generated by ProDy for {name}")
        return None, None

    return pdb_path, sdf_path


def read_ligand_positions(sdf_path: str) -> Optional[np.ndarray]:
    """Read ligand heavy-atom 3D positions from SDF file (same as MULTICOM read_molecule).

    :param sdf_path: Path to SDF file.
    :return: numpy array of shape (N_atoms, 3) or None on failure.
    """
    try:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
        mol = supplier[0]
        if mol is None:
            return None
        return mol.GetConformer().GetPositions()
    except Exception as e:
        log.warning(f"Failed to read ligand positions from {sdf_path}: {e}")
        return None


def align_complex_to_reference(
    ref_pdb: str,
    mob_pdb: str, mob_sdf: str,
    mob_lig_pos: np.ndarray,
    aligned_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Superimpose mobile protein onto reference protein (CA atoms), apply same
    rotation/translation to mobile ligand positions.

    Numerically equivalent to MULTICOM's align_complex_to_protein_only
    (verified: 0.000000 Å difference vs scipy Rotation.align_vectors).

    :param ref_pdb: Reference protein PDB path.
    :param mob_pdb: Mobile protein PDB path.
    :param mob_sdf: Mobile ligand SDF path (used to save aligned SDF).
    :param mob_lig_pos: Mobile ligand atom positions (N, 3).
    :param aligned_dir: If provided, save aligned PDB and SDF to this directory.
    :return: Aligned ligand positions, or None on failure.
    """
    try:
        from Bio import PDB
        from Bio.PDB.Superimposer import Superimposer
        from Bio.PDB import PDBIO

        parser = PDB.PDBParser(QUIET=True)
        ref_struct = parser.get_structure("ref", ref_pdb)
        mob_struct = parser.get_structure("mob", mob_pdb)

        ref_cas = [r["CA"] for r in ref_struct.get_residues() if "CA" in r]
        mob_cas = [r["CA"] for r in mob_struct.get_residues() if "CA" in r]
        n = min(len(ref_cas), len(mob_cas))
        if n < 3:
            return None

        sup = Superimposer()
        sup.set_atoms(ref_cas[:n], mob_cas[:n])
        rot = np.array(sup.rotran[0])
        tran = np.array(sup.rotran[1])
        aligned_lig = mob_lig_pos @ rot + tran

        if aligned_dir is not None:
            os.makedirs(aligned_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(mob_pdb))[0]
            # Save aligned protein PDB
            sup.apply(mob_struct.get_atoms())
            io = PDBIO()
            io.set_structure(mob_struct)
            io.save(os.path.join(aligned_dir, f"{base}_aligned.pdb"))

            # Save aligned ligand SDF
            mol = Chem.SDMolSupplier(mob_sdf, sanitize=True)[0]
            if mol is not None:
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, Point3D(*aligned_lig[i]))
                sdf_base = os.path.splitext(os.path.basename(mob_sdf))[0]
                writer = Chem.SDWriter(os.path.join(aligned_dir, f"{sdf_base}_aligned.sdf"))
                writer.write(mol)
                writer.close()

        return aligned_lig
    except Exception as e:
        log.warning(f"Protein alignment failed: {e}")
        return None


def rmsd_consensus_rank(
    predictions: List[Tuple[str, str, str]],
    aligned_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> RankedPredictions:
    """RMSD-based consensus ranking with pairwise protein alignment and incremental caching."""
    import json
    
    raw_positions = []
    valid_predictions = []
    cached_cas = []
    seen_positions = {}
    
    # Load cache if exists
    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load RMSD cache: {e}")
            pairwise_cache = {}
    
    from Bio import PDB
    from Bio.PDB.Superimposer import Superimposer
    parser = PDB.PDBParser(QUIET=True)

    # First pass: parse coordinates and CA arrays for all requested models
    for pred in predictions:
        method, pdb, sdf = pred
        pos = read_ligand_positions(sdf)
        if pos is None:
            log.warning(f"Skipping {sdf}: could not read positions")
            continue
            
        pos_key = tuple(map(tuple, pos))
        if pos_key in seen_positions:
            log.debug(f"Duplicate pose skipped: {sdf}")
            continue
            
        try:
            struct = parser.get_structure("X", pdb)
            cas = [r["CA"] for r in struct.get_residues() if "CA" in r]
        except Exception:
            continue

        seen_positions[pos_key] = True
        raw_positions.append(pos)
        valid_predictions.append(pred)
        cached_cas.append(cas)

    if len(valid_predictions) == 0:
        return {}
    if len(valid_predictions) == 1:
        method, pdb, sdf = valid_predictions[0]
        return {1: (method, pdb, sdf, 0.0)}

    n = len(valid_predictions)
    avg_rmsd = []
    cache_updated = False
    
    # Second pass: compute pairwise similarities
    import concurrent.futures

    def compute_rmsd_row(i):
        rmsd_vals = []
        method_i = valid_predictions[i][0]
        ref_cas = cached_cas[i]
        row_cache = {}
        for j in range(n):
            if i == j: continue
            method_j = valid_predictions[j][0]
            # Check cache!
            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                rmsd_vals.append(pairwise_cache[method_i][method_j])
                continue
                
            mob_cas = cached_cas[j]
            num_ca = min(len(ref_cas), len(mob_cas))
            if num_ca < 3: continue
            
            sup = Superimposer()
            sup.set_atoms(ref_cas[:num_ca], mob_cas[:num_ca])
            rot = np.array(sup.rotran[0])
            tran = np.array(sup.rotran[1])
            
            aligned_pos_j = raw_positions[j] @ rot + tran
            diff = aligned_pos_j - raw_positions[i]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            rmsd = float(np.sqrt(np.mean(dist**2)))
            rmsd_vals.append(rmsd)
            row_cache[method_j] = rmsd
        return i, method_i, rmsd_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_rmsd_row, i) for i in range(n)]
        
        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_rmsd_vals, row_cache = future.result()
            
            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}
                
            for method_j, rmsd in row_cache.items():
                pairwise_cache[method_i][method_j] = rmsd
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = rmsd
                cache_updated = True
                
            avg_rmsd.append((i, np.mean(row_rmsd_vals) if row_rmsd_vals else 0.0))

    avg_rmsd.sort(key=lambda x: x[0])
    avg_rmsd_arr = np.array([x[1] for x in avg_rmsd])

    # Save cache incrementally
    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save RMSD cache: {e}")

    # Lower RMSD is better
    sorted_idx = np.argsort(avg_rmsd_arr)

    return {
        rank + 1: (*valid_predictions[idx], float(avg_rmsd_arr[idx]))
        for rank, idx in enumerate(sorted_idx)
    }


def rmsd_6a_consensus_rank(
    predictions: List[Tuple[str, str, str]],
    cache_path: Optional[str] = None,
    pocket_cutoff: float = 6.0,
) -> RankedPredictions:
    """RMSD-based consensus ranking using pocket-only CA alignment (6A around ligand).

    Instead of aligning all protein CA atoms, only aligns CA atoms of residues
    within pocket_cutoff Å of the ligand. This focuses the alignment on the
    binding site and can improve ligand RMSD accuracy.
    """
    import json
    from Bio import PDB
    from Bio.PDB.NeighborSearch import NeighborSearch

    parser = PDB.PDBParser(QUIET=True)

    # Load cache if exists
    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load RMSD_6A cache: {e}")
            pairwise_cache = {}

    # First pass: parse ligand positions, protein CA coords, and pocket residue keys
    raw_positions = []
    valid_predictions = []
    # For each model: dict mapping (chain_id, resseq) -> CA coordinate array
    all_ca_by_reskey = []
    # For each model: set of (chain_id, resseq) keys for pocket residues
    all_pocket_reskeys = []
    seen_positions = {}

    for pred in predictions:
        method, pdb, sdf = pred
        pos = read_ligand_positions(sdf)
        if pos is None:
            log.warning(f"Skipping {sdf}: could not read positions")
            continue

        pos_key = tuple(map(tuple, pos))
        if pos_key in seen_positions:
            log.debug(f"Duplicate pose skipped: {sdf}")
            continue

        try:
            struct = parser.get_structure("X", pdb)
        except Exception:
            continue

        # Build CA coordinate map by residue key
        ca_by_reskey = {}
        for r in struct.get_residues():
            if r.id[0] != " ":
                continue
            if "CA" in r:
                chain_id = r.get_parent().id
                resseq = r.id[1]
                ca_by_reskey[(chain_id, resseq)] = r["CA"].get_vector().get_array()

        if len(ca_by_reskey) < 3:
            continue

        # Find pocket residues: protein residues with any atom within cutoff of ligand
        all_atoms = list(struct[0].get_atoms())
        ns = NeighborSearch(all_atoms)
        pocket_reskeys = set()
        for lx, ly, lz in pos:
            for r in ns.search(np.array([lx, ly, lz]), pocket_cutoff, level="R"):
                if r.id[0] == " " and "CA" in r:
                    chain_id = r.get_parent().id
                    resseq = r.id[1]
                    pocket_reskeys.add((chain_id, resseq))

        if len(pocket_reskeys) < 3:
            log.debug(f"Too few pocket residues ({len(pocket_reskeys)}) for {method}, using all CAs")
            pocket_reskeys = set(ca_by_reskey.keys())

        seen_positions[pos_key] = True
        raw_positions.append(pos)
        valid_predictions.append(pred)
        all_ca_by_reskey.append(ca_by_reskey)
        all_pocket_reskeys.append(pocket_reskeys)

    if len(valid_predictions) == 0:
        return {}
    if len(valid_predictions) == 1:
        method, pdb, sdf = valid_predictions[0]
        return {1: (method, pdb, sdf, 0.0)}

    n = len(valid_predictions)
    avg_rmsd = []
    cache_updated = False

    import concurrent.futures

    def compute_rmsd_6a_row(i):
        rmsd_vals = []
        method_i = valid_predictions[i][0]
        pocket_i = all_pocket_reskeys[i]
        ca_i = all_ca_by_reskey[i]
        row_cache = {}

        for j in range(n):
            if i == j:
                continue
            method_j = valid_predictions[j][0]

            # Check cache
            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                rmsd_vals.append(pairwise_cache[method_i][method_j])
                continue

            ca_j = all_ca_by_reskey[j]
            # Find common pocket residues (use model i's pocket definition)
            common_keys = sorted(k for k in pocket_i if k in ca_j)
            if len(common_keys) < 3:
                continue

            ref_coords = np.array([ca_i[k] for k in common_keys])
            mob_coords = np.array([ca_j[k] for k in common_keys])

            # Kabsch alignment on pocket CAs
            ref_center = ref_coords.mean(axis=0)
            mob_center = mob_coords.mean(axis=0)
            ref_c = ref_coords - ref_center
            mob_c = mob_coords - mob_center
            H = mob_c.T @ ref_c
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            sign_matrix = np.diag([1, 1, d])
            rot = (Vt.T @ sign_matrix @ U.T).T  # rot such that mob @ rot + tran ≈ ref
            tran = ref_center - mob_center @ rot

            aligned_pos_j = raw_positions[j] @ rot + tran
            diff = aligned_pos_j - raw_positions[i]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            rmsd = float(np.sqrt(np.mean(dist ** 2)))
            rmsd_vals.append(rmsd)
            row_cache[method_j] = rmsd

        return i, method_i, rmsd_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_rmsd_6a_row, i) for i in range(n)]

        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_rmsd_vals, row_cache = future.result()

            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}

            for method_j, rmsd in row_cache.items():
                pairwise_cache[method_i][method_j] = rmsd
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = rmsd
                cache_updated = True

            avg_rmsd.append((i, np.mean(row_rmsd_vals) if row_rmsd_vals else 0.0))

    avg_rmsd.sort(key=lambda x: x[0])
    avg_rmsd_arr = np.array([x[1] for x in avg_rmsd])

    # Save cache
    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save RMSD_6A cache: {e}")

    # Lower RMSD is better
    sorted_idx = np.argsort(avg_rmsd_arr)

    return {
        rank + 1: (*valid_predictions[idx], float(avg_rmsd_arr[idx]))
        for rank, idx in enumerate(sorted_idx)
    }


def rmsd_8a_consensus_rank(predictions, cache_path=None):
    return rmsd_6a_consensus_rank(predictions, cache_path=cache_path, pocket_cutoff=8.0)


def rmsd_10a_consensus_rank(predictions, cache_path=None):
    return rmsd_6a_consensus_rank(predictions, cache_path=cache_path, pocket_cutoff=10.0)


# ---------------------------------------------------------------------------
# Multi-ligand consensus ranking functions (for duplicate-ligand targets)
# ---------------------------------------------------------------------------

def rmsd_consensus_rank_multilig(
    predictions_by_chain: Dict[str, List[Tuple[str, str, str]]],
    aligned_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    chain_smiles: Optional[Dict[str, str]] = None,
) -> RankedPredictions:
    """RMSD consensus ranking across multiple ligand chains.

    For each pair of models (i, j): align using full protein CA atoms,
    then compute ligand RMSD for all cross-chain pairs (same SMILES only), take min.
    Models may be present in only a subset of chains (e.g. SeedFold only predicted
    one chain for large multimers). Uses model name as key, not index.
    """
    import json
    from Bio import PDB
    from Bio.PDB.Superimposer import Superimposer

    chain_ids = list(predictions_by_chain.keys())
    if len(chain_ids) < 2:
        return rmsd_consensus_rank(predictions_by_chain[chain_ids[0]], aligned_dir=aligned_dir, cache_path=cache_path)

    # Build per-chain lookup by model name (handles different model counts per chain)
    chain_lookup = {}
    all_names = set()
    for ch in chain_ids:
        chain_lookup[ch] = {p[0]: p for p in predictions_by_chain[ch]}
        all_names.update(chain_lookup[ch].keys())
    method_names = sorted(all_names)
    n = len(method_names)

    # unified_preds: pick PDB from any chain that has the model (protein is same)
    unified_preds = []
    for name in method_names:
        for ch in chain_ids:
            if name in chain_lookup[ch]:
                unified_preds.append(chain_lookup[ch][name])
                break

    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception:
            pairwise_cache = {}

    parser = PDB.PDBParser(QUIET=True)

    cached_cas = []
    valid_mask = [True] * n
    chain_positions = {ch: [] for ch in chain_ids}

    for i in range(n):
        _, pdb, _ = unified_preds[i]
        try:
            struct = parser.get_structure("X", pdb)
            cas = [r["CA"] for r in struct.get_residues() if "CA" in r]
        except Exception:
            cas = []

        if len(cas) < 3:
            valid_mask[i] = False
            cached_cas.append([])
            for ch in chain_ids:
                chain_positions[ch].append(None)
            continue

        cached_cas.append(cas)

        # Read ligand positions from each chain (None if model missing from chain)
        has_any = False
        for ch in chain_ids:
            pred = chain_lookup[ch].get(method_names[i])
            if pred is not None:
                pos = read_ligand_positions(pred[2])
                chain_positions[ch].append(pos)
                if pos is not None:
                    has_any = True
            else:
                chain_positions[ch].append(None)

        if not has_any:
            valid_mask[i] = False

    valid_indices = [i for i in range(n) if valid_mask[i]]
    if len(valid_indices) == 0:
        return {}
    if len(valid_indices) == 1:
        idx = valid_indices[0]
        method, pdb, sdf = unified_preds[idx]
        return {1: (method, pdb, sdf, 0.0)}

    avg_rmsd = []
    cache_updated = False

    import concurrent.futures

    def compute_rmsd_row_multilig(i):
        rmsd_vals = []
        method_i = method_names[i]
        row_cache = {}
        ref_cas = cached_cas[i]

        for j in valid_indices:
            if i == j:
                continue
            method_j = method_names[j]

            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                rmsd_vals.append(pairwise_cache[method_i][method_j])
                continue

            mob_cas = cached_cas[j]
            num_ca = min(len(ref_cas), len(mob_cas))
            if num_ca < 3:
                continue

            sup = Superimposer()
            sup.set_atoms(ref_cas[:num_ca], mob_cas[:num_ca])
            rot = np.array(sup.rotran[0])
            tran = np.array(sup.rotran[1])

            min_rmsd = float("inf")
            for ch_a in chain_ids:
                pos_i = chain_positions[ch_a][i]
                if pos_i is None:
                    continue
                for ch_b in chain_ids:
                    if chain_smiles and chain_smiles.get(ch_a) != chain_smiles.get(ch_b):
                        continue
                    pos_j = chain_positions[ch_b][j]
                    if pos_j is None:
                        continue
                    aligned_pos_j = pos_j @ rot + tran
                    diff = aligned_pos_j - pos_i
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    rmsd = float(np.sqrt(np.mean(dist ** 2)))
                    if rmsd < min_rmsd:
                        min_rmsd = rmsd

            if min_rmsd < float("inf"):
                rmsd_vals.append(min_rmsd)
                row_cache[method_j] = min_rmsd

        return i, method_i, rmsd_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_rmsd_row_multilig, i) for i in valid_indices]

        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_rmsd_vals, row_cache = future.result()

            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}

            for method_j, rmsd in row_cache.items():
                pairwise_cache[method_i][method_j] = rmsd
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = rmsd
                cache_updated = True

            avg_rmsd.append((i, np.mean(row_rmsd_vals) if row_rmsd_vals else 0.0))

    avg_rmsd.sort(key=lambda x: x[0])

    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save RMSD multilig cache: {e}")

    idx_to_avg = {entry[0]: entry[1] for entry in avg_rmsd}
    scored = [(idx, idx_to_avg.get(idx, 0.0)) for idx in valid_indices]
    scored.sort(key=lambda x: x[1])

    return {
        rank + 1: (*unified_preds[idx], float(score))
        for rank, (idx, score) in enumerate(scored)
    }


def _classify_protein_chains(struct) -> str:
    """Classify a protein structure's chain composition.

    Returns 'homo' if all chains have the same sequence (single-chain or
    homo-multimer), 'hetero' if chains have different sequences.
    """
    try:
        from Bio.PDB.Polypeptide import three_to_one
    except ImportError:
        from Bio.SeqUtils import seq1 as three_to_one
    seqs = set()
    for chain in struct[0]:
        seq = ""
        for r in chain:
            if r.id[0] == " " and "CA" in r:
                try:
                    seq += three_to_one(r.resname)
                except Exception:
                    pass
        if seq:
            seqs.add(seq)
    return "homo" if len(seqs) <= 1 else "hetero"


# Per-target reference for hetero-chain sequence alignment.
# Set by _pocket_reskeys on the first call for a hetero target, reused for all subsequent models.
_hetero_ref_seq: Optional[str] = None
_hetero_ref_residues: Optional[list] = None


def _reset_hetero_ref():
    """Reset the hetero reference. Call at the start of each target's processing."""
    global _hetero_ref_seq, _hetero_ref_residues
    _hetero_ref_seq = None
    _hetero_ref_residues = None


def _get_residue_seq(struct):
    """Extract (residue_list, sequence_string) for standard residues."""
    try:
        from Bio.PDB.Polypeptide import three_to_one
    except ImportError:
        from Bio.SeqUtils import seq1 as three_to_one
    residues = [r for r in struct.get_residues() if r.id[0] == " " and "CA" in r]
    seq = ""
    for r in residues:
        try:
            seq += three_to_one(r.resname)
        except Exception:
            seq += "X"
    return residues, seq


def _build_residue_mapping_hetero(struct):
    """For hetero-chain structures, map each Residue object to a canonical
    integer ID via global sequence alignment against a shared reference.

    The first model encountered becomes the reference. All subsequent models
    (even with different chain counts) are aligned to the same reference.

    Returns dict: {Residue_obj → canonical_int}.
    """
    global _hetero_ref_seq, _hetero_ref_residues
    from Bio import pairwise2

    residues, seq = _get_residue_seq(struct)

    if _hetero_ref_seq is None:
        # First model → becomes the reference
        _hetero_ref_seq = seq
        _hetero_ref_residues = residues
        return {r: i for i, r in enumerate(residues)}

    # Align current model's sequence to the reference
    alns = pairwise2.align.globalxx(_hetero_ref_seq, seq)
    if not alns:
        return {r: i for i, r in enumerate(residues)}
    aln_ref, aln_mob = alns[0][0], alns[0][1]
    mapping = {}
    ri, mi = 0, 0
    for rc, mc in zip(aln_ref, aln_mob):
        if rc != "-" and mc != "-":
            mapping[residues[mi]] = ri
        if rc != "-":
            ri += 1
        if mc != "-":
            mi += 1
    return mapping


def _pocket_reskeys(struct, mol, cutoff: float = 6.0):
    """Extract pocket residue keys for IoU comparison.

    For homo-chain structures (single chain or all chains identical):
      Uses resnum as key — symmetric chains share the same numbering.

    For hetero-chain structures (chains with different sequences):
      Uses sequence-alignment-mapped canonical index as key — handles
      different chain counts and residue numbering across methods.

    Returns: set of hashable keys (int).
    """
    from casp17_ligand.utils.pocket_metrics import _pocket_from_mol

    p_res = _pocket_from_mol(struct, mol, cutoff)
    if not p_res:
        return set()

    chain_type = _classify_protein_chains(struct)

    if chain_type == "homo":
        # Resnum-only: works for single chain and homo-multimers
        return set(r.id[1] for r in p_res)
    else:
        # Hetero-chain: use sequence alignment mapping
        mapping = _build_residue_mapping_hetero(struct)
        return set(mapping[r] for r in p_res if r in mapping)


def _copy_pair_scores(chain_ids, frags_i, frags_j, chain_smiles=None) -> Dict[Tuple, float]:
    """Best pocket-IoU x SuCOS for every ordered (copy_a, copy_b) chain pair.

    ``frags_i`` / ``frags_j`` map chain id -> ``[(fragment_mol, pocket_reskeys), ...]``
    for the two models being compared, or ``None`` where that model has no pose on
    that chain.  Copies of different chemical species are never paired.  Pairs
    that share no pocket residues are omitted rather than stored as 0.0, so an
    empty dict means the two models agree nowhere.

    Split out of :func:`sucos_consensus_rank_multilig` so the Butina threshold
    calibration scores copies with the exact same code that ranks them.
    """
    from casp17_ligand.utils.pocket_metrics import ligand_sucos

    scores: Dict[Tuple, float] = {}
    for ch_a in chain_ids:
        fp_i = frags_i.get(ch_a)
        if fp_i is None:
            continue
        for ch_b in chain_ids:
            if chain_smiles and chain_smiles.get(ch_a) != chain_smiles.get(ch_b):
                continue
            fp_j = frags_j.get(ch_b)
            if fp_j is None:
                continue
            best_leakage = 0.0
            for f_i, p_i in fp_i:
                for f_j, p_j in fp_j:
                    union = len(p_i | p_j)
                    iou = len(p_i & p_j) / union if union > 0 else 0.0
                    if iou == 0.0:
                        continue
                    leakage = iou * ligand_sucos(f_i, f_j)
                    if leakage > best_leakage:
                        best_leakage = leakage
            if best_leakage > 0.0:
                scores[(ch_a, ch_b)] = best_leakage
    return scores


def _greedy_copy_assignment(pair_scores: Dict[Tuple, float], n_slots: int) -> float:
    """Score a model pair by greedily matching their ligand copies one-to-one.

    ``pair_scores`` maps (copy_a, copy_b) to the similarity between copy ``a`` of
    one model and copy ``b`` of the other; absent keys are unpairable.  Copies are
    matched highest-score-first with each used at most once — deliberately greedy
    rather than optimal (Hungarian), because that is the assignment OpenStructure
    runs when CASP scores a multi-copy ligand target, and our ranking should agree
    with the scorer about which copy answers which reference ligand.

    The sum is divided by ``n_slots`` (the number of copies the target asks for),
    so unmatched copies score zero.  Taking a max over copy pairs instead — the
    previous behaviour — let a model that places one copy well and scatters the
    rest rank level with one that places every copy correctly.
    """
    if n_slots <= 0:
        return 0.0
    used_a, used_b = set(), set()
    total = 0.0
    for (a, b), s in sorted(pair_scores.items(), key=lambda kv: (-kv[1], kv[0])):
        if s <= 0.0:
            break
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        total += s
    return total / n_slots


def sucos_consensus_rank_multilig(
    predictions_by_chain: Dict[str, List[Tuple[str, str, str]]],
    aligned_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    chain_smiles: Optional[Dict[str, str]] = None,
) -> RankedPredictions:
    """SuCOS consensus ranking across multiple ligand chains.

    For each pair of models (i, j): score every copy-vs-copy combination
    (same SMILES only), pair the copies up one-to-one, and average.  Averaging
    rather than taking the best pair is what makes a model that places only one
    of its copies well rank below one that places both — see
    :func:`_greedy_copy_assignment`.
    Models may be present in only a subset of chains.
    """
    import json
    from casp17_ligand.utils.pocket_metrics import load_ligand_mol
    from Bio import PDB

    chain_ids = list(predictions_by_chain.keys())
    if len(chain_ids) < 2:
        return sucos_consensus_rank(predictions_by_chain[chain_ids[0]], aligned_dir=aligned_dir, cache_path=cache_path)

    # Build per-chain lookup by model name
    chain_lookup = {}
    all_names = set()
    for ch in chain_ids:
        chain_lookup[ch] = {p[0]: p for p in predictions_by_chain[ch]}
        all_names.update(chain_lookup[ch].keys())
    method_names = sorted(all_names)
    n = len(method_names)

    unified_preds = []
    for name in method_names:
        for ch in chain_ids:
            if name in chain_lookup[ch]:
                unified_preds.append(chain_lookup[ch][name])
                break

    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception:
            pairwise_cache = {}

    parser = PDB.PDBParser(QUIET=True)

    chain_frags = {ch: [] for ch in chain_ids}
    valid_mask = [True] * n

    for i in range(n):
        has_any = False
        for ch in chain_ids:
            pred = chain_lookup[ch].get(method_names[i])
            if pred is None:
                chain_frags[ch].append(None)
                continue

            _, pdb, sdf = pred
            mol = load_ligand_mol(sdf)
            if mol is None:
                chain_frags[ch].append(None)
                continue

            try:
                struct = parser.get_structure("X", pdb)
                from rdkit import Chem
                frags = Chem.GetMolFrags(mol, asMols=True)
                if not frags:
                    chain_frags[ch].append(None)
                    continue

                max_ha = max(f.GetNumHeavyAtoms() for f in frags)
                largest_frags = [f for f in frags if f.GetNumHeavyAtoms() == max_ha]

                frag_pocket_pairs = []
                for f in largest_frags:
                    p_keys = _pocket_reskeys(struct, f, 6.0)
                    frag_pocket_pairs.append((f, p_keys))

                if not frag_pocket_pairs:
                    chain_frags[ch].append(None)
                    continue

                chain_frags[ch].append(frag_pocket_pairs)
                has_any = True
            except Exception:
                chain_frags[ch].append(None)

        if not has_any:
            valid_mask[i] = False

    valid_indices = [i for i in range(n) if valid_mask[i]]
    if len(valid_indices) == 0:
        return {}
    if len(valid_indices) == 1:
        idx = valid_indices[0]
        return {1: (*unified_preds[idx], 1.0)}

    avg_sucos = []
    cache_updated = False

    import concurrent.futures

    def compute_sucos_row_multilig(i):
        sucos_vals = []
        method_i = method_names[i]
        row_cache = {}

        for j in valid_indices:
            if i == j:
                continue
            method_j = method_names[j]

            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                sucos_vals.append(pairwise_cache[method_i][method_j])
                continue

            # Score every copy-vs-copy combination, then pair the copies up
            # one-to-one instead of keeping only the single best pair.
            pair_scores = _copy_pair_scores(
                chain_ids,
                {ch: chain_frags[ch][i] for ch in chain_ids},
                {ch: chain_frags[ch][j] for ch in chain_ids},
                chain_smiles)

            score = _greedy_copy_assignment(pair_scores, len(chain_ids))
            sucos_vals.append(score)
            row_cache[method_j] = score

        return i, method_i, sucos_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_sucos_row_multilig, i) for i in valid_indices]

        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_sucos_vals, row_cache = future.result()

            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}

            for method_j, leakage in row_cache.items():
                pairwise_cache[method_i][method_j] = leakage
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = leakage
                cache_updated = True

            avg_sucos.append((i, np.mean(row_sucos_vals) if row_sucos_vals else 0.0))

    avg_sucos.sort(key=lambda x: x[0])

    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save SuCOS multilig cache: {e}")

    idx_to_avg = {entry[0]: entry[1] for entry in avg_sucos}
    scored = [(idx, idx_to_avg.get(idx, 0.0)) for idx in valid_indices]
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        rank + 1: (*unified_preds[idx], float(score))
        for rank, (idx, score) in enumerate(scored)
    }


def rmsd_6a_consensus_rank_multilig(
    predictions_by_chain: Dict[str, List[Tuple[str, str, str]]],
    cache_path: Optional[str] = None,
    pocket_cutoff: float = 6.0,
    chain_smiles: Optional[Dict[str, str]] = None,
) -> RankedPredictions:
    """RMSD consensus ranking with pocket-only CA alignment across multiple ligand chains.

    For each pair of models (i, j): for each cross-chain pair of ligands
    (same SMILES only), find pocket residues, do Kabsch alignment, compute
    ligand RMSD, take min. Models may be present in only a subset of chains.
    """
    import json
    from Bio import PDB
    from Bio.PDB.NeighborSearch import NeighborSearch

    chain_ids = list(predictions_by_chain.keys())
    if len(chain_ids) < 2:
        return rmsd_6a_consensus_rank(predictions_by_chain[chain_ids[0]], cache_path=cache_path, pocket_cutoff=pocket_cutoff)

    # Build per-chain lookup by model name
    chain_lookup = {}
    all_names = set()
    for ch in chain_ids:
        chain_lookup[ch] = {p[0]: p for p in predictions_by_chain[ch]}
        all_names.update(chain_lookup[ch].keys())
    method_names = sorted(all_names)
    n = len(method_names)

    unified_preds = []
    for name in method_names:
        for ch in chain_ids:
            if name in chain_lookup[ch]:
                unified_preds.append(chain_lookup[ch][name])
                break

    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception:
            pairwise_cache = {}

    parser = PDB.PDBParser(QUIET=True)

    all_ca_by_reskey = []
    chain_positions = {ch: [] for ch in chain_ids}
    chain_pocket_reskeys = {ch: [] for ch in chain_ids}
    valid_mask = [True] * n

    for i in range(n):
        _, pdb, _ = unified_preds[i]
        try:
            struct = parser.get_structure("X", pdb)
        except Exception:
            valid_mask[i] = False
            all_ca_by_reskey.append({})
            for ch in chain_ids:
                chain_positions[ch].append(None)
                chain_pocket_reskeys[ch].append(set())
            continue

        ca_by_reskey = {}
        for r in struct.get_residues():
            if r.id[0] != " ":
                continue
            if "CA" in r:
                chain_id = r.get_parent().id
                resseq = r.id[1]
                ca_by_reskey[(chain_id, resseq)] = r["CA"].get_vector().get_array()

        if len(ca_by_reskey) < 3:
            valid_mask[i] = False
            all_ca_by_reskey.append({})
            for ch in chain_ids:
                chain_positions[ch].append(None)
                chain_pocket_reskeys[ch].append(set())
            continue

        all_ca_by_reskey.append(ca_by_reskey)

        all_atoms = list(struct[0].get_atoms())
        ns = NeighborSearch(all_atoms)

        has_any = False
        for ch in chain_ids:
            pred = chain_lookup[ch].get(method_names[i])
            if pred is None:
                chain_positions[ch].append(None)
                chain_pocket_reskeys[ch].append(set())
                continue

            pos = read_ligand_positions(pred[2])
            if pos is None:
                chain_positions[ch].append(None)
                chain_pocket_reskeys[ch].append(set())
                continue

            chain_positions[ch].append(pos)
            has_any = True

            pocket_reskeys = set()
            for lx, ly, lz in pos:
                for r in ns.search(np.array([lx, ly, lz]), pocket_cutoff, level="R"):
                    if r.id[0] == " " and "CA" in r:
                        c_id = r.get_parent().id
                        resseq = r.id[1]
                        pocket_reskeys.add((c_id, resseq))

            if len(pocket_reskeys) < 3:
                pocket_reskeys = set(ca_by_reskey.keys())

            chain_pocket_reskeys[ch].append(pocket_reskeys)

        if not has_any:
            valid_mask[i] = False

    valid_indices = [i for i in range(n) if valid_mask[i]]
    if len(valid_indices) == 0:
        return {}
    if len(valid_indices) == 1:
        idx = valid_indices[0]
        return {1: (*unified_preds[idx], 0.0)}

    avg_rmsd = []
    cache_updated = False

    import concurrent.futures

    def compute_rmsd_6a_row_multilig(i):
        rmsd_vals = []
        method_i = method_names[i]
        row_cache = {}

        for j in valid_indices:
            if i == j:
                continue
            method_j = method_names[j]

            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                rmsd_vals.append(pairwise_cache[method_i][method_j])
                continue

            ca_i = all_ca_by_reskey[i]
            ca_j = all_ca_by_reskey[j]

            min_rmsd = float("inf")

            for ch_a in chain_ids:
                pos_i = chain_positions[ch_a][i]
                pocket_i = chain_pocket_reskeys[ch_a][i]
                if pos_i is None:
                    continue
                for ch_b in chain_ids:
                    if chain_smiles and chain_smiles.get(ch_a) != chain_smiles.get(ch_b):
                        continue
                    pos_j = chain_positions[ch_b][j]
                    if pos_j is None:
                        continue

                    common_keys = sorted(k for k in pocket_i if k in ca_j)
                    if len(common_keys) < 3:
                        continue

                    ref_coords = np.array([ca_i[k] for k in common_keys])
                    mob_coords = np.array([ca_j[k] for k in common_keys])

                    ref_center = ref_coords.mean(axis=0)
                    mob_center = mob_coords.mean(axis=0)
                    ref_c = ref_coords - ref_center
                    mob_c = mob_coords - mob_center
                    H = mob_c.T @ ref_c
                    U, S, Vt = np.linalg.svd(H)
                    d = np.linalg.det(Vt.T @ U.T)
                    sign_matrix = np.diag([1, 1, d])
                    rot = (Vt.T @ sign_matrix @ U.T).T
                    tran = ref_center - mob_center @ rot

                    aligned_pos_j = pos_j @ rot + tran
                    diff = aligned_pos_j - pos_i
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    rmsd = float(np.sqrt(np.mean(dist ** 2)))
                    if rmsd < min_rmsd:
                        min_rmsd = rmsd

            if min_rmsd < float("inf"):
                rmsd_vals.append(min_rmsd)
                row_cache[method_j] = min_rmsd

        return i, method_i, rmsd_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_rmsd_6a_row_multilig, i) for i in valid_indices]

        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_rmsd_vals, row_cache = future.result()

            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}

            for method_j, rmsd in row_cache.items():
                pairwise_cache[method_i][method_j] = rmsd
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = rmsd
                cache_updated = True

            avg_rmsd.append((i, np.mean(row_rmsd_vals) if row_rmsd_vals else 0.0))

    avg_rmsd.sort(key=lambda x: x[0])

    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save RMSD_6A multilig cache: {e}")

    idx_to_avg = {entry[0]: entry[1] for entry in avg_rmsd}
    scored = [(idx, idx_to_avg.get(idx, 0.0)) for idx in valid_indices]
    scored.sort(key=lambda x: x[1])

    return {
        rank + 1: (*unified_preds[idx], float(score))
        for rank, (idx, score) in enumerate(scored)
    }


def rmsd_8a_consensus_rank_multilig(predictions_by_chain, cache_path=None, chain_smiles=None):
    return rmsd_6a_consensus_rank_multilig(predictions_by_chain, cache_path=cache_path, pocket_cutoff=8.0, chain_smiles=chain_smiles)


def rmsd_10a_consensus_rank_multilig(predictions_by_chain, cache_path=None, chain_smiles=None):
    return rmsd_6a_consensus_rank_multilig(predictions_by_chain, cache_path=cache_path, pocket_cutoff=10.0, chain_smiles=chain_smiles)


def sucos_consensus_rank(
    predictions: List[Tuple[str, str, str]],
    aligned_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> RankedPredictions:
    """SuCOS-based consensus ranking utilizing cached pockets and incremental caching."""
    import json
    from casp17_ligand.utils.pocket_metrics import ligand_sucos, load_ligand_mol, _get_pocket_residues
    from Bio import PDB
    
    valid_predictions = []
    mols = []
    pockets = []
    seen_positions = {}
    
    # Load cache if exists
    pairwise_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                pairwise_cache = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load SuCOS cache: {e}")
            pairwise_cache = {}
            
    parser = PDB.PDBParser(QUIET=True)

    for pred in predictions:
        method, pdb, sdf = pred
        pos = read_ligand_positions(sdf)
        if pos is None: continue
        pos_key = tuple(map(tuple, pos))
        if pos_key in seen_positions: continue
        
        mol = load_ligand_mol(sdf)
        if mol is None: continue
        
        try:
            struct = parser.get_structure("X", pdb)
            from rdkit import Chem
            from casp17_ligand.utils.pocket_metrics import _pocket_from_mol
            
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags: continue
            
            max_ha = max(f.GetNumHeavyAtoms() for f in frags)
            largest_frags = [f for f in frags if f.GetNumHeavyAtoms() == max_ha]
            
            target_frags = []
            target_pockets = []
            for f in largest_frags:
                p_keys = _pocket_reskeys(struct, f, 6.0)
                target_frags.append(f)
                target_pockets.append(p_keys)
                
            if not target_frags:
                continue
                
        except Exception:
            continue
            
        seen_positions[pos_key] = True
        valid_predictions.append(pred)
        mols.append(target_frags)
        pockets.append(target_pockets)

    n = len(valid_predictions)
    if n == 0: return {}
    if n == 1: return {1: (valid_predictions[0][0], valid_predictions[0][1], valid_predictions[0][2], 1.0)}

    avg_sucos = []
    cache_updated = False
    
    # Precompute symmetric ligand SuCOS array (O3A self alignment) ON THE FLY using cache
    import concurrent.futures

    def compute_sucos_row(i):
        sucos_vals = []
        method_i = valid_predictions[i][0]
        row_cache = {}
        for j in range(n):
            if i == j: continue
            method_j = valid_predictions[j][0]
            if method_i in pairwise_cache and method_j in pairwise_cache[method_i]:
                sucos_vals.append(pairwise_cache[method_i][method_j])
                continue
            
            best_leakage = 0.0
            
            # Cross-pairwise maximum SuCOS score across largest fragments
            for f_i, p_i in zip(mols[i], pockets[i]):
                for f_j, p_j in zip(mols[j], pockets[j]):
                    union = len(p_i | p_j)
                    iou = len(p_i & p_j) / union if union > 0 else 0.0
                    if iou == 0.0:
                        continue
                    val = ligand_sucos(f_i, f_j)
                    leakage = iou * val
                    if leakage > best_leakage:
                        best_leakage = leakage
                        
            sucos_vals.append(best_leakage)
            row_cache[method_j] = best_leakage
            
        return i, method_i, sucos_vals, row_cache

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compute_sucos_row, i) for i in range(n)]
        for future in concurrent.futures.as_completed(futures):
            i, method_i, row_sucos_vals, row_cache = future.result()
            
            if method_i not in pairwise_cache:
                pairwise_cache[method_i] = {}
                
            for method_j, leakage in row_cache.items():
                pairwise_cache[method_i][method_j] = leakage
                if method_j not in pairwise_cache:
                    pairwise_cache[method_j] = {}
                pairwise_cache[method_j][method_i] = leakage
                cache_updated = True
                
            avg_sucos.append((i, np.mean(row_sucos_vals) if row_sucos_vals else 0.0))

    avg_sucos.sort(key=lambda x: x[0])
    avg_sucos_arr = np.array([x[1] for x in avg_sucos])

    # Save cache incrementally
    if cache_updated and cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(pairwise_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save SuCOS cache: {e}")

    # Higher SuCOS is better
    sorted_idx = np.argsort(avg_sucos_arr)[::-1]

    return {
        rank + 1: (*valid_predictions[idx], float(avg_sucos_arr[idx]))
        for rank, idx in enumerate(sorted_idx)
    }


def run_posebusters(protein_pdb: str, ligand_sdf: str) -> bool:
    """Run PoseBusters dock-mode validation on a single pose.

    Returns True if the pose passes all key checks (same criteria as MULTICOM).

    When the cofactor-bearing sidecar exists it is used in place of
    ``protein_pdb``, so this final gate sees the same receptor the relax stage
    judged against.  Without it a pose buried in ZN or SFG passes all 22 checks
    and would be written out with a ``_pb=True`` suffix.
    """
    cof_rec = receptor_with_cofactor_path(protein_pdb)
    if os.path.exists(cof_rec):
        protein_pdb = cof_rec
    try:
        buster = PoseBusters(config="dock", top_n=None)
        bust_res = buster.bust([ligand_sdf], mol_cond=protein_pdb, full_report=False)
        if "posebusters_dock" in bust_res.columns:
            return bool(bust_res["posebusters_dock"].iloc[0])
        elif "passes_testing" in bust_res.columns:
            return bool(bust_res["passes_testing"].iloc[0])
        return bool(bust_res.iloc[0].all())
    except Exception as e:
        log.warning(f"PoseBusters failed structurally on {ligand_sdf}: {e}")
        return False


def run_chem_validate(ligand_sdf: str, ref_smiles_list: List[str]) -> bool:
    """LG-format chemistry validator: atom count + bond type + MCS topology vs ref SMILES.

    Mirrors the checks the official CASP `LG_validation.py` runs on the
    submitted mol blocks. Used as a pre-submission sanity filter alongside
    PoseBusters in `_save_ranked`.

    Returns True if every mol in `ligand_sdf` matches the parallel reference
    SMILES list. Empty / falsy `ref_smiles_list` short-circuits to True (no
    reference available → don't penalize).
    """
    if not ref_smiles_list:
        return True
    try:
        ok, _ = validate_sdf_against_smiles_list(ligand_sdf, ref_smiles_list)
        return ok
    except Exception as e:
        log.warning(f"chem_validate failed structurally on {ligand_sdf}: {e}")
        return False


def _cif_has_misplaced_atoms(cif_path: str, thresh: float = 2.5) -> bool:
    """True if any polymer (protein/RNA) heavy atom sits farther than ``thresh`` Å
    from its NEAREST same-residue heavy neighbour — i.e. a detached / mis-placed atom.

    Protenix (all series) occasionally predicts terminal cap atoms — protein
    C-terminal OXT, RNA 5'-phosphate (P/OP1/OP2/OP3) — at garbage coordinates
    (60-100+ Å from their own residue) in multi-chain complexes. The fold itself
    stays intact (0 CA-CA breaks), so pair_iptm cannot flag these; only a direct
    connectivity check catches them. Every heavy atom in a well-formed residue has
    a bonded neighbour < ~1.7 Å, so a nearest-neighbour distance > 2.5 Å means the
    atom is detached. Used by the prefilter to skip such structures and backfill
    with the next-ranked clean one (see _prefilter_cifs).
    """
    import numpy as _np
    from collections import defaultdict as _dd
    try:
        obj = f"tcheck_{os.path.basename(cif_path).replace('.', '_')}"
        cmd.load(cif_path, obj)
        m = cmd.get_model(f"{obj} and polymer and not elem H")
        cmd.delete(obj)
    except Exception:
        try:
            cmd.delete("all")
        except Exception:
            pass
        return False  # cannot inspect → do not block selection
    res = _dd(list)
    for a in m.atom:
        res[(a.chain, a.resi)].append(a.coord)
    for coords in res.values():
        if len(coords) < 2:
            continue
        P = _np.asarray(coords, dtype=float)
        d = _np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        _np.fill_diagonal(d, _np.inf)
        if float(d.min(axis=1).max()) > thresh:
            return True
    return False


SP3_LINEAR_DEG = 150.0
"""A 2-coordinate heavy atom wider than this is linear, not merely strained.

Calibrated on 8 CASP17_T targets. Kept in sync with
evaluate_topn_clusters.SP3_MAX_ANGLE_DEG.
"""


def _n_linear_centres_in_smiles(smiles: Optional[str]) -> Optional[int]:
    """How many 2-connected heavy atoms the reference molecule is *supposed* to
    have at ~180 deg (nitrile / alkyne / allene carbons). None if unknown.

    Needed to tell a genuine sp centre from a collapsed sp3 one: both look like a
    linear 2-coordinate atom with a ~1.14 A bond in a bond-order-free CIF.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem as _Chem
    except ImportError:
        return None
    n = 0
    for part in str(smiles).split('.'):
        m = _Chem.MolFromSmiles(part)
        if m is None:
            return None            # cannot trust the count → disable the gate
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue
            if a.GetHybridization() == _Chem.HybridizationType.SP and \
               len([x for x in a.GetNeighbors() if x.GetAtomicNum() > 1]) == 2:
                n += 1
    return n


def _cif_ligand_has_broken_sp3(cif_path: str, n_allowed_linear: Optional[int]) -> bool:
    """True if the ligand has more near-linear 2-coordinate heavy atoms than the
    reference SMILES allows — i.e. an sp3 centre has been flattened to ~180 deg.

    Co-folding models sometimes render a saturated terminal group as if it were a
    triple bond: T2455 (2026-07-29) had SeedFold emit every Ar-CH2-NH3+ at
    1.14 A / 179 deg — 100/100 poses — with Protenix at 76/100, while AF3 and
    Boltz-2 were clean. PoseBusters' dock config passes all of them, because its
    "bond_angles" test is really a 1-3 *distance* check and the collapsed bond
    shrinks that distance in step with the opening angle.

    Bond orders are not available in the CIF, so a genuine sp centre and a
    collapsed sp3 centre are geometrically identical; we distinguish them by
    counting how many linear centres the reference SMILES actually has. If the
    SMILES is missing or unparseable the gate disables itself (returns False).
    """
    if n_allowed_linear is None:
        return False
    import numpy as _np
    try:
        obj = f"sp3chk_{os.path.basename(cif_path).replace('.', '_')}"
        cmd.load(cif_path, obj)
        m = cmd.get_model(f"{obj} and not polymer and not elem H and not solvent")
        cmd.delete(obj)
    except Exception:
        try:
            cmd.delete("all")
        except Exception:
            pass
        return False               # cannot inspect → do not block selection
    if len(m.atom) < 4:
        return False
    P = _np.asarray([a.coord for a in m.atom], dtype=float)
    d = _np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    _np.fill_diagonal(d, _np.inf)
    n_linear = 0
    for i in range(len(P)):
        nb = _np.where(d[i] < 1.75)[0]          # covers the collapsed 1.14 A case
        if len(nb) != 2:
            continue
        v1, v2 = P[nb[0]] - P[i], P[nb[1]] - P[i]
        denom = _np.linalg.norm(v1) * _np.linalg.norm(v2)
        if denom == 0:
            continue
        ang = _np.degrees(_np.arccos(_np.clip(float(_np.dot(v1, v2) / denom), -1.0, 1.0)))
        if ang > SP3_LINEAR_DEG:
            n_linear += 1
    return n_linear > n_allowed_linear


def _prefilter_cifs(
    method: str,
    target: str,
    cif_paths: List[str],
    metric: str,
    topn: int,
    method_output_dir,
    input_json_dir: str = "",
    rtmscore_cache_dir: Optional[str] = None,
    n_protein_chains: int = 0,
    smiles: Optional[str] = None,
    n_cofactor_chains: int = 0,
) -> List[str]:
    """Filter CIF paths by confidence metric, keeping top-N.

    Uses score extraction from confidence_metric_analysis.
    Falls back to original list if scores cannot be collected.

    ``method_output_dir`` may be a single str or a list/tuple of dirs
    (combined across multiple data versions, e.g. v1 + v2).
    ``rtmscore_cache_dir`` overrides the RTMScore cache lookup path
    (defaults to a dataset-derived path inside collect_scores).
    ``n_protein_chains`` must be set for multimeric targets so that the
    ``pair_iptm`` metric measures protein–ligand rather than the (much higher)
    protein–protein interface — see ``extract_pair_chains_iptm``.
    ``n_cofactor_chains`` likewise excludes a supplied cofactor (ZN / SFG) from
    that measurement.
    """
    from casp17_ligand.analysis.confidence_metric_analysis import collect_scores
    scores = collect_scores(metric, method, target, method_output_dir, input_json_dir,
                            rtmscore_cache_dir=rtmscore_cache_dir,
                            n_protein_chains=n_protein_chains,
                            n_cofactor_chains=n_cofactor_chains)
    if not scores:
        return cif_paths

    # Map ensemble index → cif path
    from casp17_ligand.analysis.self_ranking_comparison import _get_ensemble_cif_list
    ensemble_cifs = _get_ensemble_cif_list(method, target, method_output_dir)
    canon_to_idx = {os.path.realpath(p): i for i, p in enumerate(ensemble_cifs)}

    # Score each cif_path
    scored = []
    for cp in cif_paths:
        idx = canon_to_idx.get(os.path.realpath(cp))
        if idx is not None and idx in scores:
            scored.append((scores[idx], cp))
        else:
            scored.append((float('-inf'), cp))  # unscored → lowest priority

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top-N by score, but SKIP structures with mis-placed terminal atoms
    # (Protenix OXT / RNA 5'-P artifact) and backfill with the next-ranked clean
    # structure — i.e. keep the top-N *clean* structures in score order. The fold
    # is intact in these cases so pair_iptm keeps them high, hence this explicit
    # connectivity gate. If too few clean candidates exist, the best skipped ones
    # are appended (better than returning < topn).
    # The sp3 gate runs here, BEFORE the top-N cut, so a method whose poses are
    # geometrically broken loses them from the candidate pool entirely rather
    # than having them merely barred from representing a cluster later. Policy
    # set 2026-07-30; it can change MODEL 1 when a whole
    # method is affected — on T2455 SeedFold went 0/100 and dropped out.
    n_allowed_linear = _n_linear_centres_in_smiles(smiles)
    selected, skipped, skipped_sp3 = [], [], []
    for score, cp in scored:
        if len(selected) >= topn:
            break
        if _cif_has_misplaced_atoms(cp):
            skipped.append((score, cp))
            continue
        if _cif_ligand_has_broken_sp3(cp, n_allowed_linear):
            skipped_sp3.append((score, cp))
            continue
        selected.append((score, cp))
    # Backfill applies ONLY to mis-placed-atom structures: that artefact sits on
    # the receptor 60-100 A from the ligand, so the pose itself is still usable
    # and returning < topn would needlessly shrink the pool.
    # Collapsed-sp3 structures are DROPPED OUTRIGHT — the defect is in the ligand
    # we are actually predicting, so a short pool is the correct outcome. On
    # T2455 this takes SeedFold to 0 and Protenix to 24 (pool 200 -> 124).
    if len(selected) < topn and skipped:
        for score, cp in skipped:
            if len(selected) >= topn:
                break
            selected.append((score, cp))
    if skipped:
        log.info(f"  [{method}] {target}: skipped {len(skipped)} structure(s) with "
                 f"mis-placed terminal atoms (Protenix C-term OXT / RNA 5'-P at garbage "
                 f"coords) → backfilled with next-ranked clean structures")
    if skipped_sp3:
        log.info(f"  [{method}] {target}: DROPPED {len(skipped_sp3)}/{len(cif_paths)} "
                 f"structure(s) with a collapsed sp3 centre (>{SP3_LINEAR_DEG:.0f} deg; "
                 f"reference SMILES allows {n_allowed_linear} linear centre(s)) — not "
                 f"backfilled; {len(selected)} candidate(s) remain for this method")
        if not selected:
            log.warning(f"  [{method}] {target}: ALL poses failed the sp3 gate — this "
                        f"method contributes nothing to the ensemble")

    filtered = [cp for _, cp in selected]
    if selected:
        log.info(f"  [{method}] {target}: prefilter {metric} top-{topn}/{len(cif_paths)} "
                 f"(score range: {selected[0][0]:.4f} → {selected[-1][0]:.4f})")
    return filtered


def _detect_ligand_chains(cif_path: str, prot_chain: str = "A") -> List[Tuple[str, int, str]]:
    """Detect all ligand chains in a CIF and return [(chain_id, n_atoms, formula), ...].

    A ligand chain is any chain that is NOT a protein chain. Protein chains are
    identified by having polymer atoms (>100 atoms typical for proteins). This
    handles monomers, dimers, and multimers automatically.

    The formula is a sorted element string (e.g. "C10H12N2O" or "Cl" for single ions),
    used downstream to match chains to input SMILES when multiple ligand types have
    the same heavy atom count.
    """
    try:
        obj = f"detect_{os.path.basename(cif_path).replace('.', '_')}"
        cmd.load(cif_path, obj)
        chains = cmd.get_chains(obj)
        # First pass: classify chains by size (protein chains are large)
        chain_sizes = []
        for ch in chains:
            n_atoms = cmd.count_atoms(f'{obj} and chain "{ch}"')
            chain_sizes.append((ch, n_atoms))
        if not chain_sizes:
            cmd.delete(obj)
            return []
        # A ligand chain is a NON-POLYMER chain (small molecule / ion), not a
        # short RNA/peptide. The polymer selector is the correct discriminator:
        # a pure atom-count threshold misclassifies a short nucleic/peptide chain
        # (e.g. a 10-nt RNA, ~218 atoms) as the ligand in multi-chain complexes
        # (M-series) when it is larger than the real ligand (~30 atoms) yet
        # smaller than the largest chain. Fall back to the legacy size threshold
        # only if no non-polymer chain is present (keeps old behavior otherwise).
        ligand_chains = [
            (ch, n) for ch, n in chain_sizes
            if cmd.count_atoms(f'{obj} and chain "{ch}" and not polymer') > 0
        ]
        if not ligand_chains:
            max_atoms = max(n for _, n in chain_sizes)
            prot_threshold = max(100, max_atoms * 0.1)  # at least 100, or 10% of largest
            ligand_chains = [(ch, n) for ch, n in chain_sizes if 0 < n <= prot_threshold]

        # Second pass: get element composition for each ligand chain
        from collections import Counter
        result = []
        for ch, n in ligand_chains:
            model = cmd.get_model(f'{obj} and chain "{ch}"')
            elem_counts = Counter()
            for atom in model.atom:
                elem = atom.symbol.strip()
                if elem and elem != 'H':
                    elem_counts[elem] += 1
            # Build sorted formula string: "C10Cl2N3O5" style
            formula = "".join(f"{e}{c if c > 1 else ''}" for e, c in sorted(elem_counts.items()))
            result.append((ch, n, formula))

        cmd.delete(obj)
        return result
    except Exception as e:
        log.warning(f"Failed to detect ligand chains in {cif_path}: {e}")
        try:
            cmd.delete("all")
        except Exception:
            pass
        return []


def _detect_cofactor_chains(cif_path: str, cofactor_names) -> List[str]:
    """Chain IDs holding one of ``cofactor_names`` (residue name match, e.g. ZN, SFG).

    A cofactor supplied with the target is a ligand chain like any other, so
    every heuristic that picks "the ligand" by size has to be told about it
    explicitly.  Size cannot be the discriminator here: SFG carries 27 heavy
    atoms and outweighs all 109 CASP17 L-series fragments (8–21), so the
    largest-chain rule selects the cofactor and the fragment is never ranked,
    clustered or submitted.  The CIF's residue name says which chain it is
    directly, so match on that rather than inferring it.
    """
    wanted = {c.strip().upper() for c in cofactor_names if c and c.strip()}
    if not wanted:
        return []
    obj = f"cofdet_{os.path.basename(cif_path).replace('.', '_')}"
    try:
        cmd.load(cif_path, obj)
        found = []
        for ch in cmd.get_chains(obj):
            resns = {a.resn.strip().upper()
                     for a in cmd.get_model(f'{obj} and chain "{ch}"').atom}
            if resns & wanted:
                found.append(ch)
        cmd.delete(obj)
        return found
    except Exception as e:
        log.warning(f"Failed to detect cofactor chains in {cif_path}: {e}")
        try:
            cmd.delete("all")
        except Exception:
            pass
        return []


def _find_method_cifs(method_name: str, target: str, base_dir, n_models: int) -> list:
    """Find CIF prediction files for a given method and target.

    Centralizes method-specific CIF discovery logic to avoid duplication.
    Returns list of CIF paths (up to n_models), or empty list if not found.

    ``base_dir`` may be a single str (original behavior) or a list/tuple of
    dirs — in the latter case, CIFs from each dir are concatenated in the
    given order; ``n_models`` acts as a per-dir cap.
    """
    import glob as _glob

    # Accept list/tuple/omegaconf ListConfig (anything iterable that is not a str)
    if not isinstance(base_dir, (str, bytes)) and base_dir is not None:
        try:
            dirs_iter = list(base_dir)
        except TypeError:
            dirs_iter = None
        if dirs_iter is not None:
            combined = []
            for d in dirs_iter:
                combined.extend(_find_method_cifs(method_name, target, d, n_models))
            return combined

    if method_name == "boltz2":
        # Layout A (single-seed r1): {base}/boltz_results_{target}_input/predictions/{chain}/
        pred_dir = os.path.join(base_dir, f"boltz_results_{target}_input", "predictions")
        if os.path.isdir(pred_dir):
            subdirs = [d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
            if subdirs:
                subdir = subdirs[0]
                return sorted(_glob.glob(os.path.join(
                    pred_dir, subdir, f"{subdir}_model_*.cif")))[:n_models]
        # Layout B (multi-seed r2): {base}/seed_{N}/boltz_results_{target}_input/predictions/{chain}/
        seed_dirs = sorted(_glob.glob(os.path.join(base_dir, "seed_*")))
        all_cifs = []
        for sd in seed_dirs:
            pd = os.path.join(sd, f"boltz_results_{target}_input", "predictions")
            if not os.path.isdir(pd):
                continue
            subdirs = [d for d in os.listdir(pd) if os.path.isdir(os.path.join(pd, d))]
            if not subdirs:
                continue
            subdir = subdirs[0]
            all_cifs.extend(sorted(_glob.glob(os.path.join(
                pd, subdir, f"{subdir}_model_*.cif"))))
        return all_cifs[:n_models]

    elif method_name == "af3":
        # Try nested layouts first (per-target subdir)
        for d in [os.path.join(base_dir, target, target.lower()),
                  os.path.join(base_dir, target.lower()),
                  os.path.join(base_dir, target)]:
            if os.path.isdir(d):
                cifs = sorted(_glob.glob(os.path.join(d, "*", "*_model.cif")))
                if cifs:
                    return cifs[:n_models]
        # Flat layout: all targets share base_dir, CIFs under seed-*_sample-*/
        # with per-target filename prefix. Match either case (R2314 used lower,
        # R2317/R2318 use capital); MUST stay in sync with _get_ensemble_cif_list.
        if os.path.isdir(base_dir):
            patterns = [
                f"{target.lower()}*_seed-*_sample-*_model.cif",
                f"{target}*_seed-*_sample-*_model.cif",
            ]
            out: list = []
            seen: set = set()
            for pat in patterns:
                for cif in sorted(_glob.glob(os.path.join(
                        base_dir, "seed-*_sample-*", pat))):
                    if cif in seen:
                        continue
                    seen.add(cif)
                    out.append(cif)
            if out:
                return out[:n_models]
        return []

    elif method_name == "protenix":
        d = os.path.join(base_dir, target)
        if not os.path.isdir(d):
            return []
        return sorted(_glob.glob(os.path.join(d, "seed_*", "predictions", f"{target}_sample_*.cif")))[:n_models]

    elif method_name == "seedfold":
        d = os.path.join(base_dir, target)
        if not os.path.isdir(d):
            return []
        return sorted(_glob.glob(os.path.join(d, "*", "*.cif")))[:n_models]

    elif method_name == "rf3":
        if not os.path.isdir(base_dir):
            return []
        cifs = sorted(_glob.glob(os.path.join(base_dir, f"{target}_seed-*", "**", "*.cif"), recursive=True))
        return [p for p in cifs if p.endswith("_model.cif")][:n_models]

    return []


def process_target(
    target: str,
    methods_cfg: DictConfig,
    ensemble_output_dir: str,
    smiles: Optional[str] = None,
    export_top_n: int = 5,
    prefilter_metric: Optional[str] = None,
    prefilter_topn: Optional[int] = None,
    lig_chain_override: Optional[str] = None,
    intra_workers: int = 1,
    n_protein_chains: int = 0,
    cofactor: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Run full ensemble pipeline for one target (both RMSD and SuCOS ranking).

    :param target: Target name, e.g. 'L1001'.
    :param methods_cfg: Configuration dictionary mapping prediction methods to their specific output folders / parameters.
    :param ensemble_output_dir: Root dir for ensemble outputs (under outputs/).
    :param smiles: SMILES string for bond order assignment in SDF.
    :param export_top_n: How many ranked poses to save per method.
    :param prefilter_metric: Optional confidence metric for pre-filtering (pair_iptm/ligand_plddt/ranking_score).
    :param prefilter_topn: Keep top-N models per method by prefilter_metric before ensemble.
    :param n_protein_chains: Protein chain count for multimeric targets (default
        0 → 1). Drives the pair_iptm protein/ligand chain split; without it a
        homodimer scores its own protein–protein interface. Also read per method
        from ``methods_cfg``: ``input_json_dir`` (enables exact chain typing) and
        ``allowed_models_json`` (conformation-group filter applied *before* the
        confidence prefilter).
    :param cofactor: Residue name(s) of any cofactor modelled alongside the
        fragment but supplied with the target rather than predicted by us
        (CASP17 L-series: ``"ZN"`` or ``"SFG"``; comma-separate for several).
        Those chains are excluded from ligand selection, from pair_iptm and from
        the SDF, and are written to a separate receptor file for PoseBusters.
        Default None keeps every ligand chain, which is the behaviour for every
        target that ships without one.
    :return: DataFrame with ranking results from both methods, or None on failure.
    """
    target_out = os.path.join(ensemble_output_dir, "targets", target)
    cif_out = os.path.join(target_out, "cif_converted")
    aligned_rmsd_out = os.path.join(target_out, "aligned_rmsd")
    aligned_sucos_out = os.path.join(target_out, "aligned_sucos")
    os.makedirs(cif_out, exist_ok=True)
    err_log_path = os.path.join(ensemble_output_dir, "err.log")

    # ── Step 1: Ligand-centric detection ─────────────────────────────────────
    # Detect ligand chains per method, identify main ligand instances,
    # and build a cross-method chain mapping by atom count + order.
    import glob as glob_mod
    from collections import defaultdict

    method_lig_info = {}   # method -> [(chain_id, n_atoms, formula), ...]
    method_cif_paths = {}  # method -> [cif_path, ...]
    method_canon_idx = {}  # method -> {realpath: idx}
    method_cof_chains = {}  # method -> [cofactor chain_id, ...]

    cofactor_names = [c.strip().upper()
                      for c in (cofactor or "").replace(";", ",").split(",")
                      if c.strip()]

    # Build RTMScore cache dir from ensemble_output_dir (forward-looking;
    # only used when prefilter_metric="rtmscore")
    rtm_cache_dir = os.path.join(ensemble_output_dir, "rtmscore_self_ranking")

    for method_name, method_cfg in methods_cfg.items():
        # Support both single `output_dir: str` and multi `output_dirs: list`
        # (combining v1 + v2 etc.). `output_dirs` takes precedence.
        base_dir = method_cfg.get("output_dirs") or method_cfg.get("output_dir", "")
        n_models = method_cfg.get("n_models", 5)
        cif_paths = _find_method_cifs(method_name, target, base_dir, n_models)
        if not cif_paths:
            continue

        ensemble_cifs = _get_ensemble_cif_list(method_name, target, base_dir)
        canon_to_idx = {os.path.realpath(p): idx for idx, p in enumerate(ensemble_cifs)}

        # ── Conformation-group / explicit allow-list filter ──────────────────
        # Applied BEFORE the confidence prefilter so that the top-N is taken
        # *within* the group. Doing it the other way round lets a global
        # ranking discard a minority conformation before it is ever seen.
        # Indices are the ORIGINAL global indices from the (uncapped)
        # _get_ensemble_cif_list enumeration, so `af3_model137` means the same
        # structure in every group, in the diagnostic, and in the submission.
        allow_json = method_cfg.get("allowed_models_json", "")
        if allow_json:
            with open(allow_json) as _f:
                _spec = json.load(_f)
            # Two layouts. Flat `{method: [idx]}` is the original single-target
            # form (T2451 conformation groups). Nested `{target: {method: [idx]}}`
            # lets one file cover a whole batch, which is what the L-series
            # pocket restriction needs -- the allow-list differs per fragment.
            if target in _spec and isinstance(_spec[target], dict):
                _allow = set(_spec[target].get(method_name, []))
            else:
                _allow = set(_spec.get(method_name, []))
            _before = len(cif_paths)
            cif_paths = [p for p in cif_paths
                         if canon_to_idx.get(os.path.realpath(p)) in _allow]
            log.info(f"  [{method_name}] {target}: allow-list "
                     f"{os.path.basename(allow_json)} → {len(cif_paths)}/{_before} models")
            if not cif_paths:
                continue

        if prefilter_metric and prefilter_topn and prefilter_topn < len(cif_paths):
            cif_paths = _prefilter_cifs(method_name, target, cif_paths,
                                        prefilter_metric, prefilter_topn, base_dir,
                                        input_json_dir=method_cfg.get("input_json_dir", ""),
                                        rtmscore_cache_dir=rtm_cache_dir,
                                        n_protein_chains=n_protein_chains,
                                        smiles=smiles,
                                        n_cofactor_chains=len(cofactor_names))

        if not cif_paths:
            # The sp3 gate can empty a whole method (T2455: SeedFold 0/100).
            # Drop it from the ensemble rather than indexing into an empty list.
            log.warning(f"  [{method_name}] {target}: no candidates left after "
                        f"filtering — method excluded from this target's ensemble")
            continue

        method_cif_paths[method_name] = cif_paths
        method_canon_idx[method_name] = canon_to_idx

        # Detect ligand chains from this method's first CIF
        prot_ch_hint = "A0" if method_name == "seedfold" else "A"
        info = _detect_ligand_chains(cif_paths[0], prot_chain=prot_ch_hint)
        if info and cofactor_names:
            cof_chains = _detect_cofactor_chains(cif_paths[0], cofactor_names)
            remaining = [t for t in info if t[0] not in cof_chains]
            if cof_chains and remaining:
                method_cof_chains[method_name] = cof_chains
                log.info(f"  [{method_name}] {target}: cofactor {cofactor_names} on "
                         f"chain(s) {cof_chains} — excluded from ligand selection")
                info = remaining
            elif cof_chains:
                # Every ligand chain looked like a cofactor. Keeping them is
                # wrong, but dropping them all leaves nothing to rank, so keep
                # going on the un-filtered list and make the mismatch loud.
                log.warning(f"  [{method_name}] {target}: cofactor {cofactor_names} "
                            f"matched ALL ligand chains {cof_chains} — not excluding")
            else:
                log.warning(f"  [{method_name}] {target}: expected cofactor "
                            f"{cofactor_names} but no chain carries it")
        if info:
            method_lig_info[method_name] = info

    if not method_cif_paths:
        log.warning(f"No CIF files found for any method for {target}")
        return None

    # Pick a reference method (prefer non-seedfold) for canonical chain IDs
    ref_method = None
    ref_lig_info = None
    for m in method_lig_info:
        if m != "seedfold":
            ref_method = m
            ref_lig_info = method_lig_info[m]
            break
    if ref_lig_info is None and method_lig_info:
        ref_method = next(iter(method_lig_info))
        ref_lig_info = method_lig_info[ref_method]

    if not ref_lig_info:
        log.warning(f"Could not detect ligand chains for {target}")
        return None

    # Main ligand chains = chains with atom count equal to max (the largest ligands)
    max_atoms = max(n for _, n, _ in ref_lig_info)
    main_chains = sorted([ch for ch, n, _ in ref_lig_info if n == max_atoms])
    is_multichain = len(main_chains) >= 2
    log.info(f"{target}: detected {len(ref_lig_info)} ligand chains, "
             f"main chains: {main_chains} ({max_atoms} atoms each)")

    # Build cross-method chain mapping: ref_chain -> method_chain (by atom count + order)
    method_chain_map = {}  # method -> {ref_chain: method_chain}
    for method_name, info in method_lig_info.items():
        if method_name == ref_method:
            method_chain_map[method_name] = {ch: ch for ch in main_chains}
            continue
        method_main = sorted([ch for ch, n, _ in info if n == max_atoms])
        mapping = dict(zip(main_chains, method_main))
        method_chain_map[method_name] = mapping
        if len(method_main) != len(main_chains):
            log.warning(f"  {method_name}: found {len(method_main)} main chains "
                        f"(expected {len(main_chains)}), mapping may be incomplete")

    # ── Step 2: Derive per-chain SMILES ───────────────────────────────────
    # Match each main chain to its SMILES using element formula from the CIF.
    # This correctly handles cases like H1135 where multiple different ligands
    # (Cl- and K+) have the same heavy atom count.
    chain_smiles = {}
    if smiles:
        smiles_norm = smiles.replace(':', '.')
        smi_parts = [s.strip() for s in smiles_norm.split('.') if s.strip()]

        # Build formula -> SMILES list mapping from input SMILES
        formula_to_smi = defaultdict(list)
        for s in smi_parts:
            mol = Chem.MolFromSmiles(s)
            if mol:
                from collections import Counter as _Counter
                elem_counts = _Counter()
                for atom in mol.GetAtoms():
                    elem_counts[atom.GetSymbol()] += 1
                formula = "".join(f"{e}{c if c > 1 else ''}" for e, c in sorted(elem_counts.items()))
                formula_to_smi[formula].append(s)

        # Build chain -> formula mapping from ref_lig_info (which has formula from CIF)
        chain_formula = {ch: f for ch, n, f in ref_lig_info if ch in main_chains}

        for ch in main_chains:
            f = chain_formula.get(ch, "")
            matched = formula_to_smi.get(f, [])
            if matched:
                chain_smiles[ch] = matched[0]
            else:
                # Fallback: match by HA count (original behavior)
                ha_to_smi = defaultdict(list)
                for s in smi_parts:
                    mol = Chem.MolFromSmiles(s)
                    if mol:
                        ha_to_smi[mol.GetNumHeavyAtoms()].append(s)
                ha_matched = ha_to_smi.get(max_atoms, [])
                chain_smiles[ch] = ha_matched[0] if ha_matched else smiles
    else:
        for ch in main_chains:
            chain_smiles[ch] = None

    # ── Step 3: Ligand-centric extraction (unified single/multi) ──────────
    predictions_by_chain = {}

    for lig_ch in main_chains:
        chain_cif_out = os.path.join(cif_out, f"lig_{lig_ch}") if is_multichain else cif_out
        os.makedirs(chain_cif_out, exist_ok=True)
        chain_preds = []
        ch_smiles = chain_smiles.get(lig_ch, smiles)

        # Collect tasks across all methods for this ligand chain
        tasks = []
        for method_name in method_cif_paths:
            actual_ch = method_chain_map.get(method_name, {}).get(lig_ch)
            if actual_ch is None:
                continue
            cif_paths = method_cif_paths[method_name]
            canon_to_idx = method_canon_idx[method_name]

            for cif_path in cif_paths:
                orig_idx = canon_to_idx.get(os.path.realpath(cif_path))
                if orig_idx is None:
                    continue
                name_suffix = f"_lig{lig_ch}" if is_multichain else ""
                name = f"{target}_{method_name}_model_{orig_idx}{name_suffix}"
                tasks.append((method_name, orig_idx, cif_path, chain_cif_out,
                              name, ch_smiles, actual_ch, err_log_path,
                              method_cof_chains.get(method_name, [])))

        # Execute cif_to_pdb_sdf in parallel (ProcessPoolExecutor with spawn ctx
        # for PyMOL state isolation — PyMOL `cmd` is a singleton, NOT thread-safe).
        if intra_workers > 1 and len(tasks) > 1:
            import concurrent.futures
            n_w = min(intra_workers, len(tasks))
            ctx = multiprocessing.get_context("spawn")
            log.info(f"{target} chain {lig_ch}: dispatching {len(tasks)} cif→PDB+SDF "
                     f"tasks to {n_w} parallel workers")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_w,
                mp_context=ctx,
                initializer=_cif_worker_init,
            ) as executor:
                results = list(executor.map(_cif_to_pdb_sdf_task, tasks))
        else:
            results = [_cif_to_pdb_sdf_task(t) for t in tasks]

        for method_name_r, orig_idx, pdb, sdf in results:
            if pdb and sdf:
                chain_preds.append((f"{method_name_r}_model{orig_idx}", pdb, sdf))

        # Cap at 200 models per chain when multi-chain (preserve single-chain behavior)
        if is_multichain and len(chain_preds) > 200:
            MAX_MODELS_PER_CHAIN = 200
            DEFAULT_PER_METHOD = 50
            by_method = defaultdict(list)
            for pred in chain_preds:
                method_prefix = pred[0].split("_model")[0]
                by_method[method_prefix].append(pred)
            capped = []
            non_af3_count = 0
            for method, preds in sorted(by_method.items()):
                if method != "af3":
                    selected = preds[:DEFAULT_PER_METHOD]
                    capped.extend(selected)
                    non_af3_count += len(selected)
            af3_preds = by_method.get("af3", [])
            af3_limit = min(100, MAX_MODELS_PER_CHAIN - non_af3_count)
            capped.extend(af3_preds[:max(af3_limit, DEFAULT_PER_METHOD)])
            chain_preds = capped[:MAX_MODELS_PER_CHAIN]
            log.info(f"  {target} chain {lig_ch}: capped to {len(chain_preds)} models")

        predictions_by_chain[lig_ch] = chain_preds

    total_preds = sum(len(v) for v in predictions_by_chain.values())
    if total_preds == 0:
        log.warning(f"No valid predictions across any method for {target}")
        return None
    log.info(f"{target}: converted {total_preds} CIF files across {len(main_chains)} chain(s)")

    # ── Step 1.5: PB-aware early-stop relax ───────────────────────────────
    # Per-pose PB check; PB-fail → OpenMM early-stop relax (step=30, k=100,
    # max=300 + retry 500). Removes clash that contaminates SuCOS landscape
    # downstream. Boltz-2 usually passes ORIG; AF3/Protenix on RNA fail and
    # need relax. SDFs updated in place; persistent invalids get _pb=False
    # at ranking_sucos. See: casp17_ligand/utils/relax_pose_batch.py
    #
    # Cross-env subprocess: OpenMM/openff/pdbfixer live in `PoseBench` env
    # (mamba solve into casp17_ligand env takes 10+h on shared NFS). The
    # batch runner is module-callable from that env via `-m`.
    # Disable with env var `DISABLE_EARLYSTOP_RELAX=1`.
    if not os.environ.get("DISABLE_EARLYSTOP_RELAX") and total_preds > 0:
        import json as _json
        import subprocess as _subprocess
        import tempfile as _tempfile
        pose_list = []
        n_cof_rec = 0
        for lig_ch, preds in predictions_by_chain.items():
            for _name, _pdb, _sdf in preds:
                # Third element (when present) is the receptor PoseBusters judges
                # against; OpenMM keeps using the cofactor-free _pdb.
                _cof_rec = receptor_with_cofactor_path(_pdb)
                if os.path.exists(_cof_rec):
                    pose_list.append([_pdb, _sdf, _cof_rec])
                    n_cof_rec += 1
                else:
                    pose_list.append([_pdb, _sdf])
        if cofactor_names:
            log.info(f"  {target}: {n_cof_rec}/{len(pose_list)} poses will be PB-checked "
                     f"against a receptor carrying {cofactor_names}")
            if n_cof_rec == 0:
                log.warning(f"  {target}: cofactor {cofactor_names} requested but no "
                            f"_receptor_cofactor.pdb was written — PB will not see it")
        manifest = {"target": target, "pose_list": pose_list}
        with _tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{target}_relax_manifest.json",
                delete=False) as _mf:
            _json.dump(manifest, _mf)
            manifest_path = _mf.name
        summary_path = manifest_path.replace("_manifest.json", "_summary.json")
        # PROJECT_ROOT so PoseBench env can import casp17_ligand.utils.*
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))
        sub_env = os.environ.copy()
        sub_env["PYTHONPATH"] = project_root + ":" + sub_env.get("PYTHONPATH", "")
        # WARNING: ensemble may run multiple targets in parallel (num_workers
        # in config); each target's Stage 1.5 launches N_RELAX OpenMM workers.
        # Total CPU load = num_workers_ensemble × N_RELAX × ~2 threads each.
        # Default 4 keeps total ≤ 4×4×2 = ~32 CPU threads (safe on 64-core box).
        # Override via RELAX_WORKERS env var when running single-target.
        n_relax_workers = max(1, int(os.environ.get("RELAX_WORKERS", "4")))
        # MUST default to PoseBench, not FlowDock: AM1-BCC charging needs the
        # AmberTools toolkit (`sqm`), which only PoseBench has. In FlowDock the
        # openff registry is ['The RDKit', 'Built-in Toolkit'], so every pose
        # dies with `No registered toolkits can provide the capability
        # "assign_partial_charges"`. The relax subprocess then exits 1 and the
        # pipeline CONTINUES — clustering silently proceeds on unrelaxed,
        # clash-contaminated poses (which corrupts the
        # SuCOS landscape). T2413v1 lost a full run to this on 2026-07-30.
        relax_env = os.environ.get("RELAX_ENV", "PoseBench")
        cmd = [
            "conda", "run", "--no-capture-output", "-n", relax_env,
            "python", "-m", "casp17_ligand.utils.relax_pose_batch",
            "--manifest", manifest_path,
            "--summary-out", summary_path,
            "--workers", str(n_relax_workers),
        ]
        # Iteration budget. Defaults match relax_pose_batch's own (300, retry
        # 500) so behaviour is unchanged unless asked. Lower them when a target
        # has poses that grind for hours: we measured that of 178
        # poses ever rescued by relax, none needed more than ~210 iterations,
        # and the 450-570 band had a 0/177 success rate. Past that point the
        # minimizer has converged and PoseBusters still says no -- almost always
        # because the failure is ligand-INTERNAL geometry (22/28 in that audit),
        # which no amount of further minimization can fix. Capping just makes
        # the hopeless ones give up early; they land as `_pb=False` and are
        # ranked last, which is the documented behaviour (§4).
        relax_max_iter = os.environ.get("RELAX_MAX_ITER")
        relax_retry_max_iter = os.environ.get("RELAX_RETRY_MAX_ITER")
        if relax_max_iter:
            cmd += ["--max-iter", str(int(relax_max_iter))]
        if relax_retry_max_iter:
            cmd += ["--retry-max-iter", str(int(relax_retry_max_iter))]
        # Pocket-local relax (opt-in via POCKET_CUTOFF Å env): minimize only the
        # residues near the ligand instead of the full receptor — essential for
        # huge multi-chain complexes (M-series) where full-receptor OpenMM
        # minimization is impractically slow. Off by default so R/T behavior is
        # unchanged; PB check still runs against the full receptor.
        pocket_cutoff = os.environ.get("POCKET_CUTOFF")
        if pocket_cutoff:
            cmd += ["--pocket-cutoff", str(pocket_cutoff)]
        log.info(f"{target}: launching PB-aware relax via env={relax_env} "
                 f"({len(pose_list)} poses, {n_relax_workers} workers)")
        # A failed relax is NOT a recoverable warning. Clustering downstream
        # would run on clash-contaminated poses and still emit a confident,
        # plausible-looking top-5. That is indistinguishable
        # from a good run unless someone greps the log, so abort instead.
        # `DISABLE_EARLYSTOP_RELAX=1` is the way to deliberately skip relax;
        # `RELAX_FAILURE_OK=1` downgrades this back to a warning if a target
        # genuinely cannot be relaxed and you accept the degraded ranking.
        relax_error = None
        try:
            _subprocess.run(cmd, check=True, env=sub_env)
            with open(summary_path) as f:
                stats = _json.load(f)
            log.info(f"{target}: relax stats = {stats}")
        except _subprocess.CalledProcessError as e:
            relax_error = (f"{target}: PB-aware relax subprocess failed "
                           f"(exit {e.returncode}, env={relax_env})")
            log.error(relax_error)
        except Exception as e:
            relax_error = f"{target}: PB-aware relax wrapper failed: {e}"
            log.error(relax_error, exc_info=True)
        finally:
            for p in (manifest_path, summary_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if relax_error and not os.environ.get("RELAX_FAILURE_OK"):
            raise RuntimeError(
                relax_error
                + "\nRefusing to cluster unrelaxed poses — the resulting top-5 "
                  "would look normal but be ranked on a clash-contaminated SuCOS "
                  "landscape.\nHints: RELAX_ENV must name an env with the "
                  "AmberTools toolkit (PoseBench, not FlowDock — check with "
                  "`conda run -n <env> which sqm`).\nSet RELAX_FAILURE_OK=1 to "
                  "proceed anyway, or DISABLE_EARLYSTOP_RELAX=1 to skip relax "
                  "deliberately."
            )

    rows = []

    def _ref_smiles_for_sdf(sdf_path: str) -> List[str]:
        """Resolve the per-chain reference SMILES list for an SDF, used by chem_validate.

        Single-chain target → `chain_smiles[only_chain]`. Multi-chain → parse
        `lig_{ch}/` segment from the SDF path. Returns split-by-'.' list (each
        item one ligand SMILES). Empty list disables chem validation.
        """
        if len(main_chains) == 1:
            smi = chain_smiles.get(main_chains[0])
        else:
            ch = None
            for p in os.path.normpath(sdf_path).split(os.sep):
                if p.startswith("lig_"):
                    ch = p[4:]
                    break
            smi = chain_smiles.get(ch) if ch else None
        if not smi:
            return []
        # Strip salt counterions to match the SDF, which cif_to_pdb_sdf built from
        # the salt-stripped fragment (else the fragment SDF fails chem_validate
        # against the full salt reference → spurious pb=False on salt targets).
        smi = _strip_salt_smiles(smi.replace(":", "."))
        return [s.strip() for s in smi.split(".") if s.strip()]

    # model name → {ligand_chain: sdf_path}. For multi-copy ligands (homodimer
    # with one ligand per protomer) the consensus ranker keeps only the FIRST
    # chain's pose per model, so the ranked SDF would carry a single molecule
    # and the submission would emit only `LIGAND 000`, silently dropping the
    # second binding site. This index lets _save_ranked re-assemble all copies.
    sdf_by_name_chain: Dict[str, Dict[str, str]] = {}
    for _ch in main_chains:
        for _nm, _pdb, _sdf in predictions_by_chain.get(_ch, []):
            sdf_by_name_chain.setdefault(_nm, {})[_ch] = _sdf

    def _save_ranked(ranked, score_key, score_fmt, method_suffix, out_subdir):
        if os.path.exists(out_subdir):
            shutil.rmtree(out_subdir)
        os.makedirs(out_subdir, exist_ok=True)

        candidates = []
        for orig_rank, (method, pdb, sdf, score) in ranked.items():
            if export_top_n is not None and orig_rank > export_top_n:
                candidates.append({
                    "orig_rank": orig_rank, "method": method, "pdb": pdb, "sdf": sdf, "score": score,
                    "pb_valid": False, "chem_valid": False, "tested": False
                })
            else:
                pb_valid = run_posebusters(pdb, sdf)
                chem_valid = run_chem_validate(sdf, _ref_smiles_for_sdf(sdf))
                candidates.append({
                    "orig_rank": orig_rank, "method": method, "pdb": pdb, "sdf": sdf, "score": score,
                    "pb_valid": pb_valid, "chem_valid": chem_valid, "tested": True
                })

        tested = [c for c in candidates if c["tested"]]
        untested = [c for c in candidates if not c["tested"]]
        # sort: (passed pb AND chem) first, then by original rank
        tested.sort(key=lambda x: (not (x["pb_valid"] and x["chem_valid"]), x["orig_rank"]))

        if tested and all(not (c["pb_valid"] and c["chem_valid"]) for c in tested):
            log_msg = f"[{target}] [{method_suffix}] All top-{len(tested)} models failed PB+chem checks."
            log.warning(log_msg)
            with open(os.path.join(ensemble_output_dir, "err.log"), "a") as f:
                f.write(log_msg + "\n")

        final_candidates = tested + untested

        for final_rank, cand in enumerate(final_candidates, start=1):
            method = cand["method"]
            pdb = cand["pdb"]
            sdf = cand["sdf"]
            score = cand["score"]
            pb_valid = cand["pb_valid"]
            chem_valid = cand["chem_valid"]
            ranked_base = f"{method}_rank{final_rank}_orig{cand['orig_rank']}_{score_fmt.format(score)}"
            # `_pb=` filename suffix carries the *combined* QC flag (PoseBusters
            # AND LG chemistry validation). Naming kept as `_pb=` so existing
            # downstream regex (evaluate_topn_clusters.py:219, generate_submission.py)
            # still parses; pick_consensus_pb_rep automatically prefers the
            # combined-True samples without modification. The CSV row above
            # still records pb_valid and chem_valid separately for debugging.
            qc_passed = pb_valid and chem_valid
            dst_sdf = os.path.join(out_subdir, f"{ranked_base}_pb={qc_passed}.sdf")
            if is_multichain:
                # Write ALL ligand copies as a multi-mol SDF so that
                # generate_submission.build_model_block emits one LIGAND block
                # per SMILES TSV row (LIGAND 000 + LIGAND 001 for a homodimer).
                per_ch = sdf_by_name_chain.get(method, {})
                mols = []
                for ch in main_chains:            # deterministic order (sorted)
                    p = per_ch.get(ch)
                    if not p:
                        continue
                    for m in Chem.SDMolSupplier(p, removeHs=False):
                        if m is not None:
                            mols.append(m)
                if len(mols) == len(main_chains):
                    w = Chem.SDWriter(dst_sdf)
                    for m in mols:
                        w.write(m)
                    w.close()
                else:
                    log.warning(f"{target}/{method}: got {len(mols)} ligand mols for "
                                f"{len(main_chains)} chains; writing single-mol SDF")
                    shutil.copy2(sdf, dst_sdf)
            else:
                shutil.copy2(sdf, dst_sdf)
            shutil.copy2(pdb, os.path.join(out_subdir, f"{ranked_base}.pdb"))
            rows.append({
                "target": target,
                "ranking_method": method_suffix,
                "rank": final_rank,
                "original_rank": cand["orig_rank"],
                "method": method,
                "score": score,
                "pb_valid": pb_valid,
                "chem_valid": chem_valid,
            })

    def _ranking_exists(subdir_name):
        d = os.path.join(target_out, subdir_name)
        return os.path.isdir(d) and any(f.endswith(".sdf") for f in os.listdir(d))

    # ── Step 4: Consensus ranking (always multilig, auto-degrades for single chain) ──
    _reset_hetero_ref()  # Reset per-target sequence alignment reference

    # Edge case: single-atom ligands (ions like Cl-, K+) have no shape/pharmacophore
    # features → SuCOS is always ~0. Fall back to RMSD consensus for ranking_sucos.
    is_single_atom_ligand = max_atoms <= 1

    rmsd_cache_path = os.path.join(target_out, "pairwise_rmsd_cache.json")
    ranked_rmsd = rmsd_consensus_rank_multilig(
        predictions_by_chain, aligned_dir=None,
        cache_path=rmsd_cache_path, chain_smiles=chain_smiles)

    if is_single_atom_ligand:
        log.info(f"{target}: single-atom ligand (max_atoms={max_atoms}), "
                 f"using RMSD consensus as SuCOS fallback")
        ranked_sucos = ranked_rmsd
    else:
        sucos_cache_path = os.path.join(target_out, "pairwise_sucos_cache.json")
        ranked_sucos = sucos_consensus_rank_multilig(
            predictions_by_chain, aligned_dir=None,
            cache_path=sucos_cache_path, chain_smiles=chain_smiles)

    rmsd_6a_cache_path = os.path.join(target_out, "pairwise_rmsd_6a_cache.json")
    ranked_rmsd_6a = rmsd_6a_consensus_rank_multilig(
        predictions_by_chain, cache_path=rmsd_6a_cache_path,
        chain_smiles=chain_smiles)

    if ranked_rmsd and not _ranking_exists("ranking_rmsd"):
        _save_ranked(ranked_rmsd, "avg_rmsd", "rmsd{:.2e}", "rmsd", os.path.join(target_out, "ranking_rmsd"))
    if ranked_sucos and not _ranking_exists("ranking_sucos"):
        _save_ranked(ranked_sucos, "avg_sucos", "sucos{:.3f}", "sucos", os.path.join(target_out, "ranking_sucos"))
    if ranked_rmsd_6a and not _ranking_exists("ranking_rmsd_6a"):
        _save_ranked(ranked_rmsd_6a, "avg_rmsd_6a", "rmsd6a{:.2e}", "rmsd_6a", os.path.join(target_out, "ranking_rmsd_6a"))

    # Free per-target cif_converted intermediates once ranking outputs exist
    # (ranking_sucos/_rmsd hold independent shutil.copy2 copies, so this is safe;
    # _target_is_complete checks ranking_* only). Large screens (L-series: 1856
    # targets × ~27 MB cif_converted) otherwise fill the shared disk. Opt-in via
    # env var so single-target R/T/M runs keep the cache for cheap re-runs.
    if os.environ.get("CLEAN_CIF_CONVERTED") and _ranking_exists("ranking_sucos"):
        shutil.rmtree(cif_out, ignore_errors=True)

    return pd.DataFrame(rows) if rows else None


def _target_is_complete(ensemble_output_dir: str, target: str) -> bool:
    """Check if a target already has completed ranking outputs (RMSD + SuCOS).

    rmsd_6a is NOT required because it relies on Cα backbone atoms — RNA
    targets have no Cα and ranking_rmsd_6a is always empty for the R-series.
    Requiring it would mark all RNA targets perpetually "incomplete" and
    trigger Stage 1.5 re-relax on every ensemble run.
    """
    target_out = os.path.join(ensemble_output_dir, "targets", target)
    for subdir in ["ranking_rmsd", "ranking_sucos"]:
        d = os.path.join(target_out, subdir)
        if not os.path.isdir(d):
            return False
        if not any(f.endswith(".sdf") for f in os.listdir(d)):
            return False
    return True


def _worker_init():
    """Initialize PyMOL in each worker process."""
    import pymol
    pymol.finish_launching(["pymol", "-qc"])


def _cif_worker_init():
    """Initialize PyMOL in each ProcessPoolExecutor worker (spawn context).

    Used by the intra-target cif_to_pdb_sdf parallelism. Each worker gets
    a fresh Python interpreter (spawn context) → fresh PyMOL singleton.
    """
    import pymol
    pymol.finish_launching(["pymol", "-qc"])


def _cif_to_pdb_sdf_task(args):
    """Picklable wrapper around cif_to_pdb_sdf for ProcessPoolExecutor.

    Returns (method_name, orig_idx, pdb_path, sdf_path) — pdb/sdf are None
    on failure. Errors are swallowed and logged in-worker via stderr; the
    parent process treats None outputs as failed conversions.
    """
    cofactor_chains = []
    if len(args) == 9:
        (method_name, orig_idx, cif_path, out_dir, name, smiles, lig_chain,
         err_log_path, cofactor_chains) = args
    else:
        method_name, orig_idx, cif_path, out_dir, name, smiles, lig_chain, err_log_path = args
    try:
        pdb, sdf = cif_to_pdb_sdf(
            cif_path, out_dir, name, smiles=smiles,
            lig_chain=lig_chain, err_log_path=err_log_path,
            cofactor_chains=cofactor_chains,
        )
        return method_name, orig_idx, pdb, sdf
    except Exception as e:
        import sys
        print(f"[worker] cif_to_pdb_sdf failed for {name}: {e}", file=sys.stderr)
        return method_name, orig_idx, None, None


def _process_target_worker(args):
    """Worker function for multiprocessing. Wraps process_target with picklable args."""
    n_protein_chains = 0
    cofactor = None
    if len(args) == 10:
        (target, methods_cfg_dict, ensemble_output_dir, smiles,
         export_top_n, prefilter_metric, prefilter_topn, intra_workers,
         n_protein_chains, cofactor) = args
    elif len(args) == 9:
        (target, methods_cfg_dict, ensemble_output_dir, smiles,
         export_top_n, prefilter_metric, prefilter_topn, intra_workers,
         n_protein_chains) = args
    elif len(args) == 8:
        # Legacy 8-tuple (before multimer support)
        (target, methods_cfg_dict, ensemble_output_dir, smiles,
         export_top_n, prefilter_metric, prefilter_topn, intra_workers) = args
    else:
        # Legacy 7-tuple (multi-target mode without intra parallelism)
        target, methods_cfg_dict, ensemble_output_dir, smiles, export_top_n, prefilter_metric, prefilter_topn = args
        intra_workers = 1
    from omegaconf import OmegaConf
    methods_cfg = OmegaConf.create(methods_cfg_dict)
    try:
        result = process_target(
            target=target,
            methods_cfg=methods_cfg,
            ensemble_output_dir=ensemble_output_dir,
            smiles=smiles,
            export_top_n=export_top_n,
            prefilter_metric=prefilter_metric,
            prefilter_topn=prefilter_topn,
            intra_workers=intra_workers,
            n_protein_chains=n_protein_chains,
            cofactor=cofactor,
        )
        return target, result
    except Exception as e:
        logging.getLogger(__name__).error(f"Worker failed for {target}: {e}")
        return target, None


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="ensemble_generation",
)
def main(cfg: DictConfig) -> None:
    """Run ensemble generation for all targets in a dataset."""
    from omegaconf import OmegaConf

    os.makedirs(cfg.ensemble_output_dir, exist_ok=True)
    df = pd.read_csv(cfg.input_csv)
    targets = df["target"].tolist()
    smiles_map = dict(zip(df["target"], df["ligand_smiles"]))
    # Optional per-target cofactor column (CCD codes, e.g. "ZN" / "SFG"). Targets
    # whose structure ships with a cofactor need it excluded from ligand selection
    # and included in the PoseBusters receptor; see _detect_cofactor_chains.
    if "cofactor" in df.columns:
        cofactor_map = {t: (c if isinstance(c, str) and c.strip() else None)
                        for t, c in zip(df["target"], df["cofactor"])}
        n_cof = sum(1 for v in cofactor_map.values() if v)
        log.info(f"Cofactor column present: {n_cof}/{len(targets)} target(s) carry one")
    else:
        cofactor_map = {}

    # Skip already-completed targets
    pending_targets = []
    skipped = 0
    for target in targets:
        if _target_is_complete(cfg.ensemble_output_dir, target):
            skipped += 1
        else:
            pending_targets.append(target)
    log.info(f"Dataset {cfg.dataset}: {len(targets)} total, {skipped} skipped (complete), {len(pending_targets)} pending")

    if not pending_targets:
        log.info("All targets already complete, nothing to do")
        return

    # Prepare picklable args for workers
    methods_cfg_dict = OmegaConf.to_container(cfg.get("methods", {}), resolve=True)
    export_top_n = cfg.get("export_top_n", None)
    prefilter_metric = cfg.get("prefilter_metric", None)
    prefilter_topn = cfg.get("prefilter_topn", None)
    if prefilter_metric:
        log.info(f"Pre-filtering enabled: {prefilter_metric} top-{prefilter_topn}")
    num_workers = cfg.get("num_workers", 16)

    # Parallelism strategy:
    #   - Multi-target (CASP16): outer Pool, 1 outer worker per target, serial inside (intra_workers=1)
    #   - Single-target (CASP17): bypass outer Pool, run process_target in main process
    #     with intra-target ProcessPoolExecutor (intra_workers=num_workers).
    #     Bypass needed because multiprocessing.Pool workers are daemonic by
    #     default → cannot spawn child ProcessPoolExecutor processes inside.
    single_target_mode = len(pending_targets) == 1 and num_workers > 1
    intra_workers = num_workers if single_target_mode else 1

    # Number of protein chains in the target (multimers only; default 0 → 1).
    # Needed so pair_iptm measures protein–ligand and not the protein–protein
    # interface — see extract_pair_chains_iptm.
    n_protein_chains = cfg.get("n_protein_chains", 0)
    if n_protein_chains > 1:
        log.info(f"Multimer mode: n_protein_chains={n_protein_chains} "
                 f"(pair_iptm fallback treats chains 0..{n_protein_chains - 1} as protein)")

    worker_args = [
        (target, methods_cfg_dict, cfg.ensemble_output_dir, smiles_map.get(target),
         export_top_n, prefilter_metric, prefilter_topn, intra_workers, n_protein_chains,
         cofactor_map.get(target))
        for target in pending_targets
    ]

    all_results = []

    if num_workers <= 1:
        # Serial mode for debugging
        pymol.finish_launching(["pymol", "-qc"])
        for args in worker_args:
            target, result = _process_target_worker(args)
            if result is not None:
                all_results.append(result)
    elif single_target_mode:
        # Single-target with intra-target parallelism: bypass outer Pool so
        # ProcessPoolExecutor can spawn workers without the daemonic restriction.
        log.info(f"Single target: bypassing outer Pool, using {num_workers}-way "
                 f"intra-target parallelism inside process_target")
        pymol.finish_launching(["pymol", "-qc"])
        for args in worker_args:
            target, result = _process_target_worker(args)
            if result is not None:
                all_results.append(result)
                log.info(f"Completed {target} ({len(all_results)}/{len(pending_targets)})")
    else:
        # Multi-target outer Pool (each outer worker handles 1 target serially)
        log.info(f"Launching {num_workers} parallel workers (multi-target mode)")
        with multiprocessing.Pool(processes=num_workers, initializer=_worker_init) as pool:
            for target, result in pool.imap_unordered(_process_target_worker, worker_args):
                if result is not None:
                    all_results.append(result)
                    log.info(f"Completed {target} ({len(all_results)}/{len(pending_targets)})")

    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(cfg.ensemble_output_dir, "ranking_summary.csv")
        summary.to_csv(summary_path, index=False)
        log.info(f"Saved ranking summary → {summary_path}")
        for method in ["rmsd", "sucos", "rmsd_6a", "rmsd_8a", "rmsd_10a"]:
            sub = summary[summary["ranking_method"] == method]
            rank1 = sub[sub["rank"] == 1]
            if len(rank1):
                log.info(
                    f"Pass rate [{method}] rank-1: PB {rank1['pb_valid'].sum()}/{len(rank1)} | "
                    f"chem {rank1['chem_valid'].sum()}/{len(rank1)}"
                )


if __name__ == "__main__":
    main()
