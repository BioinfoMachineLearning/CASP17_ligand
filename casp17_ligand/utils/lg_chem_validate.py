"""LG submission chemistry validators (atom / bond / MCS topology).

Functions here are 1:1 ports of `casp17/scripts/LG_validation.py`'s
`compare_atoms`, `compare_bonds`, `get_bonds_dic`, and
`maximum_common_substructure`. The only intentional change is removing
the validator's stdout `print("# ERROR! ...")` calls — failures are returned
as a list of reasons instead, so this can run silently inside the ensemble
pipeline alongside PoseBusters.

When updating: keep these functions in lockstep with the official validator.
Don't refactor for style. If the upstream validator changes, mirror the
change here verbatim.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS

log = logging.getLogger(__name__)


def get_bonds_dic(mol: Chem.Mol) -> Counter:
    """Mirror of LG_validation.get_bonds_dic — bond-type multiset."""
    all_bonds = []
    for i in mol.GetBonds():
        bond_type = sorted([i.GetBeginAtom().GetSymbol(), i.GetEndAtom().GetSymbol()])
        bond_type = bond_type[0] + bond_type[1] + str(i.GetBondType())
        all_bonds.append(bond_type)
    return Counter(all_bonds)


def compare_atoms(ref_mol: Chem.Mol, query_mol: Chem.Mol) -> bool:
    """Mirror of LG_validation.compare_atoms (sans the print/lig_id arg)."""
    gt_atoms = Counter([i.GetSymbol() for i in ref_mol.GetAtoms()])
    mol_atoms = Counter([i.GetSymbol() for i in query_mol.GetAtoms()])
    if sorted(gt_atoms.items(), key=lambda x: x[0]) != sorted(
        mol_atoms.items(), key=lambda x: x[0]
    ):
        return False
    return True


def compare_bonds(ref_mol: Chem.Mol, query_mol: Chem.Mol) -> bool:
    """Mirror of LG_validation.compare_bonds (sans the print/lig_id arg)."""
    gt_bonds = get_bonds_dic(ref_mol)
    mol_bonds = get_bonds_dic(query_mol)
    if sorted(gt_bonds.items(), key=lambda x: x[0]) != sorted(
        mol_bonds.items(), key=lambda x: x[0]
    ):
        return False
    return True


def maximum_common_substructure(ref_mol: Chem.Mol, query_mol: Chem.Mol) -> bool:
    """Mirror of LG_validation.maximum_common_substructure (sans print/lig_id).

    Note the call order is preserved: FindMCS first, then the n==1 short-circuit,
    matching the upstream exactly.
    """
    res = rdFMCS.FindMCS([query_mol, ref_mol])

    if ref_mol.GetNumAtoms() == 1 and query_mol.GetNumAtoms() == 1:
        return True

    if (
        res.numAtoms == ref_mol.GetNumAtoms()
        and res.numAtoms == query_mol.GetNumAtoms()
    ):
        return True
    else:
        return False


def validate_mol_against_smiles(
    query_mol: Chem.Mol, ref_smiles: str
) -> Tuple[bool, List[str]]:
    """Validate one molecule against a reference SMILES (3 checks, same as
    LG_validation.validate_ligands inner loop).

    Returns (passed, [reasons-failed]). Reasons are short tags from
    {"atoms", "bonds", "topology"}.
    """
    ref = Chem.MolFromSmiles(ref_smiles)
    if ref is None:
        return False, ["ref_smiles_unparseable"]
    ref = Chem.RemoveAllHs(ref)
    qry = Chem.RemoveAllHs(query_mol)

    fails = []
    if not compare_atoms(ref, qry):
        fails.append("atoms")
    if not compare_bonds(ref, qry):
        fails.append("bonds")
    if not maximum_common_substructure(ref, qry):
        fails.append("topology")
    return (len(fails) == 0), fails


def validate_sdf_against_smiles_list(
    sdf_path: str, ref_smiles_list: List[str]
) -> Tuple[bool, List[str]]:
    """Validate every mol in an SDF against a parallel list of reference SMILES.

    The SDF is expected to contain `len(ref_smiles_list)` molecules in the
    same order as the reference list (matches `cif_to_pdb_sdf` output for
    multi-ligand targets, which is a single multi-mol SDF).

    Reader uses `SDMolSupplier` defaults (removeHs=True, sanitize=True) to
    match the upstream validator, which uses `ForwardSDMolSupplier` defaults
    (also removeHs=True, sanitize=True).

    Returns (all_passed, reasons_or_messages). On structural failures the
    reason list is "<idx>:<tag>" entries.
    """
    if not ref_smiles_list:
        return True, []

    suppl = Chem.SDMolSupplier(sdf_path)  # defaults: removeHs=True, sanitize=True
    mols = [m for m in suppl if m is not None]
    if len(mols) != len(ref_smiles_list):
        return False, [f"mol_count_mismatch:{len(mols)}vs{len(ref_smiles_list)}"]

    all_fails: List[str] = []
    for i, (mol, smi) in enumerate(zip(mols, ref_smiles_list)):
        ok, fails = validate_mol_against_smiles(mol, smi)
        if not ok:
            all_fails.extend(f"{i}:{f}" for f in fails)
    return (len(all_fails) == 0), all_fails


__all__ = [
    "get_bonds_dic",
    "compare_atoms",
    "compare_bonds",
    "maximum_common_substructure",
    "validate_mol_against_smiles",
    "validate_sdf_against_smiles_list",
]
