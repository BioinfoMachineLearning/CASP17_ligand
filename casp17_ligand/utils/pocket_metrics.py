"""Pocket alignment + SuCOS scoring utilities.

Adapted from runs-n-poses/batch_pocket_comparison.py.
Core: align mobile protein to reference using binding pocket residues
(within 6.0 Å of reference ligand), then compute SuCOS similarity.

Metrics:
  P_SuCOS  — SuCOS after pocket-based protein alignment (binding pose quality)
  P_Cov    — Pocket coverage IoU (do both models agree on where the pocket is?)
  L_SuCOS  — SuCOS after ligand O3A self-alignment (ligand conformation quality)
"""

import os
import warnings
import numpy as np
from Bio.PDB import PDBParser, Superimposer, NeighborSearch, PDBIO
from Bio import BiopythonWarning, pairwise2
warnings.simplefilter("ignore", BiopythonWarning)

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdShapeHelpers, rdMolAlign
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Geometry import Point3D


def load_ligand_mol(path: str):
    """Load RDKit Mol from SDF or PDB file (sanitized, with 3D conformer)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".sdf":
            mol = Chem.SDMolSupplier(path, sanitize=True)[0]
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(path, sanitize=True, removeHs=False)
        else:
            return None
        return mol if mol is not None and mol.GetNumConformers() > 0 else None
    except Exception:
        return None


def read_ligand_positions(path: str):
    """Read heavy-atom 3D positions from SDF or PDB ligand file.

    Returns numpy array (N, 3) or None on failure.
    """
    mol = load_ligand_mol(path)
    if mol is None:
        return None
    try:
        return mol.GetConformer().GetPositions()
    except Exception:
        return None

# SuCOS constants (module-level, built once)
_FDEF = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
_FEAT_MAP_PARAMS = {k: FeatMaps.FeatMapParams() for k in _FDEF.GetFeatureFamilies()}
_PHARM_FAMILIES = (
    "Donor", "Acceptor", "NegIonizable", "PosIonizable",
    "ZnBinder", "Aromatic", "Hydrophobe", "LumpedHydrophobe",
)


def _get_pocket_residues(structure, ligand_sdf: str, cutoff: float = 6.0):
    """Return set of protein residues within cutoff Å of ligand atoms."""
    try:
        mol = Chem.SDMolSupplier(ligand_sdf, sanitize=True)[0]
        if mol is None:
            return set()
        conf = mol.GetConformer()
        lig_coords = [(conf.GetAtomPosition(i).x,
                       conf.GetAtomPosition(i).y,
                       conf.GetAtomPosition(i).z)
                      for i in range(mol.GetNumAtoms())]
    except Exception:
        return set()
    return _pocket_from_coords(structure, lig_coords, cutoff)


def _pocket_from_coords(structure, lig_coords, cutoff: float = 6.0):
    """Return set of standard protein residues within cutoff Å of coordinates."""
    ns = NeighborSearch(list(structure[0].get_atoms()))
    pocket = set()
    for lx, ly, lz in lig_coords:
        for r in ns.search(np.array([lx, ly, lz]), cutoff, level="R"):
            if r.id[0] == " ":
                pocket.add(r)
    return pocket


def _pocket_from_mol(structure, mol, cutoff: float = 6.0):
    """Return set of standard protein residues within cutoff Å of mol's conformer atoms."""
    try:
        conf = mol.GetConformer()
        lig_coords = [(conf.GetAtomPosition(i).x,
                       conf.GetAtomPosition(i).y,
                       conf.GetAtomPosition(i).z)
                      for i in range(mol.GetNumAtoms())]
    except Exception:
        return set()
    return _pocket_from_coords(structure, lig_coords, cutoff)


def _get_ca_sequence(structure):
    """Return (seq_str, [CA_atoms], [residues]) for standard residues."""
    seq, cas, res_list = "", [], []
    for chain in structure[0]:
        for res in chain:
            if res.id[0] == " " and "CA" in res:
                try:
                    seq += three_to_one(res.resname)
                    cas.append(res["CA"])
                    res_list.append(res)
                except Exception:
                    pass
    return seq, cas, res_list


def _map_residues(ref_struct, mob_struct):
    """Map ref residues → mob residues via global sequence alignment."""
    ref_seq, _, ref_res = _get_ca_sequence(ref_struct)
    mob_seq, _, mob_res = _get_ca_sequence(mob_struct)
    if not ref_seq or not mob_seq:
        return {}
    alns = pairwise2.align.globalxx(ref_seq, mob_seq)
    if not alns:
        return {}
    aln_ref, aln_mob = alns[0][0], alns[0][1]
    ref_to_mob, ri, mi = {}, 0, 0
    for rc, mc in zip(aln_ref, aln_mob):
        if rc != "-" and mc != "-":
            ref_to_mob[ref_res[ri]] = mob_res[mi]
        if rc != "-":
            ri += 1
        if mc != "-":
            mi += 1
    return ref_to_mob


def _feature_map_score(mol1, mol2) -> float:
    feat_lists = []
    for mol in [mol1, mol2]:
        raw = _FDEF.GetFeaturesForMol(mol)
        feat_lists.append([f for f in raw if f.GetFamily() in _PHARM_FAMILIES])
    if not feat_lists[0] or not feat_lists[1]:
        return 0.0
    fmaps = [
        FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=_FEAT_MAP_PARAMS)
        for x in feat_lists
    ]
    fmaps[0].scoreMode = FeatMaps.FeatMapScoreMode.All
    score = fmaps[0].ScoreFeats(feat_lists[1])
    return float(np.clip(score / min(fmaps[0].GetNumFeatures(), len(feat_lists[1])), 0, 1))


def get_sucos_score(mol1, mol2):
    """Compute SuCOS score between two RDKit mols with 3D conformers.

    Returns (sucos, feature_map_score, shape_score).
    """
    try:
        fm = _feature_map_score(mol1, mol2)
        protrude = rdShapeHelpers.ShapeProtrudeDist(mol1, mol2, allowReordering=False)
        shape = float(np.clip(1 - protrude, 0, 1))
        return 0.5 * fm + 0.5 * shape, fm, shape
    except Exception:
        return 0.0, 0.0, 0.0


def pocket_coverage_iou(
    ref_struct, ref_sdf: str,
    mob_struct, mob_sdf: str,
    ref_to_mob: dict,
    cutoff: float = 6.0,
) -> float:
    """Compute pocket coverage as IoU (Intersection over Union).

    Both ref and mob define their own pocket (residues within cutoff of their
    respective ligand). IoU = |intersection| / |union|, where intersection is
    counted via the sequence alignment mapping.
    """
    ref_pocket = _get_pocket_residues(ref_struct, ref_sdf, cutoff)
    mob_pocket = _get_pocket_residues(mob_struct, mob_sdf, cutoff)
    if not ref_pocket and not mob_pocket:
        return 0.0
    hits = 0
    for r_ref in ref_pocket:
        if r_ref in ref_to_mob and ref_to_mob[r_ref] in mob_pocket:
            hits += 1
    union = len(ref_pocket) + len(mob_pocket) - hits
    return hits / union if union > 0 else 0.0


def ligand_sucos(mol_ref, mol_mob):
    """Compute L_SuCOS: SuCOS after optimal ligand self-alignment via O3A.

    This measures ligand conformational similarity independent of protein
    alignment. High L_SuCOS + low P_SuCOS → protein pocket is wrong but
    ligand conformation is OK.

    Returns L_SuCOS score (float).
    """
    if mol_ref is None or mol_mob is None:
        return 0.0
    try:
        probe = Chem.Mol(mol_mob)
        o3a = rdMolAlign.GetO3A(probe, mol_ref)
        o3a.Align()
        sucos, _, _ = get_sucos_score(mol_ref, probe)
        return sucos
    except Exception:
        pass
    # Fallback: RMSD alignment if same atom count
    try:
        if mol_ref.GetNumAtoms() == mol_mob.GetNumAtoms():
            probe = Chem.Mol(mol_mob)
            rdMolAlign.AlignMol(probe, mol_ref)
            sucos, _, _ = get_sucos_score(mol_ref, probe)
            return sucos
    except Exception:
        pass
    return 0.0


def pocket_align_complex_to_reference(
    ref_pdb: str,
    ref_sdf: str,
    mob_pdb: str,
    mob_sdf: str,
    mob_lig_pos: np.ndarray,
    cutoff: float = 6.0,
    aligned_dir=None,
):
    """Align mobile protein to reference using binding pocket residues.

    Pocket = residues within `cutoff` Å of the reference ligand.

    :return: dict with keys:
        aligned_lig_pos, p_sucos, pocket_rmsd, pocket_iou, l_sucos
        Returns failure dict on error.
    """
    fail = {"aligned_lig_pos": None, "p_sucos": np.nan, "pocket_rmsd": np.nan,
            "pocket_iou": np.nan, "l_sucos": np.nan, "leakage": np.nan}

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", ref_pdb)
    mob_struct = parser.get_structure("mob", mob_pdb)

    pocket_res = _get_pocket_residues(ref_struct, ref_sdf, cutoff)
    if len(pocket_res) < 3:
        return fail

    ref_to_mob = _map_residues(ref_struct, mob_struct)
    fixed, moving = [], []
    for r_ref in pocket_res:
        if r_ref in ref_to_mob:
            r_mob = ref_to_mob[r_ref]
            if "CA" in r_ref and "CA" in r_mob:
                fixed.append(r_ref["CA"])
                moving.append(r_mob["CA"])

    if len(fixed) < 3:
        return fail

    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    pocket_rmsd = float(sup.rms)
    rot = np.array(sup.rotran[0])
    tran = np.array(sup.rotran[1])
    aligned_lig = mob_lig_pos @ rot + tran

    # Pocket coverage IoU
    p_iou = pocket_coverage_iou(
        ref_struct, ref_sdf, mob_struct, mob_sdf, ref_to_mob, cutoff
    )

    # P_SuCOS and L_SuCOS
    p_sucos, l_sucos = 0.0, 0.0
    try:
        mol_ref = load_ligand_mol(ref_sdf)
        mol_mob_orig = load_ligand_mol(mob_sdf)
        if mol_ref is not None and mol_mob_orig is not None:
            # P_SuCOS: apply pocket alignment transform to mob ligand, then SuCOS
            mol_mob = Chem.RWMol(mol_mob_orig)
            conf = mol_mob.GetConformer()
            for i in range(mol_mob.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                new_p = np.array([p.x, p.y, p.z]) @ rot + tran
                conf.SetAtomPosition(i, Point3D(*new_p))
            p_sucos, _, _ = get_sucos_score(mol_ref, mol_mob)

            # L_SuCOS: ligand O3A self-alignment (independent of protein)
            l_sucos = ligand_sucos(mol_ref, mol_mob_orig)

            if aligned_dir is not None:
                os.makedirs(aligned_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(mob_pdb))[0]
                sdf_base = os.path.splitext(os.path.basename(mob_sdf))[0]
                sup.apply(mob_struct.get_atoms())
                io = PDBIO()
                io.set_structure(mob_struct)
                io.save(os.path.join(aligned_dir, f"{base}_pocket_aligned.pdb"))
                writer = Chem.SDWriter(
                    os.path.join(aligned_dir, f"{sdf_base}_pocket_aligned.sdf")
                )
                writer.write(mol_mob)
                writer.close()
    except Exception:
        pass

    leakage = p_iou * l_sucos

    return {
        "aligned_lig_pos": aligned_lig,
        "p_sucos": p_sucos,
        "pocket_rmsd": pocket_rmsd,
        "pocket_iou": p_iou,
        "l_sucos": l_sucos,
        "leakage": leakage,
    }
