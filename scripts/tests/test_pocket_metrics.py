"""Test pocket_metrics using boltz2 L1001 pairwise model comparison.

Uses converted boltz2 outputs (model_0 vs model_1) to verify that
pocket_align_complex_to_reference and related functions work correctly.
"""

from pathlib import Path

import pytest
import numpy as np

from casp17_ligand.utils.pocket_metrics import (
    pocket_align_complex_to_reference,
    read_ligand_positions,
    pocket_coverage_iou,
    ligand_sucos,
    load_ligand_mol,
    get_sucos_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONV_DIR = (
    PROJECT_ROOT / "data/test_cases/casp16_l1000/ensemble_outputs/L1001/cif_converted"
)

M0 = {
    "pdb": f"{CONV_DIR}/L1001_model_0_protein.pdb",
    "sdf": f"{CONV_DIR}/L1001_model_0_ligand.sdf",
}
M1 = {
    "pdb": f"{CONV_DIR}/L1001_model_1_protein.pdb",
    "sdf": f"{CONV_DIR}/L1001_model_1_ligand.sdf",
}


class TestPairwiseAlignment:
    """Model 0 vs Model 1 pairwise comparison."""

    def test_basic_run(self):
        mob_pos = read_ligand_positions(M1["sdf"])
        assert mob_pos is not None
        r = pocket_align_complex_to_reference(
            M0["pdb"], M0["sdf"], M1["pdb"], M1["sdf"], mob_pos
        )
        assert isinstance(r, dict)
        for key in ("aligned_lig_pos", "p_sucos", "pocket_rmsd",
                     "pocket_iou", "l_sucos"):
            assert key in r
        assert r["aligned_lig_pos"] is not None
        assert r["aligned_lig_pos"].shape == mob_pos.shape

    def test_scores_in_range(self):
        mob_pos = read_ligand_positions(M1["sdf"])
        r = pocket_align_complex_to_reference(
            M0["pdb"], M0["sdf"], M1["pdb"], M1["sdf"], mob_pos
        )
        assert 0.0 <= r["p_sucos"] <= 1.0
        assert 0.0 <= r["pocket_iou"] <= 1.0
        assert 0.0 <= r["l_sucos"] <= 1.0
        assert r["pocket_rmsd"] >= 0.0
        print(f"\np_sucos={r['p_sucos']:.3f}  pocket_rmsd={r['pocket_rmsd']:.3f}"
              f"  pocket_iou={r['pocket_iou']:.3f}  l_sucos={r['l_sucos']:.3f}")

    def test_self_comparison(self):
        """Model compared to itself should give perfect scores."""
        mob_pos = read_ligand_positions(M0["sdf"])
        r = pocket_align_complex_to_reference(
            M0["pdb"], M0["sdf"], M0["pdb"], M0["sdf"], mob_pos
        )
        assert r["p_sucos"] > 0.99
        assert r["pocket_iou"] > 0.99
        assert r["pocket_rmsd"] < 0.01


class TestIndividualMetrics:

    def test_sucos_score(self):
        mol0 = load_ligand_mol(M0["sdf"])
        mol1 = load_ligand_mol(M1["sdf"])
        assert mol0 is not None and mol1 is not None
        sucos, fm, shape = get_sucos_score(mol0, mol1)
        assert 0.0 <= sucos <= 1.0
        assert 0.0 <= fm <= 1.0
        assert 0.0 <= shape <= 1.0
        print(f"\nSuCOS={sucos:.3f}  feature_map={fm:.3f}  shape={shape:.3f}")

    def test_ligand_sucos(self):
        mol0 = load_ligand_mol(M0["sdf"])
        mol1 = load_ligand_mol(M1["sdf"])
        l_sucos = ligand_sucos(mol0, mol1)
        assert 0.0 <= l_sucos <= 1.0
        print(f"\nL_SuCOS={l_sucos:.3f}")

    def test_ligand_sucos_self(self):
        mol0 = load_ligand_mol(M0["sdf"])
        l_sucos = ligand_sucos(mol0, mol0)
        assert l_sucos > 0.99
