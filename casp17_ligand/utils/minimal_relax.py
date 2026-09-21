"""Minimal OpenMM ligand relaxation in fixed protein/RNA pocket.

Stack (PoseBusters paper canonical):
  - AMBER14 (protein + RNA + DNA)
  - OpenFF SMIRNOFF (organic ligand) via openmmforcefields
  - implicit GBN2 solvent
  - all protein/RNA atoms fixed (mass=0), only ligand moves

Usage:
  conda run -n posebench_em python casp17_ligand/utils/minimal_relax.py \\
      --protein <input.pdb> --ligand <input.sdf> --output <relaxed.sdf>
"""
import argparse
import os
import sys
from openff.toolkit.topology import Molecule
from openmm import LangevinIntegrator, Platform
from openmm.app import ForceField, HBonds, Modeller, PDBFile, Simulation
from openmm.unit import kelvin, picosecond, nanometer, kilojoule, mole
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem


def _strip_extra_phosphate_oxygens(input_pdb: str, output_pdb: str) -> None:
    """Strip 5'-phosphate group (P/OP1/OP2/OP3) from residue 1.

    AMBER's G5/A5/C5/U5 templates model the 5'-terminus WITHOUT a 5'-phosphate
    (since natural RNA 5' end has a 5'-OH). AF3/Boltz-2/Protenix output a
    5'-phosphate on residue 1 because the input had one; strip it for OpenMM.
    """
    skipped_atoms = {"P", "OP1", "OP2", "OP3"}
    with open(input_pdb) as f, open(output_pdb, "w") as g:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resnum = line[22:26].strip()
                atom = line[12:16].strip()
                if resnum == "1" and atom in skipped_atoms:
                    continue
            g.write(line)


def relax(protein_pdb: str, ligand_sdf: str, output_sdf: str,
          max_iterations: int = 200, tolerance: float = 10.0,
          platform_name: str = "CPU") -> dict:
    # Pre-process PDB: strip 5'-OP3 atom (AMBER14 internal G template doesn't include it)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_pdb = tmp.name
    _strip_extra_phosphate_oxygens(protein_pdb, tmp_pdb)

    # Load protein/RNA via OpenMM PDBFile
    pdb = PDBFile(tmp_pdb)

    # Load ligand via OpenFF Molecule (auto-detects bond orders / charges from SDF)
    rdmol = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=True)
    if rdmol is None:
        raise ValueError(f"RDKit could not parse {ligand_sdf}")
    rdmol = Chem.AddHs(rdmol, addCoords=True)
    off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    # Build system: AMBER14 (protein/RNA) + SMIRNOFF (ligand) + implicit GBN2
    forcefield_kwargs = {
        "constraints": HBonds,
        "rigidWater": False,
        "hydrogenMass": None,
    }
    sysgen = SystemGenerator(
        forcefields=["amber14-all.xml", "implicit/gbn2.xml"],
        small_molecule_forcefield="openff-2.0.0",
        molecules=[off_mol],
        forcefield_kwargs=forcefield_kwargs,
    )

    # Build modeller from RNA, add missing hydrogens via AMBER14
    modeller = Modeller(pdb.topology, pdb.positions)
    # addHydrogens needs an OpenMM ForceField (not a SystemGenerator)
    base_ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield=base_ff)

    # Add ligand to topology
    lig_top = off_mol.to_topology().to_openmm()
    lig_pos = off_mol.conformers[0].to_openmm()
    modeller.add(lig_top, lig_pos)

    # Build system via SystemGenerator (handles ligand SMIRNOFF)
    system = sysgen.create_system(modeller.topology)

    # Freeze every atom that's NOT in the ligand (set mass=0)
    n_prot = pdb.topology.getNumAtoms()
    for atom_idx in range(system.getNumParticles()):
        if atom_idx < n_prot:
            system.setParticleMass(atom_idx, 0.0)

    # Minimize
    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picosecond)
    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy().value_in_unit(kilojoule / mole)

    sim.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)

    state1 = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state1.getPotentialEnergy().value_in_unit(kilojoule / mole)
    final_positions = state1.getPositions(asNumpy=True).value_in_unit(nanometer)

    # Extract ligand positions, push back into RDKit mol, write SDF
    lig_pos_nm = final_positions[n_prot:]  # nanometers
    lig_pos_A = lig_pos_nm * 10.0  # to angstroms

    # The off_mol order may differ from the input rdmol order; use off_mol → rdkit roundtrip
    final_rdmol = off_mol.to_rdkit()
    conf = final_rdmol.GetConformer()
    for i in range(final_rdmol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(lig_pos_A[i, 0]),
                                 float(lig_pos_A[i, 1]),
                                 float(lig_pos_A[i, 2])))
    # Strip Hs back to match the original SDF convention if needed
    final_no_h = Chem.RemoveHs(final_rdmol)

    writer = Chem.SDWriter(output_sdf)
    writer.write(final_no_h)
    writer.close()

    return {"e_init_kJ/mol": e0, "e_final_kJ/mol": e1, "delta_kJ/mol": e1 - e0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--platform", default="CPU")
    args = ap.parse_args()

    info = relax(args.protein, args.ligand, args.output,
                 max_iterations=args.max_iter, tolerance=args.tol,
                 platform_name=args.platform)
    print(f"  e_init  = {info['e_init_kJ/mol']:>12.2f} kJ/mol")
    print(f"  e_final = {info['e_final_kJ/mol']:>12.2f} kJ/mol")
    print(f"  Δ       = {info['delta_kJ/mol']:>+12.2f} kJ/mol")
    print(f"  output  = {args.output}")


if __name__ == "__main__":
    main()
