"""Universal protein-or-RNA + ligand pose relaxation via OpenMM.

Design principles (per user constraints):
  - Receptor PDB stays UNCHANGED in the output (only ligand SDF is overwritten)
  - No big-block structural modification: only add missing Hs + strip the single
    5'-OP3 atom from RNA residue 1 (AMBER14 5'-terminal templates lack 5'-PO4).
    Both of these are at most 1-atom-per-chain adjustments; no residues are
    deleted/inserted.
  - Auto-detect protein vs RNA from residue name conventions:
      protein: 3-letter codes (ALA, GLY, ...)
      RNA:     1-letter codes (A, C, G, U)
  - Harmonic restraint on ligand center (k=1000 kJ/mol/nm²) so that minimization
    relieves local clashes without letting the ligand drift out of the pocket.

Stack: OpenMM 8.x + AMBER14 (protein+RNA force field) + OpenFF SMIRNOFF
(small-molecule auto-parametrize) + PDBFixer (add missing atoms/Hs).

Usage:
    conda run -n PoseBench python casp17_ligand/utils/relax_pose.py \\
        --receptor input_protein.pdb --ligand input_ligand.sdf \\
        --output relaxed_ligand.sdf [--platform CPU|CUDA]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings

import numpy as np
from openff.toolkit.topology import Molecule
from openmm import CustomExternalForce, LangevinIntegrator, Platform, unit
from openmm.app import (
    ForceField, HBonds, Modeller, PDBFile, Simulation,
)
from openmm.unit import kelvin, kilojoule, mole, nanometer, picosecond
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem

# 1-letter residue names that indicate RNA chains
RNA_RESIDUES = {"A", "C", "G", "U"}
# 3-letter codes for canonical amino acids
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def detect_chain_type(pdb_path: str) -> str:
    """Return 'rna' if PDB residues use 1-letter A/C/G/U convention,
    'protein' if 3-letter amino-acid codes, else 'unknown'.
    """
    rna_hits = prot_hits = 0
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            rn = line[17:20].strip()
            if rn in RNA_RESIDUES:
                rna_hits += 1
            elif rn in PROTEIN_RESIDUES:
                prot_hits += 1
    if rna_hits > prot_hits:
        return "rna"
    if prot_hits > rna_hits:
        return "protein"
    return "unknown"


def make_temp_receptor(orig_pdb: str, chain_type: str, temp_pdb: str) -> None:
    """Build a TEMP receptor PDB suitable for OpenMM AMBER14:
      - For RNA: strip 5'-OP3 atom from residue 1 (AMBER14 G5/A5/C5/U5 lack
        5'-PO4); also rename the first residue to {X}5 and last to {X}3
        so OpenMM picks the correct 5'/3' terminal template.
      - Run PDBFixer.addMissingHydrogens(pH=7.0).

    The original PDB on disk is left untouched.
    """
    # Step 1: scan PDB to identify per-chain first/last residue numbers
    chain_first_last = {}  # chain_id -> (first_resnum, last_resnum)
    with open(orig_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21]
                resnum = line[22:26].strip()
                if ch not in chain_first_last:
                    chain_first_last[ch] = (resnum, resnum)
                else:
                    chain_first_last[ch] = (chain_first_last[ch][0], resnum)

    # Determine per-chain RNA-ness so MIXED protein+RNA receptors (M-series)
    # get the 5'-phosphate strip on their RNA chains even when the receptor is
    # majority-protein (global chain_type would be 'protein' and skip it). For
    # pure-RNA (R) / pure-protein (T) receptors this reduces to the old behavior.
    _rna_hits = {}
    _prot_hits = {}
    with open(orig_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21]
                rn = line[17:20].strip()
                if rn in RNA_RESIDUES:
                    _rna_hits[ch] = _rna_hits.get(ch, 0) + 1
                elif rn in PROTEIN_RESIDUES:
                    _prot_hits[ch] = _prot_hits.get(ch, 0) + 1
    chain_is_rna = {ch: _rna_hits.get(ch, 0) > _prot_hits.get(ch, 0)
                    for ch in chain_first_last}

    # Step 2: strip the entire 5'-phosphate (P/OP1/OP2/OP3 = 4 atoms) from
    # residue 1 of each RNA chain — AMBER14 5'-terminal templates (G5/A5/C5/U5)
    # model the 5' end as 5'-OH (no 5'-phosphate, the standard MD convention).
    # Original PDB on disk is left intact; this only affects the OpenMM temp.
    PHOSPHATE_ATOMS_5PRIME = {"P", "OP1", "OP2", "OP3"}
    intermediate = temp_pdb + ".pre.pdb"
    with open(orig_pdb) as f, open(intermediate, "w") as g:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21]
                resnum = line[22:26].strip()
                atom = line[12:16].strip()
                rn = line[17:20].strip()
                if chain_is_rna.get(ch, chain_type == "rna"):
                    if resnum == chain_first_last[ch][0] and atom in PHOSPHATE_ATOMS_5PRIME:
                        continue
            g.write(line)

    # Step 3: PDBFixer add missing Hs (will respect renamed templates)
    fixer = PDBFixer(filename=intermediate)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    with open(temp_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    os.unlink(intermediate)


def extract_pocket_receptor(receptor_pdb: str, ligand_sdf: str,
                            cutoff: float, out_pdb: str) -> int:
    """Write a truncated receptor PDB keeping only WHOLE residues that have any
    atom within ``cutoff`` Å of any ligand atom. Returns the number of residues
    kept. This shrinks the OpenMM system from a full 10k-atom complex to just
    the binding pocket, so per-pose minimization is fast — the receptor atoms
    are position-restrained anyway (only the ligand moves), and PoseBusters
    validation is still run against the FULL receptor by the caller, so a
    generous cutoff (≥8Å) captures every atom the ligand could clash with.

    Whole residues are kept so backbone templates stay intact. Two extra steps
    make the truncated system templatable by AMBER14 even when the ligand sits
    at a multi-chain interface (M-series RNP) and drags in scattered residues:
      1. Each within-cutoff residue is expanded by its ±1 sequential neighbours
         (from the full structure) so no residue is left isolated — a lone
         residue cut from mid-chain has no AMBER free-amino-acid template.
      2. Every contiguous fragment is written onto its own chain ID + TER, so
         PDBFixer (in make_temp_receptor) caps each fragment's ends as real
         termini (adds OXT / terminal H). Without this, internal-numbered cut
         residues look like mid-chain residues missing their peptide bonds and
         fail with "No template found for residue N".
    """
    lig = Chem.MolFromMolFile(ligand_sdf, sanitize=False, removeHs=True)
    conf = lig.GetConformer()
    lig_xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(lig.GetNumAtoms())])

    # Pass 1: collect per-residue atom coords (key = chain+resnum+icode).
    res_atoms = {}
    res_lines = {}
    order = []
    with open(receptor_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            key = (line[21], line[22:27])  # chain, resSeq+iCode
            if key not in res_atoms:
                res_atoms[key] = []
                res_lines[key] = []
                order.append(key)
            res_atoms[key].append([float(line[30:38]), float(line[38:46]),
                                   float(line[46:54])])
            res_lines[key].append(line)

    # resSeq integer per residue key (cols 23-26; iCode ignored for adjacency).
    key_resseq = {key: int(key[1][:4]) for key in order}

    # Pass 2: keep residues with min distance to ligand ≤ cutoff.
    c2 = cutoff * cutoff
    keep = set()
    for key in order:
        ra = np.array(res_atoms[key])
        d2 = ((ra[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1)
        if d2.min() <= c2:
            keep.add(key)

    # Pass 2b: expand by ±1 sequential neighbour (same original chain) so no kept
    # residue is isolated — turns each cut site into a proper N-/C-terminus.
    present_by_chain = {}
    for key in order:
        present_by_chain.setdefault(key[0], {})[key_resseq[key]] = key
    for key in list(keep):
        ch, rs = key[0], key_resseq[key]
        for nb_rs in (rs - 1, rs + 1):
            nb = present_by_chain.get(ch, {}).get(nb_rs)
            if nb is not None:
                keep.add(nb)

    # Pass 3: write kept residues, re-chaining each CONTIGUOUS fragment (same
    # orig chain + consecutive resSeq) onto its own chain ID + TER so PDBFixer
    # caps every fragment's termini.
    import string as _string
    chain_pool = _string.ascii_uppercase + _string.ascii_lowercase + _string.digits
    kept_keys = [k for k in order if k in keep]
    kept = 0
    frag_idx = -1
    new_ch = "A"
    prev = None  # (orig_chain, resSeq_int)
    with open(out_pdb, "w") as g:
        for key in kept_keys:
            orig_ch, rs = key[0], key_resseq[key]
            if prev is None or prev[0] != orig_ch or rs - prev[1] > 1:
                if prev is not None:
                    g.write("TER\n")
                frag_idx += 1
                new_ch = chain_pool[frag_idx % len(chain_pool)]
            for line in res_lines[key]:
                g.write(line[:21] + new_ch + line[22:])
            prev = (orig_ch, rs)
            kept += 1
        if prev is not None:
            g.write("TER\n")
        g.write("END\n")
    return kept


def _build_simulation(orig_receptor_pdb: str, ligand_sdf: str,
                      platform_name: str, ligand_restraint_k: float,
                      cached_off_molecule=None):
    """Internal: build (sim, off_mol, n_receptor, temp_pdb, chain_type).
    Caller is responsible for os.unlink(temp_pdb).
    """
    chain_type = detect_chain_type(orig_receptor_pdb)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        temp_pdb = tmp.name
    make_temp_receptor(orig_receptor_pdb, chain_type, temp_pdb)

    pdb = PDBFile(temp_pdb)
    rdmol = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=True)
    if rdmol is None:
        raise ValueError(f"RDKit failed to parse {ligand_sdf}")
    rdmol = Chem.AddHs(rdmol, addCoords=True)

    # The cached molecule's GRAPH is what ends up in the output SDF (see
    # _write_ligand_sdf) and its atom i receives this pose's atom i. Reuse is
    # therefore only safe for a molecule that is genuinely the same, in the same
    # atom order. The caller keys its cache on exactly that, but verify here too:
    # a mismatched cache must degrade to a fresh (slower) build, never to a crash
    # or to relaxing somebody else's molecule.
    off_mol = None
    if cached_off_molecule is not None:
        import copy
        cand = copy.deepcopy(cached_off_molecule)
        same = (cand.n_atoms == rdmol.GetNumAtoms() and
                all(a.atomic_number == rdmol.GetAtomWithIdx(i).GetAtomicNum()
                    for i, a in enumerate(cand.atoms)))
        if same:
            from openff.units import unit as off_unit
            conf = rdmol.GetConformer()
            new_conf = np.array([[conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]
                                 for i in range(rdmol.GetNumAtoms())])
            cand._conformers = [off_unit.Quantity(new_conf, off_unit.angstrom)]
            off_mol = cand
        else:
            reason = ("atom count" if cand.n_atoms != rdmol.GetNumAtoms()
                      else "atom ordering")
            warnings.warn(f"charge cache mismatch ({reason}) for {ligand_sdf} "
                          f"({cand.n_atoms} cached vs {rdmol.GetNumAtoms()} "
                          f"atoms); rebuilding without cache")
    if off_mol is None:
        off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    sysgen = SystemGenerator(
        forcefields=["amber14-all.xml", "implicit/gbn2.xml"],
        small_molecule_forcefield="openff-2.0.0",
        molecules=[off_mol],
        forcefield_kwargs={"constraints": HBonds, "rigidWater": False,
                           "hydrogenMass": None},
    )

    modeller = Modeller(pdb.topology, pdb.positions)
    lig_topology = off_mol.to_topology().to_openmm()
    lig_positions = off_mol.conformers[0].to_openmm()
    modeller.add(lig_topology, lig_positions)

    residue_templates = {}
    if chain_type == "rna":
        for mod_chain in modeller.topology.chains():
            mod_residues = list(mod_chain.residues())
            rna_residues = [r for r in mod_residues if r.name in RNA_RESIDUES]
            if not rna_residues:
                continue
            first = rna_residues[0]
            last = rna_residues[-1]
            residue_templates[first] = first.name + "5"
            residue_templates[last] = last.name + "3"

    system = sysgen.forcefield.createSystem(
        modeller.topology,
        constraints=HBonds,
        rigidWater=False,
        residueTemplates=residue_templates,
    )

    n_receptor = pdb.topology.getNumAtoms()
    K_RECEPTOR = 50000.0
    positions = modeller.positions
    restraint = CustomExternalForce(
        "0.5 * k_atom * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addPerParticleParameter("k_atom")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i in range(system.getNumParticles()):
        p = positions[i].value_in_unit(nanometer)
        k = (K_RECEPTOR if i < n_receptor else ligand_restraint_k) * kilojoule / mole / nanometer ** 2
        restraint.addParticle(i, [k.value_in_unit(kilojoule / mole / nanometer ** 2),
                                  p[0], p[1], p[2]])
    system.addForce(restraint)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picosecond)
    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    return sim, off_mol, n_receptor, temp_pdb, chain_type


def _write_ligand_sdf(sim, off_mol, n_receptor, output_sdf):
    """Internal: extract current ligand positions from sim and write SDF."""
    state = sim.context.getState(getPositions=True)
    final_positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    lig_pos_A = final_positions[n_receptor:] * 10.0
    final_rdmol = off_mol.to_rdkit()
    conf = final_rdmol.GetConformer()
    for i in range(final_rdmol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(lig_pos_A[i, 0]),
                                 float(lig_pos_A[i, 1]),
                                 float(lig_pos_A[i, 2])))
    final_no_h = Chem.RemoveHs(final_rdmol)
    writer = Chem.SDWriter(output_sdf)
    writer.write(final_no_h)
    writer.close()
    return final_no_h


def relax_iterative(orig_receptor_pdb: str, ligand_sdf: str, output_sdf: str,
                    pb_check_fn,
                    platform_name: str = "CPU",
                    step_iter: int = 20, max_total_iter: int = 200,
                    ligand_restraint_k: float = 1000.0,
                    tolerance_kJ: float = 10.0,
                    cached_off_molecule=None,
                    pocket_cutoff: float = None) -> dict:
    """Early-stop relax: minimize in small chunks, write SDF after each chunk,
    invoke pb_check_fn(sdf_path, pdb_path) -> bool, stop on first True.

    Always writes the final state (passed or not) to output_sdf.

    ``pocket_cutoff`` (Å): if set, the OpenMM system is built from only the
    residues within that distance of the ligand (see extract_pocket_receptor),
    making minimization fast for huge multi-chain receptors. PoseBusters is
    still validated against the FULL ``orig_receptor_pdb``.
    """
    # Cache original ligand heavy-atom positions BEFORE any writeback, so we can
    # compute true drift even when output_sdf == ligand_sdf (in-place overwrite).
    _orig_mol = Chem.MolFromMolFile(ligand_sdf, removeHs=True, sanitize=True)
    orig_xyz_for_drift = None
    if _orig_mol is not None:
        _conf = _orig_mol.GetConformer()
        orig_xyz_for_drift = np.array(
            [(_conf.GetAtomPosition(i).x, _conf.GetAtomPosition(i).y,
              _conf.GetAtomPosition(i).z)
             for i in range(_orig_mol.GetNumAtoms())])

    # Pocket-local relax: build the OpenMM system from only the binding-pocket
    # residues (fast for huge receptors). PB check below still uses the FULL
    # orig_receptor_pdb. Receptor atoms are position-restrained either way.
    sim_receptor_pdb = orig_receptor_pdb
    _pocket_tmp = None
    if pocket_cutoff:
        with tempfile.NamedTemporaryFile(suffix="_pocket.pdb", delete=False) as _pt:
            _pocket_tmp = _pt.name
        extract_pocket_receptor(orig_receptor_pdb, ligand_sdf, pocket_cutoff, _pocket_tmp)
        sim_receptor_pdb = _pocket_tmp

    sim, off_mol, n_receptor, temp_pdb, chain_type = _build_simulation(
        sim_receptor_pdb, ligand_sdf, platform_name, ligand_restraint_k,
        cached_off_molecule)
    try:
        state0 = sim.context.getState(getEnergy=True)
        e0 = state0.getPotentialEnergy().value_in_unit(kilojoule / mole)

        # First, check if ORIGINAL already passes (caller should ideally pre-filter,
        # but this is a safe net).
        _write_ligand_sdf(sim, off_mol, n_receptor, output_sdf)
        if pb_check_fn(output_sdf, orig_receptor_pdb):
            return dict(chain_type=chain_type, iters_done=0, passed=True,
                        e_init_kJ_mol=e0, e_final_kJ_mol=e0,
                        ligand_drift_RMSD_A=0.0)

        iters_done = 0
        tol = tolerance_kJ * kilojoule / mole / nanometer
        passed = False
        while iters_done < max_total_iter:
            sim.minimizeEnergy(tolerance=tol, maxIterations=step_iter)
            iters_done += step_iter
            _write_ligand_sdf(sim, off_mol, n_receptor, output_sdf)
            if pb_check_fn(output_sdf, orig_receptor_pdb):
                passed = True
                break

        state1 = sim.context.getState(getEnergy=True)
        e1 = state1.getPotentialEnergy().value_in_unit(kilojoule / mole)

        # RMSD vs original input ligand (use cached orig positions; output_sdf
        # may have been overwritten in place during the loop).
        final = Chem.MolFromMolFile(output_sdf, removeHs=True)
        if final is None or orig_xyz_for_drift is None:
            rmsd = float("nan")
        else:
            relax_xyz = np.array(
                [(final.GetConformer().GetAtomPosition(i).x,
                  final.GetConformer().GetAtomPosition(i).y,
                  final.GetConformer().GetAtomPosition(i).z)
                 for i in range(final.GetNumAtoms())])
            rmsd = (float(np.sqrt(((orig_xyz_for_drift - relax_xyz) ** 2)
                                  .sum(axis=1).mean()))
                    if orig_xyz_for_drift.shape == relax_xyz.shape else float("nan"))

        return dict(chain_type=chain_type, iters_done=iters_done, passed=passed,
                    e_init_kJ_mol=e0, e_final_kJ_mol=e1,
                    ligand_drift_RMSD_A=rmsd)
    finally:
        os.unlink(temp_pdb)
        if _pocket_tmp and os.path.exists(_pocket_tmp):
            os.unlink(_pocket_tmp)


def relax(orig_receptor_pdb: str, ligand_sdf: str, output_sdf: str,
          platform_name: str = "CPU", max_iter: int = 200,
          ligand_restraint_k: float = 1000.0,
          tolerance_kJ: float = 10.0,
          cached_off_molecule=None) -> dict:
    """Run constrained ligand-only relaxation.

    Returns: dict with 'chain_type', 'e_init', 'e_final', 'delta_e' (all kJ/mol),
    'rmsd_ligand_drift_A' (RMSD between input and relaxed ligand positions).
    """
    chain_type = detect_chain_type(orig_receptor_pdb)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        temp_pdb = tmp.name
    make_temp_receptor(orig_receptor_pdb, chain_type, temp_pdb)

    # Load receptor topology + ligand molecule
    pdb = PDBFile(temp_pdb)
    if cached_off_molecule is not None:
        # Reuse cached Molecule (with AM1-BCC charges pre-computed). Just
        # overwrite its conformer with this specific pose's coordinates so
        # we skip the slow per-pose AM1-BCC charge derivation.
        import copy
        off_mol = copy.deepcopy(cached_off_molecule)
        rdmol = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=True)
        if rdmol is None:
            raise ValueError(f"RDKit failed to parse {ligand_sdf}")
        rdmol = Chem.AddHs(rdmol, addCoords=True)
        # Update cached off_mol's conformer with current pose coordinates
        from openff.units import unit as off_unit
        import numpy as np
        new_conf = np.array([[rdmol.GetConformer().GetAtomPosition(i).x,
                              rdmol.GetConformer().GetAtomPosition(i).y,
                              rdmol.GetConformer().GetAtomPosition(i).z]
                             for i in range(rdmol.GetNumAtoms())])
        off_mol._conformers = [off_unit.Quantity(new_conf, off_unit.angstrom)]
    else:
        rdmol = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=True)
        if rdmol is None:
            raise ValueError(f"RDKit failed to parse {ligand_sdf}")
        rdmol = Chem.AddHs(rdmol, addCoords=True)
        off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    sysgen = SystemGenerator(
        forcefields=["amber14-all.xml", "implicit/gbn2.xml"],
        small_molecule_forcefield="openff-2.0.0",
        molecules=[off_mol],
        forcefield_kwargs={"constraints": HBonds, "rigidWater": False,
                           "hydrogenMass": None},
    )

    # Combine receptor topology + ligand topology
    modeller = Modeller(pdb.topology, pdb.positions)
    lig_topology = off_mol.to_topology().to_openmm()
    lig_positions = off_mol.conformers[0].to_openmm()
    modeller.add(lig_topology, lig_positions)

    # For RNA: explicitly tell OpenMM which template (5'-terminal X5, 3'-terminal
    # X3, otherwise internal) for each receptor RNA residue.  This avoids the
    # default-matcher confusion at the 5' end (we stripped 5'-PO4) and 3' end.
    residue_templates = {}
    if chain_type == "rna":
        for mod_chain in modeller.topology.chains():
            mod_residues = list(mod_chain.residues())
            rna_residues = [r for r in mod_residues if r.name in RNA_RESIDUES]
            if not rna_residues:
                continue
            first = rna_residues[0]
            last = rna_residues[-1]
            residue_templates[first] = first.name + "5"
            residue_templates[last] = last.name + "3"

    system = sysgen.forcefield.createSystem(
        modeller.topology,
        constraints=HBonds,
        rigidWater=False,
        residueTemplates=residue_templates,
    )

    n_receptor = pdb.topology.getNumAtoms()
    # Do NOT freeze receptor (setParticleMass(0) would lock bad initial H
    # positions from PDBFixer and starve the minimizer of useful gradient).
    # Instead, apply harmonic restraints on BOTH receptor and ligand to anchor
    # near original positions while allowing local clash relief.
    #
    # Final submission discards the relaxed receptor (uses original PDB) and
    # uses only the relaxed ligand SDF.
    K_RECEPTOR = 50000.0  # kJ/mol/nm² — very stiff: keeps receptor backbone +
                          #                heavy atoms within ~0.05 Å of original
    positions = modeller.positions
    restraint = CustomExternalForce(
        "0.5 * k_atom * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addPerParticleParameter("k_atom")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i in range(system.getNumParticles()):
        p = positions[i].value_in_unit(nanometer)
        k = (K_RECEPTOR if i < n_receptor else ligand_restraint_k) * kilojoule / mole / nanometer ** 2
        restraint.addParticle(i, [k.value_in_unit(kilojoule / mole / nanometer ** 2),
                                  p[0], p[1], p[2]])
    system.addForce(restraint)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picosecond)
    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy().value_in_unit(kilojoule / mole)

    sim.minimizeEnergy(tolerance=tolerance_kJ * kilojoule / mole / nanometer,
                       maxIterations=max_iter)

    state1 = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state1.getPotentialEnergy().value_in_unit(kilojoule / mole)
    final_positions = state1.getPositions(asNumpy=True).value_in_unit(nanometer)

    # Extract ligand positions, write back to SDF (preserve original atom
    # ordering by writing into the original rdmol's conformer, then drop Hs).
    lig_pos_A = final_positions[n_receptor:] * 10.0  # nm → Å

    # Replace ligand's conformer with relaxed positions.  off_mol's atom order
    # may differ from rdmol's; safest is to write a fresh RDKit mol from off_mol.
    final_rdmol = off_mol.to_rdkit()
    conf = final_rdmol.GetConformer()
    for i in range(final_rdmol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(lig_pos_A[i, 0]),
                                 float(lig_pos_A[i, 1]),
                                 float(lig_pos_A[i, 2])))
    final_no_h = Chem.RemoveHs(final_rdmol)
    writer = Chem.SDWriter(output_sdf)
    writer.write(final_no_h)
    writer.close()

    # Compute ligand RMSD (drift): compare relaxed heavy-atom positions vs orig
    orig = Chem.MolFromMolFile(ligand_sdf, removeHs=True)
    orig_xyz = np.array([(orig.GetConformer().GetAtomPosition(i).x,
                          orig.GetConformer().GetAtomPosition(i).y,
                          orig.GetConformer().GetAtomPosition(i).z)
                         for i in range(orig.GetNumAtoms())])
    relax_xyz = np.array([(final_no_h.GetConformer().GetAtomPosition(i).x,
                           final_no_h.GetConformer().GetAtomPosition(i).y,
                           final_no_h.GetConformer().GetAtomPosition(i).z)
                          for i in range(final_no_h.GetNumAtoms())])
    rmsd = float(np.sqrt(((orig_xyz - relax_xyz) ** 2).sum(axis=1).mean()))

    os.unlink(temp_pdb)
    return {
        "chain_type": chain_type,
        "e_init_kJ/mol": e0,
        "e_final_kJ/mol": e1,
        "delta_e_kJ/mol": e1 - e0,
        "ligand_drift_RMSD_A": rmsd,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receptor", required=True,
                    help="Input protein-or-RNA PDB (kept unchanged in submission)")
    ap.add_argument("--ligand", required=True,
                    help="Input ligand SDF (will be relaxed)")
    ap.add_argument("--output", required=True,
                    help="Output relaxed ligand SDF path")
    ap.add_argument("--platform", default="CPU", choices=["CPU", "CUDA", "OpenCL"])
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--restraint-k", type=float, default=1000.0,
                    help="Ligand harmonic restraint stiffness in kJ/mol/nm^2 "
                         "(default 1000; ≈1 kcal/mol/Å^2 — stiff enough to hold pose)")
    args = ap.parse_args()

    info = relax(args.receptor, args.ligand, args.output,
                 platform_name=args.platform, max_iter=args.max_iter,
                 ligand_restraint_k=args.restraint_k)
    print(f"  chain_type       : {info['chain_type']}")
    print(f"  e_init           : {info['e_init_kJ/mol']:>14.2f} kJ/mol")
    print(f"  e_final          : {info['e_final_kJ/mol']:>14.2f} kJ/mol")
    print(f"  Δe               : {info['delta_e_kJ/mol']:>+14.2f} kJ/mol")
    print(f"  ligand drift RMSD: {info['ligand_drift_RMSD_A']:>14.3f} Å")
    print(f"  output           : {args.output}")


if __name__ == "__main__":
    main()
