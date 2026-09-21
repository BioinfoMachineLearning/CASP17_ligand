import logging
from pathlib import Path
from itertools import combinations
import numpy as np

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def get_center_atom_boltz_name(smiles: str) -> tuple[str | None, float]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None: return None, 0.0
    mol = AllChem.AddHs(mol)
    canonical_order = list(AllChem.CanonicalRankAtoms(mol))
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    
    # Generate 3D conformer to get coordinates
    success = AllChem.EmbedMolecule(mol, randomSeed=42)
    if success == -1:
        success = AllChem.EmbedMolecule(mol)
        if success == -1:
            return None, 0.0
            
    # Calculate geometric center of heavy atoms
    conf = mol.GetConformer()
    heavy_coords = []
    heavy_indices = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() != 1:  # Not hydrogen
            pos = conf.GetAtomPosition(i)
            heavy_coords.append([pos.x, pos.y, pos.z])
            heavy_indices.append(i)
            
    if not heavy_coords:
        return None, 0.0
        
    heavy_coords = np.array(heavy_coords)
    center = np.mean(heavy_coords, axis=0)
    
    # Calculate max radius (distance from center to furthest heavy atom)
    distances_to_center = np.linalg.norm(heavy_coords - center, axis=1)
    max_radius = float(np.max(distances_to_center))
    
    # Find heavy atom closest to the center
    closest_idx = heavy_indices[int(np.argmin(distances_to_center))]
    closest_atom = mol.GetAtomWithIdx(closest_idx)
    
    # Get its Boltz canonical name
    can_idx = canonical_order[closest_idx]
    atom_name = closest_atom.GetSymbol().upper() + str(can_idx + 1)
    
    return atom_name, max_radius

    # L3000 autotaxin cofactors: 2× ZN ions + 1× NAG glycosylation
# From PDB 5M7M (public reference, CASP-recommended), residue numbering with +28 offset
L3000_COFACTORS = {
    "zn1": {  # Catalytic zinc #1
        "ccd": "ZN",
        "contacts": [  # (chain, residue, max_distance)
            ("A", 284, 3.0),  # Asp284
            ("A", 288, 3.0),  # His288
            ("A", 447, 3.0),  # His447
        ],
    },
    "zn2": {  # Catalytic zinc #2
        "ccd": "ZN",
        "contacts": [
            ("A", 144, 3.0),  # Asp144
            ("A", 182, 3.0),  # Thr182
            ("A", 331, 3.0),  # Asp331
            ("A", 332, 3.0),  # His332
        ],
    },
    "nag": {  # N-glycosylation on Asn497
        "ccd": "NAG",
        "bond": {
            "protein_atom": ("A", 497, "ND2"),  # Asn497 ND2
            "ligand_atom": (None, 1, "C1"),     # NAG C1 (chain filled at runtime)
        },
    },
}


def create_boltz2_yaml(
    target,  # TargetData
    output_path: Path,
    predict_affinity: bool = False,
    msa_path: str | None = None,
    protonation: bool = False,
    apply_contacts: bool = True,
    cofactors: dict | None = None,
) -> None:
    """Write a single Boltz-2 input YAML file from a TargetData object."""
    lines = [
        "version: 1",
        "sequences:",
    ]

    # Write all polymer chains (protein or RNA — Boltz-2 supports both).
    # Boltz-2 docs: msa is "only for protein"; RNA chains have no MSA field.
    chain_kind = "rna" if target.entity_type == "rna" else "protein"
    for chain_id, seq in target.protein_sequences.items():
        lines += [
            f"  - {chain_kind}:",
            f"      id: {chain_id}",
            f"      sequence: {seq}",
        ]
        if msa_path and target.entity_type == "protein":
            lines.append(f"      msa: {msa_path}")

    # Process and write all ligands
    binder_ids = []
    ligand_meta = []

    # Use ascii letters for ligand IDs (A, B, C...) after protein chains
    start_char = chr(ord('A') + len(target.protein_sequences))

    for i, lig in enumerate(target.ligands):
        lig_id = chr(ord(start_char) + i)
        binder_ids.append(lig_id)

        # Apply protonation if requested
        smiles = lig.smiles
        if protonation:
            from casp17_ligand.data.components import protonate_smiles
            smiles = protonate_smiles(smiles)

        lines += [
            "  - ligand:",
            f"      id: {lig_id}",
            f"      smiles: '{smiles}'",
        ]

        if apply_contacts:
            center_name, radius = get_center_atom_boltz_name(smiles)
            if center_name:
                ligand_meta.append((lig_id, center_name, radius))

    # Add cofactor entities (ZN ions, NAG, etc.)
    cofactor_constraint_lines = []
    next_id = chr(ord(start_char) + len(target.ligands))
    if cofactors:
        for cof_name, cof_def in cofactors.items():
            cof_id = next_id
            next_id = chr(ord(next_id) + 1)
            lines += [
                "  - ligand:",
                f"      id: {cof_id}",
                f"      ccd: [{cof_def['ccd']}]",
            ]
            # Contact constraints for ions
            # For NONPOLYMER (ions), token uses atom name not residue index
            # ZN atom name in CCD is "ZN", NAG atoms are "C1","O1" etc.
            if "contacts" in cof_def:
                atom_name = cof_def["ccd"]  # e.g. "ZN" — atom name = CCD code for single-atom ions
                for prot_chain, res_idx, max_dist in cof_def["contacts"]:
                    cofactor_constraint_lines += [
                        "  - contact:",
                        f"      token1: [{cof_id}, {atom_name}]",
                        f"      token2: [{prot_chain}, {res_idx}]",
                        f"      max_distance: {max_dist}",
                        "      force: true",
                    ]
            # Bond constraint for covalent cofactors (e.g., NAG)
            if "bond" in cof_def:
                b = cof_def["bond"]
                pc, pr, pa = b["protein_atom"]
                _, lr, la = b["ligand_atom"]
                cofactor_constraint_lines += [
                    "  - bond:",
                    f"      atom1: [{pc}, {pr}, {pa}]",
                    f"      atom2: [{cof_id}, {lr}, {la}]",
                ]
            log.info(f"    + cofactor {cof_name}: chain {cof_id} ({cof_def['ccd']})")

    # Apply Contact Constraints + Pocket Constraints
    # MPro active site residues around Cys145 for pocket anchoring
    MPRO_ACTIVE_SITE_RESIDUES = [144, 145, 146, 163, 166]
    protein_chain_ids = list(target.protein_sequences.keys())

    # IMPORTANT: MPro Cys145 pocket anchoring is L4000-specific.
    # Do NOT apply to other dimers (e.g. CASP15 T1187, T1158v4 etc.).
    is_mpro = target.target_id.startswith("L4")

    if apply_contacts and target.is_dimer and is_mpro:
        from casp17_ligand.data.components import group_ligands_for_dimer
        group_a, group_b = group_ligands_for_dimer(target.ligands)

        constraint_lines = []
        meta_by_id = {lid: (lid, atom, rad) for lid, atom, rad in ligand_meta}

        for site_idx, group in enumerate([group_a, group_b]):
            if not group:
                continue
            # Contact constraints within group
            group_ids = [chr(ord(start_char) + orig_idx) for orig_idx, _ in group]
            group_meta = [meta_by_id[gid] for gid in group_ids if gid in meta_by_id]
            for (l1_id, a1, r1), (l2_id, a2, r2) in combinations(group_meta, 2):
                max_dist = round(r1 + r2 + 4.5, 2)
                constraint_lines += [
                    "  - contact:",
                    f"      token1: [{l1_id}, {a1}]",
                    f"      token2: [{l2_id}, {a2}]",
                    f"      max_distance: {max_dist}",
                    "      force: true",
                ]

            # Pocket constraint: main ligand → corresponding chain's MPro active site
            main_entry = max(group, key=lambda x: len(x[1].smiles))
            main_lig_id = chr(ord(start_char) + main_entry[0])
            prot_chain = protein_chain_ids[site_idx]
            contacts_str = ", ".join(
                f"[{prot_chain}, {r}]" for r in MPRO_ACTIVE_SITE_RESIDUES
            )
            constraint_lines += [
                "  - pocket:",
                f"      binder: {main_lig_id}",
                f"      contacts: [{contacts_str}]",
                "      max_distance: 10",
            ]

        constraint_lines.extend(cofactor_constraint_lines)
        if constraint_lines:
            lines.append("constraints:")
            lines.extend(constraint_lines)

        log.info(f"  {target.target_id}: MPro dimer grouping + pocket — "
                 f"Site A: {[l.name for _, l in group_a]}, "
                 f"Site B: {[l.name for _, l in group_b]}")

    elif apply_contacts and target.is_dimer and not is_mpro:
        # Non-MPro dimer: no protein-ligand pocket anchor (Cys145 is MPro-specific).
        # Only add cofactor constraints if present (e.g. L3000 ZN coordination).
        if cofactor_constraint_lines:
            lines.append("constraints:")
            lines.extend(cofactor_constraint_lines)
        log.info(f"  {target.target_id}: non-MPro dimer — no pocket/ligand-ligand constraints")

    elif apply_contacts and len(ligand_meta) > 1 and target.target_id.startswith("L3"):
        # L3000 non-dimer multi-ligand: co-crystallized nearby ligands kept within
        # radius-sum + 4.5Å of each other to prevent scatter.
        # WARNING: do NOT apply to other series (CASP15 multi-ligand targets have
        # completely different co-location assumptions).
        constraint_lines = []
        for (lig1_id, atom1, rad1), (lig2_id, atom2, rad2) in combinations(ligand_meta, 2):
            max_dist = round(rad1 + rad2 + 4.5, 2)
            constraint_lines += [
                "  - contact:",
                f"      token1: [{lig1_id}, {atom1}]",
                f"      token2: [{lig2_id}, {atom2}]",
                f"      max_distance: {max_dist}",
                "      force: true",
            ]
        # Append cofactor constraints
        constraint_lines.extend(cofactor_constraint_lines)
        if constraint_lines:
            lines.append("constraints:")
            lines.extend(constraint_lines)

    elif apply_contacts and len(ligand_meta) > 1:
        # Other non-dimer multi-ligand targets: no inter-ligand constraints.
        # (No universal co-location assumption outside L3000.)
        if cofactor_constraint_lines:
            lines.append("constraints:")
            lines.extend(cofactor_constraint_lines)
        log.info(f"  {target.target_id}: non-L3000 multi-ligand non-dimer — no inter-ligand constraints")

    elif cofactor_constraint_lines:
        # No ligand contacts but have cofactor constraints
        lines.append("constraints:")
        lines.extend(cofactor_constraint_lines)

    # Predict affinity config (Boltz 2 currently fails if a ligand has multiple identical copies)
    # So we disable affinity prediction if there is more than 1 ligand for safety
    if predict_affinity and binder_ids and len(target.ligands) == 1:
        lines += [
            "properties:",
            "  - affinity:",
            f"      binder: {binder_ids[0]}",
        ]
        
    output_path.write_text("\n".join(lines) + "\n")


@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="boltz2_input_preparation")
def main(cfg: DictConfig) -> None:
    from casp17_ligand.data.components import load_all_targets
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    series = cfg.get("series", "L1000")
    data_root_cfg = cfg.get("data_root", "data/casp16_data")
    data_root = root / data_root_cfg
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    protonation = cfg.get("protonation", False)
    log.info(f"Loading {series} targets from {data_root} (protonation={protonation})")
    
    # Load all targets that have structural references (struct_only configurable;
    # CASP17 in-progress targets have no GT struct → set struct_only=false).
    struct_only = cfg.get("struct_only", True)
    # `targets=` restricts prep to specific ids (competition: regenerate one
    # target without rewriting inputs for targets already running/submitted).
    # Unset/null = whole series (benchmark & eval mode).
    targets = load_all_targets(
        series, data_root, struct_only=struct_only, only=cfg.get("targets", None)
    )
    log.info(f"Preparing Boltz-2 inputs for {len(targets)} targets → {output_dir}")

    msa_path = cfg.get("msa_path", None)

    # Cofactors: L3000 autotaxin has 2× ZN + 1× NAG
    cofactors = None
    if cfg.get("cofactors", False) and series == "L3000":
        cofactors = L3000_COFACTORS
        log.info("Cofactors enabled: 2× ZN + 1× NAG (autotaxin)")

    for target in targets:
        yaml_path = output_dir / f"{target.target_id}_input.yaml"
        create_boltz2_yaml(
            target,
            yaml_path,
            predict_affinity=cfg.predict_affinity,
            msa_path=msa_path,
            protonation=protonation,
            cofactors=cofactors,
        )
        log.info(f"  Created: {yaml_path.name}")

    log.info(f"Done. {len(targets)} YAML files written to {output_dir}")


if __name__ == "__main__":
    main()
