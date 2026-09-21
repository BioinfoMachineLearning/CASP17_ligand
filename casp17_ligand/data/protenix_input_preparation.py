"""Prepare Protenix input JSON files from an ensemble inputs CSV.

Generates JSON files in the Protenix format
(see forks/Protenix/docs/infer_json_format.md).
Supports injecting pre-computed MSA/Template paths for reuse.

L4000 dimer handling:
  - Small ligands are grouped with main ligands by pocket (Site A / Site B)
  - Contact constraints are set within each pocket group only
  - Covalent bonds are declared for covalent inhibitor targets
"""

import json
import logging
from collections import Counter
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Covalent inhibitor definitions: target_id -> {smarts, cys_pos}
# The SMARTS first atom is the attachment point to Cys SG.
COVALENT_TARGETS = {
    "L4003": {"smarts": "[cX3][nX2][cX3][nX2]", "cys_pos": 145},
    "L4013": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
    "L4019": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
    "L4023": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
}


def _get_protenix_atom_name(smiles: str, atom_idx: int) -> str:
    """Get the Protenix-internal atom name for a given atom index in a SMILES.

    Protenix names atoms as {ELEMENT}{count}, e.g. C1, C2, N1, O1, C3, ...
    following the order atoms appear in the RDKit molecule (heavy atoms only).
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    element_count = Counter()
    idx_to_name = {}
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        idx_to_name[atom.GetIdx()] = f"{element}{element_count[element]}"

    if atom_idx not in idx_to_name:
        raise ValueError(f"Atom index {atom_idx} not found in SMILES: {smiles}")
    return idx_to_name[atom_idx]


def find_covalent_attachment_atom(smiles: str, smarts: str) -> str:
    """Find the Protenix atom name for the SMARTS attachment point.

    The first atom in the SMARTS match is the covalent attachment point.
    Returns the Protenix-style atom name (e.g. "C4").
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"Failed to parse SMARTS: {smarts}")

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        raise ValueError(f"SMARTS '{smarts}' did not match SMILES '{smiles}'")

    attach_idx = matches[0][0]
    atom_name = _get_protenix_atom_name(smiles, attach_idx)
    log.info(f"  Covalent attachment: SMARTS '{smarts}' → atom {attach_idx} → {atom_name}")
    return atom_name


# group_ligands_for_dimer moved to casp17_ligand.data.components.target_data
from casp17_ligand.data.components import group_ligands_for_dimer


# Autotaxin (L3000) catalytic ZN coordination residues (L3000 sequence numbering).
# Derived from 5M7M crystal structure analysis: PDB→L3000 offset = 28.
# Atom-level constraints: specific coordinating atoms from 5M7M structure.
ATX_ZN_COORDINATION = {
    "zn1": {  # PDB: Asp312.OD1(2.01Å), His316.NE2(2.19Å), His475.NE2(2.07Å)
        "contacts": [
            {"position": 284, "atom": "OD1"},  # Asp284
            {"position": 288, "atom": "NE2"},  # His288
            {"position": 447, "atom": "NE2"},  # His447
        ],
    },
    "zn2": {  # PDB: Asp172.OD1(1.99Å), Asp359.OD2(1.99Å), His360.NE2(2.07Å), Thr210.OG1(2.01Å)
        "contacts": [
            {"position": 144, "atom": "OD1"},  # Asp144
            {"position": 331, "atom": "OD2"},  # Asp331
            {"position": 332, "atom": "NE2"},  # His332
            {"position": 182, "atom": "OG1"},  # Thr182
        ],
    },
}


def create_protenix_json(
    target,  # TargetData
    output_path: Path,
    paired_msa_path: str | None = None,
    unpaired_msa_path: str | None = None,
    templates_path: str | None = None,
    protonation: bool = False,
    add_zn: bool = False,
    add_glycan: bool = False,
    rna_unpaired_msa_path: str | None = None,
) -> None:
    """Write a single Protenix input JSON file from a TargetData object.

    For RNA targets (entity_type=="rna") chains are emitted as `rnaSequence`
    instead of `proteinChain`; protein-MSA / templates / cofactor / covalent
    branches all skip naturally because the relevant target_id prefixes
    (L3/L4/CASP15) don't match CASP17 RNA target ids.
    """
    sequences = []
    is_rna = target.entity_type == "rna"

    # Build polymer chain entries (protein or RNA).
    for chain_id, seq in target.protein_sequences.items():
        if is_rna:
            rna_entry = {"rnaSequence": {"sequence": seq, "count": 1}}
            if rna_unpaired_msa_path:
                rna_entry["rnaSequence"]["unpairedMsaPath"] = rna_unpaired_msa_path
            sequences.append(rna_entry)
        else:
            protein_entry = {
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                }
            }
            if paired_msa_path:
                protein_entry["proteinChain"]["pairedMsaPath"] = paired_msa_path
            if unpaired_msa_path:
                protein_entry["proteinChain"]["unpairedMsaPath"] = unpaired_msa_path
            if templates_path:
                protein_entry["proteinChain"]["templatesPath"] = templates_path
            sequences.append(protein_entry)

    n_protein = len(target.protein_sequences)

    # Add ZN ions if requested (Autotaxin catalytic dual-zinc center)
    n_ions = 0
    if add_zn:
        sequences.append({"ion": {"ion": "ZN", "count": 1}})  # ZN#1
        sequences.append({"ion": {"ion": "ZN", "count": 1}})  # ZN#2
        n_ions = 2

    # Add N-glycan if requested (Autotaxin Asn497 glycosylation)
    n_glycan = 0
    if add_glycan:
        sequences.append({"ligand": {"ligand": "CCD_NAG", "count": 1}})
        n_glycan = 1

    # Build ligand entries (all ligands as separate entities)
    for lig in target.ligands:
        smiles = lig.smiles
        if protonation:
            from casp17_ligand.data.components import protonate_smiles
            smiles = protonate_smiles(smiles)
        sequences.append({
            "ligand": {
                "ligand": smiles,
                "count": 1,
            }
        })

    job = {
        "name": target.target_id,
        "sequences": sequences,
    }

    n_ligands = len(target.ligands)
    # Entity numbering: protein(1..n_protein), ions(..), glycan(..), ligands(..)
    lig_entity_base = n_protein + n_ions + n_glycan  # first drug ligand entity = lig_entity_base + 1

    # --- Contact constraints ---
    # IMPORTANT: Cys145 pocket anchoring is MPro (L4000) specific.
    # Do NOT apply it to other dimers (e.g. CASP15 targets like T1187).
    # The "is_dimer" condition alone is NOT sufficient — L4000 must be explicitly checked.
    MPRO_ACTIVE_SITE_RESIDUE = 145
    is_mpro = target.target_id.startswith("L4")  # L4000 series = MPro

    if n_ligands > 1 and target.is_dimer and is_mpro:
        # MPro dimer: group ligands by pocket, constraints only within groups
        group_a, group_b = group_ligands_for_dimer(target.ligands)
        contacts = []

        # 1) Ligand-ligand contacts within each group
        for group in [group_a, group_b]:
            if len(group) <= 1:
                continue
            for gi in range(len(group)):
                for gj in range(gi + 1, len(group)):
                    entity_i = lig_entity_base + group[gi][0] + 1
                    entity_j = lig_entity_base + group[gj][0] + 1
                    contacts.append({
                        "entity1": entity_i,
                        "copy1": 1,
                        "position1": 1,
                        "entity2": entity_j,
                        "copy2": 1,
                        "position2": 1,
                        "max_distance": 4.5,
                        "min_distance": 3,
                    })

        # 2) Protein-ligand pocket anchoring: main LIG → corresponding chain's Cys145
        for site_idx, group in enumerate([group_a, group_b]):
            if not group:
                continue
            main_entry = max(group, key=lambda x: len(x[1].smiles))
            protein_entity = site_idx + 1  # chain A=1, chain B=2
            ligand_entity = lig_entity_base + main_entry[0] + 1
            contacts.append({
                "entity1": protein_entity,
                "copy1": 1,
                "position1": MPRO_ACTIVE_SITE_RESIDUE,
                "entity2": ligand_entity,
                "copy2": 1,
                "position2": 1,
                "max_distance": 10,
                "min_distance": 0,
            })

        if contacts:
            job["constraint"] = {"contact": contacts}

        log.info(f"  {target.target_id}: MPro dimer grouping — "
                 f"Site A: {[l.name for _, l in group_a]}, "
                 f"Site B: {[l.name for _, l in group_b]}")

    elif n_ligands > 1 and target.is_dimer and not is_mpro:
        # Non-MPro dimer: no constraints added.
        # group_ligands_for_dimer and the 4.5Å ligand-ligand anchoring are L4000-specific
        # logic and should NOT be applied to other systems (e.g. CASP15 dimers with
        # nucleotides/glycans, which have completely different binding site geometries).
        log.info(f"  {target.target_id}: non-MPro dimer with {n_ligands} ligands — no constraints added")

    elif target.is_dimer and n_ligands == 1 and is_mpro:
        # MPro single-ligand dimer: anchor to chain A's Cys145
        contacts = [{
            "entity1": 1,
            "copy1": 1,
            "position1": MPRO_ACTIVE_SITE_RESIDUE,
            "entity2": lig_entity_base + 1,
            "copy2": 1,
            "position2": 1,
            "max_distance": 10,
            "min_distance": 0,
        }]
        job["constraint"] = {"contact": contacts}

    elif target.is_dimer and n_ligands == 1 and not is_mpro:
        # Non-MPro single-ligand dimer: no protein-ligand anchor
        # (no universal "Cys145" equivalent exists for arbitrary dimers)
        pass

    elif n_ligands > 1 and target.target_id.startswith("L3"):
        # L3000 non-dimer multi-ligand: co-crystallized nearby ligands are kept
        # within 4.5Å of each other to prevent scatter.
        # WARNING: do NOT apply to other series (e.g. CASP15 multi-ligand targets
        # have completely different co-location assumptions).
        contacts = []
        for i in range(n_ligands):
            for j in range(i + 1, n_ligands):
                contacts.append({
                    "entity1": lig_entity_base + i + 1,
                    "copy1": 1,
                    "position1": 1,
                    "entity2": lig_entity_base + j + 1,
                    "copy2": 1,
                    "position2": 1,
                    "max_distance": 4.5,
                    "min_distance": 3,
                })
        job["constraint"] = {"contact": contacts}

    elif n_ligands > 1:
        # Other multi-ligand targets: no constraints (no universal co-location assumption).
        log.info(f"  {target.target_id}: multi-ligand non-dimer non-L3000 — no constraints added")

    # --- ZN ion coordination constraints (Autotaxin) ---
    if add_zn and n_ions == 2:
        zn_contacts = []
        protein_entity = 1  # single-chain protein
        zn1_entity = n_protein + 1
        zn2_entity = n_protein + 2
        for zn_entity, zn_key in [(zn1_entity, "zn1"), (zn2_entity, "zn2")]:
            for contact in ATX_ZN_COORDINATION[zn_key]["contacts"]:
                zn_contacts.append({
                    "entity1": protein_entity,
                    "copy1": 1,
                    "position1": contact["position"],
                    "atom1": contact["atom"],
                    "entity2": zn_entity,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "ZN",
                    "max_distance": 3.0,
                    "min_distance": 0,
                })
        # Merge with existing contacts
        if "constraint" in job:
            job["constraint"]["contact"].extend(zn_contacts)
        else:
            job["constraint"] = {"contact": zn_contacts}
        log.info(f"  {target.target_id}: added {len(zn_contacts)} ZN atom-level coordination constraints")

    # --- Covalent bonds ---
    covalent_bonds = []

    # Glycan covalent bond: Asn497.ND2 → NAG.C1
    if add_glycan and n_glycan == 1:
        glycan_entity = n_protein + n_ions + 1  # NAG is right after ions
        covalent_bonds.append({
            "entity1": "1",
            "copy1": 1,
            "position1": "497",
            "atom1": "ND2",
            "entity2": str(glycan_entity),
            "copy2": 1,
            "position2": "1",
            "atom2": "C1",
        })
        log.info(f"  {target.target_id}: added glycan covalent bond (Asn497.ND2 → NAG entity{glycan_entity}.C1)")

    # L4000 covalent inhibitors
    cov_info = COVALENT_TARGETS.get(target.target_id)
    if cov_info and target.is_dimer:
        group_a, group_b = group_ligands_for_dimer(target.ligands)
        protein_chain_ids = list(target.protein_sequences.keys())

        for site_idx, group in enumerate([group_a, group_b]):
            main_lig_entry = max(group, key=lambda x: len(x[1].smiles))
            orig_idx, main_lig = main_lig_entry

            attach_atom = find_covalent_attachment_atom(
                main_lig.smiles, cov_info["smarts"]
            )

            protein_entity = site_idx + 1
            ligand_entity = lig_entity_base + orig_idx + 1

            covalent_bonds.append({
                "entity1": str(protein_entity),
                "copy1": 1,
                "position1": str(cov_info["cys_pos"]),
                "atom1": "SG",
                "entity2": str(ligand_entity),
                "copy2": 1,
                "position2": "1",
                "atom2": attach_atom,
            })

        log.info(f"  {target.target_id}: added {len(covalent_bonds)} covalent bonds")

    if covalent_bonds:
        job["covalent_bonds"] = covalent_bonds

    output_path.write_text(json.dumps([job], indent=2) + "\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="protenix_input_preparation",
)
def main(cfg: DictConfig) -> None:
    from casp17_ligand.data.components import load_all_targets

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    series = cfg.get("series", "L1000")
    data_root_cfg = cfg.get("data_root", "data/casp16_data")
    data_root = root / data_root_cfg
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protonation = cfg.get("protonation", False)
    add_zn = cfg.get("add_zn", False)
    add_glycan = cfg.get("add_glycan", False)
    log.info(f"Loading {series} targets from {data_root} (protonation={protonation}, add_zn={add_zn}, add_glycan={add_glycan})")

    struct_only = cfg.get("struct_only", True)
    # `targets=` restricts prep to specific ids (competition: regenerate one
    # target without rewriting inputs for targets already running/submitted).
    # Unset/null = whole series (benchmark & eval mode).
    targets = load_all_targets(
        series, data_root, struct_only=struct_only, only=cfg.get("targets", None)
    )
    log.info(f"Preparing Protenix inputs for {len(targets)} targets → {output_dir}")

    # Resolve MSA/Template paths (convert relative → absolute)
    paired_msa = cfg.get("paired_msa_path", None)
    unpaired_msa = cfg.get("unpaired_msa_path", None)
    templates = cfg.get("templates_path", None)
    rna_unpaired_msa = cfg.get("rna_unpaired_msa_path", None)

    if paired_msa and not Path(paired_msa).is_absolute():
        paired_msa = str(root / paired_msa)
    if unpaired_msa and not Path(unpaired_msa).is_absolute():
        unpaired_msa = str(root / unpaired_msa)
    if templates and not Path(templates).is_absolute():
        templates = str(root / templates)
    if rna_unpaired_msa and not Path(rna_unpaired_msa).is_absolute():
        rna_unpaired_msa = str(root / rna_unpaired_msa)

    for target in targets:
        json_path = output_dir / f"{target.target_id}.json"
        create_protenix_json(
            target,
            json_path,
            paired_msa_path=paired_msa,
            unpaired_msa_path=unpaired_msa,
            templates_path=templates,
            protonation=protonation,
            add_zn=add_zn,
            add_glycan=add_glycan,
            rna_unpaired_msa_path=rna_unpaired_msa,
        )
        cov = " [COVALENT]" if target.target_id in COVALENT_TARGETS else ""
        extras = ""
        if add_zn:
            extras += " [+2ZN]"
        if add_glycan:
            extras += " [+NAG]"
        log.info(f"  Created: {json_path.name} ({target.num_ligands} ligands){cov}{extras}")

    log.info(f"Done. {len(targets)} JSON files written to {output_dir}")


if __name__ == "__main__":
    main()
