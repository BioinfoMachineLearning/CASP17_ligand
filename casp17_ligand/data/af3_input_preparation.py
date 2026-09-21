"""Prepare AlphaFold3 input JSON files using shared target data loading.

Supports:
  - Standard SMILES-based ligand input
  - Covalent ligands via userCCD + bondedAtomPairs (required by AF3 for bonds)
  - Homodimer targets (e.g., L4000 MPro)
  - MSA injection from pre-computed data pipeline output
"""

import json
import logging
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Covalent inhibitor definitions: target_id -> {smarts, cys_position}
# The SMARTS first atom is the attachment point to Cys SG.
COVALENT_TARGETS = {
    "L4003": {"smarts": "[cX3][nX2][cX3][nX2]", "cys_pos": 145},
    "L4013": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
    "L4019": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
    "L4023": {"smarts": "[C](=[N])[c]", "cys_pos": 145},
}


def load_msa_from_ref(ref_data_json: str) -> dict:
    """Extract protein MSA fields from a completed AF3 data JSON.

    AF3 stores MSA/templates in its *_data.json output after running the data
    pipeline. This function extracts them so they can be injected into new
    input JSONs (enabling --norun_data_pipeline for same-protein targets).
    """
    with open(ref_data_json) as f:
        data = json.load(f)
    for seq in data["sequences"]:
        if "protein" in seq:
            p = seq["protein"]
            return {
                "unpairedMsa": p.get("unpairedMsa"),
                "pairedMsa": p.get("pairedMsa"),
                "templates": p.get("templates"),
            }
    raise ValueError(f"No protein entity found in {ref_data_json}")


def smiles_to_user_ccd(smiles: str, comp_id: str) -> tuple[str, dict[int, str]]:
    """Generate AF3 userCCD mmCIF string from SMILES using RDKit.

    Returns:
        (ccd_string, atom_idx_to_name): CCD mmCIF string and mapping from
            RDKit atom index to CCD atom name.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=1000)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=5000,
                              useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # Assign unique atom names: element + index (e.g., C01, N02, O03)
    atom_idx_to_name = {}
    elem_count = {}
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        elem_count[elem] = elem_count.get(elem, 0) + 1
        name = f"{elem}{elem_count[elem]:02d}"
        atom_idx_to_name[atom.GetIdx()] = name

    # Build CCD mmCIF
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.ExactMolWt(mol)
    lines = [
        f"data_{comp_id}",
        "#",
        f"_chem_comp.id {comp_id}",
        f"_chem_comp.name '{comp_id}'",
        "_chem_comp.type non-polymer",
        f"_chem_comp.formula '{formula}'",
        "_chem_comp.mon_nstd_parent_comp_id ?",
        "_chem_comp.pdbx_synonyms ?",
        f"_chem_comp.formula_weight {mw:.3f}",
        "#",
        "loop_",
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_leaving_atom_flag",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    ]

    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_idx_to_name[idx]
        elem = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        if conf:
            pos = conf.GetAtomPosition(idx)
            x, y, z = pos.x, pos.y, pos.z
        else:
            x, y, z = 0.0, 0.0, 0.0
        lines.append(f"{comp_id} {name} {elem} {charge} N {x:.3f} {y:.3f} {z:.3f}")

    lines.extend(["#", "loop_",
                   "_chem_comp_bond.atom_id_1",
                   "_chem_comp_bond.atom_id_2",
                   "_chem_comp_bond.value_order",
                   "_chem_comp_bond.pdbx_aromatic_flag"])

    bond_order_map = {
        Chem.BondType.SINGLE: "SING",
        Chem.BondType.DOUBLE: "DOUB",
        Chem.BondType.TRIPLE: "TRIP",
        Chem.BondType.AROMATIC: "SING",  # aromatic flag handles this
    }
    for bond in mol.GetBonds():
        a1 = atom_idx_to_name[bond.GetBeginAtomIdx()]
        a2 = atom_idx_to_name[bond.GetEndAtomIdx()]
        bt = bond_order_map.get(bond.GetBondType(), "SING")
        arom = "Y" if bond.GetIsAromatic() else "N"
        lines.append(f"{a1} {a2} {bt} {arom}")

    lines.append("#")
    return "\n".join(lines), atom_idx_to_name


def find_covalent_atom_name(smiles: str, smarts: str, atom_idx_to_name: dict[int, str]) -> str:
    """Find the CCD atom name for the SMARTS attachment point.

    The SMARTS first atom (index 0 in match tuple) is the attachment point.
    We match on the molecule WITHOUT explicit Hs to get canonical heavy-atom indices,
    then map to the atom names from smiles_to_user_ccd (which includes Hs).
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

    # First atom of first match is the attachment point
    attach_idx = matches[0][0]

    # Map from no-H mol atom idx to with-H mol atom idx
    # smiles_to_user_ccd adds Hs, so indices shift. We need to find the
    # corresponding atom in the H-added mol.
    mol_h = Chem.AddHs(mol)
    # Heavy atom indices in mol_h correspond to the original indices
    # because AddHs appends Hs at the end
    atom_name = atom_idx_to_name[attach_idx]
    log.info(f"  Covalent attachment: SMARTS '{smarts}' → atom {attach_idx} → {atom_name}")
    return atom_name


def create_af3_json(
    target,  # TargetData
    output_path: Path,
    num_seeds: int = 5,
    msa_data: dict | None = None,
    rna_unpaired_msa: str | None = None,
) -> None:
    """Write a single AF3 input JSON from a TargetData object.

    For RNA chains: by default we emit NO `unpairedMsa` field, which lets AF3's
    built-in data pipeline run a real Rfam / RNAcentral search at inference time
    (single-sequence baseline is significantly weaker for RNA structure
    prediction). To inject a precomputed a3m string (e.g. reused from a previous
    `--norun_inference` run), pass `rna_unpaired_msa`.

    NOTE: setting `rna_unpaired_msa = ">query\\n{seq}\\n"` (a self-only dummy)
    explicitly DISABLES AF3 MSA search — only do this if you know that's what
    you want.
    """
    sequences = []
    bonded_atom_pairs = []
    user_ccd_parts = []

    cov_info = COVALENT_TARGETS.get(target.target_id)
    protein_chain_ids = list(target.protein_sequences.keys())

    # Write all polymer chains (protein or RNA).
    is_rna = target.entity_type == "rna"
    chain_key = "rna" if is_rna else "protein"
    for chain_id, seq in target.protein_sequences.items():
        chain_entry = {"id": chain_id, "sequence": seq}
        if is_rna:
            if rna_unpaired_msa is not None:
                chain_entry["unpairedMsa"] = rna_unpaired_msa
            # else: omit — AF3 data pipeline will search RNA MSAs
        elif msa_data:
            chain_entry.update(msa_data)
        sequences.append({chain_key: chain_entry})

    # Write all ligands (IDs continue after protein chain IDs)
    start_char = chr(ord('A') + len(target.protein_sequences))
    lig_chain_ids = []
    for i, lig in enumerate(target.ligands):
        lig_id = chr(ord(start_char) + i)
        lig_chain_ids.append(lig_id)

        is_cov_lig = cov_info and lig.name == "LIG"

        if is_cov_lig:
            # Use userCCD for covalent ligands (AF3 requires atom names for bonds)
            comp_id = f"LIG-{lig_id}"
            ccd_str, atom_idx_to_name = smiles_to_user_ccd(lig.smiles, comp_id)
            user_ccd_parts.append(ccd_str)
            sequences.append({"ligand": {"id": lig_id, "ccdCodes": [comp_id]}})

            # Find attachment atom and create bond to Cys SG
            attach_atom = find_covalent_atom_name(
                lig.smiles, cov_info["smarts"], atom_idx_to_name
            )
            # Bond to matching protein chain (ligand i pairs with protein chain i if dimer)
            prot_chain = protein_chain_ids[i % len(protein_chain_ids)]
            bonded_atom_pairs.append([
                [prot_chain, cov_info["cys_pos"], "SG"],
                [lig_id, 1, attach_atom],
            ])
        else:
            sequences.append({"ligand": {"id": lig_id, "smiles": lig.smiles}})

    input_json = {
        "name": target.target_id,
        "modelSeeds": list(range(1, num_seeds + 1)),
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }

    if bonded_atom_pairs:
        input_json["bondedAtomPairs"] = bonded_atom_pairs

    if user_ccd_parts:
        input_json["userCCD"] = "\n".join(user_ccd_parts)

    output_path.write_text(json.dumps(input_json, indent=2))


@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="af3_input_preparation")
def main(cfg: DictConfig) -> None:
    from casp17_ligand.data.components import load_all_targets
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    series = cfg.get("series", "L1000")
    data_root_cfg = cfg.get("data_root", "data/casp16_data")
    data_root = root / data_root_cfg
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    struct_only = cfg.get("struct_only", True)
    log.info(f"Loading {series} targets from {data_root}")

    # `targets=` restricts prep to specific ids (competition: regenerate one
    # target without rewriting inputs for targets already running/submitted).
    # Unset/null = whole series (benchmark & eval mode).
    targets = load_all_targets(
        series, data_root, struct_only=struct_only, only=cfg.get("targets", None)
    )
    log.info(f"Preparing AF3 inputs for {len(targets)} targets → {output_dir}")

    # Load MSA from reference if provided
    msa_data = None
    ref_data_json = cfg.get("ref_data_json", None)
    if ref_data_json:
        msa_data = load_msa_from_ref(ref_data_json)
        log.info(f"Loaded MSA from {ref_data_json}")

    num_seeds = cfg.get("num_seeds", 5)
    rna_unpaired_msa = cfg.get("rna_unpaired_msa", None)
    if rna_unpaired_msa:
        log.info("RNA `unpairedMsa` will be injected from cfg.rna_unpaired_msa "
                 "(disables AF3's RNA MSA search pipeline)")

    for target in targets:
        json_path = output_dir / f"{target.target_id}.json"
        create_af3_json(
            target,
            json_path,
            num_seeds=num_seeds,
            msa_data=msa_data,
            rna_unpaired_msa=rna_unpaired_msa,
        )
        cov = " [COVALENT]" if target.target_id in COVALENT_TARGETS else ""
        log.info(f"  Created: {json_path.name} ({target.num_ligands} ligands){cov}")

    log.info(f"Done. {len(targets)} JSON files written to {output_dir}")


if __name__ == "__main__":
    main()
