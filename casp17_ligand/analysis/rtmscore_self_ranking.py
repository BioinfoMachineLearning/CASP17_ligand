"""RTMScore self-ranking: score ALL models per method per target.

Produces per-target JSON caches: {model_index: rtmscore}
that confidence_metric_analysis.py can consume as metric="rtmscore".

Usage:
  conda run --no-capture-output -n rtmscore bash -c '
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    python casp17_ligand/analysis/rtmscore_self_ranking.py \
      --datasets casp16_l1000 casp16_l2000_struct casp16_l3000_struct casp16_l4000
  '
"""
import os
import sys
import json
import glob
import re
import argparse
import tempfile
import traceback

# Set BABEL env before any openbabel import
_conda_prefix = os.environ.get('CONDA_PREFIX', '')
if _conda_prefix:
    os.environ.setdefault('BABEL_LIBDIR', os.path.join(_conda_prefix, 'lib', 'openbabel', '3.1.0'))
    os.environ.setdefault('BABEL_DATADIR', os.path.join(_conda_prefix, 'share', 'openbabel', '3.1.0'))
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    lib_dir = os.path.join(_conda_prefix, 'lib')
    if lib_dir not in ld:
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{ld}" if ld else lib_dir

import numpy as np

RTMSCORE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'forks', 'RTMScore')
sys.path.insert(0, os.path.join(RTMSCORE_DIR, 'example'))
sys.path.insert(0, RTMSCORE_DIR)

METHODS = ["af3", "boltz2", "protenix", "seedfold"]


def extract_pocket_pdb(protein_pdb, ref_ligand_sdf, cutoff=10.0):
    """Extract pocket from protein PDB, return path to clean pocket PDB."""
    import prody as pr
    from rdkit import Chem
    from scipy.spatial.distance import cdist

    suppl = Chem.SDMolSupplier(ref_ligand_sdf, removeHs=False)
    mol = next(iter(suppl))
    if mol is None:
        return None
    conf = mol.GetConformer()
    lig_coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])

    prot = pr.parsePDB(protein_pdb)
    dists = cdist(prot.getCoords(), lig_coords)
    pocket_mask = dists.min(axis=1) <= cutoff

    pocket_residues = set()
    chids = prot.getChids()
    resnums = prot.getResnums()
    for i in range(len(pocket_mask)):
        if pocket_mask[i]:
            pocket_residues.add((chids[i], resnums[i]))

    if not pocket_residues:
        return None

    output_path = tempfile.mktemp(suffix='_pocket.pdb')
    with open(protein_pdb) as fin, open(output_path, 'w') as fout:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')):
                chain = line[21]
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                if (chain, resnum) in pocket_residues:
                    fout.write(line)
        fout.write("END\n")
    return output_path


def score_models_rtmscore(pocket_mol, sdf_paths, model_names, model_path):
    """Score multiple ligand SDFs against a pocket. Returns {name: score}."""
    import torch as th
    from rdkit import Chem

    # Combine SDFs
    combined = tempfile.mktemp(suffix='.sdf')
    written = 0
    try:
        with open(combined, 'w') as fout:
            for sdf_path, name in zip(sdf_paths, model_names):
                suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                for mol in suppl:
                    if mol is not None:
                        mol.SetProp("_Name", name)
                        block = Chem.MolToMolBlock(mol)
                        lines = block.split('\n')
                        lines[0] = name
                        fout.write('\n'.join(lines) + '\n$$$$\n')
                        written += 1
                        break

        if written == 0:
            return {}

        from rtmscore import scoring
        args = {
            "batch_size": 128, "dist_threhold": 5,
            "device": 'cuda' if th.cuda.is_available() else 'cpu',
            "num_workers": 0, "num_node_featsp": 41, "num_node_featsl": 41,
            "num_edge_featsp": 5, "num_edge_featsl": 10,
            "hidden_dim0": 128, "hidden_dim": 128,
            "n_gaussians": 10, "dropout_rate": 0.10,
        }
        ids, scores = scoring(
            prot=pocket_mol, lig=combined, modpath=model_path,
            gen_pocket=False, explicit_H=False, use_chirality=True,
            parallel=False, **args,
        )
        result = {}
        for mol_id, score in zip(ids, scores):
            base_name = mol_id.rsplit('-', 1)[0]
            result[base_name] = float(score)
        return result
    finally:
        if os.path.exists(combined):
            os.unlink(combined)


def process_target_method(target_dir, target_id, method, model_path):
    """Score all models of one method for one target. Returns {model_idx: score}."""
    from rdkit import Chem

    ranking_dir = os.path.join(target_dir, 'ranking_sucos')
    cif_dir = os.path.join(target_dir, 'cif_converted')
    if not os.path.isdir(ranking_dir) or not os.path.isdir(cif_dir):
        return {}

    # Find all SDFs for this method
    pattern = re.compile(rf'^{re.escape(method)}_model(\d+)_rank\d+_orig\d+_sucos[\d.]+_pb=(True|False)\.sdf$')
    sdf_map = {}  # {model_idx: sdf_path}
    for fname in os.listdir(ranking_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            if idx not in sdf_map:  # Take first match per model
                sdf_map[idx] = os.path.join(ranking_dir, fname)

    if not sdf_map:
        return {}

    # Find a protein PDB and reference ligand for pocket extraction
    # Use the first available model's protein
    first_idx = min(sdf_map.keys())
    prot_pdb = None
    # Try direct name (single-chain targets)
    candidate = os.path.join(cif_dir, f"{target_id}_{method}_model_{first_idx}_protein.pdb")
    if os.path.exists(candidate):
        prot_pdb = candidate
    if not prot_pdb:
        # Fallback: search in cif_converted and subdirs (multi-chain targets use lig_X/ subdirs)
        pdbs = glob.glob(os.path.join(cif_dir, f"{target_id}_*_protein.pdb")) + \
               glob.glob(os.path.join(cif_dir, "*", f"{target_id}_*_protein.pdb"))
        if pdbs:
            prot_pdb = pdbs[0]
    if not prot_pdb:
        return {}

    ref_sdf = sdf_map[first_idx]

    # Extract pocket
    pocket_pdb = extract_pocket_pdb(prot_pdb, ref_sdf, cutoff=10.0)
    if not pocket_pdb:
        return {}

    try:
        pocket_mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=True, sanitize=False)
        if pocket_mol is None:
            return {}

        # Score in batches of 50
        sorted_indices = sorted(sdf_map.keys())
        sdf_paths = [sdf_map[i] for i in sorted_indices]
        model_names = [f"{method}_model{i}" for i in sorted_indices]

        scores = score_models_rtmscore(pocket_mol, sdf_paths, model_names, model_path)

        # Map back to model indices
        result = {}
        for i, idx in enumerate(sorted_indices):
            name = f"{method}_model{idx}"
            if name in scores:
                result[idx] = scores[name]
        return result
    except Exception as e:
        print(f"    Error scoring {target_id}/{method}: {e}")
        return {}
    finally:
        if pocket_pdb and os.path.exists(pocket_pdb):
            os.unlink(pocket_pdb)


def main():
    parser = argparse.ArgumentParser(description='RTMScore self-ranking for all models')
    parser.add_argument('--datasets', nargs='+',
                        default=['casp16_l1000', 'casp16_l2000_struct',
                                 'casp16_l3000_struct', 'casp16_l4000'])
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--ensemble_dir', default='outputs/ensemble')
    parser.add_argument('--rtmscore_model', default=None)
    args = parser.parse_args()

    if args.rtmscore_model is None:
        args.rtmscore_model = os.path.join(RTMSCORE_DIR, 'trained_models', 'rtmscore_model1.pth')

    for dataset in args.datasets:
        ens_base = os.path.join(args.ensemble_dir, dataset)
        targets_dir = os.path.join(ens_base, 'targets')
        if not os.path.isdir(targets_dir):
            print(f"Skipping {dataset}: no targets dir")
            continue

        # Output dir for caches
        cache_dir = os.path.join(ens_base, 'rtmscore_self_ranking')
        os.makedirs(cache_dir, exist_ok=True)

        targets = sorted([d for d in os.listdir(targets_dir)
                         if os.path.isdir(os.path.join(targets_dir, d))])

        from casp17_ligand.data.components.target_data import BLACKLISTED_TARGETS
        targets = [t for t in targets if t not in BLACKLISTED_TARGETS]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({len(targets)} targets)")
        print(f"{'='*60}")

        for ti, target_id in enumerate(targets):
            target_dir = os.path.join(targets_dir, target_id)
            print(f"[{ti+1}/{len(targets)}] {target_id}", end="", flush=True)

            for method in args.methods:
                cache_path = os.path.join(cache_dir, f"{target_id}_{method}.json")
                if os.path.exists(cache_path):
                    print(f" {method}:cached", end="")
                    continue

                scores = process_target_method(target_dir, target_id, method, args.rtmscore_model)
                if scores:
                    with open(cache_path, 'w') as f:
                        json.dump(scores, f)
                    print(f" {method}:{len(scores)}", end="")
                else:
                    # Write empty cache to avoid re-processing
                    with open(cache_path, 'w') as f:
                        json.dump({}, f)
                    print(f" {method}:0", end="")

            print()

        print(f"Caches saved to: {cache_dir}")


if __name__ == '__main__':
    main()
