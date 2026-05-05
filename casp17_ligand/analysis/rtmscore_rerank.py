"""
RTMScore Re-ranking of Top-5 Cluster Representatives

对聚类产出的 top5 代表模型用 RTMScore 重新打分排序，
看能否提升 top1 lddt-pli。

用法:
  conda run --no-capture-output -n rtmscore bash -c '
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    python casp17_ligand/analysis/rtmscore_rerank.py \
      --cluster_csv outputs/ensemble/cluster_experiment_detail_latest.csv \
      --config "自适应: 40HA->0.70" \
      --output outputs/ensemble/rtmscore_rerank_top5.csv
  '
"""
import os
import sys

# Must set BABEL env BEFORE any openbabel import (happens transitively via RTMScore)
_conda_prefix = os.environ.get('CONDA_PREFIX', '')
if _conda_prefix:
    os.environ.setdefault('BABEL_LIBDIR', os.path.join(_conda_prefix, 'lib', 'openbabel', '3.1.0'))
    os.environ.setdefault('BABEL_DATADIR', os.path.join(_conda_prefix, 'share', 'openbabel', '3.1.0'))
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    lib_dir = os.path.join(_conda_prefix, 'lib')
    if lib_dir not in ld:
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{ld}" if ld else lib_dir

import csv
import glob
import json
import argparse
import tempfile
import traceback
from collections import defaultdict

import numpy as np

# Add RTMScore to path
RTMSCORE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'forks', 'RTMScore')
sys.path.insert(0, os.path.join(RTMSCORE_DIR, 'example'))
sys.path.insert(0, RTMSCORE_DIR)


def find_target_dir(target_id):
    """Find the target directory across datasets."""
    base = "/bmlfast/Lyuwei/0.Projects/CASP17_ligand/outputs/ensemble"
    # Priority order: struct > non-struct (struct has more methods)
    datasets = ["casp16_l1000", "casp16_l2000_struct", "casp16_l3000_struct", "casp16_l4000"]
    for d in datasets:
        path = os.path.join(base, d, "targets", target_id)
        if os.path.isdir(path):
            return path
    return None


def find_sdf_for_model(target_dir, model_name):
    """Find SDF file for a model in ranking_sucos/."""
    ranking_dir = os.path.join(target_dir, 'ranking_sucos')
    if not os.path.isdir(ranking_dir):
        return None
    # Pattern: {model_name}_rank*_sucos*_pb=*.sdf
    pattern = os.path.join(ranking_dir, f"{model_name}_rank*_sucos*_pb=*.sdf")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def find_protein_pdb(target_dir, target_id, model_name):
    """Find protein PDB in cif_converted/."""
    cif_dir = os.path.join(target_dir, 'cif_converted')
    if not os.path.isdir(cif_dir):
        return None
    # model_name = "af3_model25" -> need "L1001_af3_model_25_protein.pdb"
    # Parse: source = af3, num = 25
    parts = model_name.rsplit('_model', 1)
    if len(parts) == 2:
        source = parts[0]   # e.g. "af3"
        num = parts[1]      # e.g. "25"
        # Convention in cif_converted: {target}_{source}_model_{num}_protein.pdb
        pdb_name = f"{target_id}_{source}_model_{num}_protein.pdb"
        pdb_path = os.path.join(cif_dir, pdb_name)
        if os.path.exists(pdb_path):
            return pdb_path
    # Fallback: glob
    pattern = os.path.join(cif_dir, f"{target_id}_*_protein.pdb")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def combine_sdfs(sdf_paths, model_names, output_path):
    """Combine multiple SDF files into one, setting molecule names for identification.

    RTMScore uses _sdf_split -> MolFromMolBlock which reads the first line of each
    mol block as _Name. So we write SDF blocks manually with model_name as the title.
    """
    from rdkit import Chem
    written = 0
    with open(output_path, 'w') as fout:
        for sdf_path, model_name in zip(sdf_paths, model_names):
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            for mol in suppl:
                if mol is not None:
                    mol.SetProp("_Name", model_name)
                    block = Chem.MolToMolBlock(mol)
                    # Replace first line (title) with model_name
                    lines = block.split('\n')
                    lines[0] = model_name
                    fout.write('\n'.join(lines))
                    fout.write('\n$$$$\n')
                    written += 1
                    break
    return written


def extract_pocket_pdb(protein_pdb, ref_ligand_sdf, cutoff=10.0):
    """Extract pocket from protein PDB, return path to clean pocket PDB.

    Uses ProDy for selection, writes minimal ATOM-only PDB for RDKit compatibility.
    """
    import prody as pr
    from rdkit import Chem

    # Get ligand atom coordinates from SDF
    suppl = Chem.SDMolSupplier(ref_ligand_sdf, removeHs=False)
    mol = next(iter(suppl))
    if mol is None:
        raise ValueError(f"Cannot read ligand from {ref_ligand_sdf}")
    conf = mol.GetConformer()
    lig_coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])

    # Parse protein and find pocket residues
    prot = pr.parsePDB(protein_pdb)
    from scipy.spatial.distance import cdist
    dists = cdist(prot.getCoords(), lig_coords)
    pocket_mask = dists.min(axis=1) <= cutoff

    pocket_residues = set()
    chids = prot.getChids()
    resnums = prot.getResnums()
    for i in range(len(pocket_mask)):
        if pocket_mask[i]:
            pocket_residues.add((chids[i], resnums[i]))

    if not pocket_residues:
        raise ValueError("Empty pocket")

    # Write clean PDB (ATOM lines only, no CONECT/END issues)
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

    # Verify RDKit can read it
    test_mol = Chem.MolFromPDBFile(output_path, removeHs=True, sanitize=False)
    if test_mol is None:
        raise ValueError("RDKit cannot parse extracted pocket PDB")

    return output_path


def score_with_rtmscore(pocket_pdb, combined_sdf, model_path):
    """Run RTMScore scoring with pre-extracted pocket. Returns (ids, scores) or None."""
    try:
        import torch as th
        from rdkit import Chem
        from rtmscore import scoring

        # Load pocket with sanitize=False to handle non-standard residues
        pocket_mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=True, sanitize=False)
        if pocket_mol is None:
            raise ValueError(f"RDKit cannot parse pocket PDB: {pocket_pdb}")

        args = {
            "batch_size": 128,
            "dist_threhold": 5,
            "device": 'cuda' if th.cuda.is_available() else 'cpu',
            "num_workers": 0,
            "num_node_featsp": 41,
            "num_node_featsl": 41,
            "num_edge_featsp": 5,
            "num_edge_featsl": 10,
            "hidden_dim0": 128,
            "hidden_dim": 128,
            "n_gaussians": 10,
            "dropout_rate": 0.10,
        }
        # Pass Mol object directly -> bypasses load_mol and its sanitize issues
        ids, scores = scoring(
            prot=pocket_mol,
            lig=combined_sdf,
            modpath=model_path,
            gen_pocket=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            **args,
        )
        return ids, scores
    except Exception as e:
        print(f"  RTMScore error: {e}")
        traceback.print_exc()
        return None, None


def load_cluster_csv(csv_path, config_filter):
    """Load cluster experiment CSV, filter by config, return per-target data."""
    targets = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['config'] != config_filter:
                continue
            target = row['target']
            top5_reps = row.get('top5_reps', '')
            top5_lddts = row.get('top5_lddts', '')
            if not top5_reps:
                continue
            targets[target] = {
                'top1_rep': row['top1_rep'],
                'top1_lddt': float(row['top1_lddt']) if row['top1_lddt'] else None,
                'top5_best_lddt': float(row['top5_best_lddt']) if row['top5_best_lddt'] else None,
                'top5_reps': top5_reps.split(';'),
                'top5_lddts_str': top5_lddts,
                'top5_lddts': [float(x) if x != 'NA' else None for x in top5_lddts.split(';')],
                'top5_sdf_paths': row.get('top5_sdf_paths', '').split(';') if row.get('top5_sdf_paths') else [],
                'target_dir': row.get('target_dir', ''),
            }
    return targets


def process_target(target_id, target_data, rtmscore_model_path):
    """Process a single target: find files, score with RTMScore, re-rank."""
    # Use target_dir from CSV if available, otherwise search
    target_dir = target_data.get('target_dir') or find_target_dir(target_id)
    if not target_dir:
        return {'target': target_id, 'error': f'target dir not found'}

    top5_reps = target_data['top5_reps']
    top5_lddts = target_data['top5_lddts']
    csv_sdf_paths = target_data.get('top5_sdf_paths', [])

    # Find SDF files: prefer CSV paths, fallback to glob search
    sdf_paths = []
    valid_reps = []
    valid_lddts = []
    for i, (rep, lddt) in enumerate(zip(top5_reps, top5_lddts)):
        sdf = None
        if i < len(csv_sdf_paths) and csv_sdf_paths[i] and os.path.exists(csv_sdf_paths[i]):
            sdf = csv_sdf_paths[i]
        else:
            sdf = find_sdf_for_model(target_dir, rep)
        if sdf:
            sdf_paths.append(sdf)
            valid_reps.append(rep)
            valid_lddts.append(lddt)
        else:
            print(f"  WARNING: SDF not found for {target_id}/{rep}")

    if not sdf_paths:
        return {'target': target_id, 'error': 'no SDF files found'}

    # Use first rep's protein PDB
    protein_pdb = find_protein_pdb(target_dir, target_id, valid_reps[0])
    if not protein_pdb:
        return {'target': target_id, 'error': 'protein PDB not found'}

    # Reference ligand for pocket extraction = first SDF
    ref_ligand_sdf = sdf_paths[0]

    # Combine SDFs
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
        combined_sdf = tmp.name
    pocket_pdb = None

    try:
        n_written = combine_sdfs(sdf_paths, valid_reps, combined_sdf)
        if n_written == 0:
            return {'target': target_id, 'error': 'no valid molecules in SDFs'}

        # Extract pocket ourselves (bypass OpenBabel)
        pocket_pdb = extract_pocket_pdb(protein_pdb, ref_ligand_sdf, cutoff=10.0)

        # Score
        ids, scores = score_with_rtmscore(
            pocket_pdb, combined_sdf, rtmscore_model_path
        )

        if ids is None:
            return {'target': target_id, 'error': 'RTMScore scoring failed'}

        # Map scores back to model names
        # RTMScore returns ids as "{name}-{index}", strip the suffix
        score_map = {}
        for mol_id, score in zip(ids, scores):
            # "af3_model25-0" -> "af3_model25"
            base_name = mol_id.rsplit('-', 1)[0]
            score_map[base_name] = score

        # Build results with RTMScore scores
        ranked = []
        for rep, lddt in zip(valid_reps, valid_lddts):
            rtm_score = score_map.get(rep, None)
            ranked.append({
                'model': rep,
                'lddt': lddt,
                'rtmscore': rtm_score,
            })

        # Sort by RTMScore descending
        ranked_by_rtm = sorted(ranked, key=lambda x: x['rtmscore'] if x['rtmscore'] is not None else -999, reverse=True)

        reranked_top1 = ranked_by_rtm[0]

        return {
            'target': target_id,
            'error': None,
            'orig_top1_rep': target_data['top1_rep'],
            'orig_top1_lddt': target_data['top1_lddt'],
            'orig_top5_best_lddt': target_data['top5_best_lddt'],
            'reranked_top1_rep': reranked_top1['model'],
            'reranked_top1_lddt': reranked_top1['lddt'],
            'reranked_top1_rtmscore': reranked_top1['rtmscore'],
            'top5_reps': ';'.join(valid_reps),
            'top5_lddts': ';'.join(f"{l:.6f}" if l is not None else "NA" for l in valid_lddts),
            'top5_rtmscores': ';'.join(f"{r['rtmscore']:.4f}" if r['rtmscore'] is not None else "NA" for r in ranked),
            'top5_reranked_order': ';'.join(r['model'] for r in ranked_by_rtm),
        }
    finally:
        if os.path.exists(combined_sdf):
            os.unlink(combined_sdf)
        if pocket_pdb and os.path.exists(pocket_pdb):
            os.unlink(pocket_pdb)


def main():
    parser = argparse.ArgumentParser(description='RTMScore re-ranking of cluster top-5')
    parser.add_argument('--cluster_csv', required=True, help='Path to cluster_experiment_detail CSV')
    parser.add_argument('--config', default='自适应: 40HA->0.70', help='Config filter')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--rtmscore_model', default=None, help='RTMScore model path')
    parser.add_argument('--targets', nargs='*', default=None, help='Specific targets to process (for testing)')
    args = parser.parse_args()

    if args.rtmscore_model is None:
        args.rtmscore_model = os.path.join(RTMSCORE_DIR, 'trained_models', 'rtmscore_model1.pth')

    print(f"Loading cluster CSV: {args.cluster_csv}")
    print(f"Config filter: {args.config}")
    cluster_data = load_cluster_csv(args.cluster_csv, args.config)
    print(f"Loaded {len(cluster_data)} targets")

    if args.targets:
        cluster_data = {t: v for t, v in cluster_data.items() if t in args.targets}
        print(f"Filtered to {len(cluster_data)} targets: {list(cluster_data.keys())}")

    results = []
    successes = 0
    failures = 0

    for i, (target_id, target_data) in enumerate(sorted(cluster_data.items())):
        print(f"[{i+1}/{len(cluster_data)}] Processing {target_id}...", flush=True)
        result = process_target(target_id, target_data, args.rtmscore_model)
        results.append(result)

        if result.get('error'):
            print(f"  FAILED: {result['error']}")
            failures += 1
        else:
            orig = result['orig_top1_lddt'] or 0
            new = result['reranked_top1_lddt'] or 0
            delta = new - orig
            mark = "+" if delta > 0 else ""
            print(f"  orig_top1={orig:.4f} -> reranked_top1={new:.4f} ({mark}{delta:.4f})")
            successes += 1

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        'target', 'error',
        'orig_top1_rep', 'orig_top1_lddt', 'orig_top5_best_lddt',
        'reranked_top1_rep', 'reranked_top1_lddt', 'reranked_top1_rtmscore',
        'top5_reps', 'top5_lddts', 'top5_rtmscores', 'top5_reranked_order',
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\n{'='*80}")
    print(f"RTMScore Re-ranking Results")
    print(f"{'='*80}")
    print(f"Processed: {successes + failures}, Success: {successes}, Failed: {failures}")

    # Compute metrics
    valid_results = [r for r in results if not r.get('error')]
    if valid_results:
        orig_top1s = [r['orig_top1_lddt'] for r in valid_results if r['orig_top1_lddt'] is not None]
        reranked_top1s = [r['reranked_top1_lddt'] for r in valid_results if r['reranked_top1_lddt'] is not None]
        orig_top5_bests = [r['orig_top5_best_lddt'] for r in valid_results if r['orig_top5_best_lddt'] is not None]

        print(f"\nOriginal  Top-1 Mean lDDT: {np.mean(orig_top1s):.4f} (n={len(orig_top1s)})")
        print(f"Reranked  Top-1 Mean lDDT: {np.mean(reranked_top1s):.4f} (n={len(reranked_top1s)})")
        print(f"Original  Top-5 Best lDDT: {np.mean(orig_top5_bests):.4f} (n={len(orig_top5_bests)})")
        print(f"Delta (reranked - orig):   {np.mean(reranked_top1s) - np.mean(orig_top1s):+.4f}")

        # Count improvements
        improved = sum(1 for r in valid_results
                      if r['reranked_top1_lddt'] is not None and r['orig_top1_lddt'] is not None
                      and r['reranked_top1_lddt'] > r['orig_top1_lddt'])
        same = sum(1 for r in valid_results
                  if r['reranked_top1_lddt'] is not None and r['orig_top1_lddt'] is not None
                  and r['reranked_top1_lddt'] == r['orig_top1_lddt'])
        worse = sum(1 for r in valid_results
                   if r['reranked_top1_lddt'] is not None and r['orig_top1_lddt'] is not None
                   and r['reranked_top1_lddt'] < r['orig_top1_lddt'])
        print(f"\nImproved: {improved}, Same: {same}, Worse: {worse}")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
