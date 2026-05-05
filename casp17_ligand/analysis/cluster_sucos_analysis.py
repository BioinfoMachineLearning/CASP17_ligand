import os
import sys
import json
import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from rdkit import Chem

def get_max_heavy_atoms(target):
    l1000_dir = '/bmlfast/Lyuwei/0.Projects/CASP17_ligand/data/casp16_data/smiles/L1000'
    l2000_dir = '/bmlfast/Lyuwei/0.Projects/CASP17_ligand/data/casp16_data/smiles/L2000'
    l3000_dir = '/bmlfast/Lyuwei/0.Projects/CASP17_ligand/data/casp16_data/smiles/L3000'
    if target.startswith('L1'):
        tsv_path = os.path.join(l1000_dir, f"{target}.tsv")
    elif target.startswith('L2'):
        tsv_path = os.path.join(l2000_dir, f"{target}.tsv")
    elif target.startswith('L3'):
        tsv_path = os.path.join(l3000_dir, f"{target}.tsv")
    else:
        return 0
        
    if not os.path.exists(tsv_path):
        return 0
        
    try:
        tsv_df = pd.read_csv(tsv_path, sep='\t')
        max_ha = 0
        for smiles in tsv_df['SMILES']:
            for s in str(smiles).split('.'):
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    ha = mol.GetNumHeavyAtoms()
                    if ha > max_ha:
                        max_ha = ha
        return max_ha
    except:
        return 0

def process_target(target_dir):
    target_id = os.path.basename(target_dir)
    sucos_path = os.path.join(target_dir, 'pairwise_sucos_cache.json')
    
    score_path = os.path.join(target_dir, 'score_cache_docker.json')
    if not os.path.exists(score_path):
        score_path = os.path.join(target_dir, 'score_cache.json')
        
    if not os.path.exists(sucos_path) or not os.path.exists(score_path):
        return {'target': target_id, 'error': f"Missing json files in {target_dir}"}
        
    try:
        with open(sucos_path, 'r') as f:
            sucos_data = json.load(f)
        with open(score_path, 'r') as f:
            score_data = json.load(f)
    except Exception as e:
        return {'target': target_id, 'error': f"JSON parsing error: {str(e)}"}

    # Clustering
    models = list(sucos_data.keys())
    if not models:
        return {'target': target_id, 'error': "No models found in sucos cache"}
        
    max_ha = get_max_heavy_atoms(target_id)
    if max_ha == 0:
        threshold = 0.8
    elif max_ha <= 20:
        threshold = 0.8
    else:
        threshold = 0.8 - ((max_ha - 20) / 50.0) * 0.2
        if threshold < 0.4:
            threshold = 0.4

    neighbors = {m: set([m]) for m in models}  # include self
    for mA in models:
        for mB, score in sucos_data[mA].items():
            if score >= threshold:
                neighbors[mA].add(mB)
                # neighbors_B will just be added in the outer loop for mA=mB symmetrically
                
    unassigned = set(models)
    clusters = []
    
    while unassigned:
        best_center = None
        max_neighbors = -1
        
        for m in unassigned:
            # Count how many of its neighbors are STILL unassigned
            valid_neighbors = len(neighbors[m].intersection(unassigned))
            if valid_neighbors > max_neighbors:
                max_neighbors = valid_neighbors
                best_center = m
                
        # Form a cluster
        cluster_members = neighbors[best_center].intersection(unassigned)
        clusters.append({
            'center': best_center,
            'members': cluster_members
        })
        
        # Remove assigned members
        unassigned -= cluster_members
        
    # Sort clusters by size (descending)
    clusters.sort(key=lambda x: len(x['members']), reverse=True)
    
    # Model to cluster index mapping (1-based index)
    model_to_cluster = {}
    for i, c in enumerate(clusters):
        for m in c['members']:
            model_to_cluster[m] = i + 1
            
    # Extract info
    num_clusters = len(clusters)
    
    top5_sizes = []
    top5_center_lddt = []
    
    for i in range(min(5, num_clusters)):
        c = clusters[i]
        top5_sizes.append(len(c['members']))
        
        c_model = c['center']
        metrics = score_data.get(c_model, {})
        lddt = metrics.get('lddt_pli', None) if isinstance(metrics, dict) else None
        top5_center_lddt.append(lddt)
        
    # Global top 10 models for this target
    global_models = []
    for m, metrics in score_data.items():
        if isinstance(metrics, dict) and metrics.get('lddt_pli') is not None:
            if m in model_to_cluster:
                global_models.append((m, metrics['lddt_pli']))
                
    # Sort by lddt_pli descending
    global_models.sort(key=lambda x: x[1], reverse=True)
    top10_models = global_models[:10]
    
    top10_info = []
    for m, lddt in top10_models:
        top10_info.append(model_to_cluster[m])
        
    if global_models:
        overall_best_lddt = global_models[0][1]
    else:
        overall_best_lddt = None
        
    return {
        'target': target_id,
        'num_models': len(models),
        'num_clusters': num_clusters,
        'top5_sizes': top5_sizes,
        'top5_center_lddt': top5_center_lddt,
        'top10_lddt_clusters': top10_info,
        'overall_best_lddt': overall_best_lddt,
        'adaptive_threshold': threshold,
        'max_heavy_atoms': max_ha
    }

def main():
    base_dir = "/bmlfast/Lyuwei/0.Projects/CASP17_ligand/outputs/ensemble"
    # Match explicitly the desired directories without _plddt etc.
    dataset_dirs = ["casp16_l1000", "casp16_l2000", "casp16_l2000_struct", "casp16_l3000", "casp16_l3000_struct"]
    
    target_dirs = []
    for d in dataset_dirs:
        matched = glob.glob(os.path.join(base_dir, d, "targets", "*"))
        target_dirs.extend(matched)
        
    # Blacklist defined targets
    from casp17_ligand.data.components.target_data import BLACKLISTED_TARGETS
    target_dirs = [d for d in target_dirs if os.path.basename(d) not in BLACKLISTED_TARGETS]
    
    if not target_dirs:
        print(f"No valid target directories found in {base_dir}")
        return
        
    results = []
    print(f"Found {len(target_dirs)} target directories. Processing with multi-threading...")
    
    # ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_target, d) for d in target_dirs]
        for idx, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                if 'error' in res:
                    print(f"Skipping/Error processing {res['target']}: {res['error']}")
                else:
                    results.append(res)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(target_dirs)} targets...")
                
    if not results:
        print("No valid results collected.")
        return
        
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('target')
    
    # Display some summary
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS SUMMARY (Top 3 Targets Preview)")
    print("="*80)
    
    preview_df = df.head(min(3, len(df)))
    for _, row in preview_df.iterrows():
        print(f"Target {row['target']} ({row['num_models']} models):")
        print(f"  Total clusters: {row['num_clusters']}")
        print(f"  Top 5 cluster sizes: {row['top5_sizes']}")
        lddts = [f"{l:.4f}" if l is not None else "N/A" for l in row['top5_center_lddt']]
        print(f"  Top 5 center lddt_pli: {lddts}")
        print(f"  Top 10 overall models belong to clusters: {row['top10_lddt_clusters']}")
        print(f"-"*40)
        
    # Save to CSV
    csv_path = os.path.join(base_dir, f"cluster_analysis_summary_adaptive.csv")
    
    # Flatten lists for CSV
    df_export = df.copy()
    df_export['top5_sizes'] = df_export['top5_sizes'].apply(lambda x: ','.join(map(str, x)))
    df_export['top5_center_lddt'] = df_export['top5_center_lddt'].apply(lambda x: ','.join([f"{l:.4f}" if l is not None else "N/A" for l in x]))
    df_export['top10_lddt_clusters'] = df_export['top10_lddt_clusters'].apply(lambda x: ','.join(map(str, x)))
    
    df_export.to_csv(csv_path, index=False)
    print(f"\nSuccessfully processed {len(results)} targets.")
    print(f"Full summary saved to {csv_path}")

    # Calculate average of Top 1 center lddt_pli
    top1_centers = []
    for top5_lists in df['top5_center_lddt']:
        if len(top5_lists) > 0 and top5_lists[0] is not None:
            top1_centers.append(top5_lists[0])
            
    if top1_centers:
        avg_top1_lddt = sum(top1_centers) / len(top1_centers)
        print("="*80)
        print(f"OVERALL METRIC: Average Top-1 Cluster Center lddt_pli = {avg_top1_lddt:.4f} (across {len(top1_centers)} valid targets)")
        print("="*80)
    else:
        print("Could not compute Average Top-1 lddt_pli: No valid scores found.")

if __name__ == '__main__':
    main()
