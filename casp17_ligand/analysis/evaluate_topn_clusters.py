"""
聚类阈值参数搜索实验 (全数据集: L1000 + L2000 + L3000 + L4000)

用法:
  # 固定阈值
  python evaluate_topn_clusters.py --fixed 0.9 0.8 0.7 0.6

  # 自适应阈值 (格式: ha_target:floor, 如 70:0.60 表示 70HA->0.60)
  python evaluate_topn_clusters.py --adaptive 70:0.60 40:0.70

  # 混合
  python evaluate_topn_clusters.py --fixed 0.8 --adaptive 70:0.60 40:0.70

  # 排除特定方法 (复用已有 pairwise cache, 过滤模型后重新聚类)
  python evaluate_topn_clusters.py --adaptive 40:0.70 --strategy consensus_pb \
    --exclude-methods boltz2 --output-suffix _without_boltz2

  # 指定数据集子集
  python evaluate_topn_clusters.py --adaptive 40:0.70 \
    --datasets casp16_l1000 casp16_l3000_struct

输出:
  - 终端汇总表
  - outputs/ensemble/cluster_experiment_detail_{timestamp}.csv (per-target 原始数据)
"""
from pathlib import Path
import os
import sys
import re
import json
import glob
import argparse
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from rdkit import Chem

_ROOT = Path(__file__).resolve().parents[2]


# ── 阈值计算 ──────────────────────────────────────────────────────────────────

def fixed_threshold(value, max_ha):
    return value


def adaptive_threshold(ha_target, floor, max_ha):
    """
    ≤20 HA: 0.80
    >20 HA: 线性下降, 到 ha_target 时达到 floor
    >ha_target: 保持 floor
    """
    start = 0.80
    if max_ha <= 20:
        return start
    slope = (start - floor) / (ha_target - 20)
    t = start - (max_ha - 20) * slope
    return max(t, floor)


# ── helpers ──────────────────────────────────────────────────────────────────

def get_max_heavy_atoms(target):
    """Largest ligand heavy-atom count for a target, 0 if the TSV cannot be found.

    Returning 0 is NOT harmless: `max_ha <= 2` routes process_target down the
    single-atom-ligand branch, which clusters on pairwise ligand RMSD with a
    2.0 A threshold instead of SuCOS. A drug-like ligand silently clustered by
    the wrong metric still produces plausible-looking clusters, so the mistake
    surfaces only as an odd `threshold` value in the output CSV.
    That happened once when CASP17_M was added, and again for CASP17_T2451 —
    both times because the series directory was hardcoded here. The series list
    is therefore GLOBBED rather than enumerated, so a new series works without
    touching this function.
    """
    root = str(_ROOT)
    candidate_paths = sorted(glob.glob(f'{root}/data/casp1[5-9]_data/smiles/*/{target}.tsv'))
    # keep the historical explicit paths as a fallback for non-standard layouts
    for series_dir in ['L1000', 'L2000', 'L3000', 'L4000']:
        candidate_paths.append(f'{root}/data/casp16_data/smiles/{series_dir}/{target}.tsv')
    for series_dir in ['CASP17_R', 'CASP17_T', 'CASP17_H', 'CASP17_M', 'CASP17_L']:
        candidate_paths.append(f'{root}/data/casp17_data/smiles/{series_dir}/{target}.tsv')
    candidate_paths.append(f'{root}/data/casp15_data/smiles/CASP15/{target}.tsv')
    candidate_paths.append(f'{root}/data/casp17_data/raw/{target}.smiles.txt')
    for tsv_path in candidate_paths:
        if not os.path.exists(tsv_path):
            continue
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
            pass
    return 0


def butina_cluster(models, pairwise_cache, threshold, mode='similarity'):
    """Butina clustering using pairwise scores.

    mode='similarity': neighbors if score >= threshold (SuCOS: higher = more similar)
    mode='distance':   neighbors if score <= threshold (RMSD: lower = more similar)
    """
    neighbors = {m: {m} for m in models}
    for mA in models:
        cache_A = pairwise_cache.get(mA, {})
        for mB in models:
            if mA == mB:
                continue
            score = cache_A.get(mB, float('inf') if mode == 'distance' else float('-inf'))
            if mode == 'distance':
                if score <= threshold:
                    neighbors[mA].add(mB)
            else:
                if score >= threshold:
                    neighbors[mA].add(mB)

    unassigned = set(models)
    clusters = []
    while unassigned:
        # sorted() 保证 tie-breaking 确定性（按模型名字典序）
        best_center = max(sorted(unassigned), key=lambda m: len(neighbors[m] & unassigned))
        members_set = neighbors[best_center] & unassigned
        clusters.append({'center': best_center, 'members': sorted(members_set)})
        unassigned -= members_set

    clusters.sort(key=lambda c: len(c['members']), reverse=True)
    return clusters


# ── per-target processing ────────────────────────────────────────────────────

def compute_threshold_from_config(cfg, max_ha):
    """从可序列化的配置字典计算阈值

    adaptive 字段:
      ha_start, start: 起点 (默认 20, 0.80)
      ha_target, end:  终点 (定义斜率的参考点)
      floor:           实际下限 (默认=end, 设 None 表示不设下限继续线性延伸)
    """
    if cfg['type'] == 'fixed':
        return cfg['value']
    # adaptive
    start = cfg.get('start', 0.80)
    ha_start = cfg.get('ha_start', 20)
    end = cfg['end']
    if max_ha <= ha_start:
        return start
    slope = (start - end) / (cfg['ha_target'] - ha_start)
    t = start - (max_ha - ha_start) * slope
    floor = cfg.get('floor')
    if floor is not None:
        return max(t, floor)
    return max(t, 0.0)  # 安全下限


# Method -> best self-ranking metric, measured per method
SELFRANK_METRICS = {
    "af3":       "pocket_plddt_4.5",
    "boltz2":    "pair_iptm",
    "protenix":  "pocket_plddt_4.5",
    "seedfold":  "rtmscore",
}

SELFRANK_TOPN = 10  # model must be in top-N of its method's self-ranking


def _build_selfrank_top10(target_dir, target_id):
    """Build set of model names that are in their method's self-ranking top-N.

    Reuses collect_scores() from confidence_metric_analysis.py.
    Returns set like {'af3_model3', 'boltz2_model12', ...}.
    """
    from casp17_ligand.analysis.confidence_metric_analysis import collect_scores

    # Derive dataset name from target_dir:
    #   outputs/ensemble/casp16_l3000_struct/targets/L3001 -> casp16_l3000_struct
    parts = target_dir.replace('\\', '/').split('/')
    try:
        tidx = parts.index('targets')
        dataset = parts[tidx - 1]
    except ValueError:
        return set()

    methods_config = {
        "af3":       f"outputs/alphafold3/{dataset}",
        "boltz2":    f"outputs/boltz2/{dataset}",
        "protenix":  f"outputs/protenix/{dataset}",
        "seedfold":  f"outputs/seedfold/{dataset}",
    }
    input_dirs = {
        "af3":       f"data/test_cases/{dataset}/af3_inputs",
        "protenix":  f"data/test_cases/{dataset}/protenix_inputs",
        "seedfold":  f"data/test_cases/{dataset}/seedfold_inputs",
    }

    top10_set = set()
    for method, metric in SELFRANK_METRICS.items():
        method_dir = methods_config.get(method, "")
        input_dir = input_dirs.get(method, "")
        scores = collect_scores(metric, method, target_id, method_dir, input_dir)
        if not scores:
            continue
        # Sort by score descending, take top-N indices
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for model_idx, _ in ranked[:SELFRANK_TOPN]:
            top10_set.add(f"{method}_model{model_idx}")

    return top10_set


def parse_ranking_sucos(target_dir):
    """解析 ranking_sucos/ 文件名, 返回 {model_name: (sucos_score, pb_valid)}"""
    ranking_dir = os.path.join(target_dir, 'ranking_sucos')
    if not os.path.isdir(ranking_dir):
        return {}
    result = {}
    pattern = re.compile(
        r'^(.+?)_rank(\d+)_orig\d+_sucos([\d.]+)_pb=(True|False)\.sdf$'
    )
    for fname in os.listdir(ranking_dir):
        if not fname.endswith('.sdf'):
            continue
        m = pattern.match(fname)
        if m:
            model_name = m.group(1)
            sucos_score = float(m.group(3))
            pb_valid = m.group(4) == 'True'
            result[model_name] = (sucos_score, pb_valid)
    return result


SP3_MAX_ANGLE_DEG = 150.0
"""Angle above which an sp3 centre is considered collapsed rather than strained.

Calibrated 2026-07-29 by sweeping 125-165 deg over the 8 finished CASP17_T
targets (200 poses each). T2455's defect population is a clean second mode at
160-180 deg and its rejection count plateaus at exactly 75 for any threshold
>=145. Meanwhile the saturated polycyclic ligands (T2413, T2414) pile up at
125-140 deg -- genuine fused-ring strain, not collapse -- so a 125 deg cut would
have wrongly disqualified 170/200 and 182/200 of them. At 150 deg the whole
T2455 defect is caught while only 6 poses across the other 7 targets are
touched. Physically, 150 deg is 40 deg past tetrahedral: beyond any real strain.
"""


def check_sp3_geometry(sdf_path, max_angle=SP3_MAX_ANGLE_DEG):
    """Reject poses whose sp3 centres have been linearised.

    Returns (ok, worst_angle_deg). `ok` is False if any heavy-atom angle centred
    on an sp3 atom exceeds `max_angle` (tetrahedral is 109.5 deg).

    Why this exists on top of PoseBusters: PB's `bond_angles` check is really a
    check on 1-3 *distances* against RDKit distance-geometry bounds. When a bond
    collapses, the 1-3 distance shrinks with it, so an angle can open all the way
    to 180 deg while staying inside the distance bounds. T2455 (2026-07-29) hit
    exactly this: SeedFold rendered every Ar-CH2-NH3+ as a linear 1.14 A "nitrile"
    (179.2 deg) and PB's dock config passed all 50/50 -- its 1-3 distance was only
    5.3% past the upper bound, nowhere near the 25% tolerance. Tightening PB's
    threshold_bad_bond_length 0.25 -> 0.20 would have caught this particular pose,
    but only by 0.01 A of margin (1.143 A actual vs 1.153 A cutoff), and it would
    change PB behaviour for every target. This check is orthogonal and targets the
    failure mode directly.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolTransforms as _T
    except ImportError:
        return True, None
    mol = Chem.MolFromMolFile(sdf_path)
    if mol is None or mol.GetNumConformers() == 0:
        return True, None          # unparseable: leave the verdict to PB
    conf = mol.GetConformer()
    worst = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                worst = max(worst, _T.GetAngleDeg(conf, nbrs[i], atom.GetIdx(), nbrs[j]))
    return worst <= max_angle, worst


def build_geom_ok(target_dir, ranking_info):
    """{model_name: bool} sp3-geometry verdict for every model in ranking_sucos."""
    ranking_dir = os.path.join(target_dir, 'ranking_sucos')
    if not os.path.isdir(ranking_dir):
        return {}
    by_model = {}
    pattern = re.compile(r'^(.+?)_rank\d+_orig\d+_sucos[\d.]+_pb=(?:True|False)\.sdf$')
    for fname in os.listdir(ranking_dir):
        m = pattern.match(fname)
        if m and m.group(1) in ranking_info:
            by_model[m.group(1)] = os.path.join(ranking_dir, fname)
    return {name: check_sp3_geometry(path)[0] for name, path in by_model.items()}


def pick_consensus_pb_rep(cluster_members, ranking_info, selfrank_top10=None, geom_ok=None):
    """从类成员中选共识最高且通过PB的模型; 全不通过PB则取共识最高(不管PB).

    If selfrank_top10 is provided (set of model names), additionally require the
    candidate to be in its method's self-ranking top-10. Models not in top-10 are
    treated like non-PB: deferred and filled back only if not enough picks.

    If geom_ok is provided ({model: bool} from build_geom_ok), a candidate must
    ALSO pass the sp3-geometry check to count as valid. Clustering is unaffected:
    geometrically broken poses still vote for consensus, they just cannot be the
    face of a cluster. Clusters with no clean member fall back exactly as they do
    for PB (deferred, then filled back with a [PB-WARNING]).
    """
    def _valid(model, pb_valid):
        if not pb_valid:
            return False
        return True if geom_ok is None else geom_ok.get(model, True)

    pb_candidates = []
    all_candidates = []
    for m in cluster_members:
        if m in ranking_info:
            sucos_score, pb_valid = ranking_info[m]
            all_candidates.append((sucos_score, pb_valid, m))
            if _valid(m, pb_valid):
                pb_candidates.append((sucos_score, m))

    if selfrank_top10 is not None:
        # Further filter PB candidates by self-ranking top-10
        sr_pb = [(s, m) for s, m in pb_candidates if m in selfrank_top10]
        sr_all = [(s, pb, m) for s, pb, m in all_candidates if m in selfrank_top10]
        if sr_pb:
            sr_pb.sort(key=lambda x: x[0], reverse=True)
            return sr_pb[0][1], True
        if sr_all:
            # Among top-10 models, pick highest consensus (regardless of PB)
            sr_all.sort(key=lambda x: x[0], reverse=True)
            return sr_all[0][2], sr_all[0][1]
        # No top-10 model in this cluster — fall through to original logic

    if pb_candidates:
        pb_candidates.sort(key=lambda x: x[0], reverse=True)
        return pb_candidates[0][1], True
    elif all_candidates:
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        return all_candidates[0][2], False
    return None, False


def process_target(args):
    target_dir, config_name, config_dict, strategy = args[:4]
    exclude_methods = args[4] if len(args) > 4 else None
    target_id = os.path.basename(target_dir)

    max_ha = get_max_heavy_atoms(target_id)
    # Edge case: 1-2 atom ligands (ions, diatomic) — SuCOS feature_map is 0 for
    # these, making it unable to distinguish positions. Use RMSD-based clustering
    # with distance mode instead.
    single_atom = (max_ha <= 2)

    if single_atom:
        pairwise_path = os.path.join(target_dir, 'pairwise_rmsd_cache.json')
        cluster_mode = 'distance'
    else:
        pairwise_path = os.path.join(target_dir, 'pairwise_sucos_cache.json')
        cluster_mode = 'similarity'

    score_path = os.path.join(target_dir, 'score_cache_docker.json')
    if not os.path.exists(score_path):
        score_path = os.path.join(target_dir, 'score_cache.json')

    if not os.path.exists(pairwise_path):
        return None

    with open(pairwise_path) as f:
        pairwise_cache = json.load(f)
    score_data = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            score_data = json.load(f)

    # Filter out excluded methods
    if exclude_methods:
        def _is_excluded(model_name):
            return any(model_name.startswith(m + "_") for m in exclude_methods)
        pairwise_cache = {k: {k2: v2 for k2, v2 in v.items() if not _is_excluded(k2)}
                          for k, v in pairwise_cache.items() if not _is_excluded(k)}
        score_data = {k: v for k, v in score_data.items() if not _is_excluded(k)}

    lddt_map = {}
    for k, v in score_data.items():
        if isinstance(v, dict) and v.get('lddt_pli') is not None:
            lddt_map[k] = v['lddt_pli']

    models = list(pairwise_cache.keys())
    if not models:
        return None

    if single_atom:
        # RMSD clustering: fixed threshold of 2.0Å, use distance mode
        threshold = 2.0
        clusters = butina_cluster(models, pairwise_cache, threshold, mode='distance')
        # If fewer than 5 clusters, relax threshold (increase distance cutoff)
        orig_threshold = threshold
        while len(clusters) < 5 and threshold < 10.0:
            threshold += 0.5
            clusters = butina_cluster(models, pairwise_cache, threshold, mode='distance')
    elif config_dict.get('type') == 'maxclust':
        # New strategy: start at fixed threshold, lower until cluster count <= max_clusters
        start = config_dict.get('start', 0.80)
        max_clusters = config_dict.get('max_clusters', 50)
        step = config_dict.get('step', 0.05)
        floor = config_dict.get('floor', 0.30)

        threshold = start
        clusters = butina_cluster(models, pairwise_cache, threshold, mode='similarity')
        orig_threshold = threshold
        # Lower threshold until <= max_clusters, bounded by floor
        while len(clusters) > max_clusters and threshold > floor:
            threshold = max(threshold - step, floor)
            clusters = butina_cluster(models, pairwise_cache, threshold, mode='similarity')
        # If still fewer than 5 clusters at some low threshold, raise back up
        # (unlikely under maxclust logic but kept for safety)
        while len(clusters) < 5 and threshold < 0.99:
            if threshold < 0.90:
                threshold = min(threshold + 0.05, 0.90)
            else:
                threshold = min(threshold + 0.01, 0.99)
            clusters = butina_cluster(models, pairwise_cache, threshold, mode='similarity')
    else:
        threshold = compute_threshold_from_config(config_dict, max_ha)
        clusters = butina_cluster(models, pairwise_cache, threshold, mode='similarity')

        # 如果聚类不足 5 个，逐步提高阈值直到 >=5
        orig_threshold = threshold
        while len(clusters) < 5 and threshold < 0.99:
            if threshold < 0.90:
                threshold = min(threshold + 0.05, 0.90)
            else:
                threshold = min(threshold + 0.01, 0.99)
            clusters = butina_cluster(models, pairwise_cache, threshold, mode='similarity')

    if strategy == 'center':
        # 聚类中心
        top1_rep = clusters[0]['center'] if clusters else None
        top5_reps = [c['center'] for c in clusters[:5]]
    else:
        # consensus_pb / consensus_pb_selfrank
        ranking_info = parse_ranking_sucos(target_dir)

        # Build self-ranking top-10 set if needed
        selfrank_top10 = None
        if strategy == 'consensus_pb_selfrank':
            selfrank_top10 = _build_selfrank_top10(target_dir, target_id)

        # sp3-geometry gate on representatives only (clustering already done above)
        geom_ok = build_geom_ok(target_dir, ranking_info)
        n_geom_bad = sum(1 for v in geom_ok.values() if not v)
        if n_geom_bad:
            by_meth = {}
            for name, ok in geom_ok.items():
                if not ok:
                    by_meth[name.split('_model')[0]] = by_meth.get(name.split('_model')[0], 0) + 1
            print(f"  [GEOM-INFO] {target_id}: {n_geom_bad}/{len(geom_ok)} pose(s) have an sp3 "
                  f"centre wider than {SP3_MAX_ANGLE_DEG:.0f} deg (linearised/flattened) "
                  f"-> ineligible as cluster representatives: {by_meth}",
                  flush=True)

        # top-5 类别排序: size desc, 同 size 按类内最佳共识 desc
        def cluster_best_consensus(c):
            best = -1.0
            for m in c['members']:
                if m in ranking_info:
                    best = max(best, ranking_info[m][0])
            return best
        clusters.sort(key=lambda c: (len(c['members']), cluster_best_consensus(c)), reverse=True)

        # 选代表: 优先PB通过(+selfrank top10 if applicable), 不够再补回
        picks = []
        skipped = []           # (rep, cluster_size) for clusters whose best rep is PB-invalid
        no_pb_cluster_ids = [] # cluster index of clusters with NO PB-valid member
        idx = 0
        while len(picks) < 5 and idx < len(clusters):
            rep, pb_ok = pick_consensus_pb_rep(
                clusters[idx]['members'], ranking_info, selfrank_top10, geom_ok)
            if rep is not None:
                if pb_ok:
                    picks.append(rep)
                else:
                    skipped.append((rep, len(clusters[idx]['members'])))
                    no_pb_cluster_ids.append(idx + 1)
            idx += 1
        # 补回不通过PB的
        invalid_fillin = []
        for s, sz in skipped:
            if len(picks) >= 5:
                break
            picks.append(s)
            invalid_fillin.append((s, sz))

        if invalid_fillin:
            print(f"  [PB-WARNING] {target_id}: top-5 had to include {len(invalid_fillin)} "
                  f"PB-invalid pick(s) -> {[(r, f'cluster_size={sz}') for r, sz in invalid_fillin]}; "
                  f"clusters with no PB-valid candidate at ranks {no_pb_cluster_ids[:len(invalid_fillin)]}. "
                  f"Consider deeper relax (higher max_iter or fallback to FULL-relax).",
                  flush=True)
        elif skipped:
            print(f"  [PB-INFO] {target_id}: PB-fallback skipped {len(skipped)} cluster(s) "
                  f"(no valid candidate inside) but top-5 already filled from valid clusters. "
                  f"Affected cluster ranks: {no_pb_cluster_ids}.",
                  flush=True)

        top1_rep = picks[0] if picks else None
        top5_reps = picks

    # ── STRONG-CONSENSUS check: are top-5 poses actually distinct binding modes,
    # or all clones of the same pose (evaluator's threshold-raise fallback split
    # near-identical poses into pseudo-clusters)?  Compute ligand-centroid
    # dispersion AFTER receptor-CA alignment across top-5 picks.
    #
    # Guard: skip if <2 picks or SDF/PDB missing.
    try:
        import numpy as _np
        from Bio import PDB as _PDB
        from Bio.PDB.Superimposer import Superimposer as _Sup
        from rdkit import Chem as _Chem
        import glob as _glob
        ranking_dir_local = os.path.join(target_dir, 'ranking_sucos')
        if len(top5_reps) >= 2 and os.path.isdir(ranking_dir_local):
            parser = _PDB.PDBParser(QUIET=True)
            def _load(rep):
                sdfs = _glob.glob(os.path.join(ranking_dir_local, f'{rep}_rank*_pb=*.sdf'))
                pdbs = _glob.glob(os.path.join(ranking_dir_local, f'{rep}_rank*.pdb'))
                if not sdfs or not pdbs: return None, None, None
                st = parser.get_structure('X', pdbs[0])
                ca = [r["CA"] for r in st.get_residues() if r.id[0]==" " and "CA" in r]
                m = _Chem.MolFromMolFile(sdfs[0], removeHs=True)
                if m is None: return None, None, None
                c = m.GetConformer()
                xyz = _np.array([(c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z)
                                 for i in range(m.GetNumAtoms())])
                return ca, xyz, m.GetNumAtoms()
            ref_ca, ref_lig, _ = _load(top5_reps[0])
            if ref_ca is not None:
                sup = _Sup()
                centers = [ref_lig.mean(axis=0)]
                for rep in top5_reps[1:]:
                    ca, lig, _ = _load(rep)
                    if ca is None: continue
                    n_ca = min(len(ref_ca), len(ca))
                    try:
                        sup.set_atoms(ref_ca[:n_ca], ca[:n_ca])
                        R = _np.array(sup.rotran[0]); T = _np.array(sup.rotran[1])
                        centers.append((lig @ R + T).mean(axis=0))
                    except Exception:
                        continue
                if len(centers) >= 2:
                    centers = _np.array(centers)
                    max_dist = float(_np.max(_np.linalg.norm(centers - centers[0], axis=1)))
                    if max_dist < 2.0:
                        print(f"  [STRONG-CONSENSUS] {target_id}: top-5 ligand centers all within "
                              f"{max_dist:.2f}Å of top-1 → 1 binding site (pseudo-diversity from "
                              f"threshold-raise). Top-5 are near-clones of top-1; LDDT-PLI depends "
                              f"almost entirely on top-1's accuracy.", flush=True)
    except Exception as _e:
        # Non-fatal; do not block ranking
        pass

    top1_lddt = lddt_map.get(top1_rep) if top1_rep else None
    top5_lddts = [lddt_map.get(r) for r in top5_reps if lddt_map.get(r) is not None]
    top5_best = max(top5_lddts) if top5_lddts else None

    top5_sizes = [len(c['members']) for c in clusters[:5]]

    # Per-rep lddts (including None as "NA")
    top5_rep_lddts = [f"{lddt_map[r]:.6f}" if r in lddt_map else "NA" for r in top5_reps]

    # Resolve SDF paths for top5 reps (for downstream re-ranking tools)
    ranking_dir = os.path.join(target_dir, 'ranking_sucos')
    top5_sdf_paths = []
    for rep in top5_reps:
        sdf_matches = glob.glob(os.path.join(ranking_dir, f"{rep}_rank*_sucos*_pb=*.sdf"))
        top5_sdf_paths.append(sdf_matches[0] if sdf_matches else '')

    return {
        'target': target_id,
        'config': config_name,
        'strategy': strategy,
        'max_ha': max_ha,
        'threshold': round(threshold, 4),
        'num_models': len(models),
        'num_clusters': len(clusters),
        'largest_size': top5_sizes[0] if top5_sizes else 0,
        'top5_sizes': ','.join(map(str, top5_sizes)),
        'top1_rep': top1_rep or '',
        'top1_lddt': top1_lddt,
        'top5_best_lddt': top5_best,
        'top5_reps': ';'.join(top5_reps),
        'top5_lddts': ';'.join(top5_rep_lddts),
        'top5_sdf_paths': ';'.join(top5_sdf_paths),
        'target_dir': target_dir,
    }


def collect_targets(dataset_dirs=None, base_dir=None):
    if base_dir is None:
        base_dir = str(_ROOT / "outputs/ensemble")
    if dataset_dirs is None:
        dataset_dirs = ["casp16_l1000", "casp16_l2000_struct", "casp16_l3000_struct", "casp16_l4000"]
    from casp17_ligand.data.components.target_data import BLACKLISTED_TARGETS
    blacklist = BLACKLISTED_TARGETS

    seen = {}
    for d in dataset_dirs:
        matched = glob.glob(os.path.join(base_dir, d, "targets", "*"))
        for p in matched:
            if os.path.isdir(p):
                t = os.path.basename(p)
                if t not in blacklist:
                    if t not in seen or len(p) > len(seen[t]):
                        seen[t] = p
    return seen


def main():
    parser = argparse.ArgumentParser(description='聚类阈值参数搜索实验')
    parser.add_argument('--fixed', nargs='+', type=float, default=[],
                        help='固定阈值列表, 如 0.9 0.8 0.7 0.6')
    parser.add_argument('--adaptive', nargs='+', type=str, default=[],
                        help='自适应阈值. 格式1: ha_target:floor (从0.80@20HA下降). '
                             '格式2: ha_start:start:ha_target:floor (自定义起点). '
                             '如 70:0.60 或 30:0.85:50:0.75')
    parser.add_argument('--maxclust', nargs='+', type=str, default=[],
                        help='按聚类数反馈降阈值: 从 start 开始，若聚类数>max_clusters 则 -step，直到 <=max_clusters 或触底. '
                             '格式: start:max_clusters:step:floor (例: 0.80:50:0.05:0.30)')
    parser.add_argument('--strategy', choices=['center', 'consensus_pb', 'consensus_pb_selfrank'],
                        default='center',
                        help='代表选择策略: center=聚类中心, consensus_pb=类内最佳共识+PB, '
                             'consensus_pb_selfrank=共识+PB+方法自排序top10过滤')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help='数据集列表 (默认: casp16_l1000 casp16_l2000_struct casp16_l3000_struct casp16_l4000)')
    parser.add_argument('--exclude-methods', nargs='+', type=str, default=None,
                        help='排除的方法前缀列表, 如 boltz2 seedfold. '
                             '复用已有 pairwise cache, 过滤模型后重新聚类')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='输出文件名后缀, 如 _without_boltz2')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Ensemble outputs 根目录 (默认 outputs/ensemble). '
                             '多版本管线用 outputs/ensemble_r1r2')
    args = parser.parse_args()

    if not args.fixed and not args.adaptive and not args.maxclust:
        print("错误: 至少指定一个 --fixed / --adaptive / --maxclust 参数")
        sys.exit(1)

    strategy = args.strategy

    # Build configs: [(name, config_dict), ...]
    configs = []
    for v in sorted(args.fixed, reverse=True):
        name = f"固定 {v:.2f}"
        configs.append((name, {'type': 'fixed', 'value': v}))

    for spec in args.maxclust:
        parts = spec.split(':')
        if len(parts) != 4:
            print(f"错误: --maxclust 需要 4 个值 (start:max_clusters:step:floor)，得到 {spec}")
            sys.exit(1)
        start_val = float(parts[0])
        max_clusters = int(parts[1])
        step = float(parts[2])
        floor = float(parts[3])
        name = f"maxclust: start={start_val:.2f}→≤{max_clusters}类, step={step}, floor={floor}"
        configs.append((name, {
            'type': 'maxclust', 'start': start_val, 'max_clusters': max_clusters,
            'step': step, 'floor': floor,
        }))

    for spec in args.adaptive:
        parts = spec.split(':')
        if len(parts) == 4:
            # 格式: ha_start:start:ha_target:end (线性延伸, 无下限)
            ha_s = int(parts[0])
            start_val = float(parts[1])
            ha_t = int(parts[2])
            end_val = float(parts[3])
            name = f"adaptive: {start_val:.2f}@{ha_s}HA->{end_val:.2f}@{ha_t}HA"
            configs.append((name, {
                'type': 'adaptive', 'ha_start': ha_s, 'start': start_val,
                'ha_target': ha_t, 'end': end_val, 'floor': None
            }))
        else:
            # 格式: ha_target:end (默认 0.80@20HA, end 同时作为下限)
            ha_t = int(parts[0])
            end_val = float(parts[1])
            name = f"adaptive: {ha_t}HA->{end_val:.2f}"
            configs.append((name, {
                'type': 'adaptive', 'ha_target': ha_t, 'end': end_val, 'floor': end_val
            }))

    exclude_methods = args.exclude_methods
    output_suffix = args.output_suffix

    base_dir_override = args.base_dir
    seen = collect_targets(dataset_dirs=args.datasets, base_dir=base_dir_override)
    target_dirs = sorted(seen.values())

    series_counts = defaultdict(int)
    for t in seen:
        if t.startswith('L1'): series_counts['L1000'] += 1
        elif t.startswith('L2'): series_counts['L2000'] += 1
        elif t.startswith('L3'): series_counts['L3000'] += 1
        elif t.startswith('L4'): series_counts['L4000'] += 1

    strategy_label = "聚类中心" if strategy == "center" else "共识最佳+PB"
    print(f"Targets: {len(target_dirs)} ({', '.join(f'{k}={v}' for k,v in sorted(series_counts.items()))})")
    print(f"Configs: {[c[0] for c in configs]}")
    print(f"Strategy: {strategy} ({strategy_label})")
    if exclude_methods:
        print(f"Excluded methods: {exclude_methods}")

    # Build all tasks
    tasks = []
    for td in target_dirs:
        for cfg_name, cfg_dict in configs:
            tasks.append((td, cfg_name, cfg_dict, strategy, exclude_methods))

    print(f"Total jobs: {len(tasks)}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_target, t) for t in tasks]
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)}...", flush=True)

    # ── 全局汇总 ──
    print(f"\n{'='*90}")
    print(f"聚类阈值参数搜索结果 (全数据集, 策略={strategy_label})")
    print(f"{'='*90}\n")

    config_names = [c[0] for c in configs]

    print("| 配置 | 平均 Top-1 lDDT | 平均 Top-5 Best lDDT | 平均聚类数 | 平均最大类占比 | #Targets |")
    print("|:-----|:----------------|:---------------------|:-----------|:--------------|:---------|")

    for cfg_name in config_names:
        cr = [r for r in results if r['config'] == cfg_name]
        t1 = [r['top1_lddt'] for r in cr if r['top1_lddt'] is not None]
        t5 = [r['top5_best_lddt'] for r in cr if r['top5_best_lddt'] is not None]
        ncls = [r['num_clusters'] for r in cr]
        ratio = [r['largest_size'] / r['num_models'] for r in cr if r['num_models'] > 0]
        print(f"| {cfg_name} | {np.mean(t1):.4f} | {np.mean(t5):.4f} | {np.mean(ncls):.1f} | {np.mean(ratio):.1%} | {len(t1)} |")

    # ── 子集分解 ──
    for series, prefix in [("L1000", "L1"), ("L2000", "L2"), ("L3000", "L3"), ("L4000", "L4")]:
        sr = [r for r in results if r['target'].startswith(prefix)]
        if not sr:
            continue
        print(f"\n--- {series} 子集 ---")
        print("| 配置 | Top-1 | Top-5 Best | 平均聚类数 | #Targets |")
        print("|:-----|:------|:-----------|:-----------|:---------|")
        for cfg_name in config_names:
            cr = [r for r in sr if r['config'] == cfg_name]
            t1 = [r['top1_lddt'] for r in cr if r['top1_lddt'] is not None]
            t5 = [r['top5_best_lddt'] for r in cr if r['top5_best_lddt'] is not None]
            ncls = [r['num_clusters'] for r in cr]
            print(f"| {cfg_name} | {np.mean(t1):.4f} | {np.mean(t5):.4f} | {np.mean(ncls):.1f} | {len(t1)} |")

    # ── 保存原始数据 ──
    base_dir = base_dir_override or str(_ROOT / "outputs/ensemble")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    out_path = os.path.join(base_dir, f"cluster_experiment_detail_{ts}{output_suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[+] Per-target 原始数据: {out_path}")

    # 也保存一份无时间戳的 latest（带可选后缀）
    latest_path = os.path.join(base_dir, f"cluster_experiment_detail_latest{output_suffix}.csv")
    df.to_csv(latest_path, index=False)
    print(f"[+] Latest 副本: {latest_path}")


if __name__ == '__main__':
    main()
