# CASP17 比赛流程（最终固化版）

参赛信息：
- **AUTHOR**：`3282-9868-6371`
- **组类型**：server
- **Server deadline**：target 释放后 72 小时
- **提交端点**：`https://predictioncenter.org/casp17/submit`（HTTPS POST，字段 `email` + `prediction_file`）

---

## 1. Target 类型 & 接收原则

| Target 系列 | 类型 | 备注 |
|---|---|---|
| `T*` | 蛋白单体 | 大部分 protein-ligand 题 |
| `H*` | 异源多聚体 | 多链 protein-ligand |
| `R*` | RNA / RNA-ligand | RNA chain 用数字 0/1/2 命名 |
| 含 `Ligand` 字段 | 蛋白-小分子 / RNA-小分子 | LG 格式提交 |
| `server-only` 标记 | 仅 server 渠道 | 我们参赛用 server 角色，**全部接受** |

---

## 2. 数据准备

### 2.1 输入来源（**唯一真值**）

每个 target 必须从 CASP17 公开页**直接拉**：
- 详情页：`https://predictioncenter.org/casp17/target.cgi?target={ID}&view=all`（看 residues、ligand、释放/截止）
- **plain-text 序列**：`https://predictioncenter.org/casp17/target.cgi?target={ID}&view=sequence`（curl 直接拿到 FASTA）
- **官方 SMILES TSV**：`https://predictioncenter.org/download_area/CASP17/extra_experiments/ligands/{TARGET}.smiles.txt`（CASP16 同款 4 列 `ID Name SMILES Task`）

**禁止从中间链路读序列**：
- 师兄手抄镜像 `/bmlfast/casp17_ts_qs_qa/TS_run/fasta/{TARGET}.fasta` —— 实测过 R2314 抄错（97nt vs 正确 25nt）
- server task feed 镜像 `/bmlfast/casp17_ts_qs_qa/TS_run/sysbio.rnet.missouri.edu/multicom_cluster/TS/CASP.{target}.{timestamp}` —— R2314 那条链上也含错的 97nt 重发

任何中间环节都不可信，每个新 target 入库前必须 `view=sequence` 与 `view=all` 页面交叉对照。

### 2.2 数据布局（实际落地）

```
data/casp17_data/
├── raw/                                 # 官方 *.smiles.txt 直接 curl 镜像
├── sequences/
│   └── CASP17_R/                        # RNA 总伞 series
│       └── {target}.txt                 # FASTA: ">{tgt} A {len} RNA ..." + 单行序列
├── smiles/
│   └── CASP17_R/
│       └── {target}.tsv                 # CASP16-format 4 列 TSV
└── struct/
    └── CASP17_R_template/
        └── {target}.pdb.txt             # rna_str_from_fasta.py 零坐标提交模板

data/test_cases/casp17_R/
├── ensemble_inputs.csv                  # 7 列 RNA schema (target, entity_type, rna_sequence, ligand_name, ligand_smiles, experimental_affinity, notes)
├── README.md
└── {boltz2,af3,protenix}_inputs/        # 各方法 prep 产出
```

**Series 命名**：
- `CASP17_R` — RNA targets（含 RNA-ligand）
- 后续 protein-ligand：`CASP17_T`（单链）/ `CASP17_H`（多聚）
- 不沿用 CASP16 的 `target_id[:2] + "000"` 自动检测（R2314→R2000 与编号段不一致）；CASP17 一律 CASP15 风格 per-target 文件 + 显式 `series` 入参

### 2.3 ensemble 数据集命名

参考 CASP16 命名规则：`casp17_R`（RNA 系列）/ `casp17_T` / `casp17_H`，r1r2 ensemble 用 `casp17_<series>_r1r2`。pipeline 用 `--config-name=ensemble_generation_r1r2_<series>` 加载。

---

## 3. 上游 4 方法 ensemble

每个 target 由 4 个共折叠方法各跑两轮，每方法共 100 个 model：

| 方法 | r1 (batch=50) | r2 (batch=5 × 10 seeds) | RNA 支持 |
|---|---|---|---|
| AlphaFold3 | ss=1.5 | ss=1.5 | ✓ |
| Boltz-2 | ss=1.5 | ss=1.2 | ✓ |
| Protenix-v1 | seeds=10×5 | seeds=10×5 | ✓ |
| SeedFold-linear | 1 run | r2 视支持情况 | 待确认 |

**特殊场景 fallback**：
- 超大蛋白 OOM 时 boltz-2 退到 `--diffusion_samples=25 --step_scale=1.2`（参考 AGENTS.md 关键参数说明）
- RNA-ligand target 若某方法不支持，ensemble 自动降到 2-3 方法（AF3 + Protenix 至少能跑）

---

## 4. 共识聚类 + 代表选取（生产配置）

固化在 `scripts/run_ensemble_r1r2_pipeline.sh`，4 个 stage 一键跑：

### Stage 1：`ensemble_generation.py`
- 4 方法各 100 model 合并候选池
- 每方法用 `pocket_plddt_4.5` 预排序保留 top-50（method 内）
- 4 方法合并得 **200 个 model/target**
- 转 cif → PDB + SDF（pymol，模板 SMILES 套键级）
- **PoseBusters + LG 化学验证**（atom / bond / MCS vs SMILES）双重 QC，组合 flag 写入 SDF 文件名 `_pb=<combined>`（详见下"_pb 命名"小节）
- 输出 `targets/<target>/ranking_sucos/*_pb=*.sdf` + `ranking_summary.csv`（含 `pb_valid` / `chem_valid` 两列）

### Stage 2：`evaluate_ensemble.py`
- 调 OpenStructure Docker 算 lDDT-PLI（仅 CASP16 复盘用；CASP17 真实 target 没 ground truth，跳过）

### Stage 3：`evaluate_topn_clusters.py`
- 200 个 model 进 SuCOS Butina 聚类
- **maxclust 策略**：起始阈值 0.80，若聚类数 > 60 则按 step 0.05 降阈值，floor 0.30（在 §13c 验证为 Pareto 最优）
- `consensus_pb` 选代表：cluster 内 sucos consensus 最高 + `_pb=True` 的样本
  - cluster 内 sucos 第 2、3、4… 找直到拿到 `_pb=True`
  - 整 cluster 全 `_pb=False` 时回退（仍按 sucos 顺序选）
- 选 top-5 cluster 代表 → `cluster_experiment_detail_latest.csv`（列 `top5_reps`, `top5_sdf_paths`）

### Stage 4：`compare_r1_vs_r1r2.py`
- 跟 r1-only baseline 比对（CASP16 复盘用，CASP17 真实 target 跳过）

### `_pb=` 命名约定（重要）

`_pb=True/False` 文件名后缀的语义在 CASP17 改造里**升级为 "PB AND LG 化学验证 都通过"** 的组合 flag。沿用 `_pb=` 名是为了：

- 不破坏 `evaluate_topn_clusters.py:219` 的 regex（`_pb=(True|False)\.sdf`）
- 让 `pick_consensus_pb_rep` 自动选 (PB AND chem) 都过的代表，cluster 内 / cluster 间 fallback 链原样生效

CSV 仍分别记录 `pb_valid` / `chem_valid` 两列细节，**调试时看 CSV 不要看文件名推断**。AGENTS.md "重要警告" 章节也有此条记录。

---

## 5. 提交格式（CASP17 LG）

参考 `https://predictioncenter.org/casp17/index.cgi?page=format` Example 6.1。

### 5.1 文件结构

每 target 一个 `.txt`，5 个 MODEL/END 块：

```
PFRMAT LG
TARGET <target>
AUTHOR 3282-9868-6371
METHOD Multi-method co-folding ensemble (...)
MODEL  1
PARENT N/A
ATOM  ...                  # block 1: receptor (TS-style)
ATOM  ...
TER
[ATOM ... TER]             # 多 chain
LIGAND <nnn> <code>        # block 2: ligand (MDL)
LSCORE <0..1>
<RDKit MolBlock 含 "M  END">
[LIGAND ... LSCORE ... <MDL>]   # 多 ligand
END
MODEL  2
...
END
```

### 5.2 Chain 重命名（spec 强制）

`generate_submission.py` 自动检测每条 chain 第一个残基类型重映射：

- 蛋白 chain（标准 20 残基 + MSE）→ A, B, C, ...
- 核酸 chain（A/G/C/U/T 或 DA/DG/DC/DT）→ 0, 1, 2, ...

### 5.3 LSCORE 来源

从 SDF 文件名正则 `_sucos([\d.]+)_pb=` 直接提取，clamp 到 [0, 1]。SuCOS consensus = 该 model 与 ensemble 内其他 model 的平均成对 SuCOS。

### 5.4 关于 5 个 MODEL

CASP17 spec 字面：LG 类只评 **model 1**，model 2-5 会被忽略。但 organizer 不会因多交报错。我们交 5 个保留备份。

---

## 6. 提交脚本

### 6.1 生成

```bash
conda run -n casp17_ligand python casp17/scripts/generate_submission.py \
    --cluster-csv outputs/ensemble_r1r2/cluster_experiment_detail_latest.csv \
    --output-dir casp17/submissions/<dataset> \
    --smiles-dir data/casp17_data/smiles
```

输出 `casp17/submissions/<dataset>/<target>.txt`。

### 6.2 提交前 sanity check

跑 patched LG_validation（vendored 在 `casp17/scripts/LG_validation.py`，已 patch 接受 ATOM/PARENT/TER/HETATM 行）：

```bash
LG_TARGETS_PATH=data/casp17_data/smiles \
LG_LOG_PATH=/tmp/lg_logs \
    conda run -n casp17_ligand python casp17/scripts/LG_validation.py \
    casp17/submissions/<dataset>/<target>.txt
```

预期：log 里每个 MODEL 输出 `ATOMS VALID / BONDS VALID / TOPOLOGY VALID`，无 `# ERROR!`。

### 6.3 提交到 server endpoint

```bash
curl -F email=<your@domain> \
     -F prediction_file=@casp17/submissions/<dataset>/<target>.txt \
     https://predictioncenter.org/casp17/submit
```

或用 `casp17/scripts/upload.py`（参考 spec 提供的 Python 模板）。

---

## 7. 端到端 checklist（每个 target）

1. ✅ 收到 organizer 邮件触发的 target query（72h 倒计时开始）
2. ✅ 准备 SMILES TSV + sequence → `data/casp17_data/<series>/<target>.tsv|fasta`
3. ✅ 4 方法分别跑 r1+r2 共 100 model（按 `*_input_preparation.py` + `*_inference.py`）
4. ✅ `bash scripts/run_ensemble_r1r2_pipeline.sh <series_short> 16 --yes`
5. ✅ `python casp17/scripts/generate_submission.py --cluster-csv ... --output-dir ...`
6. ✅ 跑 LG_validation.py 验证（无 `# ERROR!`）
7. ✅ HTTPS POST 提交到 `predictioncenter.org/casp17/submit`
8. ✅ organizer 回邮件确认接收

---

## 8. 关键文件地图

| 文件 | 作用 |
|---|---|
| `scripts/run_ensemble_r1r2_pipeline.sh` | 生产管线一键跑（Stage 1-4，已固化 mc60 maxclust）|
| `casp17_ligand/models/ensemble_generation.py` | Stage 1：CIF→PDB+SDF + PB + LG chem validate + ranking |
| `casp17_ligand/utils/lg_chem_validate.py` | LG 化学验证（atom / bond / MCS），1:1 镜像 LG_validation.py 化学逻辑 |
| `casp17_ligand/analysis/evaluate_topn_clusters.py` | Stage 3：Butina 聚类 + maxclust + consensus_pb |
| `casp17/scripts/generate_submission.py` | 输出 LG 提交文件（5 MODEL，PARENT N/A + receptor + ligand）|
| `casp17/scripts/LG_validation.py` | vendor 副本（patched 接受 receptor 行 + 空白行 + env var 路径）|
| `casp17/submissions/<dataset>/<target>.txt` | 最终提交文件 |

---

## 9. 已知限制 / 待跑通

- **SeedFold RNA 支持未确认**：CASP17 RNA-ligand target 出现时需要先验证；不行就退到 3 方法 ensemble
- **AFFNTY 字段没生成**：本流程只负责 P-task（pose），A/PA-task 需要单独提供 affinity 估计
- **共价 ligand**（CASP16 L4000 类）：generate_submission.py 当前对"SDF mol 数 < SMILES TSV 行数" warning 后按 min 长度配对，可能漏报 covalent partner——CASP17 出题再具体看
- **RNA chain 重映射**：上游 cif_to_pdb_sdf 输出的 RNA chain 名格式 CASP16 没验证过，出题后第一个 RNA target 跑出来 spot-check PDB，必要时调整 `detect_chain_type` 的残基集合

---

## 10. RNA target 处理（R2314 实战，2026-05-04）

### 10.1 R2314 题面

| 项 | 值 |
|---|---|
| Target | R2314 |
| 类型 | RNA + ligand (holo) |
| RNA 长度 | **25 nt** |
| RNA 序列 | `CGAGGACCGGUACGGCCGCCACUCG` |
| Ligand | TRP (Tryptophan), `c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N` |
| 释放 | 2026-05-04 |
| Server 截止 | 2026-05-06（~48h，比标准 72h 短） |
| 伴侣 target | R2315（apo，25nt 同序列，本项目不纳入） |

### 10.2 三方法 RNA input prep（已实现）

`casp17_ligand/data/components/target_data.py` 加 `entity_type: str = "protein"` 字段；`RNA_SERIES = {"CASP17_R"}` 集合里的 series 在 loader 中自动设为 `"rna"`。`protein_sequences` 字段名保留（向后兼容）。

三份 input prep 在 chain emission 处按 entity_type 分流：

| 方法 | RNA chain spec | RNA MSA 支持 |
|---|---|---|
| Boltz-2 | `- rna:` with `id` + `sequence` | Boltz-2 docs 明示 msa "only for protein"，**模型层面就不接** |
| AF3 | `{"rna": {"id":, "sequence":, [unpairedMsa:]}}` | 通过 `rna_unpaired_msa` cfg 注入 a3m 字符串内容；不设则**让 AF3 自己跑 data pipeline 搜 Rfam/RNAcentral** |
| Protenix | `{"rnaSequence": {"sequence":, "count":, [unpairedMsaPath:]}}` | 通过 `rna_unpaired_msa_path` cfg 注入 a3m 文件路径；不设则 single-sequence |

各 prep 的 `data_root` 已参数化（默认 `data/casp16_data`，CASP17_R config 改为 `data/casp17_data`）；`struct_only` cfg 可关（CASP17 比赛中 GT struct 还没释放）。protein-centric 约束分支（MPro Cys145、L3000 ZN/NAG cofactor、L4000 covalent inhibitors）天然不触发——它们的判据都是 `target_id.startswith("L3"/"L4")` 或 `target_id in COVALENT_TARGETS`，CASP17 R-target ID 都不匹配。

**Hydra configs**：
- `configs/data/boltz2_input_preparation_casp17_R.yaml`
- `configs/data/af3_input_preparation_casp17_R.yaml`
- `configs/data/protenix_input_preparation_casp17_R.yaml`

**调用**：
```bash
conda run -n casp17_ligand python casp17_ligand/data/boltz2_input_preparation.py --config-name boltz2_input_preparation_casp17_R
conda run -n casp17_ligand python casp17_ligand/data/af3_input_preparation.py --config-name af3_input_preparation_casp17_R
conda run -n casp17_ligand python casp17_ligand/data/protenix_input_preparation.py --config-name protenix_input_preparation_casp17_R
```

**产出**：
- `data/test_cases/casp17_R/boltz2_inputs/R2314_input.yaml`
- `data/test_cases/casp17_R/af3_inputs/R2314.json`
- `data/test_cases/casp17_R/protenix_inputs/R2314.json`

**下游 inference 脚本暂未改动**（boltz2/af3/protenix_inference.py），接 RNA 推理时再看输出 chain 命名 / CIF 解析是否要加 RNA 分支，不预先动。

### 10.3 RNA MSA 搜索

RNA MSA 对 RNA 结构 / RNA-ligand 接触预测重要；single-sequence 模式质量显著偏弱。

**模型层 RNA MSA 接收能力**：

| 方法 | 接收 RNA MSA |
|---|---|
| Boltz-2 | ✗（模型层不接，跳过 MSA 搜索）|
| AF3 | ✓（JSON `unpairedMsa` 字段）|
| Protenix | ✓（JSON `rnaSequence.unpairedMsaPath`）|
| rf3 / rfd3 | ✓（`rna_msa_dirs` 配置项指向 a3m 目录）|

**搜索工具决策矩阵**（按 RNA query 长度 + 同源数据库可达性选）：

| 工具 | 数据库 | 适用场景 | 速度 | 状态 |
|---|---|---|---|---|
| AF3 内置 nhmmer (`run_data_pipeline=true`) | Rfam_14_9 + RNAcentral + nt_rna_2023_02_23 | 50+ nt RNA 默认首选；自动注入 `unpairedMsa` | RNA MSA 5-20 min | 现成 |
| Protenix `rna_msa_search.py` | 同上三库 | 跟 AF3 等价（同 nhmmer + 同 e_value=1e-3）；如果只跑 Protenix 就走它 | 同上 | `forks/Protenix/runner/rna_msa_search.py` |
| 直接 nhmmer (容器外) | 上述任意单独 | 想放宽 `-E` / `-F3` 时（AF3 容器内 hardcode `-E 0.001`）；50+ nt 不太需要，主要给短 RNA 兜底 | 整库 1-5 min | `casp17_ligand` env 装好 `hmmer 3.4` |
| INFERNAL `cmscan` Rfam.cm | 4000+ family CMs（含二级结构先验） | 50+ nt 短 motif；CM 比 nhmmer 对 conserved RNA family 召回率高 | 2-30 s | `casp17_ligand` env 装好 `infernal 1.1.5`；Rfam.cm + indices vendored 在 `data/rfam/`（652 MB） |
| rf3 / rfd3 | — | **没有 MSA 搜索工具**，是 consumer 不是 generator；要喂 a3m 进去 | — | 用其他工具搜出 a3m 后放到 `rna_msa_dirs` 指向的目录 |

**短 RNA (≤30 nt) 警告**：R2314 (25 nt) 实测下，5 条路径全 0 命中（AF3 内置 nhmmer / 直接 nhmmer `-E 100 --F3 0.5` 三库 / INFERNAL cmscan Rfam.cm 4 阈值含 `-E 10000`）。原因：Rfam family motif 平均 80-150 nt，25 nt 比所有 family 长度过滤都短；nhmmer 的 score formula 在短 query 上统计 power 接近 0。**≤30 nt RNA 不要再花时间搜 MSA，直接 single-sequence 跑下游模型**。50+ nt 才值得搜。

**复用 r1 的 `_data.json` 跑 r2/r3**：AF3 一次 `--norun_inference` 产物 `<target>_data.json` 含 `unpairedMsa` 字符串，r2 复用直接 `--norun_data_pipeline --json_path=<r1_data.json>`，跳过整个 RNA MSA 搜索阶段。Protenix 同理用 `prep` 一次产 a3m，下游 inference 复用 `rna_unpaired_msa_path`。

**注入第三方搜出的 MSA 到 AF3**：把 a3m 字符串塞进 `_data.json` 的 `sequences[i].rna.unpairedMsa` 字段，跑 `--norun_data_pipeline` 跳过 AF3 自带 search。INFERNAL `cmalign --outformat afa` 出的是 Stockholm/aligned-FASTA，需要先转 a3m（去掉 metadata + 大写匹配 / 小写插入）。

### 10.4 踩坑记录（按时间倒序）

#### 2026-05-04 — R2314 序列错误事件
- 我从 target.cgi `view=all` 抓到 25nt 后写入数据
- 用户提供师兄镜像 `/bmlfast/casp17_ts_qs_qa/TS_run/fasta/R2314.fasta`，里面是 97nt
- 我误信师兄镜像 + server task feed 12:26/12:27 的"重发"，把数据改成 97nt，宣称"server feed 是 latest canonical"
- 用户校验后指出：师兄复制错了，**正确序列就是 target.cgi 公开页的 25nt**
- 修正：所有数据 revert 25nt；本节 §2.1 写明"唯一真值是 target.cgi view=sequence，不要从师兄/server feed 间接读"

#### 2026-05-04 — AF3 dummy MSA 等于禁用 MSA
- 师兄的 `standard_af3.json` 里 RNA chain 用了 `unpairedMsa: ">query\n{seq}\n"`（自指 single-seq）
- 我跟着写，自以为是"匹配师兄惯例"，AGENTS.md 里也写了"自动注入自指 MSA 跳过 AF3 MSA pipeline"
- 实际语义：dummy 自指 MSA = **禁用** AF3 的 MSA pipeline = single-sequence = RNA 结构预测最弱档
- 修正：AF3 prep **默认不注入** `unpairedMsa`，让 AF3 自己跑 Rfam/RNAcentral 搜索；提供 cfg 注入预算 a3m 内容（用来后续复用搜过的 MSA）

#### 2026-05-04 — 误判师兄 pipeline 是 RNA monomer（无 ligand）
- 看到 notes.txt 一行 `rna_monomer.py` 命令就外推说"师兄 pipeline 只跑 RNA monomer，我们做 RNA-ligand 是互补"
- 实测发现 `valid/R2314/N3_monomer_structure_generation/standard_af3/` 已经在跑 **RNA + Trp ligand 复合物 AF3 预测**
- 修正：师兄通路和我们目标重叠（不是互补），但都是 single-sequence dummy MSA；本项目差异化在多方法 ensemble + 真 RNA MSA
