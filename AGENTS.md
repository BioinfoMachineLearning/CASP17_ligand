# CASP17_ligand — 项目规划与架构说明

## 项目目标

使用多个共折叠（co-folding）方法对同一个 CASP17 target 进行预测，然后通过两种排序方法得到最终提交结果。

## 设计原则

- **完全独立**：不依赖 MULTICOM_ligand 项目，所有复用代码需内化到本项目
- **可扩展**：新增方法只需在 `forks/` 放入方法代码，并添加对应的 input preparation / inference 脚本和 config
- **配置管理**：使用 Hydra，支持命令行参数覆盖和多参数扫描
- **相对路径**：所有路径使用相对于项目根目录的相对路径，避免绝对路径硬编码
- **代码复用**：公共逻辑放 `casp17_ligand/utils/`，各方法脚本复用，不重复造轮子

## 环境架构

采用**独立 orchestration 环境**方案：

```
casp17_ligand env（轻量）
  ├── hydra-core, omegaconf, pandas, rootutils
  ├── 运行所有 *_input_preparation.py 和 *_inference.py
  └── inference 脚本内部用 subprocess 调用各方法的可执行文件

forks/boltz env  → 只需要 boltz 本身
forks/af3 env    → 只需要 AF3 本身
...
```

**可执行文件路径自动解析**（`casp17_ligand/utils/env_utils.py`）：
1. 若 `boltz_exec` 为绝对路径 → 直接使用
2. 否则自动检测：`{conda info --base}/envs/{env_name}/bin/{exec_name}`
3. 最终回退到 PATH

config 中可覆盖：
```yaml
boltz_exec: boltz       # 可执行文件名或绝对路径
env_name: boltz         # conda env 名称
conda_envs_dir: null    # 留空则自动检测，或手动指定如 /opt/miniconda3/envs
```

## 拟支持的共折叠方法（逐步扩展）

| 方法 | 状态 | forks 路径 |
|------|------|-----------|
| Boltz-2 | 运行中（Exp-01） | `forks/boltz/` |
| AlphaFold3 | 进行中 | `forks/alphafold3/` |
| RoseTTAFold3 | 待实现 | `forks/rosettafold3/` |
| Protenix-v1 | 脚本就绪（待 Docker + 数据库） | `forks/Protenix/` |
| SeedFold-linear | 待实现 | `forks/seedfold/` |

## 排序方法

两种排序方法，具体逻辑由用户后续指定。

## 项目结构

```
CASP17_ligand/
├── casp17_ligand/               # 主 Python 包
│   ├── data/
│   │   ├── components/          # 共用工具
│   │   ├── boltz2_input_preparation.py
│   │   └── protenix_input_preparation.py
│   ├── models/
│   │   ├── boltz2_inference.py
│   │   ├── af3_inference.py
│   │   ├── protenix_inference.py
│   │   └── ensemble_generation.py   # 核心：汇总 + 两种排序（待实现）
│   ├── analysis/
│   │   └── inference_analysis_casp.py（待实现）
│   └── utils/
│       └── env_utils.py         # conda env 可执行文件路径解析（所有方法复用）
├── configs/
│   ├── data/
│   │   ├── boltz2_input_preparation.yaml
│   │   └── protenix_input_preparation.yaml
│   └── model/
│       ├── boltz2_inference.yaml
│       ├── af3_inference.yaml
│       └── protenix_inference.yaml
├── data/
│   ├── casp16_data/             # symlink 到 casp16 数据
│   └── test_cases/
│       └── casp16_l1000/        # L1000 验证集（17个分子）
│           ├── ensemble_inputs.csv   # 所有方法共用的输入 CSV
│           ├── boltz2_inputs/        # boltz2 生成的 YAML 文件
│           └── boltz2_outputs/       # boltz2 预测结果
├── forks/
│   ├── boltz/                   # Boltz-2 官方代码
│   └── alphafold3/              # AlphaFold3 官方代码
├── weights/
│   ├── boltz/                   # Boltz-2 权重（symlink 到 ~/.boltz）
│   └── alphafold3/              # AF3 权重（symlink 到 /bml/Lyuwei/Alphafold3_weights）
├── scripts/
│   └── setup_weights.sh         # 通用权重 symlink 脚本（各方法均可用）
├── environments/                # 各方法 conda 环境配置
├── .project-root                # rootutils 标记文件
└── setup.py
```

## data/test_cases 约定

每个数据集在 `data/test_cases/{dataset}/` 下，结构统一：
```
{dataset}/
├── ensemble_inputs.csv          # 所有方法共用：target, protein_sequence, ligand_smiles, ...
├── {method}_inputs/             # 各方法的输入文件
└── {method}_outputs/            # 各方法的预测结果
```

新增方法时只需在同一 `ensemble_inputs.csv` 基础上添加对应的 inputs/outputs 目录，无需重复准备数据。


## 统一数据预处理 (Unified Data Components)

随着方法集成数量的增加，解析蛋白质 PDB / TXT、提取配体 SMILES 并生成数据对象的逻辑存在高度重复。因此，我们在 `casp17_ligand/data/components/target_data.py` 实现了统一的数据处理基类组件：

1. **核心逻辑：** 只需要提供目标 series 列表以及对应的 data_dir。管线会统一解析序列和结构文件，读取配方文件 (.tsv)，将其转化为标准的 `protein_sequences: List[Tuple[str, str]]`（链ID+序列）和 `ligands: List[Tuple[str, str]]`（配体ID+SMILES）。
2. **多聚体与多配体兼容：** 无论目标是二聚体还是挂载了三个配体的复合物，预处理器都能将其无损转化为列表。
3. **质子化整合：** 包含 `--protonation` 支持。开启时，基础读取器自动拦截原始 SMILES 并调用 Dimorphite-DL 进行质子化变换，返回最可能的质子态。下游模型（Boltz-2、AF3等）可以直接使用规范化的输入，实现了业务逻辑的解耦。
## 每个方法的集成模式

1. `casp17_ligand/data/{method}_input_preparation.py` — 从 `ensemble_inputs.csv` 生成方法所需输入
2. `casp17_ligand/models/{method}_inference.py` — 调用方法推理，��果存入 `{method}_outputs/`
3. `configs/data/{method}_input_preparation.yaml`
4. `configs/model/{method}_inference.yaml`（含 `env_name`, `conda_envs_dir`, `{method}_exec`）

## Boltz-2 集成经验与关键参数

### 输出格式
- 结构文件：`boltz_results_{target}_input/predictions/B/B_model_0.cif`（diffusion_samples=1 时只有 model_0）
- 亲和力：`boltz_results_{target}_input/predictions/B/affinity_B.json`
  - `affinity_pred_value`：两个 affinity 模块的 ensemble 平均 + mw_correction，单位 **log10(IC50 in μM)**
  - `affinity_pred_value1/2`：两个模块各自的原始输出（无 mw_correction）
  - `affinity_probability_binary`：结合概率（0-1），用于 binder/decoy 区分
- pIC50 转换：`pIC50 = 6 - affinity_pred_value`；转 kcal/mol：`(6 - y) * 1.364`

### 关键参数说明
| 参数 | 我们的设置 | 说明 |
|------|-----------|------|
| `--model boltz2` | ✓ | 使用 boltz2 模型（boltz2_conf.ckpt + boltz2_aff.ckpt） |
| `--use_msa_server` | ✓ | 调用 ColabFold mmseqs2 server 生成 MSA |
| `--diffusion_samples` | 50 | 结构预测采样数 |
| `--recycling_steps` | 10 | 结构预测 recycling 步数 |
| `--sampling_steps 200` | 200 | 扩散采样步数 |

> **Affinity 默认关闭**：`predict_affinity` 默认为 `false`（input_preparation 不写 `properties` 块，inference 不传 affinity 相关 CLI 参数）。如需开启，在 config 中设置 `predict_affinity: true` + `affinity_mw_correction: true` + `diffusion_samples_affinity: 5` + `sampling_steps_affinity: 200`，且 YAML 输入需包含 `properties: - affinity: binder: X`。

> **超大蛋白显存退路**：当 `>1000 aa` 等超大体系下 `--diffusion_samples=50` OOM 时，退到 `--diffusion_samples=25` + `--step_scale=1.2`（而不是默认的 1.5）。低 step_scale 在小 batch 下能略微补偿采样多样性的损失。CASP16 L3000 top-5 验证：batch=25 时 ss=1.2 的 best-of-50 比 ss=1.5 高约 +0.05 lDDT-PLI。常规 batch=50 仍用 ss=1.5（默认最优）。

### 亲和力预测机制
1. 结构预测模型（boltz2_conf.ckpt）生成 `diffusion_samples` 个结构
2. affinity 模型（boltz2_aff.ckpt）独立运行，生成 `diffusion_samples_affinity`（默认5）个结构
3. 从 5 个结构中选 **iptm 最高**的那个
4. 用两个 affinity 模块（affinity_module1/2）分别预测，取平均
5. 加 mw_correction：`1.03526 * raw + (-0.59993) * mw^0.3 + 2.83288`

### 多配体/二聚体坑 (Binder 参数格式)
当遇到多配体或者二聚体时，由于 CASP17 数据解释："Molecules located nearby (<4.5 A) the main ligand(s) ... were also included in the SMILES files"。
我们需要在提供多配体输入时，强制为它们的相互作用**施加准确的最大距离约束 `contact` constraints**，否则 Boltz-2 很容易在没有空间限制的情况下发散。

**自适应距离约束方案 (4.5A)**：
在 `casp17_ligand/data/boltz2_input_preparation.py` 中，我们集成了 `RDKit` 三维坐标生成，针对多配体体系的每个 Ligand，自动计算其三维几何中心，选出离中心最近的原子作为**中心锚点**，同时计算该分子的**最大延伸半径**。
对于体系中的每对 Ligand，自动加入 `contact` constraints：
- `token1`: Ligand A 中心原子 (如 `[B, C34]`)
- `token2`: Ligand B 中心原子 (如 `[C, S5]`)
- `max_distance`: `半径 A + 半径 B + 4.5A` 
- `force: true`

对于 `affinity` 计算，Boltz-2 会要求在 YAML 里为配体配置 `binder` 属性。
**坑：** Boltz-2 解析器要求 `binder` 必须是单链的列表。如果把多个配体的 ID 写成类似 `binder: [C, D]` 甚至是 `binder: C, D`，由于解析器内部逻辑，有时会抛出 `ValueError: Binder must be a single chain.` 的报错。
**正确使用方法：** 必须在 YAML 里把每个配体分离为独立的 `binder` 声明，例如序列列表中要有：
```yaml
properties:
  - type: affinity
    binder: C
  - type: affinity
    binder: D
```
我们的 `casp17_ligand.data.boltz2_input_preparation.create_boltz2_yaml` 函数内部已对此做好兼容，针对目标的所有 Ligand ID 分别生成单个字母的 `binder` 属性。

### 实验数据格式
- 实验亲和力：`data/casp16_data/answer/L{series}_exper_affinity.csv`，`binding_affinity` 列
- 单位：与 boltz2 输出同为 log(Ki) 类单位，直接用于 Kendall τ 计算（rank-based，无需转换）

### MSA 复用（多分子同一蛋白）
当同一蛋白质对应多个配体时（如 L1000/L3000 系列），MSA 只需生成一次：
1. 先跑第一个分子（`--use_msa_server`），MSA 存于 `boltz_results_{target}_input/msa/B_0.csv`
2. 复制到 `data/test_cases/{dataset}/msa/{series}_msa.csv`
3. 后续 YAML 中指定 `msa: data/test_cases/{dataset}/msa/{series}_msa.csv`（相对于运行时工作目录）
4. 后续推断不需要 `--use_msa_server`

### 验证结果（L1000，17个分子）
- 我们的 Kendall τ = **0.221**（参考研究 τ = 0.353）
- 差异原因：扩散模型随机性（无固定 seed），主要是 L1014 预测值偏差（-1.51 vs 参考 -1.16）
- 参数和流程均正确，差异在正常随机波动范围内

## 开发进度

- [x] 项目规划与架构设计
- [x] 项目骨架初始化
- [x] `env_utils.py`：conda 可执行文件路径自动解析
- [x] Boltz-2 input preparation（支持 msa_path 参数）
- [x] Boltz-2 inference 脚本
- [x] L1000 验证集 CSV（17个分子，实验值来自 L1000_exper_affinity.csv）
- [x] weights symlink 脚本
- [x] casp17_ligand orchestration env 创建
- [x] L1000 端到端验证（τ=0.221，参考 0.353，随机波动）
- [x] L3000 推断（123个分子，进行中）
- [ ] 评估脚本 boltz2_affinity_eval.py（支持 L1000/L3000 复用）
- [x] ensemble_generation.py（RMSD + SuCOS 双排序，MCS fallback 键级修复）
- [x] Protenix-v1 input preparation + inference 脚本（JSON 格式，MSA/Template 复用策略）
- [x] Protenix-v1 conda 环境部署（放弃 Docker，CUDA ext 已编译，权重已下载）
- [x] Protenix-v1 L1001 端到端验证（iptm=0.469，no MSA）
- [ ] Protenix-v1 L1000 MSA+Template 搜索（进行中）
- [ ] Protenix-v1 L1000 全量推理（10 seeds × 5 samples = 50 models/target）
- [x] RoseTTAFold3 集成 (In-process Python API + CLI fallback)
- [x] RF3 L1000 100-model ensemble（20 seeds × 5 batch，已完成，outputs/rf3/casp16_l1000/）
- [x] RF3 L3000 100-model ensemble（Hellbender A100，SLURM array job 12653421，运行中）
- [x] SeedFold-linear 集成（ensemble_generation.py 支持 A0/B0 链名）

## 重要警告

- `/bmlfast/databases/` 是公用数据库目录，**绝对不能修改**，只读挂载
- Docker 容器必须设置 cgroups 资源限制（`--memory`, `--cpus`），防止 OOM 杀死宿主机服务
- Docker 输出文件权限问题：必须使用 `--user $(id -u):$(id -g)` 映射用户身份
- **`_pb=True/False` SDF 文件名后缀的语义不再是单纯 PoseBusters 结果**：`ensemble_generation.py:_save_ranked` 写出 ranking 子目录里的 SDF 时，文件名后缀 `_pb=<flag>` 实际承载的是 **PoseBusters AND LG 化学验证 (`lg_chem_validate`)** 的组合 flag。CSV (`ranking_summary.csv`) 仍分别记录 `pb_valid` / `chem_valid` 两列。沿用 `_pb=` 名称是为了不破坏 `evaluate_topn_clusters.py:219` 的 regex 与 `pick_consensus_pb_rep` 的选代表逻辑，让组合 QC 自动嵌入现有 cluster 内 / cluster 间 fallback 链。如需调试单独看 PB 还是 chem 失败，去 CSV 两列字段，**不要从文件名推断**。详见 `casp17.md` "提交流程" 章节。

## AlphaFold3 集成

### 运行方式
通过 Docker 容器运行，镜像 `alphafold3:latest`。

### 硬件环境
- GPU 0: NVIDIA A100 40GB → 需要 unified memory（XLA_PYTHON_CLIENT_PREALLOCATE=false, TF_FORCE_UNIFIED_MEMORY=1, XLA_CLIENT_MEM_FRACTION=3.2）
- GPU 1: NVIDIA A100 80GB → 标准配置即可（XLA_PYTHON_CLIENT_PREALLOCATE=true, XLA_CLIENT_MEM_FRACTION=0.95）
- 192 CPU 核心，1TB RAM

### 资源隔离（防止服务器卡死）
```
--memory=200g              # 物理内存上限
--memory-swap=200g         # 禁止 swap（等于 --memory）
--cpus=188.0               # 保留 4 核给系统
--shm-size=8g              # 共享内存
--user=$(id -u):$(id -g)   # 文件权限映射
```

### XLA 显存调优（40GB A100）
```
XLA_PYTHON_CLIENT_PREALLOCATE=false   # 禁止预分配
TF_FORCE_UNIFIED_MEMORY=1            # 启用统一内存
XLA_CLIENT_MEM_FRACTION=3.2          # 允许溢出到主机 RAM
```

### Docker 镜像版本说明

两个镜像均为 alphafold3==3.0.1（同一 git commit），仅构建时间不同：
- `alphafold3`：2025-07-06 构建，旧镜像（本机）
- `alphafold3_casp17`：2026-02-27 构建，新镜像（另一台机器）

**关键差异**：新镜像 `alphafold3_casp17` 的 `folding_input.py` 将 `modelSeeds` 从合法 JSON key 中注释掉了，导致行为不同（见下）。

### 输入格式（当前方案，与新镜像兼容）

- JSON 中**不包含** `modelSeeds` 字段（新镜像会拒绝该字段）
- `version: 1`（新镜像支持 1/2/3，version 字段只是 schema 声明，不影响运行时行为）
- seed 通过 CLI `--model_seed=N` 传入（`--json_path` 模式专用）
- 多 seed 需要多次调用 docker，每次传不同 `--model_seed`（**待解决**）
- SMILES 中的反斜杠需要 JSON 转义（`\\`）

**与旧镜像的兼容性**：旧镜像 `alphafold3` 要求 JSON 中有 `modelSeeds`，两者不能用同一套输入文件。目前统一使用新镜像方案，旧镜像不再维护。

### GPU 要求

- **最低 40GB 显存**，L3000 target 最大约 1000 tokens（bucket size 1024），25GB 显卡会 OOM 崩溃
- 本机 GPU 1（A100 80GB）为推荐运行环境

### 关键路径
- 权重: `weights/alphafold3/af3.bin` → `/bml/Lyuwei/Alphafold3_weights/af3.bin`
- 数据库: `/bmlfast/databases/`（只读，公用）
- 输入: `data/test_cases/{dataset}/af3_inputs/*.json`
- 输出: `outputs/alphafold3/{dataset}/`
- 脚本: `casp17_ligand/models/af3_inference.py`
- 配置: `configs/model/af3_inference.yaml`

### Docker 运行命令模板
```bash
docker run --rm \
  --gpus '"device=0"' \
  --memory=200g --memory-swap=200g --cpus=188.0 --shm-size=8g \
  --user=$(id -u):$(id -g) \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e TF_FORCE_UNIFIED_MEMORY=1 \
  -e XLA_CLIENT_MEM_FRACTION=3.2 \
  -v /path/to/af3_inputs:/root/af_input \
  -v /path/to/af3_output:/root/af_output \
  -v /bml/Lyuwei/Alphafold3_weights:/root/models \
  -v /bmlfast/databases:/root/public_databases \
  alphafold3 \
  python run_alphafold.py \
  --json_path=/root/af_input/target.json \
  --model_dir=/root/models \
  --output_dir=/root/af_output
```

### MSA 复用策略
AF3 支持预计算 MSA：先用 `--norun_inference` 只跑 data pipeline，输出带 MSA 的 JSON，后续用 `--norun_data_pipeline` 只跑推理。同一蛋白多配体场景下可大幅节省时间。

### chembl35_full 多 ligand 运行模式(aster, 2026-04-21)

"单 target 几十到几百 ligand" 场景用 `scripts/run_af3_chembl35_full.sh` + 两卡 wrapper 并行,三阶段 per series:

1. **Phase 1** — 对 target 首个 compound 跑 `--norun_inference`,产出 `_data.json`(MSA+template)。
2. **Phase 2** — Python 把 `unpairedMsa` / `pairedMsa` / `templates` 注入到该 target 所有 compound 的 JSON → `af3_inputs_msa/`。
3. **Phase 3** — 对每个 compound 独立 `--norun_data_pipeline` 推理。

Skip 标志:
- Phase 1:`find -iname "<compound>_data.json"` 有结果(新镜像输出**强制小写 compound 名**,必须 `-iname`)
- Phase 3:`find -iname "<compound>_ranking_scores.csv"` 有结果 = 该 compound 全部产出完成

#### 三机镜像差异(CLI 不通用)

| | aster `alphafold3_casp17` | Hellbender SIF | lily `alphafold3_new` |
|---|---|---|---|
| seed 来源 | CLI `--model_seed=N`,JSON **不含** `modelSeeds` | JSON **含** `modelSeeds`,不传 `--num_seeds` | JSON **含** `modelSeeds`(≥1),CLI `--num_seeds=K` 从第 1 个 seed 连续展开 K 个 |
| 多 seed 示例 | `--model_seed=1 --num_seeds=2` → [1,2] | JSON `modelSeeds:[1,2]` | JSON `modelSeeds:[1]` + `--num_seeds=2` → [1,2] |

runner 用 env `MODEL_SEED_FLAG` 切换:aster 默认 `--model_seed=1`,**lily / Hellbender 需设 `MODEL_SEED_FLAG=""`** 并预先给 JSON patch `"modelSeeds": [1]`。**输入 JSON 不通用**,三边分别注入。

#### lily 跑法(GPU 0, A100 40GB, 2026-04-21 long_small 验证)

```bash
GPU_DEVICE=0 XLA_PREALLOCATE=true TF_UNIFIED_MEM=0 \
XLA_CLIENT_MEM_FRACTION=0.9 XLA_MEM_FRACTION=0.9 \
MEMORY=80g CPUS=48 NUM_SEEDS=2 \
DOCKER_IMAGE=alphafold3_new MODEL_SEED_FLAG= \
setsid nohup bash scripts/run_af3_chembl35_full_long_small.sh \
  > outputs/alphafold3/chembl35_full_long_small_run.nohup.log 2>&1 < /dev/null & disown
```

- 跑前必须给该子集所有 `af3_inputs/*.json` 注入 `"modelSeeds": [1]`(只改本子集,不碰 aster 正在用的 short JSON)。
- 实测 P49841 (351aa) 推理速度 ~2 min 14 s / compound(10 models = 2 seeds × 5 samples),约 13 s/model。
- lily / aster 共用 bmlfast 磁盘,输出同一 `outputs/alphafold3/chembl35_full/` 目录;long_small 和 short 的 compound 前缀不重叠所以共存无冲突。

#### 扁平输出布局(新镜像 `--json_path` 单 JSON 模式,与 chembl35_A 的 per-compound 子目录布局**不同**)

```
outputs/alphafold3/<dataset>/
├── <lower>_{model.cif,ranking_scores.csv,summary_confidences.json,confidences.json,data.json}
└── seed-N_sample-M/<lower>_seed-N_sample-M_{model.cif,confidences.json,summary_confidences.json}
       ^ 所有 compound 共享 seed-sample 目录,用 compound 前缀区分
```

`_ranking_scores.csv` 是 compound 完成的唯一可靠 marker。

#### compound 命名

prep 脚本给每个 CSV 行分配 `<TARGET>_<NNN>`(N 按 target 内出现顺序 000 起),原始 CHEMBL ID 的回查表在 `data/test_cases/<dataset>/compound_id_map.csv`(`af3_name, compound_id, target_id, index_within_target`)。下游分析 join 此表。

### AF3 Hellbender 修复记录（2026-03-08）

**问题**：AF3 在 Hellbender 上全部报错 `ModuleNotFoundError: No module named 'tokamax'`

**根因**：SLURM 脚本通过 `PYTHONPATH` overlay 加载了 `forks/alphafold3/` 的最新版代码，该版本依赖 `tokamax`，但 SIF 镜像是旧版本没有此包。

**修复**：
1. 去掉 fork overlay，直接使用 SIF 内置的 `/app/alphafold/run_alphafold.py`
2. 挂载路径从 `/root/` 改为 `/tmp/`（Singularity 非 root 用户权限）
3. 移除 `--num_seeds`（seeds 已在 JSON 的 `modelSeeds` 中指定，10 seeds × 5 samples = 50 models/target）
4. 环境变量通过 `--env` 传入而非 `bash -c "export ..."`

**验证**：本机 Docker + Hellbender Singularity 均成功（L3001，~15min/target on A100）

**当前运行**：
- 本机 GPU 1：前 100 targets (L3001-L3109)，PID 3409723
- Hellbender A100：后 89 targets (L3110-L3201)，job 12677583

## RoseTTAFold3 集成

### 运行方式
推荐在配置好的 `rf3` Conda 环境中直接运行（使用 In-process Python API）。
如果不在 `rf3` 环境中运行，脚本会自动回退到使用 `subprocess` 调用 `rf3 fold` CLI。

### 硬件环境
- GPU 0: NVIDIA A100 40GB
- 资源要求：显存消耗视复合物大小而定（支持 `CUDA_VISIBLE_DEVICES` 调度）。

### 多种子（Multi-seed）推理优化
我们重写了 RF3 的推理脚本 `casp17_ligand/models/rf3_inference.py`，引入了在内存中换 Seed 的机制：
对于 Python API 模式，`RF3InferenceEngine` 只会在初始化时加载一次模型权重和参数到 VRAM，然后在遍历不同 seed 的预测时，通过 `seed_everything(seed)` 和 `engine.seed = seed` 来动态改变随机状态。这比命令行每次预测都重新加载模型要**快得多**。

### 关键路径和配置
- 环境: `conda activate rf3` (需包含 `foundry[rf3]`)
- 权重: `weights/rf3/rf3_foundry_01_24_latest_remapped.ckpt` (大小约 2.3GB)
- 输入生成: `casp17_ligand/data/rf3_input_preparation.py`
  - RF3 接受 JSON 格式输入，直接支持解析 SMILES（不需要转 CIF）。
  - 支持传入 `msa_dir` 寻找已有的 MSA。
- 推理脚本: `casp17_ligand/models/rf3_inference.py`
  - 配置: `configs/model/rf3_inference.yaml`
- 输出: `outputs/rf3/{dataset}/L{X}_seed-{N}/L{X}/`
  - `L{X}_model.cif` (包含结构)
  - `L{X}_ranking_scores.csv` (包含 plddt, ptm, iptm 等排行得分)
  - `L{X}_summary_confidences.json`

### 运行命令模板
```bash
# 1. 准备 JSON 输入
conda run -n casp17_ligand python casp17_ligand/data/rf3_input_preparation.py dataset=casp16_l1000 series=L1000

# 2. 运行推理 (强烈建议在 rf3 环境内运行以开启 Python API 加速)
PYTHONPATH=$PWD conda run -n rf3 python casp17_ligand/models/rf3_inference.py dataset=casp16_l1000 num_seeds=10 diffusion_batch_size=5
```

## Boltz-2 Exp-01 运行状态

### 参数
| 参数 | 值 |
|------|-----|
| diffusion_samples | 50 |
| recycling_steps | 10 |
| sampling_steps | 200 |
| step_scale | 1.5 |
| use_potentials | true |
| use_msa_server | true（本地 MSA 优先） |

> **注意：从现在起，所有 Boltz-2 运行均不再计算 affinity（不传 `--affinity_mw_correction`、`--diffusion_samples_affinity`、`--sampling_steps_affinity`）。Affinity 预测耗时且当前不需要。**

## Boltz-2 Exp-02（Round 2 补充数据）

目标：每个 CASP16 target 再补充 50 个结构（与 Exp-01 合计 100 个）。

### 参数变更
| 参数 | Exp-01 | Exp-02 |
|------|--------|--------|
| diffusion_samples | 50 | **5** |
| step_scale | 1.5 | **1.2** |
| seed | null（随机） | **10 个固定 seed**（42,123,256,314,500,617,789,888,1024,2025） |
| affinity | 开启 | **关闭** |

每个 seed 生成 5 个结构，10 seeds × 5 = 50 个/target。

### 输出路径
`outputs/boltz2/{dataset}_r2/seed_{N}/boltz_results_{target}_input/predictions/B/`

### 运行脚本
- L1000: `scripts/run_boltz2_r2_l1000.sh`（GPU 0）

### 数据集
- L1000: 17 struct targets → GPU 0
- L2000 struct: 2 struct targets (L2001, L2002) → **不做 affinity**（节省时间，predict_affinity=false）
  - L2002（单体 ~255aa）：✅ 50 CIF 生成完毕（`outputs/boltz2/casp16_l2000_struct/`）
  - L2001（单体 ~255aa）
- L3000 struct: 189 struct targets → GPU 1
- 输入来源: `data/casp16_data/sequences/L{X}000.txt` + `data/casp16_data/smiles/L{X}000/{target}.tsv`
- struct 目标列表: `data/casp16_data/struct/L{X}000_prepared/`

## Hellbender HPC 自动化工作流

针对在本地和远程 HPC 集群 (Hellbender) 之间穿梭执行预测的任务，我们在项目根目录的 `hellbender/` 文件夹下实现了完全自动化的工作流脚本，分 Boltz-2 和 RF3 两套。

### 目录结构
```bash
hellbender/
├── # ── Boltz-2 （逐 target 独立 SLURM job）──────────────────
├── 01_sync_inputs.sh        # 同步 YAML inputs 到集群
├── 02_submit_jobs.sh        # 按 target 生成 SLURM 脚本并提交
├── 03_sync_outputs.sh       # 结果回传本地
├── run_boltz2_template.sh   # SLURM 单 target 模板
│
└── # ── RF3 （SLURM array job，一次提交全部 targets）──────────
    ├── rf3_00_setup_env.sh      # 一次性：同步 foundry 源码+权重，建 conda env
    ├── rf3_01_sync_inputs.sh    # 同步 rf3_inputs/*.json 到集群
    ├── rf3_02_submit_array.sh   # 生成并提交 SLURM array job
    ├── rf3_03_sync_outputs.sh   # 结果回传本地（支持增量，可边跑边同步）
    ├── rf3_slurm_template.sh    # SLURM array job 模板（含 {dataset} 占位符）
    └── rf3_infer_single.py      # Python 推理脚本（随 rf3_01 同步到集群）
```

### RF3 完整使用流程

```bash
# 在 LOCAL 项目根目录执行以下步骤：

# Step 0：一次性环境搭建（foundry 更新后也需重跑）
bash hellbender/rf3_00_setup_env.sh

# Step 1：同步输入数据（每次新数据集执行）
bash hellbender/rf3_01_sync_inputs.sh casp16_l3000

# Step 2：提交计算（自动识别 target 数量，生成 array job）
bash hellbender/rf3_02_submit_array.sh casp16_l3000        # 默认 20 并发
bash hellbender/rf3_02_submit_array.sh casp16_l3000 10     # 指定并发数

# Step 3：结果回传（支持增量/边跑边同步）
bash hellbender/rf3_03_sync_outputs.sh casp16_l3000
# 如有历史遗留的非标准路径：
bash hellbender/rf3_03_sync_outputs.sh casp16_l3000 CASP17_rf3_l3000
```

### RF3 踩坑记录
- **分区权限**：`chengji-lab-gpu` 仅限 chengji-lab 组，需用 `gpu` 分区（2天时限足够，每 target ~36min）
- **logging 走 stderr**：Python 推理日志在 `.err` 文件而非 `.log`（`tail -f rf3_{JOB}_{TASK}.err`）
- **skip_existing**：`rf3_02_submit_array.sh` 可重复提交，已完成的 seed 自动跳过，支持断点续跑
- **foundry 路径**：Hellbender `~/data/` → `/mnt/pixstor/data/lwfvx/`，RF3 从此路径加载

### RF3 效果排查（2026-03-09）

**现象**：RF3 对 L1001 的 ranking_score 仅 0.49-0.55，iptm ~0.56，效果远低于预期。

**已排查问题**：

| 问题 | 状态 | 影响 | 详情 |
|------|------|------|------|
| num_steps=50 vs 官方推荐 200 | ⚠️ 轻微 | 200-step 平均分高 ~0.01，下限更稳定，但非致命 | 50-step 最高 0.5512 vs 200-step 最高 0.5519 |
| MACE-OMOL 原子级嵌入全部缺失 | ❌ **严重** | 384 维核心特征全部置零，相当于永远 100% dropout | 见下方详细说明 |
| GitHub Issue #97（CCD 命名冲突） | ✅ 不适用 | 我们用 SMILES 输入，AtomWorks 自动分配 `L:0` 不在 CCD 中 | 不触发模板冲突 |
| GitHub Issue #139（形式电荷/氢原子） | ✅ 不适用 | SMILES 路径不走 PDB 的 `fix_formal_charges` 流程 | `smiles_to_rdkit(sanitize=True)` 已处理 |
| 水分子剔除 | ✅ 无问题 | 输入本身不含水，`parse_atom_array` 默认 `remove_waters=True` | — |

**MACE-OMOL 嵌入缺失（关键问题）**：

- **根因**：`forks/foundry/models/rf3/src/rf3/data/pipelines.py:154` 硬编码了 IPD 内部路径 `/net/tukwila/lschaaf/datahub/MACE-OMOL-Jul2025/mace_embeddings`，我们的环境不存在此路径
- **影响链路**：`LoadCachedResidueLevelData` → 所有残基 miss → `RandomSubsampleCachedConformers` 返回空 → `FeaturizeAtomLevelEmbeddings` 返回全零张量
  - `atom_level_embedding`: zeros(n_conformers, L, 384)
  - `has_atom_level_embedding`: zeros(L)（全部标记为"无嵌入"）
- **模型架构**：`c_atom_1d_features = 393 = 392 + 1(has_atom_level_embedding)`，`use_atom_level_embedding = true`，嵌入占 384/393 维
- **训练时**：`p_dropout_atom_level_embeddings = 0.5`（50% 概率 dropout）
- **推理时配置**：`p_dropout_atom_level_embeddings = 0.0`（应始终提供）
- **我们的实际情况**：等效于 100% dropout，偏离训练分布，可能显著降低配体预测质量
- **解决方案**：需获取 MACE-OMOL 预计算嵌入数据，或用 MACE-OMOL 公开模型自行计算。`foundry install` 不含此数据，可能需联系 IPD 或在 GitHub 提 issue

### Hellbender 远端目录约定（RF3）
```
~/data/
├── rf3_env/               # conda 环境（Python 3.12 + PyTorch 2.7.1+cu126 + foundry[rf3]）
├── rf3_cache/             # RF3 权重（rf3_foundry_01_24_latest_remapped.ckpt, 2.9GB）
├── foundry/               # rc-foundry 源码（editable install）
└── rf3_{dataset}/         # 每个数据集独立目录
    ├── inputs/            # *.json 输入文件（从本地 rf3_inputs/ 同步）
    ├── outputs/           # 预测结果（{target}_seed-{N}/...）
    ├── logs/              # SLURM 日志（*.log stdout, *.err stderr+推理日志）
    └── scripts/           # SLURM 脚本 + rf3_infer_single.py
```

### Boltz-2 使用范例
```bash
# 同步输入
./hellbender/01_sync_inputs.sh data/test_cases/casp16_l3000/boltz2_inputs CASP17_2lig

# 提交（按 target，支持 A100/H100）
./hellbender/02_submit_jobs.sh CASP17_2lig casp16_l3000_2lig A100 L3007 L3008 L3019

# 结果回传
./hellbender/03_sync_outputs.sh casp16_l3000_2lig outputs/boltz2/casp16_l3000_struct
```

### Boltz-2 踩坑记录
- **OOM**：A100 80GB 跑单配体足够；2配体（846aa）需 H100 94GB
- **`--no_kernels`**：Hellbender boltz 环境缺 `cuequivariance_ops_torch`，模板已加此参数
- **动态输出路径**：输出目录名是 `boltz_results_{target}_input/predictions/{target}_input/`

## 配体质子化预处理 (Protonation)

针对 AlphaFold3 / Boltz-2 等模型，配体的输入形式为 SMILES。实验证明部分模型对质子化状态敏感，而基础的 SMILES 往往是非质子化的纯拓扑结构。
我们的 `casp17_ligand.data.components` 模块集成了基于 **Dimorphite-DL** 的自动质子化流程（pH 默认 7.4）：

1. **工作流**：在生成 YAML/JSON 输入时，若开启 `protonation=True`（或对应命令参数）选项，管线会自动将 TSV 提取的原始 SMILES 字符串传递给 Dimorphite-DL。
2. **后处理**：Dimorphite-DL 返回最有可能的质子化 SMILES。为了兼容各级底层解析器（如 RDKit），质子化后的异构分子会被统一进行进一步的标准化和重对齐清洗后再输出。
3. **评测结论 (Boltz-2)**：经过对 L4001/L4003/L4025 的针对性实测，发现：在 Boltz-2 中，**输入质子化与非质子化的 SMILES，最终预测出的 lDDT-PLI 得分几乎完全一致**（差距 < ±0.005，在扩散模型的随机性误差范围内）。目前的初步结论是，对于 Boltz-2 而言，提前做质子化对最终结构准确度没有显著的定向提升。

## OpenStructure 评测（lDDT-PLI / RMSD）

两种方式评测配体预测质量，结果完全一致（L1001 验证：`lddt_pli = 0.98918`）。

### 方式一：Docker（现有管线）

使用项目已有的 Docker 评测脚本，调用 `ost compare-ligand-structures`。

```bash
# 参考 casp17_ligand/utils/pocket_metrics.py 中的 Docker 调用逻辑
docker run --rm \
  -v $(pwd):/data \
  openstructure:latest \
  ost compare-ligand-structures \
    -m /data/model.pdb \
    -ml /data/model.sdf \
    -r /data/ref_protein.pdb \
    -rl /data/ref_ligand.sdf \
    -o /data/out.json \
    --lddt-pli --rmsd
```

### 方式二：本地 Conda 环境

#### 环境搭建

```bash
conda create -n ost_test python=3.7 -c conda-forge
conda activate ost_test
conda install -c conda-forge -c zjack openstructure==2.5.0
pip install networkx rdkit pandas scipy spyrmsd
```

#### 参考配体处理：RDKit 模板匹配

**核心问题**：我们的参考配体 PDB 文件（如 `ligand_201_C_1.pdb`）没有键级信息（单键/双键），且原子名非标准（不匹配 `compounds.chemlib` 字典），无法直接用于 OpenStructure。

**解决方案**：用 RDKit 的 `AssignBondOrdersFromTemplate` 从 SMILES 模板恢复正确的键级：

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 从 PDB 读坐标（键级全是单键）
pdb_mol = Chem.MolFromPDBFile("ligand.pdb", removeHs=True, sanitize=False)

# 2. 从 SMILES 读正确的化学拓扑
template = Chem.MolFromSmiles("CCOC(=O)N(C)Cc1c...")  # 来自 TSV 文件

# 3. 合并：PDB 坐标 + SMILES 键级
fixed_mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
Chem.SanitizeMol(fixed_mol)

# 4. 导出 SDF
writer = Chem.SDWriter("ref_ligand.sdf")
writer.write(fixed_mol)
writer.close()
```

SMILES 来源：`data/casp16_data/smiles/L{X}000/{target}.tsv`（第3列）。

#### 评测命令

```bash
conda run -n ost_test python tests/test_ost_eval.py
# 或者直接调用 CLI：
conda run -n ost_test ost compare-ligand-structures \
  -m model.pdb -ml model.sdf \
  -r ref_protein.pdb -rl ref_ligand.sdf \
  -o out.json --lddt-pli --rmsd
```

#### 输入文件格式

| 文件 | 格式 | 来源 | 说明 |
|------|------|------|------|
| 模型蛋白 | PDB | 预测输出 | 只含 ATOM 记录 |
| 模型配体 | SDF | 预测输出 | 已有正确键级 |
| 参考蛋白 | PDB | `L{X}000_prepared/{target}/protein_aligned.pdb` | 标准残基 |
| 参考配体 | SDF | RDKit 模板匹配生成 | 从配体 PDB + SMILES 重建 |

#### 已探索但不采用的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| `RuleBasedProcessor` + `compounds.chemlib` | ❌ 失败 | 我们的 PDB 原子名非标准，字典匹配不上 |
| `HeuristicProcessor`（距离+CONECT建键） | ⚠️ 能跑通 | 只建连通性不建键级，对称性判断可能有误 |
| RDKit `AssignBondOrdersFromTemplate` | ✅ **采用** | 精确恢复键级，结果与 Docker 一致 |

#### 待确认

- [x] 找到 MULTICOM 处理 CASP16 评测的正确脚本后，对比其做法是否有优化空间 (已检查 `data_utils.py`，他们使用了 prody 和 openbabel，我们的 RDKit 模板匹配更直接精确)
- [x] 基于 `test_ost_eval.py` 开发批量评测脚本 `openstructure_eval.py` (已实现 `compare_lddt.py` 进行多靶点对比评测)

### 测试脚本

`tests/test_ost_eval.py`：完整的单 case 评测示例（L1001），验证本地评测与 Docker 结果一致。

## Protenix-v1 集成

### 运行方式
**Conda 环境直接运行**（已放弃 Docker，兼容性问题）。conda env: `protenix`。

### 环境部署（一次性，已完成）

```bash
# 1. 创建环境（protenix 已以 editable 模式安装）
# conda env 在 /home/lwfvx/miniforge3/envs/protenix

# 2. 安装 CUDA 编译依赖（V100 + CUDA 12.6）
conda install -n protenix -c conda-forge cuda-nvcc=12.6.77 cuda-toolkit=12.6.2 cuda-cudart-dev=12.6.77 -y

# 3. 编译 CUDA extension（首次运行自动触发，或手动：）
TORCH_CUDA_ARCH_LIST="7.0;8.0" conda run -n protenix python /tmp/compile_layer_norm.py
# → 生成 forks/Protenix/protenix/model/layer_norm/fast_layer_norm_cuda_v2.so

# 4. 下载模型权重（首次 pred 自动下载，存至 weights/protenix/）
# weights/protenix/checkpoint/protenix_base_default_v1.0.0.pt  (1.4GB)
# weights/protenix/common/components.cif + .rdkit_mol.pkl
```

**关键说明**：
- V100 (Compute Capability 7.x) → protenix 自动强制 FP32（`dtype=bf16` 会被覆盖）
- `torch_ext_compile.py` 已修改：自动检测 conda CUDA 头文件路径（`targets/x86_64-linux/include/`）

### MSA/Template 搜索（同蛋白只需一次）

```bash
# L1000 全部17个 target 共享同一蛋白序列，只需搜索一次
cd /bmlfast/Lyuwei/0.Projects/CASP17_ligand
PROTENIX_ROOT_DIR=weights/protenix CUDA_VISIBLE_DEVICES=2 \
conda run --no-capture-output -n protenix protenix mt \
  --input data/test_cases/casp16_l1000/protenix_inputs/L1001.json \
  --out_dir data/test_cases/casp16_l1000/protenix_msa \
  --seqres_database_path /bmlfast/databases/pdb_seqres_2022_09_28.fasta

# 输出（protenix_msa/L1001/ 下）：
#   pairing.a3m        → paired MSA
#   non_pairing.a3m    → unpaired MSA
#   hmmsearch.a3m      → template 搜索结果（供 use_template=true 使用）
```

### Input Preparation（注入 MSA/Template 路径）

```bash
# 搜索完成后，重新生成带 MSA 路径的输入 JSON
conda run -n MULTICOM_ligand python casp17_ligand/data/protenix_input_preparation.py \
  paired_msa_path=data/test_cases/casp16_l1000/protenix_msa/L1001/pairing.a3m \
  unpaired_msa_path=data/test_cases/casp16_l1000/protenix_msa/L1001/non_pairing.a3m \
  templates_path=data/test_cases/casp16_l1000/protenix_msa/L1001/hmmsearch.a3m
```

### 推理（最优参数）

```bash
conda run -n MULTICOM_ligand python casp17_ligand/models/protenix_inference.py dataset=casp16_l1000
```

Config（`configs/model/protenix_inference.yaml`）关键参数：
```yaml
env_name: protenix
gpu_device: 2
protenix_root_dir: /bmlfast/Lyuwei/0.Projects/CASP17_ligand/weights/protenix
model_name: protenix_base_default_v1.0.0
seeds: "101,102,103,104,105,106,107,108,109,110"   # 10 seeds
n_sample: 5           # 每 seed 5 个样本 → 共 50 个模型/target
cycle: 10
step: 200
use_msa: true
use_template: true    # 需要 protenix_msa 路径已注入 JSON
enable_cache: true
dtype: bf16           # V100 自动降为 fp32
```

### 输出结构

```
outputs/protenix/casp16_l1000/
└── L1001/
    └── seed_101/
        └── predictions/
            ├── L1001_sample_0.cif
            └── L1001_summary_confidence_sample_0.json
```

置信度字段：`plddt`, `gpde`, `ptm`, `iptm`, `ranking_score`。

### 硬件环境
- GPU 2: NVIDIA Tesla V100 32GB
- 资源限制：CPU 核心和内存最多独占服务器的 **三分之一**（总计 192 核、1TB 内存，即最多约 **64 CPU**, **320G 内存**）。

### 资源隔离
为防止 OOM 杀死宿主机，使用相似的 docker 限制参数：
```bash
--memory=320g              # 物理内存上限 (总 1TB 的 1/3)
--memory-swap=320g         # 禁止 swap
--cpus=64.0                # CPU 限制 (总 192 核的 1/3)
--shm-size=8g              # 共享内存
--user=$(id -u):$(id -g)   # 文件权限映射
```

### 高质量预测推荐参数 (CASP17 配置)
Protenix 强烈推荐在 V1.0.0 模型上开启以下参数，以追求对复杂配体的高精度：
| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | `protenix_base_default_v1.0.0` | 精度最高的 v1 模型 |
| `use_template` | `true` | 使用 Template 模板进行预测 |
| `seeds` | `101,102,...,110...` | 跑 10~20 个 seed (逗号分隔) |
| `sample` | `5` | 每个 seed 采样的数量 (配合 20 个 seed，生成 100 个模型) |
| `cycle` | `10` | 迭代精化 recycling steps |
| `step` | `200` | 扩散步数 |
| `enable_cache` | `true` | 必须开启：共享缓存加速 |

### 动态显存控制
Protenix 对显存占用有自己的动态调整策略。针对我们的 V100 32GB 显卡，遇到大体系时，推理代码中会自动检测 `N_token` 数量。遇到超大系统会触发 BF16 AMP mixed precision，无需手动改动代码，能保证在 32GB 上安全运行。

### MSA/Template 复用策略（与 AF3/Boltz-2 同理）

Protenix 拥有自己的 MSA/Template 搜索管线，**不复用 AF3 的 MSA**。策略与 AF3 的 `--norun_inference` / `--norun_data_pipeline` 和 Boltz-2 的 MSA CSV 复用完全同理：**同一蛋白质序列只搜索一次 MSA+Template，之后所有配体共享结果。**

#### 数据管线 CLI

| 命令 | 说明 |
|------|------|
| `protenix msa` | 仅蛋白质 MSA 搜索（支持 JSON 或 FASTA 输入） |
| `protenix mt` | MSA + Template 搜索 |
| `protenix prep` | 全流程：蛋白质 MSA + Template + RNA MSA 搜索 |

```bash
# Step 1: 用第一个配体的 JSON（或仅含蛋白质的 JSON）跑 MSA+Template
protenix mt -i first_target.json -o ./msa_output

# 输出文件（在 msa_output 目录下）：
#   pairing.a3m        → paired MSA（含 taxonomy ID 用于 pairing）
#   non_pairing.a3m    → unpaired MSA
#   hmmsearch.a3m      → template 搜索结果
```

#### 复用方式

后续所有配体的 JSON 里，直接引用预计算的 MSA/Template 路径：

```json
{
  "proteinChain": {
    "sequence": "MKTAYIAKQ...",
    "count": 1,
    "pairedMsaPath": "/abs/path/to/pairing.a3m",
    "unpairedMsaPath": "/abs/path/to/non_pairing.a3m",
    "templatesPath": "/abs/path/to/hmmsearch.a3m"
  }
}
```

推理时即可跳过 MSA 搜索步骤，用 `protenix pred` 直接推断：
```bash
protenix pred -i target_with_msa.json -o ./output --use_template true
```

#### 外部依赖

`protenix mt` / `protenix prep` 依赖 `kalign` 和 `hmmer`（Docker 镜像已预装）。非 Docker 环境需手动安装：
```bash
apt-get install -y kalign hmmer
```
还需 `pdb_seqres` 数据库（`pdb_seqres_2022_09_28.fasta`），首次运行自动下载至 `$PROTENIX_ROOT_DIR/search_database`。

### 配体输入格式
Protenix 支持三种配体输入方式，完全覆盖我们的 CASP 场景：
1. **SMILES 字符串**：直接传入（如 `"ligand": "Nc1ncnc2..."`）
2. **结构文件**：SDF/MOL/PDB（如 `"ligand": "FILE_/path/to/ligand.sdf"`）
3. **CCD 编码**：（如 `"ligand": "CCD_ATP"`）

还支持 `covalent_bonds`（共价抑制剂）和 `pocket` constraint（口袋约束）。

## Ensemble Generation（排序管线）

### 新数据集接入 Checklist

当需要为新数据集（如 `casp16_l2000_struct`）接入 ensemble 管线时，需完成以下前置步骤：

**Step 0：确认前置数据已就绪**

各方法（boltz2/af3/protenix/seedfold）的推理输出必须已存在：
```
outputs/boltz2/{dataset}/boltz_results_{target}_input/predictions/...
outputs/alphafold3/{dataset}/{target}/...
outputs/protenix/{dataset}/{target}/seed_*/predictions/...
outputs/seedfold/{dataset}/{target}/...
```

同时确认参考数据（用于评测）存在：
```
data/casp16_data/struct/L{X}000_prepared/{target}/protein_aligned.pdb  # 参考蛋白
data/casp16_data/struct/L{X}000_prepared/{target}/ligand_*.pdb         # 参考配体
data/casp16_data/smiles/L{X}000/{target}.tsv                          # SMILES
```

**Step 1：创建 `ensemble_inputs.csv`**

位置：`data/test_cases/{dataset}/ensemble_inputs.csv`

格式（逗号分隔，4 列）：
```csv
target,protein_sequence,ligand_smiles,experimental_affinity
L2001,MQPLLLLL...,CC(=O)NCC...,
L2002,MQPLLLLL...,COC(=O)N(C)...,
```

数据来源：
- `protein_sequence`：从 `data/casp16_data/sequences/L{X}000.txt` 提取（去掉 `>` header，拼接多行）
- `ligand_smiles`：从 `data/casp16_data/smiles/L{X}000/{target}.tsv` 第 3 列提取
- `experimental_affinity`：可留空（仅 affinity 评测需要）

**Step 2：创建 Hydra config**

位置：`configs/model/ensemble_generation_{dataset_suffix}.yaml`

复制已有 config（如 `ensemble_generation.yaml`）并修改 `dataset` 字段：
```yaml
dataset: casp16_l2000_struct          # 改为新数据集名
input_csv: data/test_cases/${dataset}/ensemble_inputs.csv
ensemble_output_dir: outputs/ensemble/${dataset}

methods:
  boltz2:
    output_dir: outputs/boltz2/${dataset}
    n_models: 100
  af3:
    output_dir: outputs/alphafold3/${dataset}
    n_models: 100
  protenix:
    output_dir: outputs/protenix/${dataset}
    n_models: 100
  seedfold:
    output_dir: outputs/seedfold/${dataset}
    n_models: 100

num_workers: 16                       # 按 target 数量和 CPU 调整
export_top_n: null
prefilter_metric: null
prefilter_topn: null
```

**Step 3：运行 ensemble + 评测 + 参数搜索全流程**

```bash
# 3a. Ensemble 生成（CIF→PDB+SDF 转换 + 5 种共识排序 + PoseBusters 验证）
conda run -n casp17_ligand python casp17_ligand/models/ensemble_generation.py \
    --config-name ensemble_generation_{dataset_suffix}

# 3b. Ground Truth 评测（OST Docker lDDT-PLI / RMSD）
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/evaluate_ensemble.py \
    --ensemble_dir outputs/ensemble/{dataset} \
    --reference_dir data/casp16_data/struct/L{X}000_prepared \
    --smiles_dir data/casp16_data/smiles/L{X}000

# 3c. 参数搜索（7 种 metric 并行）
for metric in pair_iptm ligand_plddt combined_iptm_plddt pocket_plddt_4.5 pocket_plddt_6 pocket_plddt_8 pocket_plddt_10; do
    conda run -n casp17_ligand python casp17_ligand/analysis/evaluate_topn_benefits.py \
        --dataset {dataset} --metric $metric &
done
wait

# 3d. 汇总参数搜索结果（多 dataset 合并）
conda run -n casp17_ligand python casp17_ligand/analysis/aggregate_topn_results.py \
    --datasets casp16_l1000 casp16_l2000_struct casp16_l3000_struct
```

### 运行方式
```bash
conda run --no-capture-output -n casp17_ligand python casp17_ligand/models/ensemble_generation.py
```

### 两阶段参数搜索流程

最优 ensemble 参数（metric、Top-N、consensus 方法）的搜索分两个阶段：

**阶段一：全量 ensemble（不 prefilter）**
- 配置 `prefilter_metric: null`，`prefilter_topn: null`
- 所有模型参与 pairwise 计算 → 生成完整的 `pairwise_{rmsd,sucos,rmsd_6a}_cache.json`
- `ensemble_generation.py` 只负责生成 pairwise cache 和排序输出

**阶段二：参数搜索（从 cache 中模拟 Top-N 子集共识）**
- `evaluate_topn_benefits.py --dataset X --metric Y`：从全量 cache 中取 Top-N 子集，在子集内算共识 Rank-1
- 搜索空间：7 种 metric × Top-{10,20,30,40,all} × 5 种 consensus = 175 种组合
- 输出 `topn_benefits_{metric}.json`，`aggregate_topn_results.py` 汇总成表格

**阶段三（生产部署）：确定最优参数后**
- 配置 `prefilter_metric: pair_iptm`，`prefilter_topn: 30`（举例）
- 直接跑 ensemble，只对 Top-30 做 pairwise，节省计算时间

### 模型命名规范
每个方法的模型用 `{method}_model{canonical_idx}` 命名，`canonical_idx` 由 `_get_ensemble_cif_list()` 的规范排序决定（即 sorted glob 的顺序）。无论是否 prefilter，同一个 CIF 文件始终对应同一个 `canonical_idx`，保证 pairwise cache 的 key 跨运行可比。

### Pairwise Cache 格式
位置：`outputs/ensemble/{dataset}/targets/{target}/pairwise_{rmsd,sucos,rmsd_6a,rmsd_8a,rmsd_10a}_cache.json`
```json
{
  "af3_model0": {
    "af3_model1": 1.215,
    "boltz2_model3": 0.554,
    ...
  },
  ...
}
```
每个 key 对是一对模型之间的 RMSD/SuCOS/RMSD_6A/RMSD_8A/RMSD_10A 值。增量更新：只计算 cache 中缺失的模型对。

### casp17_ligand 环境
专用 orchestration 环境，包含 ensemble 管线所有依赖：
- `conda create -n casp17_ligand python=3.10`
- `conda install -n casp17_ligand -c conda-forge pymol-open-source`
- `pip install hydra-core omegaconf pandas numpy rootutils rdkit prody biopython biopandas beartype posebusters scipy spyrmsd requests tqdm`
- `pip install -e .`（casp17_ligand 包本身）

### CIF → PDB + SDF 转换（`cif_to_pdb_sdf`）

支持的链名格式：
| 方法 | 蛋白链 | 配体链 |
|------|--------|--------|
| boltz2, af3, protenix, rf3 | `A` | `B` |
| seedfold | `A0` | `B0` |

#### MCS Fallback 键级修复

**问题**：Protenix CIF 输出在经过 RDKit `MolFromPDBBlock` 距离推断时，会多推断出一根额外的键（如 L1008 出现 36 bonds 而非正确的 35 bonds）。`AssignBondOrdersFromTemplate` 因为连通性不匹配而失败。

**解决方案**：当 ProDy 提取的 SDF 键数与 SMILES 模板不匹配时，自动使用 MCS fallback：
1. PyMOL 从 CIF 导出配体为 MOL2（坐标正确，键可能错）
2. SMILES → RDKit 模板（拓扑正确）
3. `rdFMCS.FindMCS` + `BondCompare.CompareAny` 做原子映射
4. 将 MOL2 坐标搬到 SMILES 模板上
5. 写出正确的 SDF

**验证结果**：L1008 protenix 50 个模型全部修复（36→35 bonds）。

### SeedFold 集成

输出目录结构：
```
outputs/seedfold/{dataset}/
└── {target}/
    └── {target}_model_{N}/
        ├── {target}_model_{N}_model_{M}.cif
        └── confidence_{target}_model_{N}_model_{M}.json
```

配置：`configs/model/ensemble_generation.yaml` 中 `seedfold.output_dir`。

## 预测模型与评测指标 (Empirical Findings)

我们在 L1000 测试集上对多个模型置信度指标（Confidence Metrics）进行了深度消融实验与分析，得到以下指导性结论：

1. **统一的高分偏差与靶点难度耦合效应**
   - 无论是 `pair_chains_iptm` 还是其他复合指标，在跨 target 的总体相关性计算中看似有不错的正相关（Pearson 0.5 - 0.7），但这种相关性主要来自于"容易靶点高分、困难靶点低分"。
   - **单 target 内部方差极小**：在同一个靶点的 50 个模型之间，`pair_chains_iptm` 差异往往不到 `0.02`，这导致该指标在同靶点模型内部筛选时**缺乏足够的区分度**。

2. **跨模型横向比对（预过滤最佳指标）：`pair_chains_iptm` vs `ligand_pLDDT`**
   - 提取了各个模型输出 `.cif` 文件的 ligand `B_iso_or_equiv` 值作为 **局部 pLDDT (`ligand_pLDDT`)**。测试表明，尽管它和 `pair_chains_iptm` 一样能指示相对置信区间，但它的 Self-Top1 表现与 `pair_chains_iptm` 近似甚至稍弱，在筛选好坏模型时也没有展现出显著更大的 intra-target 区分度。
   - **PAE/PDE 指标缺陷**：测试了 AF3 (`chain_pair_pae_min`)、Boltz-2 (`complex_ipde`) 与 Protenix (`chain_pair_gpde`) 的距离误差指标。在多个 Target 上，这些分数与 lDDT-PLI 甚至呈现**强负相关**，不适合提取作跨模型通用排序。

**系统集成策略总结：**
推荐在主干筛选流程中保留 **`pair_chains_iptm` 和 `ligand_pLDDT` 作为参数化可切换指标**。但这二者主要用作 **预筛池（Pre-filtering Top-K 控制）** 或剔除明显崩溃的模型，最终单目标的精细化 Top-1 排序依然由更为敏感的 **跨方法 Structural Consensus (RMSD/SuCOS)** 模块决定。

## L3000 Autotaxin 双金属 ZN 中心与 N-糖基化

### 背景（来自 CASP 竞赛描述）

> The autotaxin protein carries a N-glycosylation on residue Asn497, and was considered as part of the structure (see pdb id 5m7m as example). It contains as well multiple ions: 2 zinc ions are present in the active site. These ions were also considered as part of the autotaxin structure. The position of the carbohydrate and ions won't be taken into account for the final scoring.

参考结构 5M7M 中共有 4 个 ZN 离子，但只有 2 个是催化活性中心的双金属离子，另外 2 个是晶体假象（artifact）。

### 5M7M ZN 配位分析

| ZN | PDB resnum | 配位残基 (5M7M PDB编号) | 身份 |
|----|------------|------------------------|------|
| **#1 (res 915)** | — | **Asp312.OD1 (2.01Å), His316.NE2 (2.19Å), His475.NE2 (2.07Å)** + 水 | **催化 ZN** ✓ |
| **#2 (res 916)** | — | **Asp172.OD1 (1.99Å), Asp359.OD2 (1.99Å), His360.NE2 (2.07Å), Thr210.OG1 (2.01Å)** | **催化 ZN** ✓ |
| #3 (res 917) | — | 无蛋白接触 (< 3Å) | 晶体 artifact ✗ |
| #4 (res 918) | — | 表面 Asp740,742,744,748 | 表面 artifact ✗ |

两个催化 ZN 通过桥接水分子 HOH1003 连接，Thr210 是执行磷酸二酯水解的核心残基。

### L3000 序列编号映射

L3000 序列（846 残基）与 5M7M PDB 编号存在 **偏移量 = 28**（L3000 pos N → PDB resnum N + 28）。

催化残基在 L3000 序列中的验证：

| 5M7M PDB resnum | L3000 pos | 氨基酸 | 角色 |
|-----------------|-----------|--------|------|
| Asp312 | **284** | D ✓ | ZN#1 配位 |
| His316 | **288** | H ✓ | ZN#1 配位 |
| His475 | **447** | H ✓ | ZN#1 配位 |
| Asp172 | **144** | D ✓ | ZN#2 配位 |
| Asp359 | **331** | D ✓ | ZN#2 配位 |
| His360 | **332** | H ✓ | ZN#2 配位 |
| Thr210 | **182** | T ✓ | 催化核心 |

### Protenix 输入方案（_zn 变体）

为与原始无 ZN 的 L3000 输入区分，带 ZN 的输入和输出使用 `_zn` 后缀：

- 输入 JSON：`data/test_cases/casp16_l3000_struct_zn/protenix_inputs/`
- 输出：`outputs/protenix/casp16_l3000_struct_zn/`
- Config：`configs/model/protenix_inference_l3000_zn.yaml`

每个 L3000 target 的 JSON 中增加：
1. **2 个 ZN ion entity**：`{"ion": {"ion": "ZN", "count": 1}}` × 2
2. **Contact constraints**：将每个 ZN 锚定到对应配位残基（使用 L3000 序列编号）
   - ZN#1 → Asp284, His288, His447
   - ZN#2 → Asp144, Asp331, His332
3. 原有的配体-配体 contact constraints 保持不变

N-糖基化（Asn497 → L3000 pos 469）暂不加入输入（竞赛描述说不计入评分）。

## L3000 评测结果与 Ensemble (AF3 + SeedFold)

在 189 个 struct targets 上的评测结果汇总（去除 L3103）：

### Top-1 / Top-5 核心指标 (STRUCT)
| 指标 | 值 |
|------|-----|
| Oracle Best (全模型) | 0.910 |
| AF3 Oracle | 0.879 |
| SeedFold Oracle | 0.827 |
| Consensus RMSD Top-1 | 0.795 |
| Consensus SuCOS Top-1 | 0.803 |
| Top-1 Success >0.7 | RMSD 79.8%, SuCOS 80.3% |
| Top-5 Success >0.7 | 95.7% (both) |
| Baseline Multicom | 0.589 |
| Baseline Champion | 0.681 |

### Self-Ranking 对比
| 方法 | Oracle | Self-Top1 | Self-Top5 | Self-Top10 |
|------|--------|-----------|-----------|------------|
| AF3 | 0.879 | 0.814 | 0.843 | 0.854 |
| SeedFold | 0.827 | 0.723 | 0.752 | 0.763 |
| Consensus RMSD | - | 0.795 | - | - |
| Consensus SuCOS | - | 0.803 | - | - |

**问题 target：**
- `L3103`: OST graph isomorphism 匹配失败，所有 lDDT-PLI 为空（手性SMILES/bond order 不匹配）。后续等其他两个模型出结果再看。

**近期修复记录：**
1. `evaluate_ensemble.py`: 增加多 ref ligand 支持、14 worker 多进程评测与 best-score 提取优化。
2. `self_ranking_comparison.py`: 修复 AF3 嵌套路径匹配问题，并增加对 NaN 的处理逻辑。

## 分析脚本说明 (casp17_ligand/analysis/)

### evaluate_ensemble.py — Ground Truth 评测
调用 OST Docker (`compare-ligand-structures`) 对 ensemble 排名结果进行 lDDT-PLI / RMSD 评测。

**功能**：
- 对每个 target 的 ranking_rmsd / ranking_sucos 下的 SDF 文件，运行 Docker 评测
- 支持多配体 target（多个 ref SDF）
- 结果缓存到 `targets/{target}/score_cache_docker.json`，支持断点续跑
- 并行：外层 `multiprocessing.Pool(num_workers=14)`，内层 `ThreadPoolExecutor(docker_threads=4)` 并发调用 Docker

**用法**：
```bash
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/evaluate_ensemble.py \
    --ensemble_dir outputs/ensemble/casp16_l1000 \
    --reference_dir data/casp16_data/struct/L1000_prepared \
    --smiles_dir data/casp16_data/smiles/L1000
```

**输出**：
- `evaluation_summary.csv`：每个 model 的 lDDT-PLI、RMSD、PoseBusters 状态
- `top1_top5_scores_{DATASET}.csv`：逐 target 的 Top-1/Top-5 汇总 + Oracle + Baseline 对比

### evaluate_topn_benefits.py — 参数搜索计算引擎（单 metric 单 dataset）
利用全量 pairwise cache 模拟"先全量 ensemble 再 Top-N 子集共识"的效果，不调用 Docker。

**功能**：
- 输入：一个 (dataset, metric) 组合
- 按指定 metric 对每个方法内部模型排序，取 Top-N 子集
- 从 pairwise cache JSON 读取已有距离值，计算子集内的共识 Rank-1
- 对比 Top-10/20/30/40/all × RMSD/SuCOS/RMSD_6A 的 Rank-1 lDDT
- 输出 `topn_benefits_{metric}.json`（含逐 target 中间结果）

**用法**：
```bash
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/evaluate_topn_benefits.py \
    --dataset casp16_l1000 --metric pair_iptm
```

**输出**：
- stdout：方法平均 lDDT 表 + 共识 Rank-1 lDDT 汇总
- `outputs/ensemble/{dataset}/topn_benefits_{metric}.json`：完整中间结果 JSON

**已知修复 (2026-03-14)**：
- 修复 `_parse_model_source(f"dummy_{pred_name}_")` bug，原代码因 `dummy_` 前缀导致 metric 分数查找全部失败。

### aggregate_topn_results.py — 参数搜索结果汇总（多 metric 多 dataset）
读取所有 `topn_benefits_{metric}.json`，生成 experiments.md 格式的 markdown 表格。

**功能**：
- 读取多个 dataset 下的 7 种 metric 的 JSON 结果
- 分别输出 RMSD/SuCOS/RMSD_6A 三张表（行=metric, 列=Top-N, 值=mean lDDT）
- 输出多 dataset 合并的 target 级平均表
- 标注全局最优 (metric, topn, consensus)

**与 evaluate_topn_benefits.py 的关系**：
- `evaluate_topn_benefits.py` 是**计算引擎**，每次只处理一个 (dataset, metric)，输出一个 JSON
- `aggregate_topn_results.py` 是**报表生成器**，读取所有 JSON，输出汇总 markdown 表格

**用法**：
```bash
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/aggregate_topn_results.py \
    --datasets casp16_l1000 casp16_l3000_struct
```

### prefilter_experiment.py — Oracle 保留率分析
分析预过滤是否会"误杀"各 target 的最佳模型（Oracle）。

**功能**：
- 对每个 target，按 metric Top-N 过滤后，检查 Oracle（最高 lDDT 模型）是否存活
- 统计 Oracle 保留率、过滤后 Oracle 均值、损失 delta

**用法**：
```bash
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/prefilter_experiment.py \
    --dataset casp16_l1000 --metric pair_iptm --topn 10 20 30
```

### confidence_metric_analysis.py — 置信度指标收集
提供 `collect_scores(metric, method, target, ...)` 函数，被上述脚本复用。

支持三种 metric：
- `pair_iptm`：从置信度 JSON 中提取蛋白-配体跨链 iPTM
- `ligand_plddt`：从 CIF 文件的 `B_iso_or_equiv` 字段提取配体局部 pLDDT
- `ranking_score`：从 JSON/CSV 中提取 ranking_score

### evaluate_topn_clusters.py — 聚类预过滤参数搜索引擎
用于全自动在所有有效 Target 上搜索“不同置信度预过滤 (Top-N)”结合“Butina 聚类”的表现，并将最优组合的全局平均得分整理出表。

**功能**：
- 使用 `concurrent.futures.ProcessPoolExecutor` 多进程自动化扫参（支持 L1000/L3000 多数据集同跑）。
- 从 `pairwise_sucos_cache.json` 读取拓扑距离，并运用 Butina Clustering 划分预测模型。
- **输出**：
  1. 终端打印 Markdown 参数组合对比表
  2. `outputs/ensemble/evaluate_topn_clusters_detailed.csv` (包含 205 个 Target 在横向 35 种组合下的具体 top1_center lddt-pli 明细得分)
  3. `outputs/ensemble/evaluate_topn_clusters_summary.csv` (全要素目标平均 Markdown 汇总宽表，方便绘图/查看)

**用法**：
```bash
PYTHONPATH=$PWD conda run -n casp17_ligand python casp17_ligand/analysis/evaluate_topn_clusters.py
```

### self_ranking_comparison.py — Self-Ranking 对比
对比各方法使用自身置信度排序 vs 共识排序的效果。提供 `_get_ensemble_cif_list()` 等工具函数。

### 代码复用关系
- `evaluate_topn_benefits.py` 和 `prefilter_experiment.py` 都 import `confidence_metric_analysis.collect_scores`
- `evaluate_topn_benefits.py` import `ensemble_generation.rmsd_consensus_rank` / `sucos_consensus_rank`（但修复后不再调用，改用缓存查表）
- `evaluate_ensemble.py` 是独立的 Docker 评测脚本，不与其他分析脚本共享核心逻辑
- `_parse_model_source` 在 `evaluate_topn_benefits.py` 和 `prefilter_experiment.py` 中各有一份定义（格式相同）

## L4000 系列特殊说明

### 靶点概述
L4000 系列为 SARS-CoV-2 MPro（主蛋白酶）同源二聚体，306 残基 × 2 链（A+B），共 25 个靶点。每个靶点有 2-6 个配体（同一配体在两个活性位点各一份，部分含溶剂分子 DMS）。

### 黑名单靶点（数据泄露）
**L4006, L4007, L4008, L4009, L4010** 因评测中出现数据泄露，已从 CASP16 评测中作废。所有管线（input preparation、inference、ensemble）应跳过这 5 个靶点。已在 `casp17_ligand/data/components/target_data.py` 的 `BLACKLISTED_TARGETS` 中实现自动过滤。

### 共价抑制剂（Covalent Inhibitors）
以下 4 个靶点的配体与催化位点 **Cys145** 的 SG 原子形成共价键：

| Target | Ligand | SMARTS (attachment atom = first atom) | 说明 |
|--------|--------|---------------------------------------|------|
| L4003 | LIG | `[cX3][nX2][cX3][nX2]` | 芳香碳连接 |
| L4013 | LIG | `[C](=[N])[c]` | 亚胺碳连接 |
| L4019 | LIG | `[C](=[N])[c]` | 亚胺碳连接 |
| L4023 | LIG | `[C](=[N])[c]` | 亚胺碳连接 |

**AF3 共价键处理**：AF3 的 SMILES 输入不支持 `bondedAtomPairs`（无原子名），必须使用 **userCCD** 格式定义共价配体。`af3_input_preparation.py` 中使用 RDKit 从 SMILES 生成 CCD mmCIF，通过 SMARTS 匹配找到连接原子，然后在 JSON 中声明 `bondedAtomPairs: [[["A", 145, "SG"], ["C", 1, "XX"]], [["B", 145, "SG"], ["D", 1, "XX"]]]`（XX 为匹配到的原子名）。

### 活性位点锚定（Pocket Anchoring）

L4000 MPro 的活性位点在 Cys145 附近。为引导配体到正确口袋，各方法的锚定支持情况：

| 方法 | 锚定方式 | 状态 |
|------|---------|------|
| Boltz-2 | `pocket` constraint: 主配体 → 对应 chain 的残基 144/145/146/163/166，`max_distance=10` | ✅ 已实现（`boltz2_inputs_pocket/`） |
| Protenix | `contact` constraint: 主配体 → 对应 chain 的 Cys145，`max_distance=10` | ✅ 已实现 |
| AF3 | JSON 格式不支持 pocket/contact 软约束，仅支持 `bondedAtomPairs`（共价键） | ❌ 无法锚定，依赖 template 引导 |
| SeedFold | 输入格式不支持任何约束 | ❌ 无法锚定 |

Boltz-2 做了两组对比实验：
- `boltz2_inputs/` → `outputs/boltz2/casp16_l4000_no_Cys145/`：仅 dimer 分组 + contact constraints（无 pocket 锚定）
- `boltz2_inputs_pocket/` → `outputs/boltz2/casp16_l4000/`：dimer 分组 + contact + pocket 锚定（**主力版本**）

### 输入格式
- **蛋白质**：使用 `data/casp16_data/sequences/L4000.txt` 中的序列，不使用 PDB 文件
- **二聚体**：`read_protein_sequence()` 已处理，返回 `{"A": seq, "B": seq}`
- **配体**：从 TSV 读取 SMILES，非共价用 SMILES 输入，共价用 userCCD 输入

### 小配体分组与 Contact Constraints（二聚体多配体处理）

MPro 二聚体有两个对称活性位点（chain A 和 chain B）。TSV 中的配体包含主配体（药物分子）和小配体（结晶溶剂 DMS、EDO、2PE 等）。需要将小配体正确分配到两个活性位点的主配体组中。

**注意**：不能通过查看参考结构来确定分组（那是看答案），必须仅根据 TSV 数据推断。

**主配体识别**：按 SMILES 长度排序，最长的两个相同 SMILES 即为主配体（不依赖 "LIG" 命名，因为 L3000 系列中主配体可能叫 L0R 等其他名字）。单配体靶点（如 L4016、L4024）只有 1 个主配体，不需要分组。

**小配体分配算法**：
1. 找出两个主配体 → 分别分配到 Site A 和 Site B
2. 将剩余小配体按 SMILES 相同性分组
3. **相同的小配体先平分**：每组 N 个相同小配体 → Site A 得 floor(N/2)，Site B 得 floor(N/2)，若 N 为奇数则剩 1 个
4. **剩余不同的小配体循环分配**：所有未分配的小配体（包括第3步的余数和独特小配体），round-robin 依次分给 Site A、Site B

**Contact Constraints**：
1. **组内配体-配体约束**：同一 Group 内的配体之间设置 contact constraint（`max_distance=4.5, min_distance=3`），不在跨 Group 之间设置。参考写法见 `data/test_cases/casp16_l3000_struct/protenix_inputs/L3006.json`。
2. **蛋白-配体 pocket 锚定**：每个 Group 的主配体需要通过 contact constraint 锚定到对应 chain 的活性位点残基 **Cys145**（`max_distance=10, min_distance=0`）。没有这个约束，Protenix 无法区分两个对称活性位点，配体可能聚集到同一个 site 或者游离在蛋白表面。

**Cys145 作为 pocket 锚点的依据**（来自 CASP 竞赛描述）：
- 竞赛描述明确指出 "In L4007, Cys145 is chemically modified to S-Hydroperoxycysteine"，确认活性位点半胱氨酸为第 145 位
- 共价抑制剂（L4003, L4013, L4019, L4023）与 "a cysteine residue" 形成共价键，结合上述信息确认为 Cys145
- 所有 25 个复合物均为 MPro 活性位点抑制剂（His41-Cys145 催化二联体）

**示例**：

| Target | TSV 配体 | 主配体 | 小配体分组 | Site A | Site B |
|--------|---------|--------|-----------|--------|--------|
| L4001 | 2×LIG | LIG×2 | 无 | LIG | LIG |
| L4004 | 2×LIG+3×DMS+1×EDO | LIG×2 | DMS×3平分→A:1,B:1,余1; EDO×1余 → round-robin DMS→A, EDO→B | LIG,DMS,DMS | LIG,DMS,EDO |
| L4011 | 2×LIG+1×2PE | LIG×2 | 2PE×1 → round-robin→A | LIG,2PE | LIG |
| L4013 | 2×LIG+2×DMS | LIG×2 | DMS×2平分→A:1,B:1 | LIG,DMS | LIG,DMS |
| L4016 | 1×LIG | LIG×1 | 无 | LIG | — |

**Protenix 共价键处理**：

对于共价靶点（L4003, L4013, L4019, L4023），在 JSON 中添加 `covalent_bonds` 字段：
- Group A 中的主配体 LIG 与 **entity 1（chain A）的 Cys145 SG** 形成共价键
- Group B 中的主配体 LIG 与 **entity 2（chain B）的 Cys145 SG** 形成共价键
- 连接原子通过 SMARTS 匹配从 SMILES 中找到（与 AF3 相同的 SMARTS 表）

Protenix `covalent_bonds` JSON 格式：
```json
"covalent_bonds": [
  {
    "entity1": "1",
    "copy1": 1,
    "position1": "145",
    "atom1": "SG",
    "entity2": "3",
    "copy2": 1,
    "position2": "1",
    "atom2": "C4"
  }
]
```
- `entity1/2`: 实体编号（字符串，从"1"开始，按 sequences 列表顺序）
- `position1`: 蛋白残基位置（Cys145 → "145"）
- `atom1`: 蛋白原子名（"SG"）
- `position2`: 配体内部位置（SMILES 配体固定为 "1"）
- `atom2`: 配体连接原子名（需 SMARTS 匹配后用 RDKit 获取）

## 结构聚类与最优模型筛选 (Structure Clustering)

为从众多模型中稳健地选出具有代表性的高优模型，项目引入了基于共识得分（SuCOS/RMSD）的无监督结构聚类机制。

### 方法设计与实现 (Butina 聚类)
1. **相似度依据**：利用全局预计算产出的 `pairwise_sucos_cache.json` 等两两比对矩阵，复用现成的距离分数以避免成本高昂的重复计算。
2. **逻辑内核 (算法选择)**：采用 Cheminformatics 中小分子聚类金标准 **Butina 聚类算法**：
   - 扫描找到“周围环境（即相似度 $\ge 0.9$ 的邻居）最多”的模态确立为**聚类中心 (Cluster Center)**。
   - 将该中心及与其达标的邻居合并归为一类（Cluster），并从全体集合候选池中抛离。
   - 不断迭代至所有模型分配完毕。
   - 该法无需事先指定类簇数量，$N$ 值自适应，所得类簇大小将直接且客观地反映系统在该构象空间采样的稳定集聚密度。
3. **脚本参数与运行路径**：
   - 脚本位置：`casp17_ligand/analysis/cluster_sucos_analysis.py`
   - 运行方式（支持 `concurrent.futures` 级多线程并行）：
     ```bash
     conda run -n casp17_ligand python casp17_ligand/analysis/cluster_sucos_analysis.py
     ```
   - **自动化匹配与输入控制**：自动遍历主集合目录（`casp16_l1000`, `casp16_l3000_struct` 等）。特别设置了精准的正则过滤/列表限定，规避匹配到同源衍生的过滤后缀目录（如 `_plddt4.5_top10`）而导致的重复采样偏差，通过内嵌安全黑名单跳过异常或撤回的靶点序列（如 `L3103`）。
   - **评估与输出归纳点**：
     汇总生成 CSV 数据台账（至 `outputs/ensemble/cluster_analysis_summary_adaptive.csv`）。直观反馈每靶点的聚类总数、最大重原子数 (max_heavy_atoms)、当下应用的动态阈值 (adaptive_threshold)、Top-n 门类规模（sizes）、中心模型的质量（真实验证 `lddt_pli`），且揭示全局打分顶尖的好模型在各聚类下的分布情况。

### 多聚配体/同源拷贝的交叉评估修复
在 `ensemble_generation.py` 计算 SuCOS 相似度时，面对 B 类靶点（如同一个大口袋包含多个同源独立配体，例如 L3134），由于上游提取管线 `cif_to_pdb_sdf` 会将其物理拆分为单链副拷贝，若模型间的配体链命名错置，会导致极大的测算盲区（错误给出 IoU=0）。
- **解决方案**：引入 `RDKit.Chem.GetMolFrags` 完整拆解模型上的所有多聚体残片。计算最大的所有独立碎片的口袋集合，并在跨模型间执行全排列对比，选取最大碰撞重合分 $max(iou \times sucos)$ 作为真实相似度。此机制彻底封堵了对齐疏漏。

### 尺寸自适应的动态类簇面宽 (Size-Adaptive Clustering)
针对大配体在同等柔性摆动下，体积重合率天然断崖下降的现象（存在 Spearman ~0.51 的尺寸偏见），目前的聚类管线已内置**自适应聚类阈值计算**。
- **动态阈值策略**：基础刚性小分子 ($\le 20$ 重原子) 受限于严格的 `0.80 SuCOS` 准入阈值；随着系统检测到 `max_heavy_atoms` 的增加，阈值按照固定的斜率进行线性放宽下降，直至大型长轴、多结合位点柔性复合物 ($\ge 70$ 重原子) 下探至保底的 `0.60`。该策略从根本上消灭了因分子体积天然排斥而导致的极端类簇孤岛分布问题。

## Protenix 含辅因子（ZN/NAG）版本评测实验（2026-03-27）

### 实验背景

对 `casp16_l3000_struct_zn` 数据集（带 Zn 离子 + NAG 糖基输入）的 Protenix 预测结果，进行 Oracle lddt-pli 评测，与不含辅因子的标准版结果对比。

### 预测结果位置

- CIF 输出：`outputs/protenix/casp16_l3000_struct_zn/{TARGET}/seed_*/predictions/{TARGET}_sample_*.cif`
- 每 target：10 seeds × 5 samples = **50 个 CIF**，共 189 个 target

### CIF 链结构说明

Protenix ZN 版 CIF 包含多条链，OST 评测时需排除辅因子链：

| Chain | 类型 | 原子数 |
|-------|------|--------|
| A | 蛋白质 | ~6800 |
| **E** | **目标配体（用于评测）** | ~29 |
| D | NAG 糖基 | ~14 |
| B / C | ZN 离子 | 各 1 |

自动链检测逻辑（`detect_protein_and_ligand_chains()`）：解析 CIF atom_site，取原子数最大的链为蛋白，取原子数 >1 的非蛋白链中最大者为配体，自然排除 ZN/NAG。

### 评测脚本

**核心脚本**：`casp17_ligand/analysis/eval_protenix_lddt.py`

- 不走 ensemble ranking，直接对每个 CIF 运行 Docker OST 计算 lddt-pli
- 支持按 CIF 文件名缓存 score，断点续跑
- 参考配体中也自动跳过 ZN/NAG 残基名的 `ligand_*.pdb`

**标准运行命令**：

```bash
conda run -n casp17_ligand python casp17_ligand/analysis/eval_protenix_lddt.py \
    --protenix_dir outputs/protenix/casp16_l3000_struct_zn \
    --reference_dir data/casp16_data/struct/L3000_prepared \
    --smiles_dir data/casp16_data/smiles/L3000 \
    --output_dir outputs/ensemble/casp16_l3000_struct_zn \
    --n_eval 50 \
    --num_workers 12
```

### Oracle 评测结果

输出 CSV：`outputs/ensemble/casp16_l3000_struct_zn/eval_protenix_zn_oracle.csv`

| 指标 | 定义 | 值（n=188/189 targets）|
|------|------|----------------------|
| **Oracle Top-1 lddt-pli** | 50 个模型里 lddt 最高的 | **0.6960** |
| **Oracle Top-5 avg lddt-pli** | 按 lddt 排序前 5 名的均值 | **0.6668** |

中间工作文件（CIF→PDB+SDF + OST JSON + score cache）：`outputs/ensemble/casp16_l3000_struct_zn/eval_work/{TARGET}/`
运行日志：`outputs/ensemble/casp16_l3000_struct_zn/eval_protenix_zn_oracle.log`

### 与标准 Protenix（无辅因子）的对比

**重要说明**：`eval_protenix_lddt.py` 与 `evaluate_ensemble.py` 的参考 SDF 构建方式略有差异（前者只读第一个 SMILES，后者按残基名匹配所有 SMILES），导致同一数据集的 Oracle 值有 ~0.04 偏差。下表使用**同一脚本** `eval_protenix_lddt.py` 评测两个版本，确保对比公平。

| 数据集 | 输入条件 | 每 target 模型数 | Oracle Top-1 (eval_protenix_lddt.py) | Oracle Top-1 (evaluate_ensemble.py) |
|--------|----------|------------------|--------------------------------------|-------------------------------------|
| `casp16_l3000_struct` | 不含 ZN/NAG | 50 | **0.8299** (n=188) | **0.8706** |
| `casp16_l3000_struct_zn` | 含 ZN + NAG | 50 | **0.6960** (n=188) | — |

**同脚本对比结论**：加入 ZN/NAG 辅因子后，Oracle Top-1 lddt-pli 从 0.8299 下降到 0.6960（Δ = **-0.134**），说明辅因子输入对 Protenix 的配体放置质量有明显**负面**影响。可能原因：
1. ZN atom-level contact constraint 是 soft constraint，部分 seed 的 ZN 配位不准确（距离 4-8Å vs 参考 ~2Å），错位的金属离子可能排斥配体
2. NAG 共价键约束成功（Asn497.ND2→NAG.C1 = 1.4Å），但 NAG 占据的空间可能干扰配体 docking 的扩散采样
3. 额外 entity（2 ZN + 1 NAG）增加了 token 数（877→931+），模型在更大搜索空间中更难找到正确构象

### top1_top5_scores_STRUCT.csv 各列含义说明

`outputs/ensemble/casp16_l3000_struct/top1_top5_scores_STRUCT.csv` 各关键列的定义：

| 列名 | 是否 Oracle | 含义 |
|------|:-----------:|------|
| `Overall Best LDDT-PLI` | ✅ | 所有方法所有模型中最高 lddt，均值 = 0.9250 |
| `protenix Best LDDT-PLI` | ✅ | 仅 Protenix 模型的最优 lddt，均值 = 0.8706 |
| `ranking_* (Top-1 LDDT)` | ❌ | 共识排序第 1 名的实际 lddt，非 oracle |
| `ranking_* (Top-5 Best)` | ❌ | 共识排序前 5 名中最好的 lddt，受 ranking 限制 |

## Boltz-2 L3000 struct zn (with Cofactors) 评测记录 (2026-03-27)

### 评测背景 & 遇到的拓扑断裂问题
对于加入了 ZN 和 NAG 辅因子作为共折叠条件生成的 189 个 Boltz-2 预测结果 (`casp16_l3000_struct_zn`)，在进行 lDDT-PLI 评测时遭遇了批量错误。
原因是 Boltz-2 将目标配体（L0R）、锌离子（ZN）和 NAG 打包为了混合式的非聚合物输出。若直接送入 OpenStructure (OST)，提取出的片段原子总数（如 46个原子）会远超参考配体（29个原子），强行进行化学图同构匹配时 OST 会报错 `Disconnected graph observed for model ligand`。

### 修复方案
我们开发了专用的高鲁棒批量评测脚本 `tests/eval_l3000_struct_zn.py`，解决了以下问题：
1. **精准子图截取 (RDKit Frag Strip)**：在从 CIF 提取预测小分子的 SDF 后，通过 RDKit `Chem.GetMolFrags` 将所有离散碎片打散，严格匹配目标配体 SMILES 的**精确原子数量**，强行剔除 ZN 和 NAG，只保留干净的主配体交给 OST 进行严格拓扑打分（弃用 heuristic 的 fault-tolerant 模式）。
2. **正确的多配体 Reference 映射**：针对自然存在的带有辅因子（ACT、BR、CL 等）的多 `ligand_*.pdb` 口袋，脚本精准解析 Reference 文件名（如 `ligand_L0R_B_1.pdb`）提取残基名 `L0R`，并到 SMILES TSV 文件中验证，杜绝了简单的 `sorted()[0]` 方法错误抓取离子作为 Ground Truth 的漏测 Bug（修复了 11 个 Failed targets）。
3. **消除并发竞态条件**：采用 4 个并行 Docker workers 带来的高频 I/O，偶发触发了 RDKit C++ 底层解析生成 NoneType 对象的并发错误。通过检测 Error 日志并通过隔离降频（单线程）重新评测这些因高并发掉线的少数 Targets（如 L3001, L3004），完成了 100% 收尾。

### 评测结果

**评测脚本**：`tests/eval_l3000_struct_zn.py`（已重写，与 `evaluate_ensemble.py` 管线一致：PyMOL 链提取 + ProDy + MCS fallback + Docker OST）

| 数据集 | 模型 | Target 数 | 成功 / 跳过 | Oracle Top-1 | Oracle Top-5 avg |
|--------|------|-----------|-------------|--------------|------------------|
| `casp16_l3000_struct_zn` | Boltz-2 | 189 | 188 / 1 (L3103) | **0.6028** | **0.5335** |
| `casp16_l3000_struct` | Boltz-2 (标准版) | 189 | 188 / 1 | **0.5494** | — |

结果 CSV：`outputs/ensemble/boltz2_l3000_struct_zn_eval/eval_oracle.csv`

### Boltz-2 配体复杂度局限性分析（跨数据集）

对 L3000 Boltz-2 预测结果（标准版 + ZN 版）进行分子特征与 Oracle lDDT-PLI 的相关性分析，发现**环数是最强的负相关因子**。

**Spearman 相关（L3000, n=188）**：

| 分子特征 | rho (std) | p (std) | rho (ZN) | p (ZN) |
|----------|-----------|---------|----------|--------|
| **环数 (n_rings)** | **-0.265** | **0.0002 \*\*\*** | **-0.202** | **0.0055 \*\*** |
| 重原子数 (n_heavy) | -0.239 | 0.0010 \*\*\* | -0.163 | 0.0256 \* |
| 可旋转键 | -0.130 | 0.076 ns | -0.104 | 0.156 ns |
| 螺环原子 (n_spiro) | — | — | 0.051 | 0.484 ns |

**按环数分层 lDDT-PLI（L3000 ZN 版）**：

| 环数 | n targets | Mean Oracle Top-1 | Mean heavy atoms |
|------|-----------|-------------------|-----------------|
| 2 环 | 19 | **0.713** | 21.3 |
| 3 环 | 30 | 0.613 | 27.4 |
| 4 环 | 58 | 0.620 | 32.9 |
| 5 环 | 49 | 0.627 | 33.4 |
| 6 环 | 13 | 0.524 | 37.3 |
| 7 环 | 15 | **0.451** | 38.5 |
| 8 环 | 3 | **0.320** | 39.0 |

**跨数据集对比**：

| 数据集 | n | Mean Oracle | rho(rings) | 7+环 targets |
|--------|---|-------------|------------|-------------|
| L1000 | 17 | 0.889 | -0.041 ns | 无 |
| L3000 std | 188 | 0.549 | **-0.265\*\*\*** | 0.418 |
| L3000 ZN | 188 | 0.603 | **-0.202\*\*** | 0.429 |
| L4000 | 20 | 0.764 | +0.233 ns | 无 |

**结论**：
1. **多环大分子（≥6 环，≥37 重原子）是 Boltz-2 的系统性弱点**，标准版和 ZN 版趋势一致（rho 方向相同）。Oracle Top-1 从 2 环的 0.71 阶梯下降到 8 环的 0.32。
2. **螺环/桥环本身不是关键因素**（p=0.48/0.58），真正的瓶颈是环数多导致的构象空间复杂度。
3. L1000/L4000 因配体普遍较小（无 7+环 target），未观测到显著相关。
4. ZN/NAG 辅因子未改变这一趋势的方向和强度，说明这是 Boltz-2 扩散采样本身的局限而非辅因子干扰。

## CASP15 数据集

### 数据来源
- PoseBench (https://github.com/BioinfoMachineLearning/PoseBench) 提供的 CASP15 数据
- Zenodo: `https://zenodo.org/records/19138652/files/casp15_set.tar.gz`
- 下载后解压至 `data/casp15_data/`

### 数据格式（与 CASP16 的区别）

CASP15 每个 target 独立，没有 series 概念（不像 CASP16 的 L1000/L3000 共享蛋白序列）。

```
data/casp15_data/targets/
├── H1135.seq.txt       # 多链 FASTA（原始比赛输入）
├── H1135.smiles.txt    # TSV: ID  Name  SMILES  Relevant
├── H1135.ligands.txt   # TSV: 同上 + Chain  Resnum  SubstructureMatch
└── H1135_lig.pdb       # 参考配体结构（ground truth，评测用）
```

### 多聚体（Multimer）—— CASP15 独有

CASP16 最多 2 链（L4000 homodimer，A=B 序列相同）。CASP15 有大量多聚体，包括高阶和异源：

| 链数 | CASP16 | CASP15 |
|------|--------|--------|
| 1 链 | 238 | 7 |
| 2 链 | 20 | 3 |
| 3 链 | 0 | 2 (T1152, T1181) |
| 7 链 | 0 | 2 (H1171v1-v2) |
| 8 链 | 0 | 4 (H1172v1-v4) |
| 12 链 | 0 | 1 (H1135) |

**异源多聚体（Heteromer）**：多条链序列不同，CASP16 无此情况。
- H1135：12 链，2 种序列（194aa × 10 + 25aa × 2）
- H1171/H1172 系列：7-8 链，4-5 种不同序列（312/309/313/48/46 aa）
- T1124：2 链异源（378aa + 361aa）
- T1127v2：2 链异源（205aa + 206aa）
- T1152：3 链，2 种序列（46aa × 2 + 47aa × 1）

**体量**：最大 T1181 = 2064 残基 + 9 配体，H1172 系列 ~1964 残基 + 8-9 配体（V100 32GB 可能 OOM）。

**对现有代码的影响**：
- `is_dimer` 属性定义为 `len(protein_sequences) > 1`，对 CASP15 异源多聚体也返回 True
- Boltz2/Protenix 的 `is_dimer` 分支包含 L4000 MPro 专用逻辑（`group_ligands_for_dimer` + `MPRO_ACTIVE_SITE_RESIDUES`），对 CASP15 不适用
- AF3/RF3/Seedfold 不依赖 `is_dimer`，直接遍历 `protein_sequences`，可正确处理任意链数

**smiles.txt 的 Relevant 列**：
- `Yes` = 评估目标配体
- `No` = 共结晶分子（如结晶条件引入的 Cl⁻），不参与评测打分
- 例：H1135 有 3 个 CL（Relevant=No）+ 9 个 K（Relevant=Yes）
- **输入时**：所有配体（包括 Relevant=No）都喂给 co-folding 模型
- **评测时**：只评 Relevant=Yes 的配体

### Target 列表（19 个 benchmarked）

| Target | 类型 | License | 配体概况 |
|--------|------|---------|----------|
| H1135 | multi | public | 3 CL (No) + 9 K (Yes) |
| H1171v1-v2 | multi | public | ADP + AGS + MG |
| H1172v1-v4 | multi | public | ADP + AGS + MG |
| T1124 | multi | public | 2 SAH + 2 TYR |
| T1127v2 | multi | private | |
| T1146 | single | private | |
| T1152 | single | public | 1 NAG |
| T1158v1-v3 | single | public | |
| T1158v4 | multi | public | |
| T1170 | multi | public | **NOT benchmarked**（excluded） |
| T1181 | multi | private | |
| T1186 | single | private | |
| T1187 | multi | public | 2 NAG |
| T1188 | multi | public | |

来源：`data/casp15_data/public_vs_private_casp15_targets.csv`

### 输入准备脚本

`casp17_ligand/data/casp15_input_preparation.py`：将 `targets/` 转换为 CASP16 兼容的目录结构，使 `load_all_targets("CASP15", data_root)` 可直接加载。

```bash
python casp17_ligand/data/casp15_input_preparation.py
```

转换结果：
```
data/casp15_data/
├── sequences/CASP15/{target_id}.txt   # per-target 多链 FASTA
├── smiles/CASP15/{target_id}.tsv      # CASP16 格式 TSV（ID Name SMILES Task）
```

`target_data.py` 的 `read_protein_sequence` 支持两种查找顺序：
1. `sequences/{series}/{target_id}.txt` — per-target（CASP15 模式）
2. `sequences/{series}.txt` — 共享（CASP16 模式）

默认排除 T1170（benchmarked=FALSE）。输出 19 个 targets。

**与 PoseBench `ensemble_prediction_inputs.csv` 的验证**：
- 19/19 targets 的 SMILES、ligand_numbers、ligand_names 完全一致
- 唯一差异：T1124 Chain B 序列，我们的 seq.txt（原始比赛输入）比 PoseBench 的 AF3 预测 PDB 提取多 1 个残基（pos 242 的 R）。这是因为 AF3 未建模该残基。**我们使用原始序列是正确做法**。

### CASP15 预测结果统计

| Target | AF3 | Protenix | Boltz2 | Seedfold | 备注 |
|--------|-----|----------|--------|----------|------|
| H1135 | 101 | 50 | 50 | 50 | Seedfold 用 2/3 |
| H1171v1 | 101 | 50 | 50 | - | 异源多聚体，Seedfold 不支持 |
| H1171v2 | 101 | 50 | 50 | - | 同上 |
| H1172v1 | 101 | 50 | 50 | - | 同上 |
| H1172v2 | 101 | 50 | 50 | - | 同上 |
| H1172v3 | 101 | 50 | 50 | - | 同上 |
| H1172v4 | 101 | 50 | 50 | - | 同上 |
| T1124 | 101 | 50 | 50 | 50 | |
| T1127v2 | 101 | 50 | 50 | 50 | |
| T1146 | 101 | 50 | 50 | 50 | |
| T1152 | 101 | 50 | 50 | 50 | |
| T1158v1 | 101 | 50 | 50 | - | Seedfold 无法生成 |
| T1158v2 | 101 | 50 | 50 | 50 | |
| T1158v3 | 101 | 50 | 50 | 50 | |
| T1158v4 | 101 | 50 | 50 | 50 | |
| T1181 | 101 | 50 | - | 50 | Blacklisted (ref ligand atom count mismatch); Seedfold 用 1/3 |
| T1186 | 101 | 50 | 50 | 50 | |
| T1187 | 101 | 50 | 50 | 50 | |
| T1188 | 101 | 50 | 50 | 50 | |

### Boltz-2 大多聚体 OOM 处理

8 个大多聚体 target 在 H100 94GB 上单次 50 samples GPU OOM。
策略：分两次跑（每次 25 samples），用完整输入，不删减链数。

**待重跑（25+25 策略）**：
- **H1135**：12 链 (9×194aa + 3×25aa) + 12 lig (9K + 3CL), 1821 res
- **T1181**：3 链 (3×688aa) + 9 lig (5OAA + 3ZN + 1CA), 2064 res
- **H1171v1/v2**：7 链异源复合物, ~1918 res + 9-11 lig
- **H1172v1-v4**：8 链异源复合物, ~1964 res + 8-9 lig

### 评测注意事项
- 参考配体结构：`targets/{name}_lig.pdb`（ground truth）
- **评测时必须按 Relevant 列过滤**：只计算 Relevant=Yes 的配体的 lDDT-PLI / RMSD
- `ligands.txt` 中的 Chain 和 Resnum 列可用于在预测结构中定位对应配体

## RTMScore 环境配置与重排序实验 (2026-03-29)

### conda env: `rtmscore` (Python 3.8)

**核心依赖**（精简安装，不使用项目自带的 requirements_conda.txt 完整 dump）：
- PyTorch 1.9.0+cu111, DGL-CUDA11.1 0.7.0, torch-scatter 2.0.9
- rdkit-pypi 2021.03.5 (pip), openbabel 3.1.1 (conda-forge)
- MDAnalysis 2.0.0, ProDy 2.1.0, dgllife 0.2.8, Cython<3
- scikit-learn 0.24.2, pandas 1.3.2, scipy, matplotlib, seaborn

**关键环境变量**（已写入 activate.d/env_vars.sh）：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export BABEL_LIBDIR=$CONDA_PREFIX/lib/openbabel/3.1.0
export BABEL_DATADIR=$CONDA_PREFIX/share/openbabel/3.1.0
```

**注意**：`conda run` 不执行 activate 脚本，需在 bash -c 中手动 export 或在 Python 脚本顶部设 `os.environ`。

**RTMScore 运行示例**：
```bash
conda run --no-capture-output -n rtmscore bash -c '
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  export BABEL_LIBDIR=$CONDA_PREFIX/lib/openbabel/3.1.0
  export CUDA_VISIBLE_DEVICES=0
  cd forks/RTMScore/example
  python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -m ../trained_models/rtmscore_model1.pth
'
```

### RTMScore 重排序实验结果

**实验设计**：对聚类最优配置（Cluster + SuCOS consensus center + PoseBusters, 20HA-0.8; 40HA→0.70）产出的每个 target top-5 代表模型，使用 RTMScore 进行蛋白-配体交互打分重排序。

**脚本**：`casp17_ligand/analysis/rtmscore_rerank.py`
- 自行用 ProDy+RDKit 提取 10Å pocket（绕过 OpenBabel 插件加载问题）
- 传入 RDKit Mol 对象（sanitize=False）绕过 RTMScore 内部 PDB 解析的化学验证错误
- RTMScore model: `forks/RTMScore/trained_models/rtmscore_model1.pth`

**运行命令**：
```bash
conda run --no-capture-output -n rtmscore bash -c '
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  export CUDA_VISIBLE_DEVICES=0
  python casp17_ligand/analysis/rtmscore_rerank.py \
    --cluster_csv outputs/ensemble/cluster_experiment_detail_latest.csv \
    --config "自适应: 40HA->0.70" \
    --output outputs/ensemble/rtmscore_rerank_top5.csv
'
```

**聚类改进**：强制每个 target 至少产出 5 个聚类代表（`evaluate_topn_clusters.py`）。当聚类数 <5 时，自动提高阈值（<0.90 时 +5%，>=0.90 时 +1%），循环直到 >=5 个 cluster。

**结果**（n=227 targets, 全部成功，全部 >=5 reps）：

| 指标 | 值 |
|------|-----|
| Cluster Top-1 Mean lDDT | **0.8114** |
| RTMScore Reranked Top-1 Mean lDDT | **0.8114** |
| Cluster Top-5 Best Mean lDDT | **0.8759**（旧版无强制 5 类时为 0.8754） |
| Delta (reranked - orig) | **+0.0000** |

**Top-1 改变数：0 / 227**（RTMScore 对所有 target 都选了与共识排序相同的 top1）

**详细分析**：
- RTMScore 改变了 2-5 名排序：177 / 227 targets
- RTMScore 给 top1 模型的分数远高于 top2（通常 3-10 倍），确认共识排序选出的 top1 在蛋白-配体交互打分上也是最优
- 强制 5 类后 Top-5 Best 从 0.8754 微升至 0.8759，Top-1 不变
- 个别 target（如 L1006）因阈值提高合并了原本独立的高分模型到同一 cluster，consensus 选代表时选了共识更高但 lddt 略低的模型，导致该 target 的 top5_best 下降（0.9406→0.9202），但全局 top5 仍提升
- **结论**：SuCOS consensus + PoseBusters 策略已经非常强，RTMScore 无法进一步提升 top1

**SuCOS 共识分说明**：`ranking_sucos/` 文件名中的 `sucos0.683` 是该模型与所有其他模型的**全局平均 SuCOS 相似度**（非类内共识）。`consensus_pb` 策略在每个 cluster 内选全局共识最高且通过 PoseBusters 的模型作为代表。

**输出文件**：
- `outputs/ensemble/rtmscore_rerank_top5.csv` — per-target top5 详细数据（模型名、lddt、RTMScore 分数、重排序结果）
- `outputs/ensemble/cluster_experiment_detail_latest.csv` — 含 `top5_reps`, `top5_lddts`, `top5_sdf_paths`, `target_dir` 列，可供下游打分工具直接使用


## CASP17 正赛（独立文档）

CASP17 比赛期间的所有约定、数据布局、RNA-ligand 处理、踩坑记录、提交流程都集中在 [`casp17.md`](./casp17.md)，避免与本文（CASP15/16 历史 benchmark 总结）混杂。

简略指引：
- 数据真值入口：`https://predictioncenter.org/casp17/target.cgi?target={ID}&view=sequence`（**禁止**从师兄镜像 / server task feed 间接读，已实测过 R2314 抄错）
- 数据落地：`data/casp17_data/{sequences,smiles,struct}/CASP17_R/{target}.{txt,tsv,pdb.txt}`
- test_cases 入口：`data/test_cases/casp17_R/ensemble_inputs.csv`（7 列 RNA-only schema）
- RNA input prep：`{boltz2,af3,protenix}_input_preparation.py` 已加 `entity_type` 分流，对应 `configs/data/*_input_preparation_casp17_R.yaml`
- RNA MSA：AF3 默认让内置 data pipeline 真搜（不注入 dummy），Protenix 待用 `protenix prep` 跑出 a3m 后注入；Boltz-2 模型层面就不接 RNA MSA
