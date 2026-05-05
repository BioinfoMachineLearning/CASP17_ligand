#!/bin/bash
# Boltz-2 step_scale 扫参实验（通用脚本）
# Targets: L3032, L3105, L3074, L3030, L3008
# 参数: diffusion_samples=25, 无 --seed, use_potentials, use_msa_server
# 输出目录: outputs/boltz2/casp16_l3000_struct_ss<ss_tag>_top5_run<N>/
#   ss_tag = step_scale 去掉小数点 (1.2 → 12, 1.638 → 1638)
#
# Usage:
#   bash scripts/run_boltz2_ss_l3000_top5.sh <step_scale> <gpu_idx> <run_name1> [run_name2 ...]
#   e.g.  bash scripts/run_boltz2_ss_l3000_top5.sh 1.2 0 run1 run2
#         bash scripts/run_boltz2_ss_l3000_top5.sh 1.2 1 run3 run4

set -uo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <step_scale> <gpu_idx> <run_name1> [run_name2 ...]"
    exit 1
fi

STEP_SCALE="$1"; shift
GPU_IDX="$1"; shift
RUNS=("$@")

SS_TAG=$(echo "${STEP_SCALE}" | tr -d '.')

export CUDA_VISIBLE_DEVICES="${GPU_IDX}"

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
BOLTZ_EXEC="/bmlfast/Lyuwei/miniconda3/envs/boltz/bin/boltz"
INPUT_DIR="${PROJECT_ROOT}/data/test_cases/casp16_l3000_struct/boltz2_inputs"
CACHE_DIR="${PROJECT_ROOT}/weights/boltz"

DIFFUSION_SAMPLES=25
MODEL="boltz2"
RECYCLING_STEPS=10
SAMPLING_STEPS=200

TARGETS=(L3032 L3105 L3074 L3030 L3008)

OUTPUT_BASE_TPL="${PROJECT_ROOT}/outputs/boltz2/casp16_l3000_struct_ss${SS_TAG}_top5"

echo "=== Boltz-2 step_scale=${STEP_SCALE}, batch=${DIFFUSION_SAMPLES} (GPU ${GPU_IDX}) ==="
echo "Targets: ${TARGETS[*]}"
echo "Runs: ${RUNS[*]}"
echo "Output base: ${OUTPUT_BASE_TPL}_<run>"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for run in "${RUNS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE_TPL}_${run}"
    mkdir -p "${OUTPUT_DIR}"

    echo "=== Run: ${run} (GPU ${GPU_IDX}) ==="
    echo "Output: ${OUTPUT_DIR}"

    for target in "${TARGETS[@]}"; do
        yaml_file="${INPUT_DIR}/${target}_input.yaml"
        if [ ! -f "${yaml_file}" ]; then
            echo "  [${target}] yaml not found, skipping"
            continue
        fi

        cif_dir="${OUTPUT_DIR}/boltz_results_${target}_input/predictions"
        if [ -d "${cif_dir}" ]; then
            cif_count=$(find "${cif_dir}" -name "*model_*.cif" 2>/dev/null | wc -l)
            if [ "${cif_count}" -ge "${DIFFUSION_SAMPLES}" ]; then
                echo "  [${target}] ${run}: skip (${cif_count} CIFs already)"
                continue
            fi
        fi

        ts=$(date '+%H:%M:%S')
        echo "  [${target}] ${run} @ ${ts}: running ${DIFFUSION_SAMPLES} samples (ss=${STEP_SCALE})..."
        "${BOLTZ_EXEC}" predict "${yaml_file}" \
            --out_dir "${OUTPUT_DIR}" \
            --cache "${CACHE_DIR}" \
            --model "${MODEL}" \
            --diffusion_samples "${DIFFUSION_SAMPLES}" \
            --recycling_steps "${RECYCLING_STEPS}" \
            --sampling_steps "${SAMPLING_STEPS}" \
            --step_scale "${STEP_SCALE}" \
            --use_potentials \
            --use_msa_server \
            --devices 1 \
            --accelerator gpu \
            > "${OUTPUT_DIR}/boltz2_${target}.log" 2>&1
        rc=$?
        ts_end=$(date '+%H:%M:%S')
        if [ ${rc} -ne 0 ]; then
            echo "  [${target}] ${run} @ ${ts_end}: FAILED (rc=${rc})"
        else
            echo "  [${target}] ${run} @ ${ts_end}: done"
        fi
    done
done

echo ""
echo "=== ss=${STEP_SCALE} GPU ${GPU_IDX} 完成 $(date '+%Y-%m-%d %H:%M:%S') ==="
