#!/bin/bash
# Run Boltz-2 Round 2 for casp16_l3000_struct: 10 seeds × 5 diffusion_samples = 50 models per target
# Parameters: step_scale=1.2, diffusion_samples=5, no affinity, no Zn/NAG
# GPU 1

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
BOLTZ_EXEC="/bmlfast/Lyuwei/miniconda3/envs/boltz/bin/boltz"
INPUT_DIR="${PROJECT_ROOT}/data/test_cases/casp16_l3000_struct/boltz2_inputs"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/boltz2/casp16_l3000_struct_r2"
CACHE_DIR="${PROJECT_ROOT}/weights/boltz"

# Parameters
DIFFUSION_SAMPLES=5
STEP_SCALE=1.2
MODEL="boltz2"
RECYCLING_STEPS=10
SAMPLING_STEPS=200

# 10 seeds
SEEDS=(42 123 256 314 500 617 789 888 1024 2025)

echo "=== Boltz-2 Round 2 (L3000 struct, 189 targets) ==="
echo "diffusion_samples=${DIFFUSION_SAMPLES}, step_scale=${STEP_SCALE}, no affinity"
echo "Seeds: ${SEEDS[*]}"
echo "Output: ${OUTPUT_BASE}"
echo ""

for seed in "${SEEDS[@]}"; do
    SEED_OUTPUT="${OUTPUT_BASE}/seed_${seed}"
    mkdir -p "${SEED_OUTPUT}"

    echo "--- Seed ${seed} ---"

    for yaml_file in "${INPUT_DIR}"/*_input.yaml; do
        target=$(basename "${yaml_file}" | sed 's/_input\.yaml//')

        # Check if already done (skip existing)
        cif_dir="${SEED_OUTPUT}/boltz_results_${target}_input/predictions"
        if [ -d "${cif_dir}" ]; then
            cif_count=$(find "${cif_dir}" -name "*.cif" 2>/dev/null | wc -l)
            if [ "${cif_count}" -ge "${DIFFUSION_SAMPLES}" ]; then
                echo "  [${target}] seed=${seed}: already done (${cif_count} CIFs), skipping"
                continue
            fi
        fi

        echo "  [${target}] seed=${seed}: running..."
        ${BOLTZ_EXEC} predict "${yaml_file}" \
            --out_dir "${SEED_OUTPUT}" \
            --cache "${CACHE_DIR}" \
            --model "${MODEL}" \
            --diffusion_samples "${DIFFUSION_SAMPLES}" \
            --recycling_steps "${RECYCLING_STEPS}" \
            --sampling_steps "${SAMPLING_STEPS}" \
            --step_scale "${STEP_SCALE}" \
            --use_potentials \
            --use_msa_server \
            --seed "${seed}" \
            --devices 1 \
            --accelerator gpu \
            2>&1 | tee -a "${SEED_OUTPUT}/boltz2_seed_${seed}.log"

        if [ $? -ne 0 ]; then
            echo "  [${target}] seed=${seed}: FAILED"
        fi
    done
done

echo ""
echo "=== All seeds complete ==="
