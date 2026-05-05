#!/bin/bash
# Run 3 CASP15 OOM targets on GPU 1: 10 seeds × 5 samples each
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

BOLTZ_EXEC="/bmlfast/Lyuwei/miniconda3/envs/boltz/bin/boltz"
PROJECT_ROOT="/bmlfast/Lyuwei/0.Projects/CASP17_ligand"
INPUT_DIR="${PROJECT_ROOT}/data/test_cases/casp15/boltz2_inputs_oom"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/boltz2/casp15_oom_local"
CACHE_DIR="${HOME}/.boltz"

TARGETS=(H1172v3 H1172v4 T1181)

for target in "${TARGETS[@]}"; do
    echo "=== ${target} ==="
    for seed in $(seq 1 10); do
        SEED_OUTPUT="${OUTPUT_BASE}/seed_${seed}"
        mkdir -p "${SEED_OUTPUT}"

        cif_dir="${SEED_OUTPUT}/boltz_results_${target}_input/predictions"
        if [ -d "${cif_dir}" ]; then
            cif_count=$(find "${cif_dir}" -name "*.cif" 2>/dev/null | wc -l)
            if [ "${cif_count}" -ge 5 ]; then
                echo "  [${target}] seed=${seed}: already done (${cif_count} CIFs), skipping"
                continue
            fi
        fi

        echo "  [${target}] seed=${seed}: running..."
        ${BOLTZ_EXEC} predict \
            "${INPUT_DIR}/${target}_input.yaml" \
            --out_dir "${SEED_OUTPUT}" \
            --cache "${CACHE_DIR}" \
            --model boltz2 \
            --diffusion_samples 5 \
            --recycling_steps 10 \
            --sampling_steps 200 \
            --step_scale 1.5 \
            --seed ${seed} \
            --use_msa_server \
            --no_kernels \
            --devices 1 --accelerator gpu \
            2>&1 | tee -a "${SEED_OUTPUT}/boltz2_${target}_seed_${seed}.log"

        if [ $? -ne 0 ]; then
            echo "  [${target}] seed=${seed}: FAILED"
        fi
    done
done

echo "=== All CASP15 OOM targets complete ==="
