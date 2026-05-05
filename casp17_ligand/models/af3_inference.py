"""Run AlphaFold3 inference via Docker on prepared input JSON files."""

import logging
import subprocess  # nosec
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def build_af3_docker_cmd(cfg: DictConfig, json_file: Path, output_dir: Path) -> list:
    """Build the docker run command for a single AF3 prediction."""
    input_dir = json_file.parent
    uid_gid = subprocess.check_output(  # nosec
        ["id", "-u"], text=True
    ).strip()
    gid = subprocess.check_output(  # nosec
        ["id", "-g"], text=True
    ).strip()

    cmd = [
        "docker", "run",
        "--rm",
        f"--gpus", f'"device={cfg.gpu_device}"',
        f"--memory={cfg.memory_limit}",
        f"--memory-swap={cfg.memory_swap}",
        f"--cpus={cfg.cpus}",
        f"--shm-size={cfg.shm_size}",
        f"--user={uid_gid}:{gid}",
        # XLA / JAX memory env vars
        f"-e", f"XLA_PYTHON_CLIENT_PREALLOCATE={'true' if cfg.xla_preallocate else 'false'}",
        f"-e", f"TF_FORCE_UNIFIED_MEMORY={'1' if cfg.tf_force_unified_memory else '0'}",
        f"-e", f"XLA_CLIENT_MEM_FRACTION={cfg.xla_client_mem_fraction}",
        # Volume mounts (use /tmp to avoid permission issues with --user)
        "-v", f"{input_dir}:/tmp/af_input",
        "-v", f"{output_dir}:/tmp/af_output",
        "-v", f"{cfg.model_dir}:/tmp/models",
        "-v", f"{cfg.db_dir}:/public_databases",
        cfg.docker_image,
        "python", "run_alphafold.py",
        f"--json_path=/tmp/af_input/{json_file.name}",
        "--model_dir=/tmp/models",
        "--output_dir=/tmp/af_output",
    ]

    if cfg.get("jax_compilation_cache_dir"):
        cmd.append(f"--jax_compilation_cache_dir={cfg.jax_compilation_cache_dir}")

    if cfg.get("num_seeds") is not None:
        cmd.append(f"--num_seeds={cfg.num_seeds}")

    if not cfg.run_data_pipeline:
        cmd.append("--norun_data_pipeline")
    if not cfg.run_inference:
        cmd.append("--norun_inference")

    return cmd


@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="af3_inference")
def main(cfg: DictConfig) -> None:
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    input_dir = root / cfg.input_dir
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        log.error(f"No JSON files found in {input_dir}")
        return

    log.info(f"Running AlphaFold3 on {len(json_files)} targets → {output_dir}")

    for json_file in json_files:
        target = json_file.stem
        pred_dir = output_dir / target.lower()
        if not cfg.override and pred_dir.exists():
            log.info(f"  Skipping {target} (already exists, use override=true to rerun)")
            continue

        cmd = build_af3_docker_cmd(cfg, json_file, output_dir)
        log.info(f"  [{target}] Running AF3...")
        log.info(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd)  # nosec
        if result.returncode != 0:
            log.error(f"  [{target}] failed with exit code {result.returncode}, skipping.")

    log.info("Done.")


if __name__ == "__main__":
    main()
