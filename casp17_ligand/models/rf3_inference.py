"""Run RoseTTAFold3 inference on prepared input JSON files.

Supports multi-seed inference to generate diverse predictions.
Each seed × diffusion_batch_size produces a set of structures.
E.g., 10 seeds × 5 batch = 50 structures per target.

Two execution modes:
  1. In-process (preferred): Uses RF3 Python API directly if running inside the
     rf3 conda environment. Model loaded once, seeds changed between runs.
  2. Subprocess fallback: Invokes `rf3 fold` CLI per seed when running outside
     the rf3 env (slower: reloads model per seed).
"""

import logging
import subprocess  # nosec
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from casp17_ligand.utils.env_utils import get_method_exec

log = logging.getLogger(__name__)


def _check_rf3_importable() -> bool:
    """Check if we can import rf3 (i.e. running inside rf3 env)."""
    try:
        import rf3  # noqa: F401

        return True
    except ImportError:
        return False


def run_rf3_inprocess(cfg: DictConfig, json_files: list[Path], output_dir: Path) -> None:
    """Run RF3 inference using the Python API (model loaded once).

    This is the efficient path when running inside the rf3 conda env.
    Loads the model once, then iterates over all targets × seeds,
    changing the RNG seed between runs.
    """
    from lightning.fabric import seed_everything
    from rf3.inference_engines.rf3 import RF3InferenceEngine

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_device", 0))

    seeds = list(range(cfg.num_seeds))
    ckpt_path = str(Path(cfg.ckpt_path).resolve()) if Path(cfg.ckpt_path).is_absolute() else cfg.ckpt_path

    log.info(f"Creating RF3InferenceEngine (ckpt={ckpt_path})...")

    # Initialize engine with seed=0 (will be changed per-run)
    engine = RF3InferenceEngine(
        ckpt_path=ckpt_path,
        n_recycles=cfg.n_recycles,
        diffusion_batch_size=cfg.diffusion_batch_size,
        num_steps=cfg.num_steps,
        seed=seeds[0],
        compress_outputs=cfg.get("compress_outputs", False),
        early_stopping_plddt_threshold=cfg.get("early_stopping_plddt_threshold", None),
    )

    for json_file in json_files:
        target = json_file.stem
        log.info(f"  [{target}] Starting ({len(seeds)} seeds)...")

        for seed in seeds:
            seed_out = output_dir / f"{target}_seed-{seed}"

            # Check if already completed
            if not cfg.override and seed_out.exists():
                ranking_csv = list(seed_out.glob("*_ranking_scores.csv"))
                if ranking_csv:
                    log.info(f"    [{target}] seed={seed} already done, skipping")
                    continue

            # Change seed for this run
            seed_everything(seed, workers=True)
            engine.seed = seed

            log.info(f"    [{target}] seed={seed}: running inference...")
            try:
                engine.run(
                    inputs=str(json_file),
                    out_dir=str(seed_out),
                    skip_existing=cfg.get("skip_existing", False),
                    annotate_b_factor_with_plddt=cfg.get("annotate_b_factor_with_plddt", False),
                )
            except Exception as e:
                log.error(f"    [{target}] seed={seed} failed: {e}")
                break

    log.info("In-process inference done.")


def build_rf3_cmd(
    rf3_exec: str,
    cfg: DictConfig,
    json_file: Path,
    output_dir: Path,
    seed: int,
) -> list:
    """Build the rf3 fold command for a single prediction.

    Args:
        rf3_exec: Path to rf3 executable in conda env.
        cfg: Hydra config with model parameters.
        json_file: Path to RF3 input JSON.
        output_dir: Output directory for predictions.
        seed: Random seed for this run.

    Returns:
        Command list for subprocess.run.
    """
    cmd = [
        rf3_exec, "fold",
        f"inputs={json_file}",
        f"out_dir={output_dir}",
        f"ckpt_path={cfg.ckpt_path}",
        f"n_recycles={cfg.n_recycles}",
        f"diffusion_batch_size={cfg.diffusion_batch_size}",
        f"num_steps={cfg.num_steps}",
        f"seed={seed}",
    ]

    # Optional parameters
    if cfg.get("early_stopping_plddt_threshold") is not None:
        cmd.append(f"early_stopping_plddt_threshold={cfg.early_stopping_plddt_threshold}")
    if cfg.get("compress_outputs", False):
        cmd.append("compress_outputs=true")
    if cfg.get("annotate_b_factor_with_plddt", False):
        cmd.append("annotate_b_factor_with_plddt=true")
    if cfg.get("skip_existing", False):
        cmd.append("skip_existing=true")

    return cmd


def run_rf3_subprocess(cfg: DictConfig, json_files: list[Path], output_dir: Path) -> None:
    """Run RF3 via subprocess (fallback when not in rf3 env)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_device", 0))

    seeds = list(range(cfg.num_seeds))

    rf3_exec = get_method_exec(
        exec_name=cfg.get("rf3_exec", "rf3"),
        env_name=cfg.get("env_name", "rf3"),
        conda_envs_dir=cfg.get("conda_envs_dir", None),
    )
    log.info(f"Using rf3 executable: {rf3_exec}")

    for json_file in json_files:
        target = json_file.stem
        log.info(f"  [{target}] Starting ({len(seeds)} seeds)...")

        for seed in seeds:
            seed_out = output_dir / f"{target}_seed-{seed}"

            # Check if already completed
            if not cfg.override and seed_out.exists():
                ranking_csv = list(seed_out.glob("*_ranking_scores.csv"))
                if ranking_csv:
                    log.info(f"    [{target}] seed={seed} already done, skipping")
                    continue

            cmd = build_rf3_cmd(rf3_exec, cfg, json_file, seed_out, seed)
            log.info(f"    [{target}] seed={seed}: {' '.join(cmd[:4])}...")
            result = subprocess.run(cmd, env={**os.environ})  # nosec
            if result.returncode != 0:
                log.error(
                    f"    [{target}] seed={seed} failed (exit {result.returncode}), "
                    "skipping remaining seeds for this target."
                )
                break

    log.info("Subprocess inference done.")


@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="rf3_inference")
def main(cfg: DictConfig) -> None:
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    input_dir = root / cfg.input_dir
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        log.error(f"No JSON files found in {input_dir}")
        return

    # Resolve seeds: list of ints for multi-seed inference
    seeds = list(range(cfg.num_seeds))
    total_samples = len(seeds) * cfg.diffusion_batch_size

    log.info(
        f"Running RF3 on {len(json_files)} targets × {len(seeds)} seeds × "
        f"{cfg.diffusion_batch_size} diffusion_batch = {total_samples} structures/target"
    )
    log.info(f"Output → {output_dir}")

    # Choose execution mode
    if _check_rf3_importable():
        log.info("RF3 importable — using in-process Python API (efficient)")
        run_rf3_inprocess(cfg, json_files, output_dir)
    else:
        log.info("RF3 not importable — falling back to subprocess (rf3 fold CLI)")
        run_rf3_subprocess(cfg, json_files, output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
