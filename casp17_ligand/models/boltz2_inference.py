"""Run Boltz-2 inference on prepared input YAML files."""

import logging
import subprocess  # nosec
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

from casp17_ligand.utils.env_utils import get_method_exec

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def build_boltz2_cmd(boltz_exec: str, cfg: DictConfig, yaml_file: Path, output_dir: Path, cache_dir: Path) -> list:
    """Build the boltz predict command from config."""
    cmd = [
        boltz_exec, "predict", str(yaml_file),
        "--out_dir", str(output_dir),
        "--cache", str(cache_dir),
        "--model", cfg.model,
        "--diffusion_samples", str(cfg.diffusion_samples),
        "--recycling_steps", str(cfg.recycling_steps),
        "--sampling_steps", str(cfg.sampling_steps),
        "--devices", str(cfg.devices),
        "--accelerator", cfg.accelerator,
    ]
    if cfg.get("step_scale") is not None:
        cmd += ["--step_scale", str(cfg.step_scale)]
    # Peak VRAM is driven by how many diffusion samples go through the structure
    # module *at once*, not by how many are requested: Boltz chunks them
    # min(max_parallel_samples, diffusion_samples) wide and defaults to 5. That
    # is why lowering diffusion_samples alone never fixed the M2415 OOMs (12/15/
    # 20/25/50 all failed identically — see run_boltz2_M2415_4x25.sh). Capping
    # the chunk instead keeps the full sample count and diversity, so it is
    # quality-neutral; only wall-clock grows.
    if cfg.get("max_parallel_samples") is not None:
        cmd += ["--max_parallel_samples", str(cfg.max_parallel_samples)]
    if cfg.get("use_potentials", False):
        cmd.append("--use_potentials")
    if cfg.use_msa_server:
        cmd.append("--use_msa_server")
    # Affinity flags (off by default, pass predict_affinity=true to enable)
    if cfg.get("affinity_mw_correction", False):
        cmd.append("--affinity_mw_correction")
    if cfg.get("diffusion_samples_affinity") is not None:
        cmd += ["--diffusion_samples_affinity", str(cfg.diffusion_samples_affinity)]
    if cfg.get("sampling_steps_affinity") is not None:
        cmd += ["--sampling_steps_affinity", str(cfg.sampling_steps_affinity)]
    if cfg.override:
        cmd.append("--override")
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    return cmd


@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="boltz2_inference")
def main(cfg: DictConfig) -> None:
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    input_dir = root / cfg.input_dir
    base_output_dir = root / cfg.output_dir
    cache_dir = root / cfg.cache_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Optional per-target subdirectory tag (e.g. "r1", "r2"). When set, each
    # target's outputs go to {base_output_dir}/{target}_{round_tag}/ so multiple
    # rounds for the same dataset don't collide. When null/empty, falls back to
    # the legacy flat layout (CASP16 behavior unchanged).
    round_tag = cfg.get("round_tag", None)

    yaml_files = sorted(input_dir.glob("*_input.yaml"))
    if not yaml_files:
        log.error(f"No *_input.yaml files found in {input_dir}")
        return

    log.info(f"Running Boltz-2 on {len(yaml_files)} targets → {base_output_dir}"
             + (f" (round_tag={round_tag})" if round_tag else ""))

    boltz_exec = get_method_exec(
        exec_name=cfg.get("boltz_exec", "boltz"),
        env_name=cfg.get("env_name", "boltz"),
        conda_envs_dir=cfg.get("conda_envs_dir", None),
    )
    log.info(f"Using boltz executable: {boltz_exec}")

    # Optional allowlist: only run targets in cfg.targets (comma-separated)
    targets_allowlist = None
    if cfg.get("targets"):
        targets_allowlist = set(cfg.targets.split(","))

    for yaml_file in yaml_files:
        target = yaml_file.stem.replace("_input", "")

        if targets_allowlist is not None and target not in targets_allowlist:
            continue

        if round_tag:
            target_output_dir = base_output_dir / f"{target}_{round_tag}"
        else:
            target_output_dir = base_output_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Boltz creates: {target_output_dir}/boltz_results_{stem}/predictions/{record}/*.cif
        # The record subdir is NOT always "B" (that was an artifact of the
        # single-ligand L-series inputs), so glob one level instead of hardcoding
        # it -- a wrong name here silently disables resume and re-runs everything.
        # Check actual CIF files, not just directory existence (empty dirs = failed runs)
        pred_dir = target_output_dir / f"boltz_results_{yaml_file.stem}" / "predictions"
        if not cfg.override and pred_dir.is_dir() and list(pred_dir.glob("*/*.cif")):
            log.info(f"  Skipping {target} (already exists, use override=true to rerun)")
            continue

        cmd = build_boltz2_cmd(boltz_exec, cfg, yaml_file, target_output_dir, cache_dir)
        log.info(f"  [{target}] {' '.join(cmd)}")
        result = subprocess.run(cmd)  # nosec
        if result.returncode != 0:
            log.error(f"  [{target}] failed with exit code {result.returncode}, skipping.")
            continue

        # Auto-symlink into the ensemble umbrella dir so ensemble glob
        # `{base}_{round}/boltz_results_{target}_input/predictions/...` matches
        # without manual `ln -s`. Skipped for legacy flat layout (round_tag=null).
        if round_tag:
            umbrella = base_output_dir.parent / f"{base_output_dir.name}_{round_tag}"
            umbrella.mkdir(parents=True, exist_ok=True)
            link = umbrella / f"boltz_results_{yaml_file.stem}"
            target_rel = Path("..") / base_output_dir.name / f"{target}_{round_tag}" / f"boltz_results_{yaml_file.stem}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target_rel)
            log.info(f"  [{target}] linked → {umbrella}/{link.name}")

    log.info("Done.")


if __name__ == "__main__":
    main()
