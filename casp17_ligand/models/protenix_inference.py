"""Run Protenix inference via conda environment on prepared input JSON files.

Runs ONE seed at a time per subprocess so each seed starts with a clean GPU
memory state (fixes V100 32GB OOM that occurred when running all seeds in a
single process).  Already-completed seeds are detected at the seed directory
level and skipped, so interrupted runs can be safely resumed.

Optional `targets` config list restricts processing to a subset of targets,
enabling parallel runs on different GPUs.
"""

import logging
import os
import shutil
import subprocess  # nosec
import tempfile
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def build_protenix_cmd(cfg: DictConfig, input_path: Path, output_dir: Path, seed: int) -> tuple:
    """Build the conda run command for a single seed. input_path is a directory."""
    env = os.environ.copy()
    # A relative protenix_root_dir is resolved against the project root, so the
    # configs can say "weights/protenix" instead of one machine's absolute path.
    protenix_root = Path(cfg.get("protenix_root_dir") or os.path.expanduser("~"))
    if not protenix_root.is_absolute():
        protenix_root = Path(_PROJECT_ROOT) / protenix_root
    env["PROTENIX_ROOT_DIR"] = str(protenix_root)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_device", 2))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", cfg.get("env_name", "protenix"),
        "protenix", "pred",
        f"--input={input_path.resolve()}",
        f"--out_dir={output_dir.resolve()}",
        f"--model_name={cfg.model_name}",
        f"--seeds={seed}",
        f"--dtype={cfg.dtype}",
    ]
    if cfg.get("use_template") is not None:
        cmd.append(f"--use_template={'true' if cfg.use_template else 'false'}")
    if cfg.get("use_msa") is not None:
        cmd.append(f"--use_msa={'true' if cfg.use_msa else 'false'}")
    if cfg.get("use_rna_msa"):
        cmd.append("--use_rna_msa=true")
    if cfg.get("enable_cache"):
        cmd.append("--enable_cache=true")
    if cfg.get("cycle") is not None:
        cmd.append(f"--cycle={cfg.cycle}")
    if cfg.get("step") is not None:
        cmd.append(f"--step={cfg.step}")
    if cfg.get("n_sample") is not None:
        cmd.append(f"--sample={cfg.n_sample}")
    if cfg.get("seqres_database_path"):
        cmd.append(f"--seqres_database_path={cfg.seqres_database_path}")
    return cmd, env


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="protenix_inference",
)
def main(cfg: DictConfig) -> None:
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    input_dir = root / cfg.input_dir
    output_dir = root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seeds = [int(s) for s in str(cfg.seeds).split(",")]
    target_filter = set(cfg.get("targets") or [])

    all_jsons = sorted(input_dir.glob("*.json"))
    if not all_jsons:
        log.error(f"No JSON files found in {input_dir}")
        return

    # Build (json_file, seed) work list with seed-level skip
    work_items = []
    skipped_targets = []
    for jf in all_jsons:
        target = jf.stem
        if target_filter and target not in target_filter:
            continue
        missing = [
            s for s in all_seeds
            if cfg.override or not (output_dir / target / f"seed_{s}").exists()
        ]
        if not missing:
            skipped_targets.append(target)
        else:
            for s in missing:
                work_items.append((jf, s))

    total_targets = len([jf for jf in all_jsons if not target_filter or jf.stem in target_filter])
    log.info(
        f"Targets in scope: {total_targets} | "
        f"Fully done: {len(skipped_targets)} | "
        f"Work items (target×seed): {len(work_items)}"
    )
    if skipped_targets:
        log.info(f"  Fully done: {skipped_targets}")
    if not work_items:
        log.info("Nothing to do.")
        return

    # Run one seed at a time — each subprocess gets a fresh GPU memory state
    for idx, (jf, seed) in enumerate(work_items, 1):
        target = jf.stem
        seed_dir = output_dir / target / f"seed_{seed}"
        log.info(f"[{idx}/{len(work_items)}] {target} seed={seed}")

        # protenix pred renames input JSONs in-place, always copy to a temp dir
        tmpdir = tempfile.mkdtemp(prefix=f"protenix_{target}_s{seed}_")
        try:
            shutil.copy2(jf, Path(tmpdir) / jf.name)
            cmd, env = build_protenix_cmd(cfg, Path(tmpdir), output_dir, seed)
            log.info(f"  CMD: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env)  # nosec
            if result.returncode != 0:
                log.error(f"  FAILED (exit {result.returncode}): {target} seed={seed}")
            elif seed_dir.exists():
                log.info(f"  OK -> {seed_dir}")
            else:
                log.warning(f"  Exit 0 but seed_dir missing: {seed_dir}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    log.info("All work items processed.")


if __name__ == "__main__":
    main()
