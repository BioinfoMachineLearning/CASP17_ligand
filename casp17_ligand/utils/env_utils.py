"""Utilities for resolving conda environment executables."""

import logging
import os
import subprocess  # nosec
from typing import Optional

log = logging.getLogger(__name__)


def get_conda_envs_dir(conda_envs_dir: Optional[str] = None) -> Optional[str]:
    """Return the conda envs directory, auto-detecting via `conda info --base` if not given."""
    if conda_envs_dir:
        return conda_envs_dir
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, check=True, timeout=10,
        )  # nosec
        conda_base = result.stdout.strip()
        return os.path.join(conda_base, "envs")
    except Exception as e:
        log.warning(f"Could not auto-detect conda base: {e}")
        return None


def get_method_exec(exec_name: str, env_name: str, conda_envs_dir: Optional[str] = None) -> str:
    """Resolve the full path to a method executable.

    Resolution priority:
    1. If exec_name is already an absolute path → use directly.
    2. Try {conda_envs_dir}/{env_name}/bin/{exec_name} (auto-detect envs dir if not given).
    3. Fall back to exec_name as-is (assumes it is on PATH).

    Args:
        exec_name: Executable name (e.g. "boltz") or absolute path.
        env_name: Conda environment name (e.g. "boltz").
        conda_envs_dir: Path to conda envs directory. Auto-detected if None.

    Returns:
        Resolved executable path.
    """
    if os.path.isabs(exec_name):
        return exec_name

    envs_dir = get_conda_envs_dir(conda_envs_dir)
    if envs_dir:
        candidate = os.path.join(envs_dir, env_name, "bin", exec_name)
        if os.path.isfile(candidate):
            log.debug(f"Resolved {exec_name} → {candidate}")
            return candidate
        log.warning(f"Executable not found at {candidate}, falling back to PATH")

    return exec_name
