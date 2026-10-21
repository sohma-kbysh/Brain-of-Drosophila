"""Result metadata: what produced this number (spec 24, addendum 23).

Every results directory carries a `metadata.json` recording both the software
parameters and the hardware that ran them, because the scaling measurements in
addendum section 22 are only meaningful alongside the machine description.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
from typing import Any

from bioquantumfly.runtime.blas import current_thread_settings
from bioquantumfly.runtime.resources import ResourceReport, detect_resources

TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "pyarrow", "networkx",
    "matplotlib", "qutip", "brian2", "neuprint-python", "pydantic",
)


def git_commit(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Current commit and whether the tree is dirty.

    A dirty tree is recorded rather than rejected: during development most runs
    happen on uncommitted code, and silently claiming a clean commit would be
    worse than saying so.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        info["branch"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        info["dirty"] = bool(status)
    except (subprocess.SubprocessError, OSError):
        pass  # not a repo, or git unavailable: leave the fields None
    return info


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = dist_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def build_metadata(
    *,
    experiment: str,
    config: dict[str, Any],
    seed: int,
    resources: ResourceReport | None = None,
    malecns_version: str | None = None,
    physical_model: str | None = None,
    speculative_coupling_enabled: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata record stored beside every result."""
    res = resources or detect_resources()
    meta: dict[str, Any] = {
        "experiment": experiment,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "config": config,
        # --- scientific provenance ---
        "malecns_dataset_version": malecns_version,
        "physical_model": physical_model,
        "speculative_coupling_enabled": speculative_coupling_enabled,
        # --- software ---
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "packages": package_versions(),
        "git": git_commit(),
        # --- hardware / runtime (addendum 23) ---
        "hostname": res.hostname,
        "cpu_model": res.cpu_model,
        "logical_cpu_count": res.logical_cpu_count,
        "usable_cpu_count": res.usable_cpu_count,
        "total_memory_gb": res.total_memory_gb,
        "blas_threads": current_thread_settings(),
        "scheduler": res.scheduler.__dict__,
        "compiler": res.compiler,
        "project_filesystem": res.project_filesystem.__dict__ if res.project_filesystem else None,
    }
    if extra:
        meta.update(extra)
    return meta


def results_dir(root: str | Path, experiment: str, timestamp: str | None = None) -> Path:
    """`results/<experiment>/<timestamp>/`, created."""
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / experiment / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    out = Path(path) / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2, default=str))
    return out
