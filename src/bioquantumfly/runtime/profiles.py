"""Explicit resource profiles (addendum section 4).

`large` is never the default. `nproc` reporting 384 CPUs is not a licence to
use 384 CPUs on a shared research server.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from bioquantumfly.runtime.resources import ResourceReport


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    workers: int
    blas_threads: int
    max_memory_gb: float
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, ResourceProfile] = {
    "development": ResourceProfile(
        "development", workers=4, blas_threads=1, max_memory_gb=32,
        description="Default. Interactive work and tests; polite on a shared machine.",
    ),
    "standard": ResourceProfile(
        "standard", workers=16, blas_threads=1, max_memory_gb=128,
        description="Routine parameter sweeps.",
    ),
    "large": ResourceProfile(
        "large", workers=64, blas_threads=1, max_memory_gb=512,
        description="Large sweeps. Opt-in only; check machine load first.",
    ),
    "linalg": ResourceProfile(
        "linalg", workers=1, blas_threads=16, max_memory_gb=128,
        description=(
            "One big multi-threaded linear-algebra job instead of many processes. "
            "Benchmark before assuming this beats process parallelism."
        ),
    ),
}

DEFAULT_PROFILE = "development"


def resolve_profile(
    name: str = DEFAULT_PROFILE,
    *,
    workers: int | None = None,
    blas_threads: int | None = None,
    max_memory_gb: float | None = None,
    resources: ResourceReport | None = None,
) -> ResourceProfile:
    """Resolve a profile, apply CLI overrides, then clamp to what we may use.

    Clamping matters inside a scheduler allocation: a profile asking for 64
    workers must not spawn 64 processes in a 8-CPU allocation.
    """
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    base = PROFILES[name]

    resolved = ResourceProfile(
        name=base.name,
        workers=workers if workers is not None else base.workers,
        blas_threads=blas_threads if blas_threads is not None else base.blas_threads,
        max_memory_gb=max_memory_gb if max_memory_gb is not None else base.max_memory_gb,
        description=base.description,
    )
    if resolved.workers < 1 or resolved.blas_threads < 1:
        raise ValueError("workers and blas_threads must be >= 1")

    if resources is not None:
        usable = resources.usable_cpu_count
        if resolved.workers * resolved.blas_threads > usable:
            capped = max(1, usable // resolved.blas_threads)
            resolved = ResourceProfile(
                resolved.name, capped, resolved.blas_threads,
                resolved.max_memory_gb, resolved.description,
            )
        if resources.available_memory_gb is not None:
            budget = 0.8 * resources.available_memory_gb
            if resolved.max_memory_gb > budget:
                resolved = ResourceProfile(
                    resolved.name, resolved.workers, resolved.blas_threads,
                    budget, resolved.description,
                )
    return resolved


def load_warnings(resources: ResourceReport, profile: ResourceProfile) -> list[str]:
    """Advisory warnings before a large local job (addendum section 25).

    Advisory only: we never kill other users' jobs or change system settings.
    """
    out: list[str] = []
    if resources.is_heavily_loaded and resources.load_average:
        out.append(
            f"machine load average is {resources.load_average[0]:.1f} on "
            f"{resources.logical_cpu_count} CPUs -- others may be running jobs; "
            f"consider a smaller profile than {profile.name!r}"
        )
    total = profile.workers * profile.blas_threads
    if total > 0.5 * resources.usable_cpu_count:
        out.append(
            f"requesting {total} threads of {resources.usable_cpu_count} usable CPUs "
            "on a shared server"
        )
    if resources.available_memory_gb and profile.max_memory_gb > resources.available_memory_gb:
        out.append(
            f"memory budget {profile.max_memory_gb:.0f} GB exceeds available "
            f"{resources.available_memory_gb:.0f} GB"
        )
    for fs, label in (
        (resources.project_filesystem, "project"),
        (resources.temporary_filesystem, "temporary"),
    ):
        if fs and fs.is_root_filesystem:
            out.append(f"{label} filesystem is the ROOT filesystem ({fs.mountpoint}) -- move it under /home")
        elif fs and fs.is_tight:
            out.append(f"{label} filesystem {fs.mountpoint} has only {fs.available_gb:.0f} GB free")
    return out
