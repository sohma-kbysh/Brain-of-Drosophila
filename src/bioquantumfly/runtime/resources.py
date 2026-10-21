"""Detect the actual machine we are running on (addendum sections 2, 3, 15, 17, 25).

Nothing here assumes hayate specifically: every probe degrades gracefully when
a tool or path is missing. The point is to *record* what we found, so results
metadata says which hardware produced them.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Candidate scratch locations, best first (addendum section 15).
#: A project-managed directory under /home beats filling the root filesystem.
SCRATCH_CANDIDATES = ("/scratch", "/local/scratch", "/local")

#: Environment variables a scheduler exports into the job (addendum section 3).
SLURM_VARS = (
    "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_NTASKS",
)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Run a probe command, returning None rather than raising if unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


@dataclass
class FilesystemInfo:
    path: str
    mountpoint: str
    total_gb: float
    available_gb: float
    is_root_filesystem: bool
    filesystem_type: str | None = None

    @property
    def is_tight(self) -> bool:
        """Under 50 GB free, or the root filesystem above 90% used."""
        used_frac = 1.0 - (self.available_gb / self.total_gb) if self.total_gb else 0.0
        return self.available_gb < 50.0 or (self.is_root_filesystem and used_frac > 0.90)


@dataclass
class SchedulerInfo:
    detected: bool = False
    kind: str | None = None
    commands_available: list[str] = field(default_factory=list)
    inside_allocation: bool = False
    allocated_cpus: int | None = None
    allocated_memory_gb: float | None = None
    job_id: str | None = None


@dataclass
class ResourceReport:
    """A snapshot of the execution environment, embedded in results metadata."""

    hostname: str
    platform: str
    cpu_model: str | None
    logical_cpu_count: int
    physical_core_count: int | None
    affinity_cpu_count: int
    total_memory_gb: float | None
    available_memory_gb: float | None
    load_average: tuple[float, float, float] | None
    numa_nodes: int | None
    numa_available: bool
    scheduler: SchedulerInfo
    project_filesystem: FilesystemInfo | None
    temporary_filesystem: FilesystemInfo | None
    shm_filesystem: FilesystemInfo | None
    python_version: str
    compiler: str | None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def usable_cpu_count(self) -> int:
        """CPUs we may actually use.

        Inside a scheduler allocation this is the allocated count, never the
        machine total -- exceeding the allocation would harm other jobs.
        """
        if self.scheduler.inside_allocation and self.scheduler.allocated_cpus:
            return self.scheduler.allocated_cpus
        return self.affinity_cpu_count

    @property
    def is_heavily_loaded(self) -> bool:
        """Is somebody else already using this shared machine? (addendum 25)"""
        if self.load_average is None:
            return False
        return self.load_average[0] > 0.7 * self.logical_cpu_count


def _filesystem_info(path: str | Path) -> FilesystemInfo | None:
    p = Path(path)
    try:
        usage = shutil.disk_usage(p)
    except OSError:
        return None
    q = p.resolve()
    while not os.path.ismount(q) and q != q.parent:
        q = q.parent
    return FilesystemInfo(
        path=str(p),
        mountpoint=str(q),
        total_gb=usage.total / 2**30,
        available_gb=usage.free / 2**30,
        is_root_filesystem=str(q) == "/",
    )


def _detect_cpu_model() -> tuple[str | None, int | None, int | None]:
    """Return (cpu model, physical cores, NUMA nodes) from lscpu when present."""
    text = _run(["lscpu"])
    if not text:
        return None, None, None
    model = cores_per_socket = sockets = numa = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Model name":
            model = value
        elif key == "Core(s) per socket":
            cores_per_socket = int(value) if value.isdigit() else None
        elif key == "Socket(s)":
            sockets = int(value) if value.isdigit() else None
        elif key == "NUMA node(s)":
            numa = int(value) if value.isdigit() else None
    physical = cores_per_socket * sockets if cores_per_socket and sockets else None
    return model, physical, numa


def _detect_memory_gb() -> tuple[float | None, float | None]:
    """Total and available RAM, read from /proc/meminfo (Linux)."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None, None
    total = avail = None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) / 2**20      # kB -> GiB
        elif line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / 2**20
    return total, avail


def _detect_scheduler() -> SchedulerInfo:
    info = SchedulerInfo()
    for cmd in ("sbatch", "srun", "qsub"):
        if shutil.which(cmd):
            info.commands_available.append(cmd)
    if info.commands_available:
        info.detected = True
        info.kind = "slurm" if {"sbatch", "srun"} & set(info.commands_available) else "pbs/sge"

    if os.environ.get("SLURM_JOB_ID"):
        info.detected = True
        info.kind = "slurm"
        info.inside_allocation = True
        info.job_id = os.environ["SLURM_JOB_ID"]
        cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
        if cpus and cpus.isdigit():
            info.allocated_cpus = int(cpus)
        mem = os.environ.get("SLURM_MEM_PER_NODE")
        if mem and mem.isdigit():
            info.allocated_memory_gb = int(mem) / 1024.0     # SLURM reports MB
        elif (per_cpu := os.environ.get("SLURM_MEM_PER_CPU")) and per_cpu.isdigit():
            n = info.allocated_cpus or 1
            info.allocated_memory_gb = int(per_cpu) * n / 1024.0
    return info


def _detect_numa_available() -> bool:
    return _run(["numactl", "--hardware"]) is not None


def detect_resources(project_path: str | Path | None = None) -> ResourceReport:
    """Probe the machine. Never raises; missing information becomes None."""
    project_path = Path(project_path) if project_path else Path.cwd()
    model, physical, numa = _detect_cpu_model()
    total_mem, avail_mem = _detect_memory_gb()

    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1

    try:
        load = os.getloadavg()
    except (AttributeError, OSError):
        load = None

    compiler = None
    if (cc := _run(["g++", "--version"])):
        compiler = cc.splitlines()[0]

    return ResourceReport(
        hostname=platform.node(),
        platform=platform.platform(),
        cpu_model=model,
        logical_cpu_count=os.cpu_count() or 1,
        physical_core_count=physical,
        affinity_cpu_count=affinity,
        total_memory_gb=total_mem,
        available_memory_gb=avail_mem,
        load_average=tuple(load) if load else None,
        numa_nodes=numa,
        numa_available=_detect_numa_available(),
        scheduler=_detect_scheduler(),
        project_filesystem=_filesystem_info(project_path),
        temporary_filesystem=_filesystem_info(os.environ.get("TMPDIR", "/tmp")),
        shm_filesystem=_filesystem_info("/dev/shm") if Path("/dev/shm").exists() else None,
        python_version=platform.python_version(),
        compiler=compiler,
    )


def resolve_scratch_dir(
    project_root: str | Path,
    requested: str = "auto",
    use_shm: bool = False,
    shm_limit_gb: float = 16.0,
) -> Path:
    """Pick a scratch directory (addendum sections 15, 16).

    Resolution order for ``auto``:
      1. a dedicated local scratch filesystem, if one exists and is writable
      2. /dev/shm -- ONLY on explicit opt-in, and it is RAM, so it is capped
      3. a project-managed directory under the project root

    The root filesystem is never chosen: on this class of machine `/` is the
    filesystem most likely to be nearly full, and filling it affects every user.
    """
    project_root = Path(project_root)
    if requested != "auto":
        p = Path(requested)
        p.mkdir(parents=True, exist_ok=True)
        return p

    if use_shm:
        shm = Path("/dev/shm")
        if shm.is_dir() and os.access(shm, os.W_OK):
            info = _filesystem_info(shm)
            if info and info.available_gb > shm_limit_gb:
                p = shm / f"bioquantumfly-{os.getuid()}"
                p.mkdir(parents=True, exist_ok=True)
                return p

    for cand in SCRATCH_CANDIDATES:
        c = Path(cand)
        if c.is_dir() and os.access(c, os.W_OK):
            info = _filesystem_info(c)
            if info and not info.is_root_filesystem and info.available_gb > 10:
                p = c / f"bioquantumfly-{os.getuid()}"
                p.mkdir(parents=True, exist_ok=True)
                return p

    p = project_root / "data" / "cache" / "scratch"
    p.mkdir(parents=True, exist_ok=True)
    return p


class DiskSpaceError(RuntimeError):
    """Raised when a planned write would not safely fit (addendum section 17)."""


def check_disk_space(
    path: str | Path,
    estimated_gb: float,
    *,
    min_headroom_gb: float = 20.0,
    raise_on_fail: bool = True,
) -> FilesystemInfo:
    """Refuse to start a large write that would not leave headroom.

    We check before downloading or generating, not after, because the failure
    mode we care about (a full root filesystem) harms other users of the machine.
    """
    info = _filesystem_info(path)
    if info is None:
        raise DiskSpaceError(f"cannot stat filesystem for {path}")

    remaining = info.available_gb - estimated_gb
    if remaining < min_headroom_gb:
        msg = (
            f"refusing to write ~{estimated_gb:.1f} GB to {path} "
            f"(mount {info.mountpoint}): only {info.available_gb:.1f} GB free, "
            f"would leave {remaining:.1f} GB (< {min_headroom_gb} GB headroom)"
        )
        if info.is_root_filesystem:
            msg += ". This is the ROOT filesystem -- write under /home instead."
        if raise_on_fail:
            raise DiskSpaceError(msg)
    return info
