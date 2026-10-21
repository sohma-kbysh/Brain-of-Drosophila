"""BLAS thread control (addendum section 5).

The failure mode this prevents: 64 worker processes each starting 16 BLAS
threads produces ~1024 runnable threads on a machine shared with other people.
Throughput collapses and everyone else suffers.

The rule is simple -- pick ONE level of parallelism:
  * process-parallel sweeps  -> 1 BLAS thread per worker
  * one big linear-algebra job -> 1 worker, many BLAS threads

These variables are read by the BLAS libraries when they are first imported, so
they must be set before numpy is imported. `configure_blas_threads` therefore
warns if numpy is already loaded, and workers set them at process start.
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator

#: Every environment variable that controls a thread pool we might inherit.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_blas_threads(n_threads: int, *, warn_if_late: bool = True) -> dict[str, str]:
    """Pin every BLAS/OpenMP thread pool to `n_threads`.

    Returns the mapping that was applied, so it can be recorded in metadata.
    """
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    if warn_if_late and "numpy" in sys.modules:
        warnings.warn(
            "numpy is already imported; BLAS thread limits may not take effect. "
            "Call configure_blas_threads() before importing numpy (worker "
            "processes do this in their initialiser).",
            RuntimeWarning,
            stacklevel=2,
        )
    applied = {var: str(n_threads) for var in THREAD_VARS}
    os.environ.update(applied)
    return applied


def current_thread_settings() -> dict[str, str | None]:
    """What the thread environment currently says, for provenance records."""
    return {var: os.environ.get(var) for var in THREAD_VARS}


@contextmanager
def blas_threads(n_threads: int) -> Iterator[None]:
    """Temporarily set the thread limits, restoring the previous values after.

    Only affects libraries that read the variables lazily; already-initialised
    pools are unaffected. Useful for benchmarking, not for correctness.
    """
    previous = {var: os.environ.get(var) for var in THREAD_VARS}
    try:
        configure_blas_threads(n_threads, warn_if_late=False)
        yield
    finally:
        for var, old in previous.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


def worker_initializer(n_blas_threads: int = 1) -> None:
    """Process-pool initialiser: runs in the child before any heavy import."""
    configure_blas_threads(n_blas_threads, warn_if_late=False)
