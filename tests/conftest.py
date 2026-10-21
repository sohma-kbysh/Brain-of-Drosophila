"""Shared pytest configuration for BioQuantumFly.

Thread limits are pinned before numpy/BLAS import so the test suite never
spawns 384 OpenMP threads on this shared server.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

# Headless plotting: no display on the research server.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test is reproducible."""
    return np.random.default_rng(20260909)


@pytest.fixture
def small_graph(rng):
    """A small synthetic connectome. Fast enough for the whole suite."""
    from bioquantumfly.connectome.synthetic import synthetic_connectome

    return synthetic_connectome(120, rng=rng)


@pytest.fixture(scope="session")
def brian_runtime():
    """Configure Brian2 once: numpy codegen, no compilation, deterministic."""
    import brian2 as b2

    b2.prefs.codegen.target = "numpy"
    b2.set_device("runtime")
    for name in ("resolution_conflict", "method_choice", "unused_brian_object"):
        b2.BrianLogger.suppress_name(name)
    return b2
