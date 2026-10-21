"""Every core dependency must import, and versions must be reportable.

This is the coarse tripwire: if the environment drifts or a package is
missing, this fails before any scientific test runs.
"""

from __future__ import annotations

import sys

import pytest

CORE_MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "networkx",
    "matplotlib",
    "sympy",
    "Cython",
    "jinja2",
    "packaging",
    "qutip",
    "brian2",
    "neuprint",
    "yaml",
    "pydantic",
    "typer",
    "rich",
    "tqdm",
]


def test_python_version_is_3_12():
    assert sys.version_info[:2] == (3, 12), (
        f"expected Python 3.12.x, got {sys.version.split()[0]}"
    )


@pytest.mark.parametrize("modname", CORE_MODULES)
def test_core_module_imports(modname):
    __import__(modname)


def test_versions_are_reportable(capsys):
    """Each core package exposes a version via module attr or dist metadata."""
    import importlib
    from importlib.metadata import PackageNotFoundError, version as dist_version

    dist_of = {"Cython": "cython", "yaml": "pyyaml", "neuprint": "neuprint-python"}
    for name in CORE_MODULES:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
        if ver is None:
            try:
                ver = dist_version(dist_of.get(name, name))
            except PackageNotFoundError:
                ver = None
        assert ver, f"no version discoverable for {name}"


def test_numpy_scipy_interop():
    """Guards against a numpy/scipy ABI mismatch, which imports fine but breaks."""
    import numpy as np
    import scipy.linalg as sla

    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    evals = np.sort(sla.eigvalsh(a))
    assert np.allclose(evals, [2.0, 3.0])
