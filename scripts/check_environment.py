#!/usr/bin/env python
"""One-shot environment validation for BioQuantumFly.

Run:  python scripts/check_environment.py

Verifies Python version, core imports, a real QuTiP Lindblad evolution, a real
Brian2 LIF simulation, C/C++ compiler presence, project filesystem location,
and (optionally) the presence of NEUPRINT_TOKEN.

Exit code 0 = environment usable. Non-zero = a required check failed.
A missing NEUPRINT_TOKEN is a WARNING, never a failure.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Conservative thread defaults must be set before numpy/BLAS import.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "16")

REQUIRED_PY = (3, 12)

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ]  {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")
    warnings.append(msg)


def fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
def check_python() -> None:
    section("1. Python interpreter")
    v = sys.version_info
    print(f"  version    : {platform.python_version()}")
    print(f"  executable : {sys.executable}")
    if (v.major, v.minor) == REQUIRED_PY:
        ok(f"Python {REQUIRED_PY[0]}.{REQUIRED_PY[1]}.x as required")
    else:
        fail(f"expected Python {REQUIRED_PY[0]}.{REQUIRED_PY[1]}.x, got {platform.python_version()}")

    expected_prefix = Path.home() / "conda-envs" / "bioquantumfly"
    exe = Path(sys.executable).resolve()
    if str(exe).startswith(str(expected_prefix.resolve())):
        ok(f"running from project env: {expected_prefix}")
    else:
        fail(f"interpreter is NOT the project env ({expected_prefix}); got {exe}")


def check_imports() -> None:
    section("2. Core imports")
    mods = [
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("pandas", "__version__"),
        ("pyarrow", "__version__"),
        ("networkx", "__version__"),
        ("matplotlib", "__version__"),
        ("sympy", "__version__"),
        ("Cython", "__version__"),
        ("jinja2", "__version__"),
        ("qutip", "__version__"),
        ("brian2", "__version__"),
        ("neuprint", "__version__"),
        ("yaml", "__version__"),
        ("pydantic", "VERSION"),
        ("typer", "__version__"),
        ("rich", "__version__"),
        ("tqdm", "__version__"),
        ("pytest", "__version__"),
    ]
    import importlib
    from importlib.metadata import PackageNotFoundError, version as dist_version

    # Import name -> distribution name, where they differ.
    dist_of = {"Cython": "cython", "yaml": "pyyaml", "neuprint": "neuprint-python"}

    for name, attr in mods:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, attr, None)
            if ver is None:  # some packages expose no module-level __version__
                try:
                    ver = dist_version(dist_of.get(name, name))
                except PackageNotFoundError:
                    ver = "unknown"
            ok(f"{name:<12} {ver}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{name}: {type(exc).__name__}: {exc}")


def check_qutip() -> None:
    section("3. QuTiP open-quantum-system smoke test")
    try:
        import numpy as np
        import qutip as qt

        # |+> = (|0> + |1>)/sqrt(2)
        plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        H = 2 * np.pi * 0.1 * qt.sigmax()          # simple driving Hamiltonian
        c_ops = [np.sqrt(0.05) * qt.sigmam()]      # amplitude damping
        tlist = np.linspace(0, 10, 101)

        res = qt.mesolve(H, plus, tlist, c_ops=c_ops, e_ops=[])
        rho = res.states[-1]
        arr = rho.full()

        assert abs(rho.tr() - 1.0) < 1e-6, f"trace = {rho.tr()}"
        assert np.allclose(arr, arr.conj().T, atol=1e-9), "density matrix not Hermitian"
        assert np.all(np.isfinite(arr)), "non-finite entries in density matrix"
        evals = np.linalg.eigvalsh(arr)
        assert evals.min() > -1e-9, f"negative eigenvalue {evals.min()}"

        ok(f"mesolve completed, trace={rho.tr().real:.12f}, Hermitian, finite, PSD")
        print("  QuTiP smoke test: PASS")
    except Exception as exc:  # noqa: BLE001
        fail(f"QuTiP smoke test: {type(exc).__name__}: {exc}")


def check_brian2() -> None:
    section("4. Brian2 simulation smoke test")
    try:
        import numpy as np
        import brian2 as b2

        # numpy codegen target: no compilation needed, deterministic in CI.
        b2.prefs.codegen.target = "numpy"
        b2.set_device("runtime")
        b2.BrianLogger.suppress_name("resolution_conflict", "method_choice")

        b2.start_scope()
        tau = 10 * b2.ms
        G = b2.NeuronGroup(1, "dv/dt = -v/tau : 1", method="exact")
        G.v = 1
        mon = b2.StateMonitor(G, "v", record=0)
        b2.run(20 * b2.ms)

        v_end = float(G.v[0])
        expected = float(np.exp(-20.0 / 10.0))  # exact solution of dv/dt = -v/tau
        assert np.isfinite(v_end), "membrane variable is not finite"
        assert abs(v_end - expected) < 1e-6, f"v={v_end}, expected {expected}"
        assert np.all(np.isfinite(mon.v[0])), "non-finite trace recorded"
        assert mon.v[0][0] > mon.v[0][-1], "variable did not decay"

        ok(f"LIF decay v(20ms)={v_end:.8f} (analytic {expected:.8f})")
        print("  Brian2 smoke test: PASS")
    except Exception as exc:  # noqa: BLE001
        fail(f"Brian2 smoke test: {type(exc).__name__}: {exc}")


def check_compiler() -> None:
    section("5. C / C++ compiler availability")
    for tool in ("gcc", "g++"):
        path = shutil.which(tool)
        if not path:
            fail(f"{tool} not found on PATH")
            continue
        try:
            out = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()[0]
            ok(f"{tool}: {out}  ({path})")
        except Exception as exc:  # noqa: BLE001
            fail(f"{tool} present but not runnable: {exc}")

    # Does Brian2 itself see a usable compiler for code generation?
    try:
        from brian2.codegen.cpp_prefs import get_compiler_and_args

        compiler, _args = get_compiler_and_args()
        ok(f"Brian2 reports C++ compiler: {compiler}")
    except Exception as exc:  # noqa: BLE001
        warn(f"could not query Brian2 compiler settings: {type(exc).__name__}: {exc}")

    try:
        import Cython  # noqa: F401

        ok("Cython available (Brian2 'cython' codegen target usable)")
    except Exception:  # noqa: BLE001
        warn("Cython not importable; Brian2 limited to numpy codegen")
    print("  NOTE: C++ standalone mode is intentionally NOT enabled at this stage.")


def check_filesystem() -> None:
    section("6. Filesystem placement")
    root = Path(__file__).resolve().parent.parent
    env = Path.home() / "conda-envs" / "bioquantumfly"

    for label, p in (("project", root), ("conda env", env)):
        if not p.exists():
            fail(f"{label} path missing: {p}")
            continue
        usage = shutil.disk_usage(p)
        free_gb = usage.free / 2**30
        total_gb = usage.total / 2**30
        mount = _mountpoint(p)
        line = f"{label:<10} {p}  -> mount {mount}  free {free_gb:,.0f} GiB / {total_gb:,.0f} GiB"
        if mount == "/":
            fail(f"{line}  [ON ROOT FILESYSTEM — must live under /home]")
        elif free_gb < 50:
            warn(f"{line}  [low free space]")
        else:
            ok(line)

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        m = _mountpoint(Path(tmpdir))
        (ok if m != "/" else warn)(f"TMPDIR={tmpdir} -> mount {m}")
    else:
        warn("TMPDIR not set; large temporary files may land on the root filesystem. "
             "Use scripts/activate_env.sh to set it.")


def _mountpoint(p: Path) -> str:
    p = p.resolve()
    while not os.path.ismount(p) and p != p.parent:
        p = p.parent
    return str(p)


def check_threads() -> None:
    section("7. CPU thread policy")
    total = os.cpu_count() or 0
    print(f"  logical CPUs available: {total}")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        val = os.environ.get(var)
        if val is None:
            warn(f"{var} not set")
        elif total and int(val) > total // 4:
            warn(f"{var}={val} is aggressive for a shared server")
        else:
            ok(f"{var}={val}")


def check_neuprint_token() -> None:
    section("8. neuPrint")
    try:
        import neuprint
        from neuprint import Client  # noqa: F401

        ok(f"neuprint-python {neuprint.__version__} importable (Client available)")
    except Exception as exc:  # noqa: BLE001
        fail(f"neuprint import failed: {type(exc).__name__}: {exc}")

    # Presence only. NEVER print or log the value.
    if os.environ.get("NEUPRINT_TOKEN"):
        ok("NEUPRINT_TOKEN is present (value intentionally not displayed)")
    else:
        warn(
            "NEUPRINT_TOKEN is not set. This is fine for setup and for the test "
            "suite; export it before fetching connectome data:\n"
            '            export NEUPRINT_TOKEN="<your token>"'
        )


def main() -> int:
    print("=" * 72)
    print("BioQuantumFly environment validation")
    print("=" * 72)

    check_python()
    check_imports()
    check_qutip()
    check_brian2()
    check_compiler()
    check_filesystem()
    check_threads()
    check_neuprint_token()

    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: FAILED  ({len(failures)} error(s), {len(warnings)} warning(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASSED  (0 errors, {len(warnings)} warning(s))")
    if warnings:
        for w in warnings:
            print(f"  ! {w.splitlines()[0]}")
    print("\nEnvironment ready for BioQuantumFly implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
