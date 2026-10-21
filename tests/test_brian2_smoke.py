"""Brian2 simulation smoke tests.

Runs real simulations (exponential decay, LIF spiking) and checks the results
against analytic values. Uses the 'numpy' codegen target so the suite never
depends on a compile step; compiler availability is reported separately.
"""

from __future__ import annotations

import numpy as np
import pytest

import brian2 as b2


@pytest.fixture(autouse=True)
def _brian_runtime():
    """Deterministic, compile-free Brian2 for every test in this module."""
    b2.prefs.codegen.target = "numpy"
    b2.set_device("runtime")
    b2.start_scope()
    yield
    b2.device.reinit()
    b2.set_device("runtime")


def test_exponential_decay_matches_analytic_solution():
    tau = 10 * b2.ms
    G = b2.NeuronGroup(1, "dv/dt = -v/tau : 1", method="exact")
    G.v = 1
    b2.run(1 * b2.ms)

    v = float(G.v[0])
    assert np.isfinite(v)
    assert v == pytest.approx(np.exp(-0.1), abs=1e-9)
    assert 0.0 < v < 1.0, "v must decay but stay positive"


def test_decay_is_monotonic_over_a_recorded_trace():
    tau = 10 * b2.ms
    G = b2.NeuronGroup(1, "dv/dt = -v/tau : 1", method="exact")
    G.v = 1
    mon = b2.StateMonitor(G, "v", record=0)
    b2.run(50 * b2.ms)

    trace = np.asarray(mon.v[0])
    assert trace.size > 10
    assert np.all(np.isfinite(trace))
    assert np.all(np.diff(trace) <= 1e-12), "trace is not monotonically decreasing"

    # StateMonitor records at the START of each timestep, so the final sample
    # sits at 50 ms - dt, not at 50 ms. Compare against its own timestamp.
    t_last = float(mon.t[-1] / b2.ms)
    assert t_last == pytest.approx(50.0 - float(b2.defaultclock.dt / b2.ms), abs=1e-9)
    assert trace[-1] == pytest.approx(np.exp(-t_last / 10.0), abs=1e-6)


def test_lif_neuron_spikes_under_constant_drive():
    """A driven LIF neuron must fire, and its rate must be finite and sane."""
    tau = 10 * b2.ms
    eqs = "dv/dt = (2 - v)/tau : 1"
    G = b2.NeuronGroup(1, eqs, threshold="v > 1", reset="v = 0", method="exact")
    G.v = 0
    spikes = b2.SpikeMonitor(G)
    b2.run(200 * b2.ms)

    assert spikes.num_spikes > 0, "driven LIF neuron never fired"
    times = np.asarray(spikes.t / b2.ms)
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0)
    # Analytic ISI for dv/dt=(I-v)/tau with I=2, threshold 1, reset 0:
    # t = tau * ln(I/(I-1)) = 10 ms * ln 2
    isi = np.diff(times)
    assert isi.mean() == pytest.approx(10.0 * np.log(2.0), rel=0.05)


def test_units_system_is_enforced():
    """Brian2's dimensional checking must be live; it catches model errors.

    Note: an undefined *name* (e.g. a missing `tau`) is resolved lazily at
    run() time, so it is not a construction error. A genuine dimensional
    inconsistency, however, is rejected immediately -- that is what we assert.
    """
    from brian2.units.fundamentalunits import DimensionMismatchError

    assert float(1 * b2.ms / b2.second) == pytest.approx(1e-3)

    # Quantity arithmetic is checked eagerly.
    with pytest.raises(DimensionMismatchError):
        _ = 1 * b2.ms + 1 * b2.mV

    # Equation-level dimensions are checked when the network is prepared, and
    # the underlying error arrives wrapped in a BrianObjectException.
    # dv/dt has dimension 1/time, so the RHS must too; "-v" alone does not.
    from brian2.core.base import BrianObjectException

    b2.start_scope()
    G = b2.NeuronGroup(1, "dv/dt = -v : 1", method="exact")  # noqa: F841
    with pytest.raises(BrianObjectException) as excinfo:
        b2.run(1 * b2.ms)
    assert isinstance(excinfo.value.__cause__, DimensionMismatchError)


def test_cpp_compiler_is_visible_to_brian2():
    """Report-only: standalone mode is not enabled yet, but note the compiler."""
    import shutil

    assert shutil.which("g++") is not None, "g++ not on PATH"
    from brian2.codegen.cpp_prefs import get_compiler_and_args

    compiler, _args = get_compiler_and_args()
    assert compiler, "Brian2 could not determine a C++ compiler"
