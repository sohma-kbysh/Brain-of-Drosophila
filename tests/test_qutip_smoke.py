"""QuTiP open-quantum-system smoke tests.

Not just an import check: we build |+>, evolve it under a Lindblad master
equation, and assert the resulting density matrix is physical. This is the
minimum machinery the radical-pair model will rely on.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

TLIST = np.linspace(0.0, 10.0, 101)


@pytest.fixture(scope="module")
def plus_state():
    """|+> = (|0> + |1>)/sqrt(2)."""
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()


@pytest.fixture(scope="module")
def evolved(plus_state):
    """Evolve |+> under a driven, damped qubit and return the final rho."""
    H = 2 * np.pi * 0.1 * qt.sigmax()
    c_ops = [np.sqrt(0.05) * qt.sigmam()]  # amplitude damping
    result = qt.mesolve(H, plus_state, TLIST, c_ops=c_ops, e_ops=[])
    return result.states[-1]


def test_plus_state_is_normalised(plus_state):
    assert plus_state.norm() == pytest.approx(1.0, abs=1e-12)
    amps = plus_state.full().ravel()
    assert np.allclose(amps, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_final_state_trace_is_one(evolved):
    assert evolved.tr().real == pytest.approx(1.0, abs=1e-8)
    assert abs(evolved.tr().imag) < 1e-12


def test_final_state_is_hermitian(evolved):
    rho = evolved.full()
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


def test_final_state_is_finite(evolved):
    assert np.all(np.isfinite(evolved.full()))


def test_final_state_is_positive_semidefinite(evolved):
    evals = np.linalg.eigvalsh(evolved.full())
    assert evals.min() > -1e-9, f"negative eigenvalue: {evals.min()}"
    assert evals.sum() == pytest.approx(1.0, abs=1e-8)


def test_damping_reduces_purity(plus_state, evolved):
    """Dissipation must actually act: the state should decohere."""
    initial_purity = (qt.ket2dm(plus_state) ** 2).tr().real
    final_purity = (evolved**2).tr().real
    assert initial_purity == pytest.approx(1.0, abs=1e-9)
    assert final_purity < initial_purity


def test_unitary_evolution_preserves_purity(plus_state):
    """With no collapse operators, a pure state stays pure (sanity control)."""
    H = 2 * np.pi * 0.1 * qt.sigmaz()
    res = qt.mesolve(H, plus_state, TLIST, c_ops=[], e_ops=[])
    rho = qt.ket2dm(res.states[-1]) if res.states[-1].type == "ket" else res.states[-1]
    assert (rho**2).tr().real == pytest.approx(1.0, abs=1e-6)


def test_expectation_values_are_bounded(plus_state):
    """<sigma_z> must stay within [-1, 1] over the whole trajectory."""
    H = 2 * np.pi * 0.1 * qt.sigmax()
    c_ops = [np.sqrt(0.05) * qt.sigmam()]
    res = qt.mesolve(H, plus_state, TLIST, c_ops=c_ops, e_ops=[qt.sigmaz()])
    sz = np.asarray(res.expect[0])
    assert np.all(np.isfinite(sz))
    assert sz.min() >= -1.0 - 1e-9 and sz.max() <= 1.0 + 1e-9
