"""The analytic Bloch backend must agree with QuTiP (addendum section 10).

Production sweeps use closed-form channels because calling `mesolve` per carrier
per time point would make the feasibility maps unaffordable. That optimisation
is only safe if the two backends agree, so this file is the contract between
them.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.analysis import metrics
from bioquantumfly.quantum import channels
from bioquantumfly.quantum.lindblad import evolve_qutip, lowering_to_ground
from bioquantumfly.quantum.states import bloch_to_density

TOLERANCE = 1e-6  # mesolve's own default integration tolerance


@pytest.mark.parametrize("T1,T2", [(2e-6, 1.5e-6), (1e-6, 2e-6), (5e-6, 1e-6), (np.inf, 1e-6)])
def test_analytic_matches_mesolve(T1, T2, rng):
    times = np.linspace(0, 3e-6, 7)
    for _ in range(4):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.2, 1.0)
        analytic = channels.t1_t2_channel(v[None, :], times[:, None], T1, T2)[:, 0, :]
        numeric = evolve_qutip(v, times, T1, T2)
        assert np.abs(analytic - numeric).max() < TOLERANCE


def test_ground_state_convention():
    """Amplitude damping must relax toward |0> = +z, not QuTiP's spin convention."""
    op = lowering_to_ground().full()
    assert np.allclose(op, [[0, 1], [0, 0]]), "expected |0><1|"
    late = channels.t1_t2_channel(np.array([[0.0, 0.0, -1.0]]), 1e-3, 1e-6, 1e-6)
    assert late[0, 2] == pytest.approx(1.0, abs=1e-9)


class TestMetricsAgainstQuTiP:
    def test_fidelity_uses_the_squared_convention(self, rng):
        import qutip as qt

        for _ in range(5):
            a, b = (rng.normal(size=3) for _ in range(2))
            a = a / np.linalg.norm(a) * rng.uniform(0, 1)
            b = b / np.linalg.norm(b) * rng.uniform(0, 1)
            qa, qb = qt.Qobj(bloch_to_density(a)), qt.Qobj(bloch_to_density(b))
            assert metrics.fidelity(a, b) == pytest.approx(qt.fidelity(qa, qb) ** 2, abs=1e-9)

    def test_trace_distance(self, rng):
        import qutip as qt

        for _ in range(5):
            a, b = (rng.normal(size=3) for _ in range(2))
            a = a / np.linalg.norm(a) * rng.uniform(0, 1)
            b = b / np.linalg.norm(b) * rng.uniform(0, 1)
            expected = qt.tracedist(qt.Qobj(bloch_to_density(a)), qt.Qobj(bloch_to_density(b)))
            assert metrics.trace_distance(a, b) == pytest.approx(expected, abs=1e-9)

    def test_von_neumann_entropy(self, rng):
        import qutip as qt

        for _ in range(5):
            a = rng.normal(size=3)
            a = a / np.linalg.norm(a) * rng.uniform(0, 0.99)
            expected = qt.entropy_vn(qt.Qobj(bloch_to_density(a)), base=2)
            assert metrics.von_neumann_entropy(a) == pytest.approx(expected, abs=1e-9)


class TestHelstrom:
    def test_orthogonal_states_are_perfectly_distinguishable(self):
        assert metrics.helstrom_guessing_probability([0, 0, 1], [0, 0, -1]) == pytest.approx(1.0)

    def test_identical_states_give_a_coin_flip(self):
        assert metrics.helstrom_guessing_probability([0, 0, 1], [0, 0, 1]) == pytest.approx(0.5)

    def test_breidbart_bound_for_raw_bb84(self):
        """The optimal unknown-basis BB84 guess is exactly cos^2(pi/8)."""
        axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        p = metrics.helstrom_guessing_probability(axis / np.sqrt(2), -axis / np.sqrt(2))
        assert p == pytest.approx(np.cos(np.pi / 8) ** 2, abs=1e-12)

    def test_never_exceeds_one(self, rng):
        for _ in range(20):
            a, b = (rng.normal(size=3) for _ in range(2))
            a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
            for prior in (0.1, 0.5, 0.9):
                p = metrics.helstrom_guessing_probability(a, b, prior)
                assert 0.5 - 1e-12 <= p <= 1.0 + 1e-12


class TestInformationMeasures:
    def test_holevo_zero_for_identical_ensemble(self):
        assert metrics.holevo_chi(np.array([[0, 0, 1.0], [0, 0, 1.0]])) == pytest.approx(0.0)

    def test_holevo_one_bit_for_orthogonal_pure_states(self):
        assert metrics.holevo_chi(np.array([[0, 0, 1.0], [0, 0, -1.0]])) == pytest.approx(1.0)

    def test_holevo_never_negative(self, rng):
        for _ in range(20):
            ens = rng.normal(size=(2, 3))
            ens /= np.linalg.norm(ens, axis=1, keepdims=True)
            assert metrics.holevo_chi(ens * rng.uniform(0, 1)) >= -1e-12

    def test_binary_entropy_endpoints(self):
        assert metrics.binary_entropy(0.5) == pytest.approx(1.0)
        assert metrics.binary_entropy(0.0) == pytest.approx(0.0)
        assert metrics.binary_entropy(1.0) == pytest.approx(0.0)

    def test_mutual_information_of_independent_variables_is_zero(self):
        joint = np.full((2, 2), 0.25)
        assert metrics.mutual_information_from_joint(joint) == pytest.approx(0.0)

    def test_mutual_information_of_perfect_correlation_is_one_bit(self):
        joint = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert metrics.mutual_information_from_joint(joint) == pytest.approx(1.0)

    def test_exact_recovery_is_the_product(self):
        p = np.array([0.99] * 10)
        assert metrics.exact_recovery_probability(p) == pytest.approx(0.99 ** 10)
