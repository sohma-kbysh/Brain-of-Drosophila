"""Quantum state and channel validity (spec section 25)."""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.quantum import channels, states


@pytest.fixture
def random_bloch(rng):
    v = rng.normal(size=(12, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0.0, 1.0, size=(12, 1))


class TestDensityMatrixValidity:
    def test_trace_is_one(self, random_bloch):
        rho = states.bloch_to_density(random_bloch)
        assert np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0)

    def test_hermitian(self, random_bloch):
        rho = states.bloch_to_density(random_bloch)
        assert np.allclose(rho, np.conj(np.swapaxes(rho, -2, -1)))

    def test_positive_semidefinite(self, random_bloch):
        rho = states.bloch_to_density(random_bloch)
        assert np.all(np.linalg.eigvalsh(rho) >= -1e-12)

    def test_validity_helper_agrees(self, random_bloch):
        for r in random_bloch:
            assert states.is_valid_density_matrix(states.bloch_to_density(r))

    def test_round_trip(self, random_bloch):
        back = states.density_to_bloch(states.bloch_to_density(random_bloch))
        assert np.allclose(back, random_bloch)

    def test_rejects_non_physical_matrix(self):
        assert not states.is_valid_density_matrix(np.array([[2.0, 0], [0, 0]]))
        assert not states.is_valid_density_matrix(np.array([[0.5, 1j], [1j, 0.5]]))


class TestBB84Encoding:
    def test_four_states_are_pure_and_correct(self):
        r = states.encode_bb84(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        assert np.allclose(np.linalg.norm(r, axis=1), 1.0)
        assert np.allclose(r, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]])

    def test_correct_basis_no_noise_gives_zero_error(self, rng):
        bits = rng.integers(0, 2, size=64)
        bases = rng.integers(0, 2, size=64)
        p = states.measure_bb84(states.encode_bb84(bits, bases), bases)
        assert np.allclose(np.where(bits == 0, p, 1 - p), 1.0)

    def test_wrong_basis_is_a_coin_flip(self, rng):
        bits = rng.integers(0, 2, size=64)
        bases = rng.integers(0, 2, size=64)
        p = states.measure_bb84(states.encode_bb84(bits, bases), 1 - bases)
        assert np.allclose(p, 0.5)

    def test_maximally_mixed_gives_half(self):
        p = states.measure_bb84(states.maximally_mixed((16,)), np.zeros(16, dtype=int))
        assert np.allclose(p, 0.5)

    def test_rejects_non_binary_input(self):
        with pytest.raises(ValueError):
            states.encode_bb84(np.array([2]), np.array([0]))


class TestChannelLimits:
    """Spec section 25: t=0 identity, large t -> off-diagonals vanish."""

    def test_dephasing_at_zero_time_is_identity(self, random_bloch):
        out = channels.t1_t2_channel(random_bloch, 0.0, np.inf, 1e-6)
        assert np.allclose(out, random_bloch)

    def test_dephasing_kills_coherence_at_long_time(self, random_bloch):
        out = channels.pure_dephasing(random_bloch, 1e-3, 1e-6)
        assert np.allclose(out[:, :2], 0.0, atol=1e-9)
        assert np.allclose(out[:, 2], random_bloch[:, 2])

    def test_amplitude_damping_relaxes_to_ground(self, random_bloch):
        out = channels.amplitude_damping(random_bloch, 1e-3, 1e-6)
        assert np.allclose(out, [[0.0, 0.0, 1.0]] * len(random_bloch), atol=1e-9)

    def test_depolarizing_to_maximally_mixed(self, random_bloch):
        assert np.allclose(channels.depolarizing(random_bloch, 1.0), 0.0)
        assert np.allclose(channels.depolarizing(random_bloch, 0.0), random_bloch)

    def test_channels_preserve_physicality(self, random_bloch):
        for t in (0.0, 1e-7, 1e-6, 1e-5, 1e-3):
            out = channels.t1_t2_channel(random_bloch, t, 2e-6, 1e-6)
            assert np.all(states.is_physical_bloch(out)), f"unphysical at t={t}"

    def test_infinite_coherence_control_does_nothing(self, random_bloch):
        out = channels.t1_t2_channel(random_bloch, 1.0, np.inf, np.inf)
        assert np.allclose(out, random_bloch)


class TestT1T2Relation:
    def test_round_trip(self):
        T1, T2 = 3e-6, 2e-6
        assert channels.t2_from_t1_t_phi(T1, channels.t_phi_from_t1_t2(T1, T2)) == pytest.approx(T2)

    def test_rejects_T2_above_2T1(self):
        with pytest.raises(ValueError, match="exceeds the physical bound"):
            channels.validate_t1_t2(T1=1e-6, T2=3e-6)

    def test_T2_equals_2T1_is_allowed(self):
        channels.validate_t1_t2(T1=1e-6, T2=2e-6)
        assert np.isinf(channels.t_phi_from_t1_t2(1e-6, 2e-6))

    def test_coherence_decays_at_exactly_T2(self):
        """|rho_01| must fall by 1/e in one T2, whatever the T1/T2 split."""
        for T1, T2 in ((1e-6, 1e-6), (5e-6, 1e-6), (np.inf, 1e-6), (0.5e-6, 1e-6)):
            out = channels.t1_t2_channel(np.array([[1.0, 0.0, 0.0]]), T2, T1, T2)
            assert out[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-9)
