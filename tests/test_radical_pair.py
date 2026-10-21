"""Radical-pair mechanism physics."""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.quantum.radical_pair import RadicalPairMechanism, RadicalPairParameters


@pytest.fixture(scope="module")
def rp():
    return RadicalPairMechanism(RadicalPairParameters(n_nuclei=1))


class TestConservation:
    def test_total_yield_is_one(self, rp):
        """Every pair recombines through one channel or the other."""
        assert rp.total_yield() == pytest.approx(1.0, abs=1e-9)

    def test_singlet_yield_is_a_probability(self, rp):
        assert 0.0 <= rp.singlet_yield() <= 1.0

    @pytest.mark.parametrize("n_nuclei", [0, 1, 2])
    def test_conservation_for_each_hilbert_size(self, n_nuclei):
        mechanism = RadicalPairMechanism(RadicalPairParameters(n_nuclei=n_nuclei))
        assert mechanism.total_yield() == pytest.approx(1.0, abs=1e-9)
        assert mechanism.params.hilbert_dim == 4 * 2**n_nuclei


class TestInitialState:
    def test_singlet_born_starts_in_the_singlet(self, rp):
        assert rp.singlet_probability(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-12)

    def test_triplet_born_starts_outside_the_singlet(self):
        mechanism = RadicalPairMechanism(RadicalPairParameters(initial_state="triplet"))
        assert mechanism.singlet_probability(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)

    def test_initial_density_matrix_is_physical(self, rp):
        rho = rp.initial_density_matrix().full()
        assert np.trace(rho).real == pytest.approx(1.0)
        assert np.allclose(rho, rho.conj().T)
        assert np.linalg.eigvalsh(rho).min() > -1e-12


class TestDynamics:
    def test_population_decays(self, rp):
        p = rp.singlet_probability(np.array([0.0, 1e-7, 1e-6, 1e-5]))
        assert np.all(np.diff(p) <= 1e-12)
        assert p[-1] < 1e-3

    def test_probabilities_stay_bounded(self, rp):
        p = rp.singlet_probability(np.logspace(-9, -4, 20))
        assert np.all(p >= -1e-12) and np.all(p <= 1.0 + 1e-12)


class TestMagneticFieldEffect:
    def test_hyperfine_coupling_produces_a_field_effect(self, rp):
        """Singlet-triplet mixing is what makes the yield field-dependent."""
        yields = rp.magnetic_field_effect(np.array([0.0, 50e-6, 1e-3]))
        assert yields.max() - yields.min() > 1e-3

    def test_without_hyperfine_there_is_no_mixing(self):
        """Control: no hyperfine, no nuclei -> singlet stays singlet, yield flat."""
        mechanism = RadicalPairMechanism(
            RadicalPairParameters(n_nuclei=0, hyperfine_T=None)
        )
        yields = mechanism.magnetic_field_effect(np.array([0.0, 50e-6, 1e-3, 1e-1]))
        assert np.allclose(yields, yields[0], atol=1e-12)
        assert yields[0] == pytest.approx(1.0, abs=1e-9)


class TestValidation:
    def test_zero_recombination_is_rejected(self):
        with pytest.raises(ValueError, match="at least one recombination rate"):
            RadicalPairParameters(k_singlet=0.0, k_triplet=0.0)

    def test_negative_rate_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            RadicalPairParameters(k_singlet=-1.0)

    def test_unknown_initial_state_is_rejected(self):
        with pytest.raises(ValueError, match="unknown initial state"):
            RadicalPairParameters(initial_state="quintet")


class TestSeparationFromTheMemoryModel:
    def test_it_is_not_an_abstract_memory(self):
        from bioquantumfly.quantum.abstract_memory import AbstractEphemeralMemory

        assert not issubclass(RadicalPairMechanism, AbstractEphemeralMemory)
        assert not hasattr(RadicalPairMechanism, "write_bb84")

    def test_the_crypto_layer_never_imports_it(self):
        """Guard the boundary: arbitrary-write must not leak into the protocols."""
        from pathlib import Path

        crypto = Path(__file__).resolve().parent.parent / "src" / "bioquantumfly" / "crypto"
        for path in crypto.glob("*.py"):
            assert "radical_pair" not in path.read_text(), path
