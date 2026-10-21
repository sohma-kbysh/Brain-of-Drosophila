"""Protocol and adversary behaviour (spec section 25).

These encode the claims the README and report make, so a regression here means
a documented claim has stopped being true.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.crypto import get_attack, get_protocol
from bioquantumfly.crypto.encoding import random_message
from bioquantumfly.crypto.eve import ATTACKS
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.quantum.abstract_memory import MemoryParams

T2 = 1e-6
ALL_ATTACK_KWARGS = {
    "none": {}, "z_intercept_resend": {}, "random_basis_intercept_resend": {},
    "breidbart": {}, "weak_measurement": {"eta": 0.6},
    "coherent_ancilla": {"alpha": 0.8}, "swap_theft": {},
}


@pytest.fixture
def masked(rng):
    protocol = get_protocol("masked_conjugate")
    return protocol, protocol.encode(random_message(48, rng), rng)


@pytest.fixture
def raw(rng):
    protocol = get_protocol("raw_bb84")
    return protocol, protocol.encode(random_message(48, rng), rng)


class TestBobReadout:
    def test_perfect_at_zero_time_no_attack(self, masked):
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("none"),
                     host=MemoryParams.from_T2(T2), t_attack=0.0, t_read=0.0)
        assert m.bob_ber == pytest.approx(0.0, abs=1e-12)
        assert m.bob_exact_recovery == pytest.approx(1.0, abs=1e-9)

    def test_ber_approaches_one_half_under_full_depolarisation(self, masked, rng):
        """Spec sanity: a maximally mixed state gives exactly BER = 1/2."""
        from bioquantumfly.crypto.bob import read_metrics
        from bioquantumfly.quantum.states import maximally_mixed

        _, encoding = masked
        r = read_metrics(maximally_mixed((encoding.n_carriers,)), encoding)
        assert r["bob_ber"] == pytest.approx(0.5)

    def test_ber_grows_monotonically_with_storage_time(self, masked):
        protocol, encoding = masked
        host = MemoryParams.from_T2(T2)
        bers = [
            evaluate(encoding, protocol, get_attack("none"), host=host,
                     t_attack=0.0, t_read=ratio * T2).bob_ber
            for ratio in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
        ]
        # Under amplitude damping every carrier's error probability rises
        # monotonically: a Z-encoded 1 relaxes toward 0, and an X-encoded bit
        # loses its coherence toward a coin flip. So the mean must too.
        assert all(b >= a - 1e-12 for a, b in zip(bers, bers[1:])), bers

    def test_infinite_coherence_never_degrades(self, masked):
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("none"),
                     host=MemoryParams.ideal(), t_attack=0.0, t_read=1.0)
        assert m.bob_ber == pytest.approx(0.0, abs=1e-12)

    def test_attack_after_read_is_rejected(self, masked):
        protocol, encoding = masked
        with pytest.raises(ValueError, match="after t_read"):
            evaluate(encoding, protocol, get_attack("none"),
                     host=MemoryParams.from_T2(T2), t_attack=1e-6, t_read=1e-9)


class TestProtocolPrivacy:
    @pytest.mark.parametrize("attack_name", sorted(ALL_ATTACK_KWARGS))
    def test_masked_leaks_nothing_before_key_reveal(self, masked, attack_name):
        """The one-time pad makes this exactly zero for EVERY attack.

        If this ever fails, the pad has stopped masking and the protocol is
        broken, not merely weakened.
        """
        protocol, encoding = masked
        m = evaluate(encoding, protocol,
                     get_attack(attack_name, **ALL_ATTACK_KWARGS[attack_name]),
                     host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.ideal(),
                     t_attack=0.1 * T2, t_read=0.5 * T2)
        assert m.eve_pguess_before_key == pytest.approx(0.5, abs=1e-9)
        assert m.eve_info_before_key == pytest.approx(0.0, abs=1e-9)
        assert m.eve_holevo_before_key == pytest.approx(0.0, abs=1e-9)

    def test_raw_bb84_leaks_with_no_key_at_all(self, raw):
        """The negative result of Experiment 2: a secret basis is not encryption."""
        protocol, encoding = raw
        m = evaluate(encoding, protocol, get_attack("breidbart"),
                     host=MemoryParams.ideal(), eve_memory=MemoryParams.ideal(),
                     t_attack=0.0, t_read=0.0)
        assert m.eve_pguess_before_key == pytest.approx(np.cos(np.pi / 8) ** 2, abs=1e-9)
        assert m.eve_info_before_key > 0.39

    def test_no_attack_yields_nothing(self, masked):
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("none"),
                     host=MemoryParams.from_T2(T2), t_attack=0.0, t_read=0.5 * T2)
        assert m.eve_pguess_after_key == pytest.approx(0.5, abs=1e-12)


class TestSwapNegativeControl:
    """Spec section 25: Eve's reduced state equals the original, up to tolerance."""

    def test_eve_receives_the_exact_state(self, rng):
        from bioquantumfly.quantum.states import encode_bb84

        state = encode_bb84(rng.integers(0, 2, 32), rng.integers(0, 2, 32))
        branch = get_attack("swap_theft").branches[0]
        assert np.allclose(branch.eve(state), state, atol=1e-12)

    def test_host_is_left_in_the_ground_state(self, rng):
        from bioquantumfly.quantum.states import encode_bb84

        state = encode_bb84(rng.integers(0, 2, 16), rng.integers(0, 2, 16))
        host_after = get_attack("swap_theft").branches[0].host(state)
        assert np.allclose(host_after, [0.0, 0.0, 1.0])

    def test_host_erases_while_eve_retains(self, masked):
        """The central negative result: decoherence is not deletion."""
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("swap_theft"),
                     host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.ideal(),
                     t_attack=0.01 * T2, t_read=50 * T2)
        assert m.host_info_after_key == pytest.approx(0.0, abs=1e-9)
        assert m.eve_pguess_after_key > 0.99
        assert m.eve_info_after_key > 0.9

    def test_eve_with_a_finite_memory_also_forgets(self, masked):
        """Eve is not magic: a decohering adversary loses information too."""
        protocol, encoding = masked
        host = MemoryParams.from_T2(T2)
        m = evaluate(encoding, protocol, get_attack("swap_theft"), host=host,
                     eve_memory=MemoryParams.from_T2(T2),
                     t_attack=0.01 * T2, t_read=50 * T2, t_reveal=50 * T2)
        assert m.eve_info_after_key < 1e-6


class TestInformationDisturbance:
    @pytest.mark.parametrize("eta", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_stronger_attacks_learn_more_and_disturb_more(self, masked, eta):
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("weak_measurement", eta=eta),
                     host=MemoryParams.ideal(), eve_memory=MemoryParams.ideal(),
                     t_attack=0.0, t_read=0.0)
        assert m.eve_pguess_after_key >= 0.5 - 1e-12
        assert m.bob_disturbance_ber >= -1e-12
        if eta == 0.0:
            assert m.eve_pguess_after_key == pytest.approx(0.5, abs=1e-12)
            assert m.bob_disturbance_ber == pytest.approx(0.0, abs=1e-12)

    def test_both_increase_monotonically_in_eta(self, masked):
        protocol, encoding = masked
        info, disturbance = [], []
        for eta in np.linspace(0, 1, 9):
            m = evaluate(encoding, protocol, get_attack("weak_measurement", eta=eta),
                         host=MemoryParams.ideal(), eve_memory=MemoryParams.ideal(),
                         t_attack=0.0, t_read=0.0)
            info.append(m.eve_info_after_key)
            disturbance.append(m.bob_disturbance_ber)
        assert all(b >= a - 1e-12 for a, b in zip(info, info[1:]))
        assert all(b >= a - 1e-12 for a, b in zip(disturbance, disturbance[1:]))

    def test_a_silent_attack_learns_nothing(self, masked):
        """No free lunch: zero disturbance must mean zero information."""
        protocol, encoding = masked
        m = evaluate(encoding, protocol, get_attack("weak_measurement", eta=0.0),
                     host=MemoryParams.ideal(), eve_memory=MemoryParams.ideal(),
                     t_attack=0.0, t_read=0.0)
        assert m.bob_disturbance_ber == pytest.approx(0.0, abs=1e-12)
        assert m.eve_info_after_key == pytest.approx(0.0, abs=1e-12)


class TestAttackWellFormedness:
    @pytest.mark.parametrize("name", sorted(ALL_ATTACK_KWARGS))
    def test_branch_probabilities_sum_to_one(self, name):
        attack = get_attack(name, **ALL_ATTACK_KWARGS[name])
        assert sum(b.probability for b in attack.branches) == pytest.approx(1.0)

    @pytest.mark.parametrize("name", sorted(ALL_ATTACK_KWARGS))
    def test_attacks_produce_physical_states(self, name, rng):
        from bioquantumfly.quantum.states import encode_bb84, is_physical_bloch

        state = encode_bb84(rng.integers(0, 2, 24), rng.integers(0, 2, 24))
        attack = get_attack(name, **ALL_ATTACK_KWARGS[name])
        assert np.all(is_physical_bloch(attack.apply_to_host(state)))
        for branch in attack.branches:
            assert np.all(is_physical_bloch(branch.host(state)))
            if branch.eve is not None:
                assert np.all(is_physical_bloch(branch.eve(state)))

    def test_coherent_ancilla_at_zero_angle_is_a_no_op(self, rng):
        from bioquantumfly.quantum.states import encode_bb84

        state = encode_bb84(rng.integers(0, 2, 16), rng.integers(0, 2, 16))
        assert np.allclose(get_attack("coherent_ancilla", alpha=0.0).apply_to_host(state),
                           state, atol=1e-12)

    def test_full_coherent_coupling_matches_z_measurement(self, rng):
        """At alpha = pi/2 the ancilla CNOTs the computational value out."""
        from bioquantumfly.quantum.states import encode_bb84

        state = encode_bb84(rng.integers(0, 2, 16), rng.integers(0, 2, 16))
        coherent = get_attack("coherent_ancilla", alpha=np.pi / 2).apply_to_host(state)
        projective = get_attack("z_intercept_resend").apply_to_host(state)
        assert np.allclose(coherent, projective, atol=1e-12)

    def test_weak_measurement_rejects_out_of_range_strength(self):
        with pytest.raises(ValueError, match=r"eta must lie in \[0, 1\]"):
            get_attack("weak_measurement", eta=1.5)

    def test_registry_is_complete(self):
        assert set(ATTACKS) == set(ALL_ATTACK_KWARGS)

    def test_unknown_attack_is_rejected(self):
        with pytest.raises(ValueError, match="unknown attack"):
            get_attack("telepathy")
