"""Post-processing, certified deletion, noisy storage, and the composed stack.

These encode the claims the primitive layer makes. A regression here means one
of the protocol-design conclusions has stopped holding.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.analysis import requirements as R
from bioquantumfly.crypto import deletion as D
from bioquantumfly.crypto import noisy_storage as NS
from bioquantumfly.crypto import postprocessing as PP
from bioquantumfly.crypto import get_attack, get_protocol
from bioquantumfly.crypto.encoding import random_message
from bioquantumfly.crypto.stack import ABLATIONS, StackConfig, run_stack
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.quantum.abstract_memory import MemoryParams

T2 = 1e-6


@pytest.fixture
def encoding(rng):
    return get_protocol("masked_conjugate").encode(random_message(2048, rng), rng)


class TestRequirementsClosedForm:
    def test_matches_the_simulator(self, rng):
        """The analytic BER must reproduce the full evaluation.

        The closed form is an ensemble average over uniform bits and bases,
        while the simulator draws one realisation, so the two agree only up to
        the sampling spread. Per-carrier error probabilities take three values
        (0, 1-e^-x, and half that), giving a standard deviation of order 0.14
        near x = 0.5; the tolerance below is ~4 standard errors at this carrier
        count rather than an arbitrary round number.
        """
        n = 65_536
        protocol = get_protocol("masked_conjugate")
        enc = protocol.encode(random_message(n, rng), rng)
        host = MemoryParams.from_T2(T2)
        tolerance = 4 * 0.15 / np.sqrt(n)
        for x in (1e-3, 1e-2, 0.1, 0.5, 1.0):
            simulated = evaluate(enc, protocol, get_attack("none"), host=host,
                                 t_attack=0.0, t_read=x * T2).bob_ber
            assert simulated == pytest.approx(float(R.ber_at(x)), abs=tolerance)

    def test_window_inverts_the_ber(self):
        for ber in (1e-4, 1e-3, 1e-2, 0.1, 0.4):
            assert float(R.ber_at(R.usable_window_over_T2(ber))) == pytest.approx(ber, rel=1e-9)

    def test_window_is_linear_in_ber_when_small(self):
        assert R.usable_window_over_T2(1e-3) == pytest.approx(2e-3, rel=2e-3)

    def test_ber_at_half_is_unreachable(self):
        with pytest.raises(ValueError, match=r"strictly in \(0, 0.5\)"):
            R.usable_window_over_T2(0.5)

    def test_error_correction_buys_two_orders(self):
        """The headline number: tolerating 10% instead of 0.1% buys ~111x."""
        gain = R.error_correction_gain()
        assert gain["storage_time_gain"] == pytest.approx(111.5, rel=0.02)
        assert 0.5 < gain["code_rate_at_tolerant"] < 0.6
        assert gain["net_bit_seconds_gain"] > 50

    def test_required_T2_scales_inversely_with_tolerance(self):
        strict = R.required_T2(1e-3, 1e-3)
        tolerant = R.required_T2(1e-3, 0.10)
        assert strict / tolerant == pytest.approx(111.5, rel=0.02)

    def test_requirement_table_is_labelled(self):
        rows = R.requirement_table()
        assert rows
        for row in rows:
            assert "required_T2_seconds" in row
            assert any(k.startswith("orders_from_") for k in row)


class TestPostProcessing:
    def test_min_entropy_of_a_certain_string_is_zero(self):
        assert PP.min_entropy_from_guessing(np.ones(64)) == pytest.approx(0.0)

    def test_min_entropy_of_an_unknown_string_is_full(self):
        assert PP.min_entropy_from_guessing(np.full(64, 0.5)) == pytest.approx(64.0)

    def test_leakage_is_zero_on_a_noiseless_channel(self):
        assert PP.ec_leakage(1000, 0.0) == pytest.approx(0.0)

    def test_leakage_is_maximal_at_half(self):
        assert PP.ec_leakage(1000, 0.5, efficiency=1.0) == pytest.approx(1000.0)

    def test_privacy_amplification_never_returns_negative(self):
        assert PP.privacy_amplification_length(10.0, 1000.0) == 0.0

    def test_secret_shrinks_as_the_adversary_learns(self):
        budgets = [PP.key_budget(1024, 0.01, p) for p in (0.5, 0.6, 0.7, 0.9)]
        secrets = [b.secret_bits for b in budgets]
        assert all(b <= a for a, b in zip(secrets, secrets[1:]))

    def test_secret_shrinks_as_errors_grow(self):
        budgets = [PP.key_budget(1024, e, 0.55) for e in (0.0, 0.02, 0.05, 0.15)]
        secrets = [b.secret_bits for b in budgets]
        assert all(b <= a for a, b in zip(secrets, secrets[1:]))

    def test_bb84_threshold(self):
        assert PP.bb84_asymptotic_rate(0.10) > 0
        assert PP.bb84_asymptotic_rate(0.12) == 0.0
        assert PP.bb84_asymptotic_rate(PP.BB84_THRESHOLD_QBER) < 1e-3

    def test_break_even_accounts_for_the_fixed_overhead(self):
        """A short message can fail on the finite-size penalty alone."""
        n = PP.break_even_carriers(0.01, 0.55)
        assert n is not None
        assert PP.key_budget(n, 0.01, 0.55).secret_bits > 0
        assert PP.key_budget(max(1, n // 2), 0.01, 0.55).secret_bits == 0.0

    def test_no_positive_rate_when_the_adversary_knows_everything(self):
        assert PP.break_even_carriers(0.01, 1.0) is None


class TestCertifiedDeletion:
    def test_honest_deletion_verifies_exactly_at_zero_time(self, encoding, rng):
        cert = D.honest_deletion(encoding.initial_state(), rng)
        result = D.verify(encoding, cert, tolerance=0.0)
        assert result.accepted
        assert result.n_mismatches == 0

    def test_a_cheater_who_kept_the_message_is_caught(self, encoding, rng):
        result = D.verify(encoding, D.cheating_deletion(encoding, rng), tolerance=0.05)
        assert not result.accepted
        assert result.mismatch_fraction == pytest.approx(0.5, abs=0.05)

    def test_waiting_produces_no_valid_certificate(self, encoding, rng):
        """The central point: decoherence is not a deletion certificate."""
        cert = D.decay_certificate(encoding, 5 * T2, T2, T2, rng)
        assert not D.verify(encoding, cert, tolerance=0.05).accepted

    def test_partial_cheating_mismatch_follows_f_over_two(self, encoding, rng):
        for f in (0.1, 0.3, 0.6):
            cert = D.partial_cheating_deletion(encoding, encoding.initial_state(), f, rng)
            observed = D.verify(encoding, cert, tolerance=1.0).mismatch_fraction
            assert observed == pytest.approx(f / 2, abs=0.05)

    def test_theft_bound_is_zero_with_no_slack(self):
        assert D.max_undetected_theft(0.0, 0.0, T2) == pytest.approx(0.0)

    def test_slack_converts_directly_into_stolen_message(self):
        """Every bit of tolerance above the honest noise is stealable."""
        assert D.max_undetected_theft(0.10, 0.0, T2) == pytest.approx(0.2, rel=1e-9)

    def test_certification_window_accounts_for_partial_cheating(self):
        """Regression: the naive honest-vs-total-cheater window was ~50x too long."""
        window = D.certification_window(10_000, T2, max_theft_fraction=0.01)
        assert window["certifiable"]
        assert window["window_over_T2"] < 0.5, "partial cheating must shorten the window"
        theft = D.max_undetected_theft(
            window["recommended_tolerance"], window["window_over_T2"] * T2, T2
        )
        assert theft <= 0.01 + 1e-6

    def test_more_check_positions_extend_the_window(self):
        windows = [D.certification_window(n, T2)["window_over_T2"]
                   for n in (1_000, 10_000, 100_000)]
        assert all(b >= a for a, b in zip(windows, windows[1:]))

    def test_deletion_outlives_readout(self):
        """Certification tolerates far more decoherence than reading does."""
        readable = R.usable_window_over_T2(1e-3)
        certifiable = D.certification_window(10_000, T2)["window_over_T2"]
        assert certifiable > 10 * readable

    def test_encoding_without_check_positions_is_rejected(self, rng):
        from bioquantumfly.crypto.encoding import Encoding

        enc = Encoding("masked_conjugate", np.zeros(8, dtype=np.int64),
                       np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64),
                       pad=np.zeros(8, dtype=np.int64))
        with pytest.raises(ValueError, match="no Hadamard-basis positions"):
            D.verify(enc, D.honest_deletion(enc.initial_state(), rng))


class TestNoisyStorage:
    def test_delay_erases_a_finite_memory(self):
        rows = NS.delay_sweep(1024, 0.02, 0.85, MemoryParams.from_T2(T2),
                              delays_over_eve_T2=[0.0, 5.0, 20.0])
        info = [r["eve_information_bits"] for r in rows]
        assert info[0] > 100
        assert info[-1] < 1e-3
        assert all(b <= a for a, b in zip(info, info[1:]))

    def test_delay_does_nothing_against_an_ideal_memory(self):
        rows = NS.delay_sweep(1024, 0.02, 0.85, MemoryParams.ideal())
        assert len(rows) == 1
        assert "buys nothing" in rows[0]["note"]

    def test_required_delay_is_infinite_for_an_ideal_adversary(self):
        assert not np.isfinite(NS.required_delay(MemoryParams.ideal(), 0.85))

    def test_required_delay_scales_with_the_adversary_memory(self):
        a = NS.required_delay(MemoryParams.from_T2(T2), 0.85)
        b = NS.required_delay(MemoryParams.from_T2(10 * T2), 0.85)
        assert b == pytest.approx(10 * a, rel=1e-9)

    def test_no_delay_needed_when_the_adversary_learned_nothing(self):
        assert NS.required_delay(MemoryParams.from_T2(T2), 0.5) == 0.0

    def test_tolerable_advantage_grows_only_linearly_with_waiting(self):
        """Patience is cheap but weak: the assumption still has to be believed."""
        a = NS.storage_advantage_threshold(0.85, 10 * T2, T2)
        b = NS.storage_advantage_threshold(0.85, 20 * T2, T2)
        assert b == pytest.approx(2 * a, rel=1e-6)


class TestComposedStack:
    @pytest.fixture
    def weak_attack(self):
        return get_attack("weak_measurement", eta=0.25)

    def test_privacy_amplification_is_what_makes_it_work(self, encoding, weak_attack, rng):
        results = {
            name: run_stack(encoding, weak_attack, host=MemoryParams.from_T2(T2),
                            eve_memory=MemoryParams.from_T2(T2), t_read=0.02 * T2,
                            t_attack=0.005 * T2, config=cfg, rng=rng)
            for name, cfg in ABLATIONS.items()
        }
        assert not results["bare (protocol B)"].usable
        assert results["+ privacy amplification"].usable
        assert results["+ privacy amplification"].secret_bits > 1000

    def test_delay_adds_secret_only_against_a_finite_memory(self, encoding, weak_attack, rng):
        kwargs = dict(host=MemoryParams.from_T2(T2), t_read=0.02 * T2,
                      t_attack=0.005 * T2, rng=rng)
        finite = run_stack(encoding, weak_attack, eve_memory=MemoryParams.from_T2(T2), **kwargs)
        ideal = run_stack(encoding, weak_attack, eve_memory=MemoryParams.ideal(), **kwargs)
        assert finite.secret_bits > finite.secret_bits_without_delay
        assert ideal.secret_bits == pytest.approx(ideal.secret_bits_without_delay)

    def test_state_theft_is_detected_and_aborts(self, encoding, rng):
        """Protocol C repairs Protocol B's fatal flaw -- by detection, not physics."""
        result = run_stack(encoding, get_attack("swap_theft"),
                           host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.ideal(),
                           t_read=0.02 * T2, t_attack=0.005 * T2, rng=rng)
        assert not result.usable
        assert "abort" in result.verdict
        assert result.bob_ber > PP.BB84_THRESHOLD_QBER

    def test_intercept_resend_is_detected(self, encoding, rng):
        result = run_stack(encoding, get_attack("breidbart"),
                           host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.from_T2(T2),
                           t_read=0.02 * T2, t_attack=0.005 * T2, rng=rng)
        assert not result.usable
        assert "abort" in result.verdict

    def test_deletion_layer_distinguishes_all_three_certificates(self, encoding, rng):
        result = run_stack(encoding, get_attack("weak_measurement", eta=0.1),
                           host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.from_T2(T2),
                           t_read=0.005 * T2, t_attack=0.001 * T2,
                           config=StackConfig(), rng=rng)
        assert result.honest_certificate_accepted is True
        assert result.cheating_certificate_accepted is False
        assert result.decay_certificate_accepted is False

    def test_no_attack_gives_the_largest_secret(self, encoding, rng):
        kwargs = dict(host=MemoryParams.from_T2(T2), eve_memory=MemoryParams.from_T2(T2),
                      t_read=0.02 * T2, t_attack=0.005 * T2, rng=rng)
        clean = run_stack(encoding, get_attack("none"), **kwargs)
        attacked = run_stack(encoding, get_attack("weak_measurement", eta=0.4), **kwargs)
        assert clean.secret_bits > attacked.secret_bits
