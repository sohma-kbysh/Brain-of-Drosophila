"""Scientific labelling and security hygiene.

These tests protect the two properties that make this repository defensible: no
number is presented without its evidence class, and no credential is ever
printed, stored or committed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bioquantumfly.params import (
    DEFAULT_T2_SWEEP_S, REGIMES, Parameter, Status, classify_T2, regime_of_T2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestParameterProvenance:
    def test_measured_parameters_must_cite_a_source(self):
        with pytest.raises(ValueError, match="no provenance"):
            Parameter(1.0, "x", Status.MEASURED)

    def test_speculative_parameters_must_not_cite_a_source(self):
        with pytest.raises(ValueError, match="should not be labelled speculative"):
            Parameter(1.0, "x", Status.SPECULATIVE, provenance="somewhere")

    def test_every_shipped_regime_parameter_is_labelled(self):
        for regime in REGIMES.values():
            for name, param in regime.parameters.items():
                assert param.status is not None, name
                if param.status is not Status.SPECULATIVE:
                    assert param.provenance, f"{regime.name}.{name} lacks provenance"

    def test_future_regime_is_marked_speculative(self):
        assert REGIMES["future"].contains_speculative

    def test_cryogenic_regime_cites_a_real_measurement(self):
        t2 = REGIMES["cryogenic_protein"]["T2_CPMG"]
        assert t2.status is Status.MEASURED
        assert float(t2) == pytest.approx(16e-6)
        assert "Feder" in t2.provenance

    @pytest.mark.parametrize("t2,expected", [
        (4e-9, "proxy"), (100e-9, "proxy"),
        (1e-6, "measured"), (16e-6, "measured"),
        (100e-6, "speculative"), (1e-3, "speculative"),
    ])
    def test_classification_of_the_default_sweep(self, t2, expected):
        assert classify_T2(t2).value == expected

    def test_every_default_sweep_point_classifies(self):
        for t2 in DEFAULT_T2_SWEEP_S:
            assert classify_T2(t2) in Status
            assert regime_of_T2(t2) in REGIMES

    def test_label_mentions_the_evidence_class(self):
        assert "SPECULATIVE" in Parameter(1e-3, "T2", Status.SPECULATIVE).label


class TestEYFPHonesty:
    def test_spectroscopic_constants_are_unknown_not_invented(self):
        from bioquantumfly.quantum.eyfp_proxy import SPECTROSCOPIC_CONSTANTS

        for name, entry in SPECTROSCOPIC_CONSTANTS.items():
            assert entry["value"] is None, f"{name} must not be guessed"
            assert entry["status"] == "unknown"

    def test_spin_one_hamiltonian_refuses_rather_than_guesses(self):
        from bioquantumfly.quantum.eyfp_proxy import spin_one_hamiltonian

        with pytest.raises(NotImplementedError, match="rather than guessed"):
            spin_one_hamiltonian()

    def test_envelope_is_not_claimed_to_be_a_fly_measurement(self):
        from bioquantumfly.quantum.eyfp_proxy import EYFPEnvelope

        for regime in ("cryogenic", "room_temperature"):
            envelope = EYFPEnvelope(regime)
            assert envelope.to_dict()["is_drosophila_measurement"] is False
            assert envelope.memory_params().status is Status.PROXY

    def test_the_two_regimes_are_not_interchangeable(self):
        from bioquantumfly.quantum.eyfp_proxy import EYFPEnvelope

        assert EYFPEnvelope("cryogenic").T2 > 100 * EYFPEnvelope("room_temperature").T2


class TestRadicalPairHonesty:
    def test_unknown_couplings_default_to_omitted(self):
        from bioquantumfly.quantum.radical_pair import RadicalPairParameters

        p = RadicalPairParameters()
        assert p.exchange_rad_s is None
        assert p.dipolar_rad_s is None
        prov = p.provenance()
        assert prov["exchange"]["omitted"] is True
        assert prov["dipolar"]["omitted"] is True

    def test_proxy_parameters_are_flagged(self):
        from bioquantumfly.quantum.radical_pair import RadicalPairParameters

        prov = RadicalPairParameters().provenance()
        assert prov["hyperfine"]["status"] == Status.PROXY.value
        assert "NOT a measured constant" in prov["hyperfine"]["note"]

    def test_it_is_not_an_arbitrary_write_memory(self):
        from bioquantumfly.quantum.radical_pair import RadicalPairMechanism

        prov = RadicalPairMechanism().provenance()
        assert prov["is_arbitrary_write_memory"] is False

    def test_hilbert_space_is_kept_small(self):
        from bioquantumfly.quantum.radical_pair import RadicalPairParameters

        with pytest.raises(ValueError, match="keep the Hilbert space small"):
            RadicalPairParameters(n_nuclei=8)


class TestCouplingIsSpeculativeAndOff:
    def test_disabled_by_default(self):
        from bioquantumfly.config import Config
        from bioquantumfly.coupling.quantum_neural import CouplingConfig, HybridMode

        assert Config().coupling.enabled is False
        default = CouplingConfig()
        assert default.enabled is False
        assert default.mode is HybridMode.STORAGE_ONLY
        assert default.is_active is False

    def test_tamper_alarm_cannot_be_switched_on_by_accident(self):
        from bioquantumfly.coupling.quantum_neural import CouplingConfig, HybridMode

        with pytest.raises(ValueError, match="requires enabled=True"):
            CouplingConfig(enabled=False, mode=HybridMode.TAMPER_ALARM)

    def test_storage_only_produces_exactly_zero_drive(self):
        from bioquantumfly.coupling.quantum_neural import CouplingConfig, quantum_drive_vector

        config = CouplingConfig(enabled=True, speculative_coupling_gain=10.0)
        drive = quantum_drive_vector(50, [1, 2, 3], np.eye(3), config)
        assert np.all(drive == 0.0)

    def test_status_is_recorded_in_metadata(self):
        from bioquantumfly.coupling.quantum_neural import CouplingConfig

        d = CouplingConfig().to_dict()
        assert d["status"] == "speculative"
        assert "empirical calibration" in d["warning"].lower()


class TestTokenSafety:
    """Spec section 17: never print, store or commit a token."""

    def test_presence_check_does_not_return_the_value(self, monkeypatch):
        from bioquantumfly.connectome import male_cns

        monkeypatch.setenv("NEUPRINT_TOKEN", "secret-value-do-not-leak")
        assert male_cns.token_is_present() is True
        assert isinstance(male_cns.token_is_present(), bool)

    def test_absent_token_reports_false(self, monkeypatch):
        from bioquantumfly.connectome import male_cns

        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        assert male_cns.token_is_present() is False

    def test_error_message_never_contains_the_token(self, monkeypatch):
        from bioquantumfly.connectome import male_cns

        monkeypatch.setenv("NEUPRINT_TOKEN", "")
        with pytest.raises(male_cns.TokenMissingError) as exc:
            male_cns._require_token()
        assert "NEUPRINT_TOKEN" in str(exc.value)
        assert "<your token>" in str(exc.value)

    def test_config_serialisation_has_no_credential_field(self):
        from bioquantumfly.connectome.male_cns import NeuPrintConfig

        d = NeuPrintConfig().to_dict()
        assert not any("token" in k.lower() for k in d)
        assert not any("secret" in str(v).lower() for v in d.values())

    def test_no_jwt_literal_in_the_tracked_source_tree(self):
        """Guard against a credential being committed."""
        jwt_prefix = "ey" + "J"
        this_file = Path(__file__).resolve()
        offenders = []
        for pattern in ("src/**/*.py", "scripts/**/*.py", "tests/**/*.py",
                        "configs/**/*.yaml", "docs/**/*.md"):
            for path in REPO_ROOT.glob(pattern):
                if path.resolve() == this_file:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if jwt_prefix in text:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"possible committed token in {offenders}"

    def test_gitignore_covers_credentials(self):
        text = (REPO_ROOT / ".gitignore").read_text()
        for pattern in (".env", "*.token", "data/raw/", "results/"):
            assert pattern in text, f"{pattern} is not git-ignored"
