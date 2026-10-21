"""Every experiment must run end-to-end without network access or a token."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioquantumfly.config import Config
from bioquantumfly.experiments.registry import EXPERIMENTS, get_experiment, make_runner


@pytest.fixture(scope="module")
def tiny_config() -> Config:
    cfg = Config()
    cfg.quantum.T2_seconds = [1e-6]
    cfg.quantum.n_carriers = 8
    cfg.connectome.synthetic_n_neurons = 80
    cfg.host_selection.n_hosts = 4
    cfg.neural.duration_s = 0.05
    cfg.coupling.gain_sweep = [0.0, 1.0]
    return cfg


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_experiment_runs_and_emits_rows(name, tiny_config, monkeypatch):
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)  # must work offline
    definition = get_experiment(name)
    specs = definition.build_specs(tiny_config)
    assert specs, f"{name} produced no run specs"

    rows = make_runner(name, tiny_config)(specs[0])
    assert rows, f"{name} produced no rows"
    df = pd.DataFrame(rows)
    assert not df.empty
    # T1/T2 columns legitimately hold `inf`: that is how an ideal (non-decohering)
    # memory is represented, and it is the default adversary model. Every other
    # numeric output must be finite.
    coherence_columns = {"T1_host", "T2_host", "T1_eve", "T2_eve"}
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in coherence_columns if c in df.columns]
    )
    assert np.isfinite(numeric.to_numpy()).all(), (
        f"{name} emitted non-finite values in "
        f"{[c for c in numeric.columns if not np.isfinite(numeric[c]).all()]}"
    )
    for column in coherence_columns & set(df.columns):
        assert not df[column].isna().any(), f"{name}.{column} contains NaN"


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_run_ids_are_unique(name, tiny_config):
    specs = get_experiment(name).build_specs(tiny_config)
    assert len({s.run_id for s in specs}) == len(specs)


def test_unknown_experiment_is_rejected():
    with pytest.raises(ValueError, match="unknown experiment"):
        get_experiment("levitation")


class TestExperimentClaims:
    """The specific scientific claims each experiment is supposed to demonstrate."""

    def test_experiment_1_decay_reaches_a_coin_flip(self, tiny_config):
        from bioquantumfly.experiments import quantum_only

        # The BER -> 1/2 claim is about the ensemble average, so it needs enough
        # carriers to be a meaningful statistic: with n = 8 the asymptote has a
        # standard deviation of order 0.1 and a single draw says little.
        config = tiny_config.model_copy(deep=True)
        config.quantum.n_carriers = 256
        spec = quantum_only.build_specs(config)[0]
        df = pd.DataFrame(quantum_only.run(spec, t_over_T2_grid=[0.0, 1.0, 50.0]))
        assert df.iloc[0]["bob_ber"] == pytest.approx(0.0, abs=1e-12)
        assert df.iloc[-1]["bob_ber"] == pytest.approx(0.5, abs=0.05)

    def test_experiment_2_raw_bb84_leaks_without_a_key(self, tiny_config):
        from bioquantumfly.experiments import eve_sweep
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("eve_attack_sweep",
                       {"T2": 1e-6, "protocol": "raw_bb84", "attack": "breidbart",
                        "n_carriers": 32, "eve_memory": "ideal"}, 42)
        df = pd.DataFrame(eve_sweep.run_attack(spec, t_over_T2_grid=[0.0]))
        assert df["eve_pguess_before_key"].iloc[0] > 0.8

    def test_experiment_3_masked_leaks_nothing_without_a_key(self, tiny_config):
        from bioquantumfly.experiments import eve_sweep
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("eve_attack_sweep",
                       {"T2": 1e-6, "protocol": "masked_conjugate", "attack": "breidbart",
                        "n_carriers": 32, "eve_memory": "ideal"}, 42)
        df = pd.DataFrame(eve_sweep.run_attack(spec, t_over_T2_grid=[0.0]))
        assert df["eve_pguess_before_key"].iloc[0] == pytest.approx(0.5, abs=1e-9)

    def test_experiment_4_tradeoff_is_monotonic(self, tiny_config):
        from bioquantumfly.experiments import eve_sweep
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("information_disturbance",
                       {"T2": 1e-6, "attack": "weak_measurement",
                        "protocol": "masked_conjugate", "n_carriers": 32}, 42)
        df = pd.DataFrame(eve_sweep.run_disturbance(spec, strengths=[0.0, 0.3, 0.6, 1.0]))
        assert df["eve_info_after_key"].is_monotonic_increasing
        assert df["bob_disturbance_ber"].is_monotonic_increasing

    def test_experiment_6_eve_outlasts_the_host(self, tiny_config):
        from bioquantumfly.experiments import eve_sweep
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("eve_ancilla_persistence",
                       {"T2": 1e-6, "attack": "coherent_ancilla", "eve_memory": "ideal",
                        "protocol": "masked_conjugate", "n_carriers": 32}, 42)
        df = pd.DataFrame(eve_sweep.run_persistence(spec, t_over_T2_grid=[0.1, 50.0]))
        late = df.iloc[-1]
        assert late["host_info_after_key"] < 1e-6
        assert late["eve_info_after_key"] > 0.0
        assert late["information_gap"] > 0.0

    def test_experiment_7_swap_is_the_worst_case(self, tiny_config):
        from bioquantumfly.experiments import eve_sweep
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("swap_negative_control",
                       {"T2": 1e-6, "attack": "swap_theft", "protocol": "masked_conjugate",
                        "n_carriers": 32}, 42)
        df = pd.DataFrame(eve_sweep.run_swap(spec, t_over_T2_grid=[100.0]))
        row = df.iloc[0]
        assert row["regime"] == "unrestricted_extraction"
        assert row["host_info_after_key"] < 1e-9
        # Eve swaps at t_attack = 0.1 T2, by which point the host has already
        # decohered a little, so her copy is good but not perfect.
        assert row["eve_info_after_key"] > 0.5

    def test_experiment_8_network_propagates(self, tiny_config):
        from bioquantumfly.experiments import neural_baseline

        spec = neural_baseline.build_specs(tiny_config)[0]
        row = neural_baseline.run(spec, tiny_config)[0]
        assert row["total_spikes"] > 0
        assert row["coupling_enabled"] is False

    def test_experiment_9_zero_gain_means_zero_perturbation(self, tiny_config):
        from bioquantumfly.experiments import hybrid
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("hybrid_tamper_propagation",
                       {"T2": 1e-6, "attack_condition": "strong",
                        "speculative_coupling_gain": 0.0, "n_hosts": 4,
                        "host_strategy": "random"}, 42)
        row = hybrid.run(spec, tiny_config)[0]
        assert row["coupling_enabled"] is False
        assert row["n_perturbed_neurons"] == 0
        assert row["max_drive_shift_volt"] == 0.0


class TestFeasibilityClassification:
    def test_regimes(self):
        from bioquantumfly.experiments.feasibility import (
            REGIME_READABLE_SECURE, REGIME_READABLE_VULNERABLE,
            REGIME_SELF_ERASED, REGIME_UNREADABLE, classify,
        )

        kw = dict(max_bob_ber=1e-3, max_eve_info=0.01)
        assert classify(1e-6, 1e-6, 0.9, **kw) == REGIME_READABLE_SECURE
        assert classify(1e-6, 0.5, 0.9, **kw) == REGIME_READABLE_VULNERABLE
        assert classify(0.4, 1e-6, 1e-9, **kw) == REGIME_SELF_ERASED
        assert classify(0.4, 0.5, 0.9, **kw) == REGIME_UNREADABLE

    def test_refinement_targets_the_boundary(self):
        from bioquantumfly.experiments.feasibility import refine_around_boundary

        df = pd.DataFrame({
            "T2": [1e-6] * 4, "eta": [0.0] * 4,
            "t_over_T2": [0.1, 0.2, 0.3, 0.4],
            "regime": ["readable_low_eve_info"] * 2 + ["unreadable_even_for_bob"] * 2,
        })
        points = refine_around_boundary(df)
        assert points and all(0.2 <= p <= 0.3 for p in points)

    def test_refinement_on_empty_input(self):
        from bioquantumfly.experiments.feasibility import refine_around_boundary

        assert refine_around_boundary(pd.DataFrame()) == []


class TestSubstrateInvariance:
    """The substrate must not change when a swept parameter changes.

    Regression guard for a real bug: because `RunSpec.run_id` hashes *all*
    parameters, seeding the connectome from `spec.rng` regenerated a different
    random graph for every coupling gain. The kappa-dependence then measured
    the difference between unrelated networks rather than the effect of the
    coupling, and produced a non-monotonic dose-response. The substrate is now
    seeded from `shared_rng` on size-related keys only.
    """

    def _graph_for(self, gain, condition, tiny_config):
        from bioquantumfly.experiments.common import GRAPH_KEYS, HOST_KEYS
        from bioquantumfly.experiments.connectome_loader import load_connectome
        from bioquantumfly.connectome.subgraph import select_hosts
        from bioquantumfly.runtime.runspec import RunSpec

        spec = RunSpec("hybrid_tamper_propagation",
                       {"T2": 1e-6, "attack_condition": condition,
                        "speculative_coupling_gain": gain, "n_hosts": 4,
                        "host_strategy": "random"}, 42)
        graph = load_connectome(tiny_config, spec.shared_rng("connectome", GRAPH_KEYS))
        hosts = select_hosts(graph, mode=tiny_config.host_selection.mode,
                             strategy="random", n_hosts=4,
                             rng=spec.shared_rng("hosts", HOST_KEYS))
        return graph, hosts

    def test_graph_is_identical_across_coupling_gain(self, tiny_config):
        a, _ = self._graph_for(0.0, "strong", tiny_config)
        b, _ = self._graph_for(10.0, "strong", tiny_config)
        assert (a.weights != b.weights).nnz == 0
        assert np.array_equal(a.body_ids, b.body_ids)

    def test_graph_is_identical_across_attack_condition(self, tiny_config):
        a, _ = self._graph_for(1.0, "none", tiny_config)
        b, _ = self._graph_for(1.0, "strong", tiny_config)
        assert (a.weights != b.weights).nnz == 0

    def test_hosts_are_identical_across_coupling_gain(self, tiny_config):
        _, a = self._graph_for(0.001, "strong", tiny_config)
        _, b = self._graph_for(10.0, "weak", tiny_config)
        assert np.array_equal(a.indices, b.indices)

    def test_baselines_are_comparable_across_gain(self, tiny_config):
        """The unattacked network must be the same network at every gain."""
        from bioquantumfly.experiments import hybrid
        from bioquantumfly.runtime.runspec import RunSpec

        rows = []
        for gain in (0.0, 1.0):
            spec = RunSpec("hybrid_tamper_propagation",
                           {"T2": 1e-6, "attack_condition": "none",
                            "speculative_coupling_gain": gain, "n_hosts": 4,
                            "host_strategy": "random", "n_repeats": 1}, 42)
            rows.append(hybrid.run(spec, tiny_config)[0])
        assert rows[0]["n_neurons"] == rows[1]["n_neurons"]
        assert rows[0]["n_edges"] == rows[1]["n_edges"]
        # With no attack there is no readout shift, so no perturbation at any gain.
        assert all(r["n_perturbed_neurons"] == 0 for r in rows)

    def test_message_is_identical_across_T2(self):
        """Same guarantee on the cryptographic side (Experiment 1 comparability)."""
        from bioquantumfly.experiments.common import encode_for
        from bioquantumfly.runtime.runspec import RunSpec

        specs = [RunSpec("quantum_memory_decay",
                         {"T2": t2, "n_carriers": 16, "protocol": "masked_conjugate"}, 42)
                 for t2 in (4e-9, 1e-3)]
        encodings = [encode_for(s, "masked_conjugate", 16)[1] for s in specs]
        assert np.array_equal(encodings[0].message, encodings[1].message)
        assert np.array_equal(encodings[0].bases, encodings[1].bases)
