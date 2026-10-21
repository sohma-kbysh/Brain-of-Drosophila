"""LIF network behaviour (spec section 25)."""

from __future__ import annotations

import numpy as np
import pytest

from bioquantumfly.neural.brian_network import compare_runs, run_network
from bioquantumfly.neural.lif import SHIU_PARAMETERS, LIFParameters


class TestLIFParameters:
    def test_shiu_defaults(self):
        p = LIFParameters()
        assert p.V_resting == pytest.approx(-52e-3)
        assert p.V_threshold == pytest.approx(-45e-3)
        assert p.T_mbr == pytest.approx(20e-3)
        assert p.spike_threshold_gap == pytest.approx(7e-3)

    def test_membrane_time_constant_is_R_times_C(self):
        """T_mbr = 10 kohm cm^2 * 2 uF/cm^2 = 20 ms."""
        r = float(SHIU_PARAMETERS["R_mbr"]) * 1e3      # kohm -> ohm
        c = float(SHIU_PARAMETERS["C_mbr"]) * 1e-6     # uF -> F
        assert r * c == pytest.approx(float(SHIU_PARAMETERS["T_mbr"]), rel=1e-9)

    def test_every_parameter_declares_provenance(self):
        for name, param in SHIU_PARAMETERS.items():
            assert param.status is not None, name
            if param.status.value in ("measured", "proxy", "transferred"):
                assert param.provenance, name

    def test_transfer_caveat_is_recorded(self):
        caveat = LIFParameters().provenance()["caveat"]
        assert "does not constitute experimental validation" in caveat

    def test_psp_peak_factor_is_below_one(self):
        """Synaptic decay outruns membrane integration, so the peak is attenuated."""
        p = LIFParameters()
        assert 0.0 < p.psp_peak_factor < 1.0
        assert p.psp_peak_factor == pytest.approx(0.1575, abs=1e-3)

    def test_min_synapses_accounts_for_filtering(self):
        p = LIFParameters()
        naive = p.spike_threshold_gap / p.W_syn
        assert p.min_synapses_to_fire > naive
        assert p.min_synapses_to_fire == pytest.approx(161.6, rel=0.01)

    def test_equal_time_constants_degenerate_case(self):
        p = LIFParameters(T_mbr=5e-3, tau_synapse=5e-3)
        assert p.psp_peak_factor == pytest.approx(1.0 / np.e, rel=1e-9)

    def test_threshold_below_rest_is_rejected(self):
        with pytest.raises(ValueError, match="must exceed rest"):
            LIFParameters(V_threshold=-60e-3)


class TestSynapticSign:
    """Spec section 25: an excitatory spike raises g, an inhibitory one lowers it."""

    def _one_synapse(self, brian_runtime, sign):
        b2 = brian_runtime
        p = LIFParameters()
        b2.start_scope()
        source = b2.SpikeGeneratorGroup(1, [0], [1] * b2.ms)
        target = b2.NeuronGroup(
            1, """dv/dt = (g - (v - V_rest))/tau_mbr : volt
                  dg/dt = -g/tau_syn                 : volt""",
            method="exact",
            namespace={"V_rest": p.V_resting * b2.volt,
                       "tau_mbr": p.T_mbr * b2.second,
                       "tau_syn": p.tau_synapse * b2.second},
        )
        target.v = p.V_resting * b2.volt
        target.g = 0 * b2.volt
        syn = b2.Synapses(source, target, model="w : volt", on_pre="g_post += w")
        syn.connect(i=0, j=0)
        syn.w = sign * 100 * p.W_syn * b2.volt
        mon = b2.StateMonitor(target, ["g", "v"], record=0)
        b2.Network(source, target, syn, mon).run(10 * b2.ms)
        return np.asarray(mon.g[0] / b2.volt), np.asarray(mon.v[0] / b2.volt), p

    def test_excitatory_spike_increases_conductance(self, brian_runtime):
        g, v, p = self._one_synapse(brian_runtime, +1)
        assert g.max() > 0
        assert v.max() > p.V_resting

    def test_inhibitory_spike_decreases_conductance(self, brian_runtime):
        g, v, p = self._one_synapse(brian_runtime, -1)
        assert g.min() < 0
        assert v.min() < p.V_resting


class TestNetworkSimulation:
    def test_unstimulated_network_is_silent(self, small_graph):
        result = run_network(small_graph, 0.1, seed=1)
        assert result.total_spikes == 0, "baseline_rate is 0 Hz, so nothing should fire"

    def test_stimulated_network_fires_and_propagates(self, small_graph):
        stim = np.arange(6)
        result = run_network(small_graph, 0.2, host_indices=stim,
                             stimulus_indices=stim, stimulus_rate_hz=120.0, seed=1)
        assert result.total_spikes > 0
        assert np.all(np.isfinite(result.firing_rates()))
        assert result.mean_rate_hz > 0

    def test_results_are_deterministic_given_a_seed(self, small_graph):
        stim = np.arange(6)
        kwargs = dict(duration_s=0.15, host_indices=stim, stimulus_indices=stim,
                      stimulus_rate_hz=120.0, seed=7)
        a = run_network(small_graph, **kwargs)
        b = run_network(small_graph, **kwargs)
        assert np.array_equal(a.spike_counts, b.spike_counts)

    def test_quantum_drive_changes_activity(self, small_graph):
        stim = np.arange(6)
        kwargs = dict(duration_s=0.2, host_indices=stim, stimulus_indices=stim,
                      stimulus_rate_hz=120.0, seed=3)
        baseline = run_network(small_graph, **kwargs)
        drive = np.zeros(small_graph.n_neurons)
        drive[stim] = 2e-3
        perturbed = run_network(small_graph, quantum_drive=drive, **kwargs)
        comparison = compare_runs(baseline, perturbed)
        assert comparison["n_perturbed_neurons"] > 0

    def test_zero_drive_is_identical_to_no_drive(self, small_graph):
        """`storage_only` mode must be numerically identical to no coupling."""
        stim = np.arange(6)
        kwargs = dict(duration_s=0.15, host_indices=stim, stimulus_indices=stim,
                      stimulus_rate_hz=120.0, seed=5)
        a = run_network(small_graph, quantum_drive=None, **kwargs)
        b = run_network(small_graph, quantum_drive=np.zeros(small_graph.n_neurons), **kwargs)
        assert np.array_equal(a.spike_counts, b.spike_counts)

    def test_comparing_different_sized_networks_is_rejected(self, small_graph, rng):
        from bioquantumfly.connectome.synthetic import synthetic_connectome

        a = run_network(small_graph, 0.05, seed=1)
        b = run_network(synthetic_connectome(60, rng=rng), 0.05, seed=1)
        with pytest.raises(ValueError, match="different sizes"):
            compare_runs(a, b)
