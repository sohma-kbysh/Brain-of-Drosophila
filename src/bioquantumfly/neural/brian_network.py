"""Brian2 network construction and execution (spec section 15, addendum 18-20).

Monitoring policy. Recording every spike and every membrane potential of
166,691 neurons produces enormous output and answers none of the questions this
project asks. The default therefore records only host neurons, a selected
downstream set, and aggregate statistics. Full spike recording is opt-in and
emits a storage warning; full voltage recording of the whole CNS is refused
outright unless explicitly forced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from bioquantumfly.connectome.graph import ConnectomeGraph
from bioquantumfly.neural.lif import LIFParameters

#: Above this neuron count, recording all membrane potentials is refused.
_VOLTAGE_RECORD_LIMIT = 5_000
#: Above this, recording all spikes warns about output size.
_SPIKE_RECORD_WARN = 20_000
#: Above this, spike *times* are not collected unless explicitly forced.
_SPIKE_TIME_LIMIT = 50_000


@dataclass
class NetworkResult:
    """Aggregate outcome of one simulation.

    Deliberately compact: counts and rates rather than raw spike trains, unless
    full recording was explicitly requested.
    """

    n_neurons: int
    duration_s: float
    spike_counts: np.ndarray                  # (N,) per-neuron spike counts
    host_indices: np.ndarray
    recorded_indices: np.ndarray
    spike_times: dict[str, np.ndarray] = field(default_factory=dict)
    voltage_trace: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_spikes(self) -> int:
        return int(self.spike_counts.sum())

    @property
    def mean_rate_hz(self) -> float:
        if self.n_neurons == 0 or self.duration_s <= 0:
            return 0.0
        return float(self.spike_counts.sum() / (self.n_neurons * self.duration_s))

    def firing_rates(self) -> np.ndarray:
        if self.duration_s <= 0:
            return np.zeros_like(self.spike_counts, dtype=float)
        return self.spike_counts / self.duration_s

    def summary(self) -> dict[str, Any]:
        rates = self.firing_rates()
        return {
            "n_neurons": self.n_neurons,
            "duration_s": self.duration_s,
            "total_spikes": self.total_spikes,
            "mean_rate_hz": self.mean_rate_hz,
            "n_active_neurons": int((self.spike_counts > 0).sum()),
            "max_rate_hz": float(rates.max()) if rates.size else 0.0,
            "host_spikes": int(self.spike_counts[self.host_indices].sum())
            if self.host_indices.size else 0,
            **self.metadata,
        }


def _brian():
    import brian2 as b2

    return b2


def configure_backend(mode: str = "runtime", build_dir: str | None = None) -> str:
    """Select the Brian2 code-generation backend (addendum section 19).

    ``runtime`` (numpy/cython) for development, tests and small subgraphs;
    ``cpp_standalone`` for large or repeated long simulations. Standalone mode
    is never forced: it compiles per model, and on this project's NFS home
    directory repeated compilation is slow.
    """
    b2 = _brian()
    if mode == "runtime":
        b2.set_device("runtime")
        b2.prefs.codegen.target = "numpy"
    elif mode == "cython":
        b2.set_device("runtime")
        b2.prefs.codegen.target = "cython"
    elif mode == "cpp_standalone":
        kwargs = {"build_on_run": True}
        if build_dir:
            kwargs["directory"] = build_dir
        b2.set_device("cpp_standalone", **kwargs)
    else:
        raise ValueError(f"unknown Brian2 backend {mode!r}")
    return mode


def build_network(
    graph: ConnectomeGraph,
    params: LIFParameters | None = None,
    *,
    host_indices: Sequence[int] | None = None,
    quantum_drive: np.ndarray | None = None,
    stimulus_indices: Sequence[int] | None = None,
    stimulus_rate_hz: float = 100.0,
    initial_voltage_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Assemble the Brian2 objects for one connectome.

    Parameters
    ----------
    quantum_drive
        Optional (N,) array of additional voltage-like drive per neuron, the
        speculative quantum-to-neural coupling term. Zero everywhere means the
        coupling is off, which is the default in every cryptographic experiment.
    stimulus_indices
        Neurons receiving Poisson input, used to make the baseline network
        actually propagate something (Experiment 8).
    """
    b2 = _brian()
    params = params or LIFParameters()
    rng = rng or np.random.default_rng(0)
    n = graph.n_neurons

    eqs = """
    dv/dt = (g + q_drive - (v - V_rest)) / tau_mbr : volt (unless refractory)
    dg/dt = -g / tau_syn                           : volt
    q_drive                                        : volt
    """
    namespace = {
        "V_rest": params.V_resting * b2.volt,
        "tau_mbr": params.T_mbr * b2.second,
        "tau_syn": params.tau_synapse * b2.second,
    }
    neurons = b2.NeuronGroup(
        n, eqs,
        threshold=f"v > {params.V_threshold} * volt",
        reset=f"v = {params.V_reset} * volt",
        refractory=params.T_refractory * b2.second,
        method="exact",
        namespace=namespace,
        name="connectome_neurons",
    )
    neurons.v = params.V_resting * b2.volt
    if initial_voltage_jitter > 0:
        neurons.v = (
            params.V_resting + rng.normal(0, initial_voltage_jitter, size=n)
        ) * b2.volt
    neurons.g = 0 * b2.volt
    neurons.q_drive = (
        np.zeros(n) if quantum_drive is None else np.asarray(quantum_drive, float)
    ) * b2.volt

    # Signed synaptic weights in volts: N_syn * s_j * W_syn.
    signed = graph.signed_weights(params.W_syn).tocoo()
    synapses = b2.Synapses(
        neurons, neurons, model="w : volt", on_pre="g_post += w",
        delay=params.T_delay * b2.second, name="connectome_synapses",
    )
    if signed.nnz:
        synapses.connect(i=signed.row.astype(np.int32), j=signed.col.astype(np.int32))
        synapses.w = signed.data * b2.volt

    objects: list[Any] = [neurons, synapses]

    if stimulus_indices is not None and len(stimulus_indices) > 0:
        idx = np.asarray(stimulus_indices, dtype=np.int32)
        poisson = b2.PoissonGroup(len(idx), rates=stimulus_rate_hz * b2.Hz, name="stimulus")
        # Size the stimulus from the actual PSP peak, not from the threshold
        # gap alone: synaptic decay (5 ms) outruns membrane integration (20 ms),
        # so only ~16% of an injected impulse reaches the membrane.
        drive_w = 1.2 * params.drive_to_fire
        stim_syn = b2.Synapses(
            poisson, neurons, model="w : volt", on_pre="g_post += w", name="stimulus_synapses"
        )
        stim_syn.connect(i=np.arange(len(idx)), j=idx)
        stim_syn.w = drive_w * b2.volt
        objects += [poisson, stim_syn]

    return neurons, synapses, objects


def run_network(
    graph: ConnectomeGraph,
    duration_s: float,
    params: LIFParameters | None = None,
    *,
    host_indices: Sequence[int] | None = None,
    record_indices: Sequence[int] | None = None,
    quantum_drive: np.ndarray | None = None,
    stimulus_indices: Sequence[int] | None = None,
    stimulus_rate_hz: float = 100.0,
    record_all_spikes: bool = False,
    record_voltage: bool = False,
    backend: str = "runtime",
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> NetworkResult:
    """Run one simulation and return compact aggregate statistics.

    `record_all_spikes` and `record_voltage` are opt-in; see the module
    docstring for why.
    """
    b2 = _brian()
    configure_backend(backend)
    b2.seed(int(seed))
    b2.start_scope()
    for _noisy in ("resolution_conflict", "method_choice", "unused_brian_object"):
        b2.BrianLogger.suppress_name(_noisy)

    params = params or LIFParameters()
    host_indices = np.asarray(host_indices if host_indices is not None else [], dtype=np.int64)
    n = graph.n_neurons

    if record_voltage and n > _VOLTAGE_RECORD_LIMIT and record_indices is None:
        raise ValueError(
            f"refusing to record membrane potentials for all {n} neurons "
            f"(limit {_VOLTAGE_RECORD_LIMIT}). Pass record_indices to select a subset."
        )
    if record_all_spikes and n > _SPIKE_RECORD_WARN:
        warnings.warn(
            f"record_all_spikes=True with {n} neurons may produce a very large "
            "output file; the default records only hosts, selected downstream "
            "neurons and aggregate counts (addendum section 18).",
            RuntimeWarning, stacklevel=2,
        )

    neurons, synapses, objects = build_network(
        graph, params, host_indices=host_indices, quantum_drive=quantum_drive,
        stimulus_indices=stimulus_indices, stimulus_rate_hz=stimulus_rate_hz, rng=rng,
    )

    # A per-neuron spike *count* is cheap at any scale; spike *times* are not.
    counter = b2.SpikeMonitor(neurons, record=False, name="spike_counter")
    monitors: list[Any] = [counter]

    recorded = (
        np.arange(n) if record_all_spikes
        else np.unique(np.concatenate([
            host_indices,
            np.asarray(record_indices if record_indices is not None else [], dtype=np.int64),
        ])).astype(np.int64) if (host_indices.size or record_indices is not None)
        else np.array([], dtype=np.int64)
    )
    # Brian2 cannot record spike *times* for an arbitrary subset: SpikeMonitor's
    # `record` is a boolean (a list is silently treated as True, recording
    # everything), and subgroups require contiguous indices, which host sets are
    # not. So spike times are collected for the whole group and filtered
    # afterwards -- affordable only below `_SPIKE_TIME_LIMIT`. Above that, the
    # per-neuron counts from `counter` are the aggregate output, which is the
    # default monitoring policy anyway (addendum section 18).
    detail = None
    collect_times = recorded.size > 0 and (record_all_spikes or n <= _SPIKE_TIME_LIMIT)
    if recorded.size and not collect_times:
        warnings.warn(
            f"network has {n} neurons (> {_SPIKE_TIME_LIMIT}); recording spike "
            "counts only. Pass record_all_spikes=True to force spike-time capture.",
            RuntimeWarning, stacklevel=2,
        )
    if collect_times:
        detail = b2.SpikeMonitor(neurons, name="spike_detail")
        monitors.append(detail)

    voltage = None
    if record_voltage:
        v_idx = recorded if recorded.size else np.arange(min(n, _VOLTAGE_RECORD_LIMIT))
        voltage = b2.StateMonitor(neurons, "v", record=v_idx.tolist(), name="voltage")
        monitors.append(voltage)

    net = b2.Network(*objects, *monitors)
    net.run(duration_s * b2.second)

    result = NetworkResult(
        n_neurons=n,
        duration_s=duration_s,
        spike_counts=np.asarray(counter.count, dtype=np.int64),
        host_indices=host_indices,
        recorded_indices=recorded,
        metadata={
            "backend": backend,
            "seed": seed,
            "n_synapses": int(len(synapses)),
            "dataset": graph.dataset,
            "graph_synthetic": graph.is_synthetic,
            "model_transfer_caveat": params.provenance()["caveat"],
        },
    )
    if detail is not None:
        idx = np.asarray(detail.i, dtype=np.int64)
        times = np.asarray(detail.t / b2.second, dtype=np.float64)
        if not record_all_spikes and recorded.size < n:
            keep = np.isin(idx, recorded)
            idx, times = idx[keep], times[keep]
        result.spike_times = {"index": idx, "time_s": times}
    if voltage is not None:
        result.voltage_trace = {
            "time_s": np.asarray(voltage.t / b2.second),
            "v": np.asarray(voltage.v / b2.volt),
        }
    return result


def compare_runs(baseline: NetworkResult, perturbed: NetworkResult) -> dict[str, float]:
    """Downstream response metrics for Experiment 9 (spec section 22).

    Reports the number of neurons whose spike count changed, the integrated
    change, and where in the network it appeared.
    """
    if baseline.n_neurons != perturbed.n_neurons:
        raise ValueError("cannot compare networks of different sizes")
    delta = perturbed.spike_counts.astype(np.int64) - baseline.spike_counts.astype(np.int64)
    changed = np.flatnonzero(delta != 0)
    host = set(perturbed.host_indices.tolist())
    non_host_changed = [i for i in changed.tolist() if i not in host]
    return {
        "n_perturbed_neurons": int(changed.size),
        "n_perturbed_non_host": len(non_host_changed),
        "fraction_perturbed": float(changed.size / max(1, baseline.n_neurons)),
        "total_spike_count_change": int(delta.sum()),
        "absolute_spike_count_change": int(np.abs(delta).sum()),
        "baseline_total_spikes": baseline.total_spikes,
        "perturbed_total_spikes": perturbed.total_spikes,
        "baseline_mean_rate_hz": baseline.mean_rate_hz,
        "perturbed_mean_rate_hz": perturbed.mean_rate_hz,
        "mean_rate_change_hz": perturbed.mean_rate_hz - baseline.mean_rate_hz,
    }
