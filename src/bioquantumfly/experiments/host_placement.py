"""Host-placement sweep (addendum section 28).

The question, stated precisely:

    For the SAME microscopic perturbation, how much does the macroscopic neural
    response depend on WHERE in the connectome the hypothetical quantum host
    sits?

This is a network-science question about the measured connectome. It is NOT
evidence that any of these neurons contains a quantum memory, and the strategy
names (descending, sensory, motor) describe network position and annotation,
nothing more.

Each strategy receives an identical normalised perturbation, so any difference
in response is attributable to placement rather than to perturbation size. The
randomised controls of addendum section 21 separate biological topology from
generic connectivity.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.connectome.subgraph import (
    HostStrategy, downstream_neighbourhood, select_hosts, shuffle_host_locations,
)
from bioquantumfly.connectome.synthetic import CONTROL_GRAPHS
from bioquantumfly.coupling.quantum_neural import CouplingConfig, HybridMode, coupling_scale
from bioquantumfly.experiments.common import GRAPH_KEYS, HOST_KEYS
from bioquantumfly.experiments.connectome_loader import load_connectome
from bioquantumfly.experiments.neural_baseline import peak_memory_mb
from bioquantumfly.neural.brian_network import compare_runs, run_network
from bioquantumfly.neural.lif import LIFParameters
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT = "host_placement"

#: Strategies that do not need class annotations, so they work on any graph.
GENERIC_STRATEGIES = (
    HostStrategy.RANDOM.value,
    HostStrategy.HIGH_OUT_DEGREE.value,
    HostStrategy.HIGH_IN_DEGREE.value,
    HostStrategy.HIGH_OUT_STRENGTH.value,
    HostStrategy.BETWEENNESS.value,
    HostStrategy.LOCALIZED.value,
)


def build_specs(config: Config, strategies: list[str] | None = None) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT,
        {
            "host_strategy": strategies or list(GENERIC_STRATEGIES),
            "graph_variant": ["measured", "degree_preserving", "weight_shuffled",
                             "host_location_shuffled"],
            "n_hosts": [config.host_selection.n_hosts],
            "perturbation": [1.0],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "host_placement", "tier": "2"},
    )


def run(spec: RunSpec, config: Config | None = None) -> list[dict[str, Any]]:
    config = config or Config()
    p = spec.params
    variant = str(p.get("graph_variant", "measured"))
    strategy = str(p.get("host_strategy", "random"))
    perturbation = float(p.get("perturbation", 1.0))

    graph = load_connectome(config, spec.shared_rng("connectome", GRAPH_KEYS))
    if variant in CONTROL_GRAPHS:
        graph = CONTROL_GRAPHS[variant](graph, spec.rng("control"))

    hosts = select_hosts(graph, mode="centrality", strategy=strategy,
                         n_hosts=int(p.get("n_hosts", 32)),
                         rng=spec.shared_rng("hosts", HOST_KEYS))
    if variant == "host_location_shuffled":
        hosts = shuffle_host_locations(graph, hosts, spec.rng("host_shuffle"))

    layers = downstream_neighbourhood(graph, hosts.indices, hops=config.neural.downstream_hops)
    params = LIFParameters()

    # An identical normalised perturbation for every placement: `perturbation`
    # multiples of the drive that just fires a resting neuron. Because the drive
    # is the same everywhere, differences in response come from topology alone.
    drive = np.zeros(graph.n_neurons)
    drive[hosts.indices] = perturbation * coupling_scale(params)

    kwargs = dict(
        duration_s=config.neural.duration_s, params=params, host_indices=hosts.indices,
        record_indices=np.concatenate([v for k, v in layers.items() if k > 0])
        if len(layers) > 1 else None,
        stimulus_indices=hosts.indices if config.neural.stimulate_hosts else None,
        stimulus_rate_hz=config.neural.stimulus_rate_hz,
        backend=config.neural.backend,
        seed=spec.seed_for("brian") % (2**31 - 1),
    )
    start = time.perf_counter()
    baseline = run_network(graph, quantum_drive=None, rng=spec.rng("net_a"), **kwargs)
    perturbed = run_network(graph, quantum_drive=drive, rng=spec.rng("net_b"), **kwargs)
    wall = time.perf_counter() - start

    comparison = compare_runs(baseline, perturbed)
    delta = perturbed.spike_counts.astype(np.int64) - baseline.spike_counts.astype(np.int64)
    changed = set(np.flatnonzero(delta != 0).tolist())
    depth = max((lvl for lvl, mem in layers.items() if changed & set(mem.tolist())), default=0)

    host_out = graph.out_degree()[hosts.indices]
    return [{
        "host_strategy": strategy,
        "graph_variant": variant,
        "is_control": variant != "measured",
        "perturbation": perturbation,
        "n_hosts": len(hosts),
        "host_mean_out_degree": float(host_out.mean()) if host_out.size else 0.0,
        "host_total_out_strength": float(graph.out_strength()[hosts.indices].sum()),
        **comparison,
        "propagation_depth": depth,
        "reachable_within_hops": int(sum(len(v) for k, v in layers.items() if k > 0)),
        "amplification_ratio": float(
            comparison["n_perturbed_non_host"] / max(1, len(hosts))
        ),
        "n_neurons": graph.n_neurons,
        "n_edges": graph.n_edges,
        "graph_synthetic": graph.is_synthetic,
        "dataset": graph.dataset,
        "wall_seconds": wall,
        "peak_rss_mb": peak_memory_mb(),
        "interpretation_note": (
            "Network-position question about the connectome; not evidence that "
            "these neurons host quantum memories."
        ),
    }]
