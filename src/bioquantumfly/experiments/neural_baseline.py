"""Experiment 8: MaleCNS neural baseline with quantum coupling OFF.

Confirms that the LIF network built from connectome connectivity propagates
spikes at all, before any hybrid claim is made. Also emits the scaling numbers
of addendum section 22 (runtime, memory, edge count) so resource estimates are
recorded alongside the science.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.connectome.subgraph import downstream_neighbourhood, select_hosts
from bioquantumfly.experiments.common import GRAPH_KEYS, HOST_KEYS
from bioquantumfly.experiments.connectome_loader import load_connectome
from bioquantumfly.neural.brian_network import run_network
from bioquantumfly.neural.lif import LIFParameters
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT = "neural_baseline"


def build_specs(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT,
        {
            "n_neurons": [config.connectome.synthetic_n_neurons],
            "duration_s": [config.neural.duration_s],
            "stimulus_rate_hz": [config.neural.stimulus_rate_hz],
            "n_hosts": [config.host_selection.n_hosts],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "neural", "tier": "1"},
    )


def peak_memory_mb() -> float:
    import resource

    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(spec: RunSpec, config: Config | None = None) -> list[dict[str, Any]]:
    config = config or Config()
    p = spec.params
    if "n_neurons" in p:
        config = config.model_copy(deep=True)
        config.connectome.synthetic_n_neurons = int(p["n_neurons"])

    graph = load_connectome(config, spec.shared_rng("connectome", GRAPH_KEYS))
    hosts = select_hosts(
        graph, mode=config.host_selection.mode, strategy=config.host_selection.strategy,
        n_hosts=int(p.get("n_hosts", config.host_selection.n_hosts)),
        rng=spec.shared_rng("hosts", HOST_KEYS),
        curated_path=config.host_selection.curated_path,
    )
    layers = downstream_neighbourhood(graph, hosts.indices, hops=config.neural.downstream_hops)

    start = time.perf_counter()
    result = run_network(
        graph,
        duration_s=float(p.get("duration_s", config.neural.duration_s)),
        params=LIFParameters(),
        host_indices=hosts.indices,
        record_indices=np.concatenate([v for k, v in layers.items() if k > 0])
        if len(layers) > 1 else None,
        stimulus_indices=hosts.indices if config.neural.stimulate_hosts else None,
        stimulus_rate_hz=float(p.get("stimulus_rate_hz", config.neural.stimulus_rate_hz)),
        record_all_spikes=config.neural.record_all_spikes,
        backend=config.neural.backend,
        seed=spec.seed_for("brian") % (2**31 - 1),
        rng=spec.rng("network"),
    )
    wall = time.perf_counter() - start

    summary = result.summary()
    return [{
        **summary,
        "n_hosts": len(hosts),
        "host_strategy": hosts.strategy.value if hosts.strategy else None,
        "propagation_depth": max(layers) if layers else 0,
        **{f"layer_{k}_size": int(len(v)) for k, v in layers.items()},
        # --- scaling measurements (addendum section 22) ---
        "wall_seconds": wall,
        "peak_rss_mb": peak_memory_mb(),
        "n_edges": graph.n_edges,
        "coupling_enabled": False,
        "graph_synthetic": graph.is_synthetic,
        "dataset": graph.dataset,
    }]
