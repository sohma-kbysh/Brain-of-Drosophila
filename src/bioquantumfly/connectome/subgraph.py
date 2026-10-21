"""Host-neuron selection (spec section 7) and host-placement strategies.

The set H of neurons hypothesised to contain a molecular quantum subsystem.

CRITICAL CONSTRAINT. No selection mode may infer that a neuron is CRY-positive,
or otherwise quantum-capable, from MaleCNS annotations. MaleCNS records
anatomy, connectivity and a predicted transmitter; it records nothing about
cryptochrome expression or spin physics. Curated mode therefore requires the
user to supply an explicit CSV, and the default `engineered` mode is honest
about being a hypothetical genetic-engineering scenario rather than a discovery.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bioquantumfly.connectome.graph import ConnectomeGraph


class HostSelectionMode(str, enum.Enum):
    ENGINEERED = "engineered"
    CENTRALITY = "centrality"
    CURATED = "curated"


class HostStrategy(str, enum.Enum):
    """Where to place hypothetical hosts (addendum section 28).

    These name network positions, not biological facts. Asking where a
    microscopic perturbation would be most efficiently amplified by the measured
    connectome is a network-science question; it is not a claim that those
    neurons contain quantum memories.
    """

    RANDOM = "random"
    HIGH_OUT_DEGREE = "high_out_degree"
    HIGH_IN_DEGREE = "high_in_degree"
    HIGH_OUT_STRENGTH = "high_out_strength"
    BETWEENNESS = "betweenness"
    DESCENDING = "descending"
    SENSORY = "sensory"
    MOTOR = "motor"
    LOCALIZED = "localized"


@dataclass
class HostSelection:
    """The chosen host set plus a record of how it was chosen."""

    indices: np.ndarray
    body_ids: np.ndarray
    mode: HostSelectionMode
    strategy: HostStrategy | None
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return int(self.indices.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_hosts": len(self),
            "mode": self.mode.value,
            "strategy": self.strategy.value if self.strategy else None,
            "body_ids": self.body_ids.tolist(),
            **self.provenance,
        }


def _class_matches(graph: ConnectomeGraph, keywords: tuple[str, ...]) -> np.ndarray:
    """Indices whose class/type annotation matches any keyword."""
    cols = [c for c in ("class", "type", "instance") if c in graph.neurons.columns]
    if not cols:
        return np.array([], dtype=np.int64)
    text = graph.neurons[cols].astype("string").fillna("").agg(" ".join, axis=1).str.lower()
    mask = np.zeros(len(text), dtype=bool)
    for kw in keywords:
        mask |= text.str.contains(kw, regex=False).to_numpy()
    return np.flatnonzero(mask)


def _betweenness_indices(graph: ConnectomeGraph, k: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate betweenness via NetworkX with sampled pivots.

    Exact betweenness is O(NM) and infeasible at connectome scale, so pivots are
    sampled; the sample is seeded from the run's RNG to stay reproducible.
    """
    import networkx as nx

    g = graph.to_networkx()
    n_pivots = min(graph.n_neurons, max(32, 4 * k))
    scores = nx.betweenness_centrality(
        g, k=n_pivots, seed=int(rng.integers(0, 2**31 - 1)), weight=None
    )
    order = sorted(scores, key=lambda b: scores[b], reverse=True)
    return graph.indices_of(order[:k])


def select_hosts(
    graph: ConnectomeGraph,
    *,
    mode: str | HostSelectionMode = HostSelectionMode.ENGINEERED,
    strategy: str | HostStrategy = HostStrategy.RANDOM,
    n_hosts: int = 32,
    rng: np.random.Generator | None = None,
    curated_path: str | Path | None = None,
) -> HostSelection:
    """Choose the host set H.

    Modes
    -----
    engineered
        Hypothetical scenario in which a spin qubit has been genetically
        expressed in a chosen population. The default, and honestly labelled.
    centrality
        Exploratory: pick hosts by a network measure. Answers a network-science
        question about amplification, not a biological one.
    curated
        Only path that uses biological identity, and only from a user-supplied
        CSV with columns ``body_id, quantum_host_type``.
    """
    rng = rng or np.random.default_rng(0)
    mode = HostSelectionMode(mode)
    strategy = HostStrategy(strategy)
    n_hosts = min(int(n_hosts), graph.n_neurons)

    if mode is HostSelectionMode.CURATED:
        if curated_path is None:
            raise ValueError(
                "curated host selection requires an explicit CSV of body_ids. "
                "CRY expression cannot be inferred from MaleCNS annotations "
                "(spec section 7)."
            )
        table = pd.read_csv(curated_path)
        if "body_id" not in table.columns:
            raise ValueError("curated CSV must contain a 'body_id' column")
        known = [b for b in table["body_id"].astype(np.int64) if int(b) in graph._index_of]
        if not known:
            raise ValueError("no curated body_id is present in this graph")
        idx = graph.indices_of(known)
        prov = {
            "source": str(curated_path),
            "n_requested": len(table),
            "n_found_in_graph": len(known),
            "note": "User-supplied curated host list; identities not inferred from MaleCNS.",
        }
        return HostSelection(idx, graph.body_ids[idx], mode, None, prov)

    note = (
        "Hypothetical genetically engineered expression; no claim that these "
        "neurons contain quantum memories."
        if mode is HostSelectionMode.ENGINEERED
        else "Exploratory network-position selection; a network-science question, "
             "not a biological identification."
    )

    if strategy is HostStrategy.RANDOM:
        idx = np.sort(rng.choice(graph.n_neurons, size=n_hosts, replace=False))
    elif strategy is HostStrategy.HIGH_OUT_DEGREE:
        idx = np.sort(np.argsort(graph.out_degree())[::-1][:n_hosts])
    elif strategy is HostStrategy.HIGH_IN_DEGREE:
        idx = np.sort(np.argsort(graph.in_degree())[::-1][:n_hosts])
    elif strategy is HostStrategy.HIGH_OUT_STRENGTH:
        idx = np.sort(np.argsort(graph.out_strength())[::-1][:n_hosts])
    elif strategy is HostStrategy.BETWEENNESS:
        idx = np.sort(_betweenness_indices(graph, n_hosts, rng))
    elif strategy is HostStrategy.LOCALIZED:
        # An anatomically contiguous group, approximated by a seed plus its
        # forward neighbourhood -- a spatially clustered rather than scattered set.
        seed = int(rng.integers(0, graph.n_neurons))
        pool = graph.k_hop([seed], hops=2)
        idx = np.sort(pool[:n_hosts]) if pool.size >= n_hosts else np.sort(
            np.concatenate([pool, rng.choice(graph.n_neurons, n_hosts - pool.size, replace=False)])
        )
    else:
        keywords = {
            HostStrategy.DESCENDING: ("descending", "dn"),
            HostStrategy.SENSORY: ("sensory", "afferent", "receptor", "orn"),
            HostStrategy.MOTOR: ("motor", "mn", "efferent"),
        }[strategy]
        pool = _class_matches(graph, keywords)
        if pool.size == 0:
            raise ValueError(
                f"no neurons match strategy {strategy.value!r} in this graph "
                f"(dataset={graph.dataset}); annotations may be absent"
            )
        take = min(n_hosts, pool.size)
        idx = np.sort(rng.choice(pool, size=take, replace=False))

    prov = {
        "n_requested": n_hosts,
        "note": note,
        "graph_dataset": graph.dataset,
        "graph_synthetic": graph.is_synthetic,
    }
    return HostSelection(idx, graph.body_ids[idx], mode, strategy, prov)


def shuffle_host_locations(
    graph: ConnectomeGraph, selection: HostSelection, rng: np.random.Generator
) -> HostSelection:
    """Host-location-shuffled control (addendum section 21).

    Same number of hosts, random positions: separates 'where the hosts are'
    from 'how many hosts there are'.
    """
    idx = np.sort(rng.choice(graph.n_neurons, size=len(selection), replace=False))
    prov = dict(selection.provenance)
    prov["control"] = "host_location_shuffled"
    return HostSelection(idx, graph.body_ids[idx], selection.mode, HostStrategy.RANDOM, prov)


def downstream_neighbourhood(
    graph: ConnectomeGraph, host_indices: np.ndarray, hops: int = 2, min_weight: int = 1
) -> dict[int, np.ndarray]:
    """Neurons at each forward distance from the hosts.

    Used to measure propagation depth without recording every neuron in the
    network (addendum section 18).
    """
    import scipy.sparse as sp

    mask = graph.weights >= min_weight if min_weight > 1 else graph.weights != 0
    adjacency = sp.csr_matrix(mask, dtype=bool)
    visited = np.zeros(graph.n_neurons, dtype=bool)
    visited[host_indices] = True
    frontier = visited.copy()
    layers: dict[int, np.ndarray] = {0: np.asarray(host_indices, dtype=np.int64)}
    for depth in range(1, hops + 1):
        nxt = (adjacency.T.dot(frontier.astype(np.int8)) > 0) & ~visited
        if not nxt.any():
            break
        layers[depth] = np.flatnonzero(nxt)
        visited |= nxt
        frontier = nxt
    return layers
