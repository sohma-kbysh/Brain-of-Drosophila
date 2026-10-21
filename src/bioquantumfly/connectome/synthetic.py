"""Offline substitute and randomised controls (spec section 23, addendum 21).

TWO DISTINCT PURPOSES, which must never be confused in a figure caption:

1. `synthetic_connectome` -- a stand-in used when no neuPrint token or network
   access is available, so the pipeline can be developed and tested. It is NOT
   MaleCNS. Any graph it produces is stamped ``synthetic: True`` in its
   provenance, and the plotting layer marks such figures.

2. `degree_preserving_shuffle`, `weight_shuffle` -- randomised *controls*
   derived from a real graph. These answer the scientific question of addendum
   section 21: is any tamper amplification we observe a property of the measured
   biological topology, or would generic connectivity of the same size and
   degree distribution do just as well?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bioquantumfly.connectome.graph import ConnectomeGraph, build_graph
from bioquantumfly.connectome.neurotransmitters import NeurotransmitterPolicy

#: Transmitter mix roughly matching the excitatory/inhibitory balance reported
#: for the fly connectome. A CONVENTION for the offline substitute, not a
#: measurement, and not fitted to MaleCNS.
_SYNTHETIC_NT_MIX = (("acetylcholine", 0.55), ("gaba", 0.25), ("glutamate", 0.20))


def synthetic_connectome(
    n_neurons: int = 500,
    *,
    rng: np.random.Generator | None = None,
    mean_out_degree: float = 30.0,
    mean_synapses_per_edge: float = 18.0,
    hub_fraction: float = 0.05,
    hub_multiplier: float = 8.0,
) -> ConnectomeGraph:
    """A synthetic directed weighted graph for offline development.

    Heavy-tailed out-degree with a small hub population, because the qualitative
    feature that matters for perturbation propagation is the presence of
    high-out-degree neurons. It is a placeholder, NOT a model of MaleCNS.

    Why these defaults. With the Shiu parameters a neuron needs roughly 162
    coincident excitatory synapses to fire (see `LIFParameters.min_synapses_to_fire`),
    so connectivity that is too sparse or too weakly weighted produces a network
    in which stimulated neurons fire but nothing propagates. The defaults were
    chosen empirically to land in a regime with clear multi-hop propagation at a
    plausible mean rate of a few Hz -- roughly 30 outgoing partners and a
    heavy-tailed ~18 synapses per connection. They are TUNED PLACEHOLDER VALUES,
    not measured MaleCNS statistics, whose convergence structure differs.
    """
    rng = rng or np.random.default_rng(0)
    n = int(n_neurons)
    if n < 2:
        raise ValueError("need at least 2 neurons")

    propensity = np.ones(n)
    n_hubs = max(1, int(hub_fraction * n))
    propensity[rng.choice(n, size=n_hubs, replace=False)] = hub_multiplier
    out_deg = rng.poisson(mean_out_degree * propensity / propensity.mean())
    out_deg = np.clip(out_deg, 0, n - 1)

    sources, targets = [], []
    for i in range(n):
        k = int(out_deg[i])
        if k == 0:
            continue
        choices = rng.choice(n - 1, size=k, replace=False)
        choices[choices >= i] += 1          # no self-connections
        sources.append(np.full(k, i, dtype=np.int64))
        targets.append(choices.astype(np.int64))

    src = np.concatenate(sources) if sources else np.empty(0, dtype=np.int64)
    dst = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    # Negative binomial: heavy-tailed synapse counts, so a few strong
    # connections dominate, as in real connectomes.
    counts = 1 + rng.negative_binomial(
        2, 2.0 / (2.0 + mean_synapses_per_edge), size=src.size
    )

    names, probs = zip(*_SYNTHETIC_NT_MIX)
    nt = rng.choice(names, size=n, p=probs)
    body_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)

    neurons = pd.DataFrame({
        "body_id": body_ids,
        "type": [f"SYN{i % 37:03d}" for i in range(n)],
        "class": np.where(np.arange(n) % 11 == 0, "descending", "interneuron"),
        "side": rng.choice(["L", "R"], size=n),
        "predicted_nt": nt,
        "confidence": rng.uniform(0.6, 0.99, size=n),
    })
    edges = pd.DataFrame({
        "source": body_ids[src], "target": body_ids[dst],
        "synapse_count": counts.astype(np.int64),
    })
    return build_graph(
        neurons, edges,
        provenance={
            "dataset": "synthetic",
            "synthetic": True,
            "generator": "synthetic_connectome",
            "n_neurons": n,
            "mean_out_degree": mean_out_degree,
            "mean_synapses_per_edge": mean_synapses_per_edge,
            "tuning_note": (
                "Connectivity tuned to give multi-hop propagation at a few Hz "
                "under the Shiu LIF parameters. Placeholder values, not MaleCNS statistics."
            ),
            "warning": (
                "SYNTHETIC GRAPH -- not MaleCNS, not measured anatomy. "
                "For offline development and as a network control only."
            ),
        },
        min_weight=1,
    )


def _rebuild(graph: ConnectomeGraph, weights: sp.csr_matrix, note: str,
             control_kind: str) -> ConnectomeGraph:
    prov = dict(graph.provenance)
    prov["control"] = control_kind
    prov["control_note"] = note
    prov["derived_from"] = graph.dataset
    return ConnectomeGraph(sp.csr_matrix(weights), graph.body_ids.copy(),
                           graph.neurons.copy(), graph.signs.copy(), prov)


def degree_preserving_shuffle(
    graph: ConnectomeGraph, rng: np.random.Generator, n_swap_multiplier: int = 10
) -> ConnectomeGraph:
    """Randomise topology while preserving in- and out-degree exactly.

    Double-edge swaps: (a->b, c->d) becomes (a->d, c->b), which leaves every
    node's in- and out-degree unchanged but destroys higher-order structure.
    A difference between this control and the real graph cannot be explained by
    degree distribution alone.
    """
    coo = graph.weights.tocoo()
    src = coo.row.copy()
    dst = coo.col.copy()
    data = coo.data.copy()
    m = len(src)
    if m < 2:
        return _rebuild(graph, graph.weights.copy(), "too few edges to shuffle",
                        "degree_preserving")

    existing = set(zip(src.tolist(), dst.tolist()))
    attempts = n_swap_multiplier * m
    for _ in range(attempts):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == d or c == b:                       # would create a self-loop
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b)); existing.discard((c, d))
        existing.add((a, d)); existing.add((c, b))
        dst[i], dst[j] = d, b
    shuffled = sp.csr_matrix((data, (src, dst)), shape=graph.weights.shape)
    shuffled.sum_duplicates()
    return _rebuild(graph, shuffled,
                    "double-edge swaps preserving in/out degree", "degree_preserving")


def weight_shuffle(graph: ConnectomeGraph, rng: np.random.Generator) -> ConnectomeGraph:
    """Keep the topology, permute the synapse counts across existing edges.

    Isolates the contribution of *edge weights* from that of wiring pattern.
    """
    coo = graph.weights.tocoo()
    data = coo.data.copy()
    rng.shuffle(data)
    shuffled = sp.csr_matrix((data, (coo.row, coo.col)), shape=graph.weights.shape)
    return _rebuild(graph, shuffled, "synapse counts permuted across edges", "weight_shuffled")


def erdos_renyi_matched(graph: ConnectomeGraph, rng: np.random.Generator) -> ConnectomeGraph:
    """Random graph with matched node and edge counts, no degree matching.

    A weaker control than `degree_preserving_shuffle`; useful to separate a
    pure size effect from a degree-distribution effect.
    """
    n, m = graph.n_neurons, graph.n_edges
    src = rng.integers(0, n, size=m)
    dst = rng.integers(0, n, size=m)
    keep = src != dst
    data = rng.permutation(graph.weights.tocoo().data)[: keep.sum()]
    shuffled = sp.csr_matrix((data, (src[keep], dst[keep])), shape=(n, n))
    shuffled.sum_duplicates()
    return _rebuild(graph, shuffled, "Erdos-Renyi with matched n and |E|", "erdos_renyi")


CONTROL_GRAPHS = {
    "degree_preserving": degree_preserving_shuffle,
    "weight_shuffled": weight_shuffle,
    "erdos_renyi": erdos_renyi_matched,
}
