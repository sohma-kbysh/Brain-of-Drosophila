"""Sparse connectome representation (addendum sections 11, 12, 13).

NetworkX is fine for a few thousand nodes, visualisation and topology
statistics, but it is not the numerical backend here: MaleCNS has 166,691
neurons, and a dense 166691 x 166691 matrix would be ~220 TB in float64. The
primary representation is therefore a `scipy.sparse` CSR matrix of synapse
counts plus separate compact arrays of node metadata:

    body_id  <-> integer index      (two ndarray/dict lookups, not a dict of dicts)
    W[i, j]  =  synapses from i to j

A `ConnectomeGraph` is designed to be built once, held in RAM, and reused
across many simulations rather than re-read from NFS per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from bioquantumfly.connectome.neurotransmitters import NeurotransmitterPolicy, assign_signs


@dataclass
class ConnectomeGraph:
    """A directed, weighted connectome as sparse arrays.

    Attributes
    ----------
    weights
        CSR matrix of raw synapse counts, shape (N, N). Unsigned.
    body_ids
        (N,) int64 neuron identifiers, index i <-> body_ids[i].
    neurons
        Per-neuron metadata, row-aligned with `body_ids`.
    signs
        (N,) int8 presynaptic sign from the neurotransmitter policy.
    provenance
        Where the data came from and how it was filtered. Carried into results
        metadata so a figure can never be silently attributed to MaleCNS when
        it was produced from a synthetic control.
    """

    weights: sp.csr_matrix
    body_ids: np.ndarray
    neurons: pd.DataFrame
    signs: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    _index_of: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError(f"weight matrix must be square, got {self.weights.shape}")
        if len(self.body_ids) != self.weights.shape[0]:
            raise ValueError("body_ids length must match the weight matrix dimension")
        if not self._index_of:
            self._index_of = {int(b): i for i, b in enumerate(self.body_ids)}

    # --- basic properties -------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.weights.nnz)

    @property
    def is_synthetic(self) -> bool:
        """True when this is a control graph, not measured MaleCNS anatomy."""
        return bool(self.provenance.get("synthetic", False))

    @property
    def dataset(self) -> str:
        return str(self.provenance.get("dataset", "unknown"))

    def index_of(self, body_id: int) -> int:
        return self._index_of[int(body_id)]

    def indices_of(self, body_ids: Iterable[int]) -> np.ndarray:
        return np.array([self._index_of[int(b)] for b in body_ids], dtype=np.int64)

    # --- topology ---------------------------------------------------------
    def out_degree(self) -> np.ndarray:
        return np.asarray((self.weights != 0).sum(axis=1)).ravel()

    def in_degree(self) -> np.ndarray:
        return np.asarray((self.weights != 0).sum(axis=0)).ravel()

    def out_strength(self) -> np.ndarray:
        return np.asarray(self.weights.sum(axis=1)).ravel()

    def in_strength(self) -> np.ndarray:
        return np.asarray(self.weights.sum(axis=0)).ravel()

    def signed_weights(self, w_syn_mv: float = 0.275) -> sp.csr_matrix:
        """Synaptic weights in mV: w_ji = N_syn * s_j * W_syn (spec section 15).

        The sign is a property of the *presynaptic* neuron, so it multiplies
        whole rows.
        """
        signed = self.weights.astype(np.float64).multiply(self.signs.reshape(-1, 1))
        return sp.csr_matrix(signed * w_syn_mv)

    def k_hop(self, seeds: Iterable[int], hops: int = 2, min_weight: int = 1) -> np.ndarray:
        """Indices reachable from `seeds` within `hops` forward steps.

        Sparse boolean frontier expansion: no dense intermediate is built, so
        this stays affordable on the full connectome.
        """
        idx = np.asarray(list(seeds), dtype=np.int64)
        if idx.size == 0:
            return idx
        mask = self.weights >= min_weight if min_weight > 1 else self.weights != 0
        adjacency = sp.csr_matrix(mask, dtype=bool)
        visited = np.zeros(self.n_neurons, dtype=bool)
        visited[idx] = True
        frontier = visited.copy()
        for _ in range(max(0, hops)):
            nxt = adjacency.T.dot(frontier.astype(np.int8)) > 0
            new = nxt & ~visited
            if not new.any():
                break
            visited |= new
            frontier = new
        return np.flatnonzero(visited)

    def subgraph(self, indices: Iterable[int], provenance_note: str = "") -> "ConnectomeGraph":
        idx = np.unique(np.asarray(list(indices), dtype=np.int64))
        prov = dict(self.provenance)
        prov["derived"] = f"subgraph of {self.n_neurons} neurons; {provenance_note}".strip()
        prov["parent_n_neurons"] = self.n_neurons
        return ConnectomeGraph(
            weights=sp.csr_matrix(self.weights[idx][:, idx]),
            body_ids=self.body_ids[idx],
            neurons=self.neurons.iloc[idx].reset_index(drop=True),
            signs=self.signs[idx],
            provenance=prov,
        )

    def to_networkx(self):
        """Convert to NetworkX. Small subgraphs and visualisation only.

        Deliberately guarded: building a NetworkX object for the full
        connectome would consume orders of magnitude more memory than the CSR
        representation for no numerical benefit.
        """
        import networkx as nx

        if self.n_neurons > 20_000:
            raise ValueError(
                f"refusing to build a NetworkX graph with {self.n_neurons} nodes; "
                "use the sparse representation for numerics (addendum section 11). "
                "Take a subgraph first if you need NetworkX for visualisation."
            )
        g = nx.DiGraph()
        for i, b in enumerate(self.body_ids):
            g.add_node(int(b), index=i, sign=int(self.signs[i]))
        coo = self.weights.tocoo()
        g.add_weighted_edges_from(
            (int(self.body_ids[i]), int(self.body_ids[j]), float(w))
            for i, j, w in zip(coo.row, coo.col, coo.data)
        )
        return g

    def summary(self) -> dict[str, Any]:
        out_deg = self.out_degree()
        return {
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "total_synapses": float(self.weights.sum()),
            "mean_out_degree": float(out_deg.mean()) if self.n_neurons else 0.0,
            "max_out_degree": int(out_deg.max()) if self.n_neurons else 0,
            "n_excitatory": int((self.signs > 0).sum()),
            "n_inhibitory": int((self.signs < 0).sum()),
            "n_zero_sign": int((self.signs == 0).sum()),
            "dataset": self.dataset,
            "synthetic": self.is_synthetic,
        }


def build_graph(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    policy: NeurotransmitterPolicy | None = None,
    provenance: dict[str, Any] | None = None,
    min_weight: int = 1,
) -> ConnectomeGraph:
    """Assemble a `ConnectomeGraph` from canonical neuron and edge tables.

    Parameters
    ----------
    neurons
        Columns: ``body_id`` (required), plus ``type``, ``class``, ``side``,
        ``predicted_nt``, ``confidence`` when available.
    edges
        Columns: ``source``, ``target``, ``synapse_count``.
    """
    if "body_id" not in neurons.columns:
        raise ValueError("neurons table must contain a 'body_id' column")
    for col in ("source", "target", "synapse_count"):
        if col not in edges.columns:
            raise ValueError(f"edges table must contain a {col!r} column")

    neurons = neurons.drop_duplicates("body_id").reset_index(drop=True)
    policy = policy or NeurotransmitterPolicy()
    signs, keep = assign_signs(neurons, policy)
    if not keep.all():
        neurons = neurons[keep].reset_index(drop=True)
        signs = signs[keep]

    body_ids = neurons["body_id"].to_numpy(dtype=np.int64)
    index_of = {int(b): i for i, b in enumerate(body_ids)}

    e = edges[edges["synapse_count"] >= min_weight]
    src = e["source"].map(index_of)
    dst = e["target"].map(index_of)
    valid = src.notna() & dst.notna()      # drop edges to neurons we excluded
    src = src[valid].to_numpy(dtype=np.int64)
    dst = dst[valid].to_numpy(dtype=np.int64)
    w = e["synapse_count"][valid].to_numpy(dtype=np.float64)

    n = len(body_ids)
    weights = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    weights.sum_duplicates()

    prov = dict(provenance or {})
    prov.setdefault("min_weight", min_weight)
    prov["neurotransmitter_policy"] = policy.to_dict()
    return ConnectomeGraph(weights, body_ids, neurons, signs, prov)


def save_processed(graph: ConnectomeGraph, directory: str | Path) -> Path:
    """Write the optimised local representation (addendum section 12).

    Parquet plus raw CSR arrays: one preprocessing pass, then every simulation
    loads this instead of re-querying neuPrint.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    graph.neurons.to_parquet(d / "neurons.parquet", index=False)

    coo = graph.weights.tocoo()
    pd.DataFrame({
        "source": graph.body_ids[coo.row],
        "target": graph.body_ids[coo.col],
        "synapse_count": coo.data.astype(np.int64),
    }).to_parquet(d / "edges.parquet", index=False)

    np.save(d / "csr_indptr.npy", graph.weights.indptr)
    np.save(d / "csr_indices.npy", graph.weights.indices)
    np.save(d / "csr_weights.npy", graph.weights.data)
    np.save(d / "body_ids.npy", graph.body_ids)
    np.save(d / "signs.npy", graph.signs)
    (d / "metadata.json").write_text(
        json.dumps({"provenance": graph.provenance, "summary": graph.summary()}, indent=2, default=str)
    )
    return d


def load_processed(directory: str | Path) -> ConnectomeGraph:
    """Load a preprocessed connectome. Fast path for every simulation."""
    d = Path(directory)
    indptr = np.load(d / "csr_indptr.npy")
    indices = np.load(d / "csr_indices.npy")
    data = np.load(d / "csr_weights.npy")
    body_ids = np.load(d / "body_ids.npy")
    signs = np.load(d / "signs.npy")
    n = len(body_ids)
    weights = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    neurons = pd.read_parquet(d / "neurons.parquet")
    meta = json.loads((d / "metadata.json").read_text())
    return ConnectomeGraph(weights, body_ids, neurons, signs, meta.get("provenance", {}))
