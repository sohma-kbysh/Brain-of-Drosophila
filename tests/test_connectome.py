"""Connectome representation, controls and host selection."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bioquantumfly.connectome.graph import build_graph, load_processed, save_processed
from bioquantumfly.connectome.neurotransmitters import (
    LowConfidencePolicy, NeurotransmitterPolicy, assign_signs,
)
from bioquantumfly.connectome.subgraph import (
    downstream_neighbourhood, select_hosts, shuffle_host_locations,
)
from bioquantumfly.connectome.synthetic import (
    degree_preserving_shuffle, erdos_renyi_matched, synthetic_connectome, weight_shuffle,
)


class TestGraphStructure:
    def test_sparse_representation(self, small_graph):
        assert sp.issparse(small_graph.weights)
        assert small_graph.weights.shape == (small_graph.n_neurons,) * 2
        assert small_graph.n_edges > 0

    def test_body_id_index_round_trip(self, small_graph):
        for i in (0, 5, small_graph.n_neurons - 1):
            assert small_graph.index_of(small_graph.body_ids[i]) == i

    def test_no_self_connections(self, small_graph):
        assert small_graph.weights.diagonal().sum() == 0

    def test_signed_weights_follow_the_presynaptic_neuron(self, small_graph):
        signed = small_graph.signed_weights(0.275e-3).tocoo()
        for row, value in zip(signed.row, signed.data):
            assert np.sign(value) == np.sign(small_graph.signs[row]) or value == 0

    def test_subgraph_preserves_identity(self, small_graph):
        keep = np.arange(20)
        sub = small_graph.subgraph(keep)
        assert sub.n_neurons == 20
        assert np.array_equal(sub.body_ids, small_graph.body_ids[keep])

    def test_networkx_conversion_refused_at_scale(self, small_graph):
        small_graph.weights = sp.csr_matrix((25_000, 25_000))
        small_graph.body_ids = np.arange(25_000)
        with pytest.raises(ValueError, match="refusing to build a NetworkX graph"):
            small_graph.to_networkx()

    def test_round_trip_through_disk(self, small_graph, tmp_path):
        save_processed(small_graph, tmp_path / "g")
        loaded = load_processed(tmp_path / "g")
        assert loaded.n_neurons == small_graph.n_neurons
        assert np.array_equal(loaded.body_ids, small_graph.body_ids)
        assert (loaded.weights != small_graph.weights).nnz == 0

    def test_synthetic_graph_is_labelled_synthetic(self, small_graph):
        assert small_graph.is_synthetic
        assert "not MaleCNS" in small_graph.provenance["warning"]


class TestNeurotransmitterPolicy:
    def _frame(self, nt, confidence):
        import pandas as pd

        return pd.DataFrame({"body_id": [1, 2], "predicted_nt": nt, "confidence": confidence})

    def test_baseline_signs(self):
        signs, keep = assign_signs(self._frame(["acetylcholine", "gaba"], [0.9, 0.9]))
        assert list(signs) == [1, -1]
        assert keep.all()

    def test_glutamate_is_inhibitory_in_the_fly(self):
        signs, _ = assign_signs(self._frame(["glutamate", "glutamate"], [0.9, 0.9]))
        assert list(signs) == [-1, -1]

    @pytest.mark.parametrize("policy,expected_sign,expected_keep", [
        (LowConfidencePolicy.EXCITATORY, 1, True),
        (LowConfidencePolicy.IGNORE, 0, True),
        (LowConfidencePolicy.EXCLUDE, 0, False),
    ])
    def test_low_confidence_policies(self, policy, expected_sign, expected_keep):
        signs, keep = assign_signs(
            self._frame(["gaba", "gaba"], [0.1, 0.1]),
            NeurotransmitterPolicy(confidence_threshold=0.5, low_confidence=policy),
        )
        assert signs[0] == expected_sign
        assert bool(keep[0]) is expected_keep

    def test_simplification_is_documented(self):
        d = NeurotransmitterPolicy().to_dict()
        assert "dopamine" in d["simplified_transmitters"]
        assert "simplification" in d["note"]


class TestControlGraphs:
    def test_degree_preserving_shuffle_preserves_degrees_exactly(self, small_graph, rng):
        shuffled = degree_preserving_shuffle(small_graph, rng)
        assert np.array_equal(shuffled.out_degree(), small_graph.out_degree())
        assert shuffled.n_edges == small_graph.n_edges

    def test_degree_preserving_shuffle_actually_rewires(self, small_graph, rng):
        shuffled = degree_preserving_shuffle(small_graph, rng)
        assert (shuffled.weights != small_graph.weights).nnz > 0

    def test_weight_shuffle_preserves_topology(self, small_graph, rng):
        shuffled = weight_shuffle(small_graph, rng)
        a = (small_graph.weights != 0).tocoo()
        b = (shuffled.weights != 0).tocoo()
        assert set(zip(a.row, a.col)) == set(zip(b.row, b.col))
        assert shuffled.weights.sum() == pytest.approx(small_graph.weights.sum())

    def test_erdos_renyi_matches_size(self, small_graph, rng):
        control = erdos_renyi_matched(small_graph, rng)
        assert control.n_neurons == small_graph.n_neurons

    def test_controls_are_labelled(self, small_graph, rng):
        for fn, label in ((degree_preserving_shuffle, "degree_preserving"),
                          (weight_shuffle, "weight_shuffled")):
            assert fn(small_graph, rng).provenance["control"] == label


class TestHostSelection:
    def test_engineered_random_default(self, small_graph, rng):
        hosts = select_hosts(small_graph, n_hosts=10, rng=rng)
        assert len(hosts) == 10
        assert len(set(hosts.indices.tolist())) == 10
        assert "no claim" in hosts.provenance["note"]

    def test_high_out_degree_picks_the_hubs(self, small_graph, rng):
        hosts = select_hosts(small_graph, mode="centrality",
                             strategy="high_out_degree", n_hosts=5, rng=rng)
        degrees = small_graph.out_degree()
        assert degrees[hosts.indices].min() >= np.sort(degrees)[-5]

    def test_curated_requires_an_explicit_file(self, small_graph, rng):
        with pytest.raises(ValueError, match="cannot be inferred from MaleCNS"):
            select_hosts(small_graph, mode="curated", rng=rng)

    def test_curated_reads_the_supplied_ids(self, small_graph, rng, tmp_path):
        import pandas as pd

        path = tmp_path / "hosts.csv"
        ids = small_graph.body_ids[:4]
        pd.DataFrame({"body_id": ids, "quantum_host_type": ["EYFP"] * 4}).to_csv(path, index=False)
        hosts = select_hosts(small_graph, mode="curated", curated_path=path, rng=rng)
        assert set(hosts.body_ids.tolist()) == set(ids.tolist())

    def test_n_hosts_is_capped_at_network_size(self, small_graph, rng):
        hosts = select_hosts(small_graph, n_hosts=10_000, rng=rng)
        assert len(hosts) == small_graph.n_neurons

    def test_selection_is_reproducible(self, small_graph):
        a = select_hosts(small_graph, n_hosts=8, rng=np.random.default_rng(3))
        b = select_hosts(small_graph, n_hosts=8, rng=np.random.default_rng(3))
        assert np.array_equal(a.indices, b.indices)

    def test_location_shuffle_keeps_the_count(self, small_graph, rng):
        hosts = select_hosts(small_graph, n_hosts=8, rng=rng)
        shuffled = shuffle_host_locations(small_graph, hosts, rng)
        assert len(shuffled) == len(hosts)
        assert shuffled.provenance["control"] == "host_location_shuffled"


class TestDownstream:
    def test_layer_zero_is_the_host_set(self, small_graph, rng):
        hosts = select_hosts(small_graph, n_hosts=5, rng=rng)
        layers = downstream_neighbourhood(small_graph, hosts.indices, hops=2)
        assert np.array_equal(layers[0], hosts.indices)

    def test_layers_are_disjoint(self, small_graph, rng):
        hosts = select_hosts(small_graph, n_hosts=5, rng=rng)
        layers = downstream_neighbourhood(small_graph, hosts.indices, hops=3)
        seen: set[int] = set()
        for members in layers.values():
            assert not (seen & set(members.tolist()))
            seen |= set(members.tolist())

    def test_k_hop_includes_the_seeds(self, small_graph):
        reached = small_graph.k_hop([0, 1], hops=2)
        assert {0, 1} <= set(reached.tolist())


class TestBuildGraphValidation:
    def test_missing_columns_are_rejected(self):
        import pandas as pd

        with pytest.raises(ValueError, match="body_id"):
            build_graph(pd.DataFrame({"x": [1]}),
                        pd.DataFrame({"source": [1], "target": [1], "synapse_count": [1]}))
        with pytest.raises(ValueError, match="synapse_count"):
            build_graph(pd.DataFrame({"body_id": [1]}),
                        pd.DataFrame({"source": [1], "target": [1]}))
