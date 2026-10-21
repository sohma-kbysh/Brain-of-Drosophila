"""Resolve a connectome from configuration, with honest provenance.

Offline (no NEUPRINT_TOKEN, no network) the synthetic substitute is used, and
it is stamped `synthetic: True` so every downstream figure and metadata file
says so. Nothing here ever presents a synthetic graph as MaleCNS.
"""

from __future__ import annotations

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.connectome.graph import ConnectomeGraph, load_processed
from bioquantumfly.connectome.male_cns import NeuPrintConfig, fetch_subgraph, token_is_present
from bioquantumfly.connectome.neurotransmitters import LowConfidencePolicy, NeurotransmitterPolicy
from bioquantumfly.connectome.synthetic import synthetic_connectome


def nt_policy(config: Config) -> NeurotransmitterPolicy:
    return NeurotransmitterPolicy(
        confidence_threshold=config.connectome.nt_confidence_threshold,
        low_confidence=LowConfidencePolicy(config.connectome.nt_low_confidence),
    )


def load_connectome(
    config: Config, rng: np.random.Generator | None = None, *, allow_fallback: bool = True
) -> ConnectomeGraph:
    """Load the configured connectome.

    Parameters
    ----------
    allow_fallback
        When the config asks for neuPrint but no token is available, fall back
        to the synthetic substitute (with a loud provenance note) instead of
        failing. Set False when a run must genuinely use MaleCNS.
    """
    cc = config.connectome
    rng = rng or np.random.default_rng(config.runtime.seed)

    if cc.source == "processed":
        return load_processed(cc.processed_dir)

    if cc.source == "neuprint":
        if not token_is_present():
            if not allow_fallback:
                from bioquantumfly.connectome.male_cns import TokenMissingError

                raise TokenMissingError(
                    "config requests neuPrint but NEUPRINT_TOKEN is not set"
                )
            graph = synthetic_connectome(cc.synthetic_n_neurons, rng=rng)
            graph.provenance["fallback_reason"] = (
                "NEUPRINT_TOKEN not set; synthetic substitute used instead of "
                f"{cc.dataset}. This graph is NOT MaleCNS."
            )
            return graph
        return fetch_subgraph(
            seed_type=cc.seed_type,
            seed_body_ids=cc.seed_body_ids,
            config=NeuPrintConfig(server=cc.server, dataset=cc.dataset,
                                  min_weight=cc.min_weight, hops=cc.hops),
            max_neurons=cc.max_neurons,
            policy=nt_policy(config),
        )

    return synthetic_connectome(cc.synthetic_n_neurons, rng=rng)
