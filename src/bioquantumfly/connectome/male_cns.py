"""neuPrint access to MaleCNS v1.0 (spec section 6, addendum section 12).

Two rules this module enforces:

1. The token is read from the ``NEUPRINT_TOKEN`` environment variable and is
   never written to disk, logged, echoed, or placed in provenance metadata.
   Only its *presence* is ever reported.
2. The full ~1.1 GB connectivity table is never downloaded implicitly. The
   default path fetches a k-hop neighbourhood around a seed and caches it.

`fetch_subgraph` requires network access and a token; `bioquantumfly.connectome
.synthetic` provides an explicitly-labelled offline substitute so that the rest
of the pipeline can be developed and tested without either.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bioquantumfly.connectome.graph import ConnectomeGraph, build_graph
from bioquantumfly.connectome.neurotransmitters import NeurotransmitterPolicy

DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"

#: Documented API example seed. It is used ONLY because the MaleCNS
#: documentation uses it; it carries no quantum-biological significance and no
#: claim is made about this cell type.
EXAMPLE_SEED_TYPE = "DNge104"

TOKEN_ENV_VAR = "NEUPRINT_TOKEN"


class TokenMissingError(RuntimeError):
    """Raised when a network operation is attempted without a token."""


def token_is_present() -> bool:
    """Whether a token is configured. Never returns or logs the value."""
    return bool(os.environ.get(TOKEN_ENV_VAR))


def _require_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise TokenMissingError(
            f"{TOKEN_ENV_VAR} is not set. Obtain a token from {DEFAULT_SERVER} "
            f"(Account -> Auth Token) and run:\n"
            f'    export {TOKEN_ENV_VAR}="<your token>"\n'
            "Never commit the token. For offline development use the synthetic "
            "control graph instead (bioquantumfly.connectome.synthetic)."
        )
    return token


@dataclass(frozen=True)
class NeuPrintConfig:
    server: str = DEFAULT_SERVER
    dataset: str = DEFAULT_DATASET
    cache_dir: str = "data/cache/neuprint"
    min_weight: int = 5
    hops: int = 2

    def to_dict(self) -> dict[str, Any]:
        # Deliberately contains no credential field.
        return {"server": self.server, "dataset": self.dataset,
                "min_weight": self.min_weight, "hops": self.hops}


def make_client(config: NeuPrintConfig | None = None):
    """Construct a neuPrint client. Requires NEUPRINT_TOKEN."""
    from neuprint import Client

    cfg = config or NeuPrintConfig()
    return Client(cfg.server, dataset=cfg.dataset, token=_require_token())


def _cache_key(**parts: Any) -> str:
    """Stable cache key. Never includes the token."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def fetch_subgraph(
    seed_type: str | None = None,
    seed_body_ids: Sequence[int] | None = None,
    *,
    config: NeuPrintConfig | None = None,
    hops: int | None = None,
    min_weight: int | None = None,
    max_neurons: int = 20_000,
    policy: NeurotransmitterPolicy | None = None,
    use_cache: bool = True,
    client=None,
) -> ConnectomeGraph:
    """Fetch a k-hop neighbourhood around a seed type or body-id list.

    `max_neurons` is a guard, not a preference: it stops an over-broad seed
    from silently pulling most of the connectome across NFS during what the
    user thought was a small query.
    """
    cfg = config or NeuPrintConfig()
    hops = cfg.hops if hops is None else hops
    min_weight = cfg.min_weight if min_weight is None else min_weight
    if seed_type is None and not seed_body_ids:
        raise ValueError("provide either seed_type or seed_body_ids")

    cache_dir = Path(cfg.cache_dir) / _cache_key(
        server=cfg.server, dataset=cfg.dataset, seed_type=seed_type,
        seed_body_ids=sorted(seed_body_ids) if seed_body_ids else None,
        hops=hops, min_weight=min_weight,
    )
    if use_cache and (cache_dir / "neurons.parquet").exists():
        neurons = pd.read_parquet(cache_dir / "neurons.parquet")
        edges = pd.read_parquet(cache_dir / "edges.parquet")
        prov = json.loads((cache_dir / "query.json").read_text())
        prov["from_cache"] = True
        return build_graph(neurons, edges, policy=policy, provenance=prov, min_weight=min_weight)

    from neuprint import NeuronCriteria, fetch_adjacencies, fetch_neurons

    client = client or make_client(cfg)
    criteria = (
        NeuronCriteria(bodyId=list(seed_body_ids), client=client)
        if seed_body_ids else
        NeuronCriteria(type=seed_type, client=client)
    )
    seed_df, _ = fetch_neurons(criteria, client=client)
    if seed_df.empty:
        raise ValueError(f"no neurons matched seed {seed_type or seed_body_ids!r} in {cfg.dataset}")

    body_ids = set(int(b) for b in seed_df["bodyId"])
    frontier = set(body_ids)
    for _ in range(hops):
        if not frontier:
            break
        _, conn = fetch_adjacencies(
            sources=NeuronCriteria(bodyId=sorted(frontier), client=client),
            targets=None, min_total_weight=min_weight, client=client,
        )
        if conn.empty:
            break
        new = set(int(b) for b in conn["bodyId_post"]) - body_ids
        if len(body_ids) + len(new) > max_neurons:
            # Truncate deterministically rather than blowing the guard.
            new = set(sorted(new)[: max_neurons - len(body_ids)])
        body_ids |= new
        frontier = new

    neuron_df, _ = fetch_neurons(NeuronCriteria(bodyId=sorted(body_ids), client=client), client=client)
    _, conn_df = fetch_adjacencies(
        sources=NeuronCriteria(bodyId=sorted(body_ids), client=client),
        targets=NeuronCriteria(bodyId=sorted(body_ids), client=client),
        min_total_weight=min_weight, client=client,
    )

    neurons = canonicalise_neurons(neuron_df)
    edges = canonicalise_edges(conn_df)
    prov = {
        "dataset": cfg.dataset, "server": cfg.server, "seed_type": seed_type,
        "seed_body_ids": sorted(seed_body_ids) if seed_body_ids else None,
        "hops": hops, "min_weight": min_weight,
        "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "synthetic": False, "from_cache": False,
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        neurons.to_parquet(cache_dir / "neurons.parquet", index=False)
        edges.to_parquet(cache_dir / "edges.parquet", index=False)
        (cache_dir / "query.json").write_text(json.dumps(prov, indent=2))
    return build_graph(neurons, edges, policy=policy, provenance=prov, min_weight=min_weight)


def canonicalise_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """neuPrint neuron table -> the canonical schema of spec section 6."""
    out = pd.DataFrame({"body_id": df["bodyId"].astype(np.int64)})
    for src, dst in (("type", "type"), ("class", "class"), ("somaSide", "side"),
                     ("predictedNt", "predicted_nt"), ("predictedNtProb", "confidence"),
                     ("instance", "instance"), ("status", "status")):
        if src in df.columns:
            out[dst] = df[src].to_numpy()
    if "predicted_nt" not in out.columns:
        out["predicted_nt"] = pd.array([None] * len(out), dtype="string")
    if "confidence" not in out.columns:
        out["confidence"] = 0.0
    return out


def canonicalise_edges(df: pd.DataFrame) -> pd.DataFrame:
    """neuPrint adjacency table -> (source, target, synapse_count)."""
    grouped = (
        df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
        if "weight" in df.columns and "bodyId_pre" in df.columns
        else df
    )
    return pd.DataFrame({
        "source": grouped["bodyId_pre"].astype(np.int64),
        "target": grouped["bodyId_post"].astype(np.int64),
        "synapse_count": grouped["weight"].astype(np.int64),
    })
