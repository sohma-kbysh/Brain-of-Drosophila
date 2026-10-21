"""Experiments 1 and 5: quantum memory decay and self-erasure (spec section 22).

Experiment 1 -- no adversary. Sweep T2 and storage time, report Bob's bit-error
rate. Sanity: BER(0) ~ 0, and under strong decoherence BER -> 1/2.

Experiment 5 -- the same sweep read as self-erasure: how the information
remaining in the host, I(M:Q_t), decays with storage time. Note carefully what
this does and does not show. The host losing information is NOT deletion in any
cryptographic sense; Experiments 6 and 7 exist to make that explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.experiments.common import (
    build_attack, encode_for, host_memory, regime_columns,
)
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT = "quantum_memory_decay"


def build_specs(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT,
        {
            "T2": list(config.quantum.T2_seconds),
            "n_carriers": [config.quantum.n_carriers],
            "protocol": [config.crypto.protocol],
            "t1_over_t2": [config.quantum.t1_over_t2],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "quantum_only", "tier": "0"},
    )


def run(spec: RunSpec, t_over_T2_grid: list[float] | None = None) -> list[dict[str, Any]]:
    """One (T2, n) point, evaluated across the whole storage-time grid."""
    p = spec.params
    T2 = float(p["T2"])
    grid = np.asarray(
        t_over_T2_grid if t_over_T2_grid is not None
        else p.get("t_over_T2_grid",
                   [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03,
                    0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]),
        dtype=float,
    )
    host = host_memory(T2, float(p.get("t1_over_t2", 1.0)))
    protocol, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))
    no_attack = build_attack("none")

    rows: list[dict[str, Any]] = []
    for ratio in grid:
        m = evaluate(
            encoding, protocol, no_attack,
            host=host, t_attack=0.0, t_read=float(ratio) * T2,
        )
        rows.append({
            **regime_columns(T2),
            "t_over_T2": float(ratio),
            "t_seconds": float(ratio) * T2,
            "n_carriers": encoding.n_carriers,
            "protocol": encoding.protocol,
            "bob_ber": m.bob_ber,
            "bob_exact_recovery": m.bob_exact_recovery,
            "host_info_at_read": m.host_info_at_read,
            "host_info_after_key": m.host_info_after_key,
            "host_holevo_at_read": m.host_holevo_at_read,
            "eve_info_after_key": m.eve_info_after_key,
        })
    return rows
