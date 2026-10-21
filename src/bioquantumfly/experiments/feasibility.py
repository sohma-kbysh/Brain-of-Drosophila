"""Cryptographic feasibility map (addendum section 29).

Sweeps (T2, t_read, t_attack, eta, n) and classifies each point into a regime.
The thresholds are CONFIGURABLE MODELLING CHOICES, not universal cryptographic
standards; `FeasibilityConfig` holds them and every output row records the
values used.

The scientific objective is the BOUNDARY between regimes, so the sweep is
logarithmic and coarse first (addendum section 9); `refine_around_boundary`
adds points only where the classification actually changes, rather than
brute-forcing an enormous Cartesian product.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bioquantumfly.config import Config
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.experiments.common import (
    build_attack, encode_for, eve_memory, host_memory, regime_columns,
)
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT = "feasibility_map"

#: Regime labels (addendum section 29).
REGIME_READABLE_SECURE = "readable_low_eve_info"
REGIME_READABLE_VULNERABLE = "readable_vulnerable"
REGIME_SELF_ERASED = "self_erased"
REGIME_UNREADABLE = "unreadable_even_for_bob"


def classify(
    bob_ber: float, eve_info: float, host_info: float,
    *, max_bob_ber: float, max_eve_info: float,
) -> str:
    """Assign one parameter point to a regime.

    Order matters: readability is checked first, because a state Bob cannot
    read is not "secure", it is useless. Self-erasure is only reported when the
    host has genuinely lost the information AND Bob can no longer read it.
    """
    bob_ok = bob_ber <= max_bob_ber
    eve_ok = eve_info <= max_eve_info
    if bob_ok and eve_ok:
        return REGIME_READABLE_SECURE
    if bob_ok and not eve_ok:
        return REGIME_READABLE_VULNERABLE
    if host_info <= max_eve_info:
        return REGIME_SELF_ERASED
    return REGIME_UNREADABLE


def build_specs(config: Config) -> list[RunSpec]:
    """Coarse logarithmic scan first (addendum section 9)."""
    return product_specs(
        EXPERIMENT,
        {
            "T2": list(config.quantum.T2_seconds),
            "n_carriers": [8, 16, 32, 64, 128],
            "attack": ["weak_measurement"],
            "eve_memory": [config.crypto.eve_memory],
            "protocol": [config.crypto.protocol],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "feasibility", "tier": "0"},
    )


def run(
    spec: RunSpec,
    config: Config | None = None,
    t_grid: list[float] | None = None,
    eta_grid: list[float] | None = None,
) -> list[dict[str, Any]]:
    config = config or Config()
    p = spec.params
    T2 = float(p["T2"])
    host = host_memory(T2, config.quantum.t1_over_t2)
    eve_mem = eve_memory(str(p.get("eve_memory", "ideal")), host,
                         config.crypto.eve_memory_advantage)
    protocol, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))

    # Logarithmic in storage time; dense near eta = 0 because that is where the
    # readable-to-vulnerable transition lives -- a uniform eta grid would step
    # straight over it (addendum section 9).
    t_ratios = np.asarray(t_grid if t_grid is not None
                          else [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03,
                                0.1, 0.3, 1.0, 3.0, 10.0], dtype=float)
    etas = np.asarray(eta_grid if eta_grid is not None
                      else [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0], dtype=float)
    fc = config.feasibility

    rows: list[dict[str, Any]] = []
    for ratio in t_ratios:
        t_read = float(ratio) * T2
        t_attack = min(0.1 * T2, t_read)
        for eta in etas:
            attack = build_attack(str(p["attack"]), float(eta))
            m = evaluate(encoding, protocol, attack, host=host, eve_memory=eve_mem,
                         t_attack=t_attack, t_read=t_read, t_reveal=t_read)
            rows.append({
                **regime_columns(T2),
                "n_carriers": encoding.n_carriers,
                "t_over_T2": float(ratio),
                "t_read_seconds": t_read,
                "eta": float(eta),
                "bob_ber": m.bob_ber,
                "bob_exact_recovery": m.bob_exact_recovery,
                "bob_disturbance_ber": m.bob_disturbance_ber,
                "eve_info_after_key": m.eve_info_after_key,
                "eve_info_before_key": m.eve_info_before_key,
                "host_info_at_read": m.host_info_at_read,
                "host_info_after_key": m.host_info_after_key,
                "regime": classify(
                    m.bob_ber, m.eve_info_after_key, m.host_info_after_key,
                    max_bob_ber=fc.max_bob_ber, max_eve_info=fc.max_eve_info,
                ),
                "threshold_max_bob_ber": fc.max_bob_ber,
                "threshold_max_eve_info": fc.max_eve_info,
                "protocol": encoding.protocol,
                "attack": str(p["attack"]),
                "eve_memory_model": str(p.get("eve_memory", "ideal")),
            })
    return rows


def refine_around_boundary(
    df: pd.DataFrame, column: str = "t_over_T2", by: tuple[str, ...] = ("T2", "eta")
) -> list[float]:
    """Suggest refinement points where the regime classification changes.

    Coarse scan -> find the transition -> refine locally, rather than an
    unnecessarily huge grid.
    """
    if df.empty or "regime" not in df.columns:
        return []
    suggestions: set[float] = set()
    group_cols = [c for c in by if c in df.columns]
    grouped = df.groupby(group_cols) if group_cols else [((), df)]
    for _, sub in grouped:
        sub = sub.sort_values(column)
        regimes = sub["regime"].to_numpy()
        values = sub[column].to_numpy(dtype=float)
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1]:
                lo, hi = values[i - 1], values[i]
                if hi > lo:
                    suggestions.update(np.geomspace(max(lo, 1e-6), hi, 5)[1:-1].tolist())
    return sorted(suggestions)
