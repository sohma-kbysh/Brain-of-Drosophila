"""Experiments 2, 3, 4, 6 and 7: the adversary (spec section 22).

2 -- raw BB84 attack. Negative result: a secret basis alone is not encryption.
3 -- masked conjugate encoding, before and after key reveal.
4 -- information-disturbance curve over attack strength eta.
6 -- Eve's ancilla persists after the host has decohered.
7 -- SWAP negative control: natural decoherence is not cryptographic deletion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.experiments.common import (
    build_attack, encode_for, eve_memory, host_memory, regime_columns,
)
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT_ATTACK = "eve_attack_sweep"
EXPERIMENT_DISTURBANCE = "information_disturbance"
EXPERIMENT_PERSISTENCE = "eve_ancilla_persistence"
EXPERIMENT_SWAP = "swap_negative_control"


def _row(m, extra: dict[str, Any]) -> dict[str, Any]:
    d = m.to_dict()
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Experiments 2 and 3
# ---------------------------------------------------------------------------
def build_specs_attack(config: Config) -> list[RunSpec]:
    """Both protocols against every configured attack.

    Protocol A is always included even when the config selects Protocol B,
    because the negative result of Experiment 2 is part of the deliverable.
    """
    return product_specs(
        EXPERIMENT_ATTACK,
        {
            "T2": list(config.quantum.T2_seconds),
            "protocol": ["raw_bb84", "masked_conjugate"],
            "attack": list(config.crypto.attacks),
            "n_carriers": [config.quantum.n_carriers],
            "eve_memory": [config.crypto.eve_memory],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "crypto", "tier": "0"},
    )


def run_attack(spec: RunSpec, t_over_T2_grid: list[float] | None = None) -> list[dict[str, Any]]:
    p = spec.params
    T2 = float(p["T2"])
    host = host_memory(T2, float(p.get("t1_over_t2", 1.0)))
    eve_mem = eve_memory(str(p.get("eve_memory", "ideal")), host,
                         float(p.get("eve_memory_advantage", 100.0)))
    protocol, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))
    attack = build_attack(str(p["attack"]), float(p.get("attack_strength", 1.0)))

    grid = np.asarray(
        t_over_T2_grid if t_over_T2_grid is not None
        else p.get("t_over_T2_grid", [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]), dtype=float
    )
    t_attack_ratio = float(p.get("t_attack_over_T2", 0.1))

    rows: list[dict[str, Any]] = []
    for ratio in grid:
        t_read = float(ratio) * T2
        t_attack = min(t_attack_ratio * T2, t_read)
        m = evaluate(encoding, protocol, attack, host=host, eve_memory=eve_mem,
                     t_attack=t_attack, t_read=t_read, t_reveal=t_read)
        rows.append(_row(m, {
            **regime_columns(T2),
            "t_over_T2": float(ratio),
            "eve_memory_model": str(p.get("eve_memory", "ideal")),
        }))
    return rows


# ---------------------------------------------------------------------------
# Experiment 4: information-disturbance tradeoff
# ---------------------------------------------------------------------------
def build_specs_disturbance(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT_DISTURBANCE,
        {
            "T2": list(config.quantum.T2_seconds),
            "attack": ["weak_measurement", "coherent_ancilla"],
            "protocol": [config.crypto.protocol],
            "n_carriers": [config.quantum.n_carriers],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "crypto", "tier": "0"},
    )


def run_disturbance(spec: RunSpec, strengths: list[float] | None = None) -> list[dict[str, Any]]:
    """Sweep attack strength eta and pair Eve's information with Bob's error."""
    p = spec.params
    T2 = float(p["T2"])
    host = host_memory(T2, float(p.get("t1_over_t2", 1.0)))
    eve_mem = eve_memory(str(p.get("eve_memory", "ideal")), host)
    protocol, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))
    etas = np.asarray(
        strengths if strengths is not None
        else p.get("attack_strengths", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]), dtype=float
    )
    t_read = float(p.get("t_read_over_T2", 0.5)) * T2
    t_attack = min(float(p.get("t_attack_over_T2", 0.1)) * T2, t_read)

    rows: list[dict[str, Any]] = []
    for eta in etas:
        attack = build_attack(str(p["attack"]), float(eta))
        m = evaluate(encoding, protocol, attack, host=host, eve_memory=eve_mem,
                     t_attack=t_attack, t_read=t_read, t_reveal=t_read)
        rows.append(_row(m, {**regime_columns(T2), "eta": float(eta)}))
    return rows


# ---------------------------------------------------------------------------
# Experiment 6: Eve's information persists past host decoherence
# ---------------------------------------------------------------------------
def build_specs_persistence(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT_PERSISTENCE,
        {
            "T2": list(config.quantum.T2_seconds),
            "attack": ["coherent_ancilla", "z_intercept_resend", "swap_theft"],
            "eve_memory": ["ideal", "same_as_host", "better"],
            "protocol": [config.crypto.protocol],
            "n_carriers": [config.quantum.n_carriers],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "crypto", "tier": "0"},
    )


def run_persistence(spec: RunSpec, t_over_T2_grid: list[float] | None = None) -> list[dict[str, Any]]:
    """Attack early (t_E < T2), then let the host decohere far past T2.

    The key comparison is `host_info_at_read` against `eve_info_after_key` at
    large t/T2: the host forgets, Eve does not.
    """
    p = spec.params
    T2 = float(p["T2"])
    host = host_memory(T2, float(p.get("t1_over_t2", 1.0)))
    eve_mem = eve_memory(str(p["eve_memory"]), host, float(p.get("eve_memory_advantage", 100.0)))
    protocol, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))
    attack = build_attack(str(p["attack"]), float(p.get("attack_strength", 1.0)))

    t_attack = float(p.get("t_attack_over_T2", 0.1)) * T2
    grid = np.asarray(
        t_over_T2_grid if t_over_T2_grid is not None
        else p.get("t_over_T2_grid", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]), dtype=float
    )

    rows: list[dict[str, Any]] = []
    for ratio in grid:
        t_read = max(float(ratio) * T2, t_attack)
        m = evaluate(encoding, protocol, attack, host=host, eve_memory=eve_mem,
                     t_attack=t_attack, t_read=t_read, t_reveal=t_read)
        rows.append(_row(m, {
            **regime_columns(T2),
            "t_over_T2": float(ratio),
            "eve_memory_model": str(p["eve_memory"]),
            "information_gap": m.eve_info_after_key - m.host_info_after_key,
        }))
    return rows


# ---------------------------------------------------------------------------
# Experiment 7: SWAP negative control
# ---------------------------------------------------------------------------
def build_specs_swap(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT_SWAP,
        {
            "T2": list(config.quantum.T2_seconds),
            "attack": ["swap_theft", "none", "z_intercept_resend"],
            "protocol": [config.crypto.protocol],
            "n_carriers": [config.quantum.n_carriers],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "crypto", "tier": "0", "role": "negative_control"},
    )


def run_swap(spec: RunSpec, t_over_T2_grid: list[float] | None = None) -> list[dict[str, Any]]:
    """The worst case: Eve extracts the state outright into an ideal memory.

    Reported explicitly rather than buried, because the failure boundary is the
    scientific output (spec section 30).
    """
    rows = run_persistence(
        RunSpec(EXPERIMENT_SWAP, {**spec.params, "eve_memory": "ideal"},
                spec.global_seed, spec.tags),
        t_over_T2_grid=t_over_T2_grid,
    )
    for row in rows:
        row["regime"] = (
            "unrestricted_extraction" if row["attack"] == "swap_theft"
            else "constrained_access"
        )
    return rows
