"""Experiment 10: which primitive repairs which failure.

An ablation over the layers of Protocol C, swept against attack strength and
storage time. The question is not "does it work" but "what is each layer
buying", so that a protocol designer can see the exchange rates directly.

Experiment 11 accompanies it: the inverted feasibility question, asking what
coherence time each error budget demands rather than what a given coherence
time delivers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bioquantumfly.analysis.requirements import requirement_table
from bioquantumfly.config import Config
from bioquantumfly.crypto import deletion as deletion_module
from bioquantumfly.crypto.stack import ABLATIONS, run_stack
from bioquantumfly.experiments.common import (
    build_attack, encode_for, eve_memory, host_memory, regime_columns,
)
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT_STACK = "protocol_stack"
EXPERIMENT_REQUIREMENTS = "coherence_requirements"


def build_specs(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT_STACK,
        {
            "T2": list(config.quantum.T2_seconds[:1]) or [1e-6],
            "attack": ["weak_measurement", "coherent_ancilla", "breidbart", "swap_theft"],
            "eve_memory": ["ideal", "same_as_host", "better"],
            "n_carriers": [max(1024, config.quantum.n_carriers)],
            "protocol": ["masked_conjugate"],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "protocol_design", "tier": "0"},
    )


def run(
    spec: RunSpec,
    config: Config | None = None,
    strengths: list[float] | None = None,
    t_read_grid: list[float] | None = None,
) -> list[dict[str, Any]]:
    config = config or Config()
    p = spec.params
    T2 = float(p["T2"])
    host = host_memory(T2, config.quantum.t1_over_t2)
    eve = eve_memory(str(p["eve_memory"]), host, config.crypto.eve_memory_advantage)
    _, encoding = encode_for(spec, p["protocol"], int(p["n_carriers"]))

    etas = np.asarray(strengths if strengths is not None else [0.0, 0.1, 0.25, 0.5, 1.0],
                      dtype=float)
    reads = np.asarray(t_read_grid if t_read_grid is not None else [0.002, 0.02, 0.1],
                       dtype=float)

    rows: list[dict[str, Any]] = []
    for eta in etas:
        attack = build_attack(str(p["attack"]), float(eta))
        for ratio in reads:
            t_read = float(ratio) * T2
            t_attack = min(0.25 * t_read, 0.005 * T2)
            for layer_name, layer_config in ABLATIONS.items():
                result = run_stack(
                    encoding, attack, host=host, eve_memory=eve,
                    t_read=t_read, t_attack=t_attack, config=layer_config,
                    rng=spec.rng(f"stack-{layer_name}-{eta:g}-{ratio:g}"),
                )
                rows.append({
                    **regime_columns(T2),
                    **result.to_dict(),
                    "layer_set": layer_name,
                    "layer_index": list(ABLATIONS).index(layer_name),
                    "eta": float(eta),
                    "eve_memory_model": str(p["eve_memory"]),
                })
    return rows


# ---------------------------------------------------------------------------
def build_specs_requirements(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT_REQUIREMENTS,
        {"n_check": [1_000, 10_000, 100_000]},
        global_seed=config.runtime.seed,
        tags={"stage": "protocol_design", "tier": "0"},
    )


def run_requirements(spec: RunSpec, config: Config | None = None) -> list[dict[str, Any]]:
    """What would have to be true, plus how long deletion stays certifiable."""
    n_check = int(spec.params["n_check"])
    rows: list[dict[str, Any]] = []
    for row in requirement_table():
        window = deletion_module.certification_window(
            n_check, row["required_T2_seconds"], max_theft_fraction=0.01
        )
        rows.append({
            **row,
            "n_check": n_check,
            "deletion_window_over_T2": window["window_over_T2"],
            "deletion_window_seconds": window["window_seconds"],
            "readable_window_over_T2": row["window_seconds"] / row["required_T2_seconds"],
            "deletion_outlives_readout_by": (
                window["window_over_T2"]
                / max(1e-12, row["window_seconds"] / row["required_T2_seconds"])
            ),
        })
    return rows
