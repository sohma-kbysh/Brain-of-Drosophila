"""Experiment 9: hybrid tamper propagation (spec sections 18, 22).

Chain being tested:

    Eve attack -> rho -> rho'  ->  molecular readout shift  ->  host neuron
    perturbation  ->  downstream neural response

SPECULATIVE. This runs in `tamper_alarm` mode, the one place where the
unvalidated quantum-to-neural coupling is switched on. Every row carries
`coupling_enabled` and `speculative_coupling_gain`, and the plotting layer
marks the figures. `storage_only` mode is included as a control and must
produce exactly zero perturbation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.connectome.subgraph import downstream_neighbourhood, select_hosts
from bioquantumfly.coupling.quantum_neural import (
    CouplingConfig, HybridMode, quantum_drive_vector, readout_shift,
)
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.experiments.common import (
    GRAPH_KEYS, HOST_KEYS, build_attack, encode_for, host_memory,
)
from bioquantumfly.experiments.connectome_loader import load_connectome
from bioquantumfly.experiments.neural_baseline import peak_memory_mb
from bioquantumfly.neural.brian_network import compare_runs, run_network
from bioquantumfly.neural.lif import LIFParameters
from bioquantumfly.quantum.channels import t1_t2_channel
from bioquantumfly.runtime.runspec import RunSpec, product_specs

EXPERIMENT = "hybrid_tamper_propagation"

#: The three adversary conditions compared (spec section 22).
ATTACK_CONDITIONS = {
    "none": ("none", 0.0),
    "weak": ("weak_measurement", 0.3),
    "strong": ("weak_measurement", 1.0),
}


def build_specs(config: Config) -> list[RunSpec]:
    return product_specs(
        EXPERIMENT,
        {
            "T2": list(config.quantum.T2_seconds[:1]) or [1e-6],
            "attack_condition": list(ATTACK_CONDITIONS),
            "speculative_coupling_gain": list(config.coupling.gain_sweep),
            "n_hosts": [config.host_selection.n_hosts],
            "host_strategy": [config.host_selection.strategy],
        },
        global_seed=config.runtime.seed,
        tags={"stage": "hybrid", "tier": "1", "speculative": "true"},
    )


def run(spec: RunSpec, config: Config | None = None) -> list[dict[str, Any]]:
    config = config or Config()
    p = spec.params
    T2 = float(p.get("T2", 1e-6))
    gain = float(p.get("speculative_coupling_gain", 0.0))
    condition = str(p.get("attack_condition", "none"))
    attack_name, strength = ATTACK_CONDITIONS[condition]

    # --- quantum layer: what does the adversary do to the molecular readout? --
    host_mem = host_memory(T2, config.quantum.t1_over_t2)
    protocol, encoding = encode_for(spec, config.crypto.protocol, config.quantum.n_carriers)
    attack = build_attack(attack_name, strength)

    t_attack = float(p.get("t_attack_over_T2", 0.1)) * T2
    t_read = float(p.get("t_read_over_T2", 0.5)) * T2

    r_at_attack = t1_t2_channel(encoding.initial_state(), t_attack, host_mem.T1, host_mem.T2)
    r_clean = t1_t2_channel(r_at_attack, t_read - t_attack, host_mem.T1, host_mem.T2)
    r_attacked = t1_t2_channel(
        attack.apply_to_host(r_at_attack), t_read - t_attack, host_mem.T1, host_mem.T2
    )
    shift = readout_shift(r_clean, r_attacked, config.coupling.readout_axis)

    security = evaluate(encoding, protocol, attack, host=host_mem,
                        t_attack=t_attack, t_read=t_read)

    # --- network layer -----------------------------------------------------
    graph = load_connectome(config, spec.shared_rng("connectome", GRAPH_KEYS))
    hosts = select_hosts(
        graph, mode=config.host_selection.mode,
        strategy=str(p.get("host_strategy", config.host_selection.strategy)),
        n_hosts=int(p.get("n_hosts", config.host_selection.n_hosts)),
        rng=spec.shared_rng("hosts", HOST_KEYS),
        curated_path=config.host_selection.curated_path,
    )
    layers = downstream_neighbourhood(graph, hosts.indices, hops=config.neural.downstream_hops)

    # Spread the carriers over the host neurons; each host averages its share.
    n_hosts = len(hosts)
    assignment = np.array_split(np.arange(encoding.n_carriers), n_hosts)
    per_host_clean = np.stack([r_clean[a].mean(axis=0) for a in assignment])
    per_host_attacked = np.stack([r_attacked[a].mean(axis=0) for a in assignment])

    coupling = CouplingConfig(
        enabled=gain > 0.0,
        mode=HybridMode.TAMPER_ALARM if gain > 0.0 else HybridMode.STORAGE_ONLY,
        speculative_coupling_gain=gain,
        readout_axis=config.coupling.readout_axis,
        baseline_readout=float(np.mean(readout_axis_values(per_host_clean, config))),
    )
    params = LIFParameters()
    drive_clean = quantum_drive_vector(graph.n_neurons, hosts.indices, per_host_clean,
                                       coupling, params)
    drive_attacked = quantum_drive_vector(graph.n_neurons, hosts.indices, per_host_attacked,
                                          coupling, params)

    record = np.concatenate([v for k, v in layers.items() if k > 0]) if len(layers) > 1 else None
    base_kwargs = dict(
        duration_s=config.neural.duration_s, params=params, host_indices=hosts.indices,
        record_indices=record,
        stimulus_indices=hosts.indices if config.neural.stimulate_hosts else None,
        stimulus_rate_hz=config.neural.stimulus_rate_hz,
        backend=config.neural.backend,
    )

    # A spiking network near threshold is chaotic: an arbitrarily small drive can
    # advance one spike across a timestep and cascade. A single realisation of
    # "how many neurons changed" is therefore dominated by that sensitivity
    # rather than by the perturbation size, and comes out non-monotonic in kappa.
    # Averaging over independent network seeds turns it into an estimate with a
    # reportable spread. Each repeat uses the SAME seed for its own baseline and
    # perturbed run, so the pair differs only by the quantum drive.
    n_repeats = int(p.get("n_repeats", config.neural.n_repeats))
    start = time.perf_counter()
    comparisons: list[dict[str, float]] = []
    depths: list[int] = []
    for repeat in range(n_repeats):
        seed = spec.seed_for(f"brian{repeat}") % (2**31 - 1)
        baseline = run_network(graph, quantum_drive=drive_clean, seed=seed,
                               rng=spec.rng(f"net_a{repeat}"), **base_kwargs)
        perturbed = run_network(graph, quantum_drive=drive_attacked, seed=seed,
                                rng=spec.rng(f"net_b{repeat}"), **base_kwargs)
        comparisons.append(compare_runs(baseline, perturbed))
        depths.append(_perturbation_depth(baseline, perturbed, layers))
    wall = time.perf_counter() - start

    comparison = {k: float(np.mean([c[k] for c in comparisons])) for k in comparisons[0]}
    comparison.update({
        f"{k}_std": float(np.std([c[k] for c in comparisons]))
        for k in ("n_perturbed_non_host", "absolute_spike_count_change")
    })
    depth = int(np.mean(depths))

    return [{
        "attack_condition": condition,
        "attack": attack_name,
        "attack_strength": strength,
        "T2": T2,
        "speculative_coupling_gain": gain,
        "coupling_enabled": coupling.is_active,
        "coupling_mode": coupling.mode.value,
        # --- microscopic ---
        "mean_abs_readout_shift": float(np.mean(np.abs(shift))),
        "max_abs_readout_shift": float(np.max(np.abs(shift))) if shift.size else 0.0,
        "max_drive_shift_volt": float(np.max(np.abs(drive_attacked - drive_clean))),
        # --- cryptographic ---
        "bob_ber": security.bob_ber,
        "bob_disturbance_ber": security.bob_disturbance_ber,
        "eve_info_after_key": security.eve_info_after_key,
        # --- macroscopic ---
        **comparison,
        "propagation_depth_perturbed": depth,
        "n_repeats": n_repeats,
        "n_hosts": n_hosts,
        "host_strategy": hosts.strategy.value if hosts.strategy else None,
        "n_neurons": graph.n_neurons,
        "n_edges": graph.n_edges,
        "graph_synthetic": graph.is_synthetic,
        "dataset": graph.dataset,
        "wall_seconds": wall,
        "peak_rss_mb": peak_memory_mb(),
    }]


def readout_axis_values(states: np.ndarray, config: Config) -> np.ndarray:
    from bioquantumfly.coupling.quantum_neural import molecular_readout

    return molecular_readout(states, config.coupling.readout_axis)


def _perturbation_depth(baseline, perturbed, layers: dict[int, np.ndarray]) -> int:
    """Deepest downstream layer containing a neuron whose spike count changed."""
    delta = perturbed.spike_counts.astype(np.int64) - baseline.spike_counts.astype(np.int64)
    changed = set(np.flatnonzero(delta != 0).tolist())
    depth = 0
    for level, members in layers.items():
        if changed & set(members.tolist()):
            depth = max(depth, level)
    return depth
