"""Experiment registry: names the CLI and sweep runner can dispatch on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from bioquantumfly.config import Config
from bioquantumfly.experiments import (
    eve_sweep, feasibility, host_placement, hybrid, neural_baseline,
    protocol_stack, quantum_only,
)
from bioquantumfly.runtime.runspec import RunSpec


@dataclass(frozen=True)
class ExperimentDef:
    name: str
    spec_number: str
    description: str
    build_specs: Callable[[Config], list[RunSpec]]
    run: Callable[..., list[dict[str, Any]]]
    needs_config: bool = False
    needs_neural: bool = False
    speculative: bool = False


EXPERIMENTS: dict[str, ExperimentDef] = {
    "quantum_memory_decay": ExperimentDef(
        "quantum_memory_decay", "Experiment 1 & 5",
        "Finite-coherence storage decay with no adversary, and host self-erasure.",
        quantum_only.build_specs, quantum_only.run,
    ),
    "eve_attack_sweep": ExperimentDef(
        "eve_attack_sweep", "Experiment 2 & 3",
        "Raw BB84 vs masked conjugate encoding under each attack, before and "
        "after key reveal.",
        eve_sweep.build_specs_attack, eve_sweep.run_attack,
    ),
    "information_disturbance": ExperimentDef(
        "information_disturbance", "Experiment 4",
        "Eve information against Bob disturbance over attack strength eta.",
        eve_sweep.build_specs_disturbance, eve_sweep.run_disturbance,
    ),
    "eve_ancilla_persistence": ExperimentDef(
        "eve_ancilla_persistence", "Experiment 6",
        "Host decoheres past T2 while Eve's ancilla retains information.",
        eve_sweep.build_specs_persistence, eve_sweep.run_persistence,
    ),
    "swap_negative_control": ExperimentDef(
        "swap_negative_control", "Experiment 7",
        "NEGATIVE CONTROL: ideal SWAP theft. Natural decoherence is not deletion.",
        eve_sweep.build_specs_swap, eve_sweep.run_swap,
    ),
    "neural_baseline": ExperimentDef(
        "neural_baseline", "Experiment 8",
        "Connectome LIF network with quantum coupling OFF; propagation sanity.",
        neural_baseline.build_specs, neural_baseline.run,
        needs_config=True, needs_neural=True,
    ),
    "hybrid_tamper_propagation": ExperimentDef(
        "hybrid_tamper_propagation", "Experiment 9",
        "SPECULATIVE: quantum disturbance coupled into neural dynamics.",
        hybrid.build_specs, hybrid.run,
        needs_config=True, needs_neural=True, speculative=True,
    ),
    "host_placement": ExperimentDef(
        "host_placement", "Addendum section 28",
        "Where would a microscopic perturbation be most efficiently amplified?",
        host_placement.build_specs, host_placement.run,
        needs_config=True, needs_neural=True,
    ),
    "protocol_stack": ExperimentDef(
        "protocol_stack", "Experiment 10",
        "Ablation over Protocol C's layers: which primitive repairs which failure.",
        protocol_stack.build_specs, protocol_stack.run, needs_config=True,
    ),
    "coherence_requirements": ExperimentDef(
        "coherence_requirements", "Experiment 11",
        "Inverted feasibility: what coherence time would each error budget demand?",
        protocol_stack.build_specs_requirements, protocol_stack.run_requirements,
        needs_config=True,
    ),
    "feasibility_map": ExperimentDef(
        "feasibility_map", "Addendum section 29",
        "Multidimensional regime classification; the feasibility boundary.",
        feasibility.build_specs, feasibility.run, needs_config=True,
    ),
}

#: Experiments that need neither a connectome nor Brian2. These run anywhere.
QUANTUM_ONLY_EXPERIMENTS = tuple(
    name for name, d in EXPERIMENTS.items() if not d.needs_neural
)


def get_experiment(name: str) -> ExperimentDef:
    if name not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}"
        )
    return EXPERIMENTS[name]


def make_runner(name: str, config: Config) -> Callable[[RunSpec], list[dict[str, Any]]]:
    """Bind an experiment to a config, producing a picklable-by-name runner.

    Worker processes receive `(spec, fn)`; `fn` must therefore be importable
    rather than a closure over local state, which is why the config is passed
    through the module-level `_Runner` class below.
    """
    return _Runner(name, config)


class _Runner:
    """Callable wrapper that survives pickling to worker processes."""

    def __init__(self, name: str, config: Config):
        self.name = name
        self.config_dict = config.to_dict()

    def __call__(self, spec: RunSpec) -> list[dict[str, Any]]:
        definition = get_experiment(self.name)
        if definition.needs_config:
            return definition.run(spec, Config.model_validate(self.config_dict))
        return definition.run(spec)
