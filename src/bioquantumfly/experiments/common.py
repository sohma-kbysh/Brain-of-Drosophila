"""Shared helpers for the experiment modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from bioquantumfly.config import Config
from bioquantumfly.crypto import get_attack, get_protocol
from bioquantumfly.crypto.encoding import random_message
from bioquantumfly.crypto.eve import Attack
from bioquantumfly.params import classify_T2, regime_of_T2
from bioquantumfly.quantum.abstract_memory import MemoryParams
from bioquantumfly.runtime.runspec import RunSpec


def host_memory(T2: float, t1_over_t2: float = 1.0) -> MemoryParams:
    return MemoryParams.from_T2(T2, t1_over_t2=t1_over_t2)


def eve_memory(kind: str, host: MemoryParams, advantage: float = 100.0) -> MemoryParams:
    """Eve's own quantum memory quality (spec section 12).

    `ideal` is the worst case for the protocol and therefore the honest default
    for a security claim: an adversary with a perfect memory.
    """
    if kind == "ideal":
        return MemoryParams.ideal()
    if kind == "same_as_host":
        return MemoryParams(T1=host.T1, T2=host.T2, label="eve_same_as_host")
    if kind == "better":
        return MemoryParams(
            T1=host.T1 * advantage, T2=host.T2 * advantage,
            label=f"eve_better_x{advantage:g}",
        )
    raise ValueError(f"unknown Eve memory model {kind!r}")


def build_attack(name: str, strength: float = 1.0) -> Attack:
    """Instantiate an attack, mapping a scalar strength onto its parameter."""
    if name == "weak_measurement":
        return get_attack(name, eta=float(strength))
    if name == "coherent_ancilla":
        # strength in [0, 1] maps onto the coupling angle in [0, pi/2], so
        # strength = 1 is a full CNOT copy of the computational-basis value.
        return get_attack(name, alpha=float(strength) * np.pi / 2.0)
    return get_attack(name)


#: The message and basis draw depend only on these, never on T2, attack or
#: timing -- so a sweep over those variables compares like with like.
MESSAGE_KEYS = ("n_carriers", "protocol")

#: The connectome depends only on its size. Sweeping the coupling gain or the
#: attack condition must NOT regenerate the graph: comparing two different
#: random networks would attribute their difference to the parameter under
#: study. Only an explicit size change may change the substrate.
GRAPH_KEYS = ("n_neurons",)

#: Host placement additionally depends on how many hosts and which strategy,
#: and on the graph variant (a control graph is a different substrate).
HOST_KEYS = ("n_neurons", "n_hosts", "host_strategy", "graph_variant")


def encode_for(spec: RunSpec, protocol_name: str, n_carriers: int):
    """Encode a random message, held fixed across the swept variables.

    Uses `RunSpec.shared_rng` rather than `rng` so that every point of a T2 or
    attack sweep encodes the SAME message in the SAME bases. Otherwise each
    point would draw a fresh message and the curves would differ by sampling
    noise as well as by the physics under study.
    """
    protocol = get_protocol(protocol_name)
    rng = spec.shared_rng("message", MESSAGE_KEYS)
    message = random_message(int(n_carriers), rng)
    return protocol, protocol.encode(message, rng)


def regime_columns(T2: float) -> dict[str, Any]:
    """Evidence labelling attached to every emitted row (spec section 28)."""
    return {
        "T2": T2,
        "T2_regime": regime_of_T2(T2),
        "T2_status": classify_T2(T2).value,
        "is_speculative": classify_T2(T2).value == "speculative",
    }


def logspace_including(values: list[float]) -> np.ndarray:
    return np.asarray(sorted(set(float(v) for v in values)), dtype=float)


def config_seed(config: Config) -> int:
    return int(config.runtime.seed)
