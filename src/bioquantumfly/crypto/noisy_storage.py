"""Security from the adversary's imperfect quantum memory.

This project's own results point at the right security assumption. In
Experiment 6 an adversary whose quantum memory is *no better than the host's*
ends with exactly zero information, while an adversary with an ideal memory
keeps hers forever. Security here does not come from no-cloning; it comes from
the adversary's storage being as bad as the biology's.

That is the noisy-storage / bounded-quantum-storage model (Damgard-Fehr-Salvail-
Schaffner; Wehner-Schaffner-Terhal), in which primitives that are impossible
unconditionally -- bit commitment, oblivious transfer -- become achievable. It
fits this setting unusually well: an adversary extracting a spin state from a
warm, wet cell faces the same decoherence physics as the host, so the
assumption is a physical claim about her apparatus rather than an unfalsifiable
computational one.

The protocol knob this buys: DELAYED KEY RELEASE. Eve cannot use her stored
ancilla until the key is public, so the longer Alice waits, the more of Eve's
memory has decayed. Waiting costs nothing and is not available to a
computationally-secured scheme. `required_delay` says how long is enough.

CAVEAT. This module computes the *operational* quantities -- how much an
adversary with a stated memory quality retains after a stated delay -- exactly,
using the same channels as the rest of the simulator. It does NOT reproduce the
formal noisy-storage security proofs, and the resulting numbers are a design
tool, not a security certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.analysis.metrics import (
    helstrom_guessing_probability, mutual_information_binary,
)
from bioquantumfly.crypto.postprocessing import key_budget
from bioquantumfly.quantum.abstract_memory import MemoryParams
from bioquantumfly.quantum.channels import t1_t2_channel

#: Named assumptions about the adversary's storage, worst case first.
STORAGE_ASSUMPTIONS: dict[str, str] = {
    "ideal": "Eve holds a perfect quantum memory. No storage assumption at all; "
             "the honest default for a security claim, and the regime in which "
             "this scheme fails.",
    "same_as_host": "Eve's memory is subject to the same decoherence as the "
                    "biological host. Physically motivated if she must extract "
                    "and hold the state in comparable conditions.",
    "bounded_advantage": "Eve's memory beats the host by a bounded factor. The "
                         "assumption an experimentalist could actually defend.",
}


def eve_retained_state(
    r_stolen: ArrayLike, delay: float, eve_memory: MemoryParams
) -> NDArray[np.float64]:
    """Eve's ancilla after waiting `delay` in her own memory."""
    return t1_t2_channel(np.asarray(r_stolen, dtype=float), delay,
                         eve_memory.T1, eve_memory.T2)


def retained_guessing_probability(
    hypothesis_pair: tuple[ArrayLike, ArrayLike],
    delay: float,
    eve_memory: MemoryParams,
) -> float:
    """Eve's optimal guessing probability after the delay.

    `hypothesis_pair` is the state she holds for encoded bit 0 and for bit 1;
    both decohere in her memory before she can use the revealed key.
    """
    r0 = eve_retained_state(hypothesis_pair[0], delay, eve_memory)
    r1 = eve_retained_state(hypothesis_pair[1], delay, eve_memory)
    return float(np.mean(helstrom_guessing_probability(r0, r1)))


def required_delay(
    eve_memory: MemoryParams,
    initial_pguess: float,
    target_pguess: float = 0.5 + 1e-6,
) -> float:
    """How long to withhold the key so Eve's advantage falls below a target.

    Her advantage over a coin flip is proportional to the trace distance between
    her two hypotheses, which decays as e^{-t/T2_E} in the worst case (the
    coherence-limited direction). Inverting,

        delay = T2_E * ln( (p0 - 1/2) / (p_target - 1/2) )

    Returns infinity for an ideal adversary: no amount of waiting helps, which
    is exactly why the storage assumption is load-bearing.
    """
    if not np.isfinite(eve_memory.T2):
        return float("inf")
    advantage = initial_pguess - 0.5
    target = target_pguess - 0.5
    if advantage <= 0:
        return 0.0
    if target <= 0:
        return float("inf")
    if advantage <= target:
        return 0.0
    return float(eve_memory.T2 * np.log(advantage / target))


@dataclass
class StorageSecurity:
    """Security summary for one (adversary memory, delay) configuration."""

    assumption: str
    eve_T2: float
    delay_seconds: float
    delay_over_eve_T2: float
    eve_pguess: float
    eve_information_bits: float
    secret_bits: float
    secret_rate: float
    secure: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyse(
    n_carriers: int,
    bob_ber: float,
    eve_pguess_at_attack: float,
    eve_memory: MemoryParams,
    delay: float,
    *,
    assumption: str = "bounded_advantage",
    information_threshold: float = 1e-3,
) -> StorageSecurity:
    """What the adversary retains, and what secret survives, after a delay.

    Combines the storage assumption with the post-processing budget: the delay
    reduces her guessing probability, privacy amplification then removes what is
    left.
    """
    advantage = max(0.0, eve_pguess_at_attack - 0.5)
    decay = 1.0 if not np.isfinite(eve_memory.T2) else float(np.exp(-delay / eve_memory.T2))
    pguess = 0.5 + advantage * decay

    budget = key_budget(n_carriers, bob_ber, pguess)
    info = float(mutual_information_binary(pguess))
    return StorageSecurity(
        assumption=assumption,
        eve_T2=eve_memory.T2,
        delay_seconds=delay,
        delay_over_eve_T2=(delay / eve_memory.T2) if np.isfinite(eve_memory.T2) else 0.0,
        eve_pguess=pguess,
        eve_information_bits=info * n_carriers,
        secret_bits=budget.secret_bits,
        secret_rate=budget.secret_rate,
        secure=info <= information_threshold and budget.secret_bits > 0,
        note=STORAGE_ASSUMPTIONS.get(assumption, ""),
    )


def delay_sweep(
    n_carriers: int,
    bob_ber: float,
    eve_pguess_at_attack: float,
    eve_memory: MemoryParams,
    delays_over_eve_T2: ArrayLike = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Eve's retained information against how long the key is withheld."""
    if not np.isfinite(eve_memory.T2):
        # An ideal adversary is unaffected by delay; report that plainly rather
        # than emitting a flat curve that looks like a bug.
        result = analyse(n_carriers, bob_ber, eve_pguess_at_attack, eve_memory, 0.0, **kwargs)
        row = result.to_dict()
        row["delay_over_eve_T2"] = 0.0
        row["note"] = "ideal adversary: delaying the key reveal buys nothing"
        return [row]
    rows = []
    for ratio in np.asarray(delays_over_eve_T2, dtype=float):
        result = analyse(n_carriers, bob_ber, eve_pguess_at_attack, eve_memory,
                         float(ratio) * eve_memory.T2, **kwargs)
        rows.append(result.to_dict())
    return rows


def storage_advantage_threshold(
    eve_pguess_at_attack: float,
    delay: float,
    host_T2: float,
    *,
    information_threshold: float = 1e-3,
    max_advantage: float = 1e6,
) -> float:
    """How much better than the host Eve's memory may be before security fails.

    Answers the question an experimentalist can act on: given that the key is
    withheld for `delay`, what bound on the adversary's memory quality does the
    scheme need? Returns the largest ratio T2_Eve / T2_host that still leaves
    her below the information threshold.
    """
    advantage = max(0.0, eve_pguess_at_attack - 0.5)
    if advantage <= 0 or delay <= 0:
        return 0.0
    # Largest p she may end with while staying under the information threshold.
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(mutual_information_binary(mid)) > information_threshold:
            hi = mid
        else:
            lo = mid
    p_allowed = lo
    allowed_decay = (p_allowed - 0.5) / advantage
    if allowed_decay >= 1.0:
        return float(max_advantage)
    if allowed_decay <= 0:
        return 0.0
    eve_T2_max = delay / np.log(1.0 / allowed_decay)
    return float(min(eve_T2_max / host_T2, max_advantage))
