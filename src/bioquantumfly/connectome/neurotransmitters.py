"""Neurotransmitter -> synaptic sign policy (spec section 15).

Baseline policy follows Shiu et al. 2024 so that the neural layer is a faithful
transfer of their model:

    GABAergic     -> inhibitory
    glutamatergic -> inhibitory     (in Drosophila, via GluCl-alpha)
    cholinergic   -> excitatory

Dopamine, octopamine and serotonin are treated as excitatory. That is a
SIMPLIFICATION inherited from Shiu et al., not a physiological claim: these are
predominantly neuromodulatory and their net effect is state- and
receptor-dependent. It is recorded as such in the metadata so a reader can see
which sign assignments are conventions rather than measurements.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Baseline sign per predicted transmitter. +1 excitatory, -1 inhibitory.
NT_POLICY_DEFAULT: dict[str, int] = {
    "gaba": -1,
    "glutamate": -1,
    "acetylcholine": +1,
    # --- Shiu et al. simplification: modulators folded into excitation ---
    "dopamine": +1,
    "octopamine": +1,
    "serotonin": +1,
    "histamine": -1,   # histaminergic photoreceptors are inhibitory in the fly
}

#: Transmitters whose sign is a modelling convention, not a measurement.
SIMPLIFIED_TRANSMITTERS = frozenset({"dopamine", "octopamine", "serotonin"})


class LowConfidencePolicy(str, enum.Enum):
    """What to do with a neuron whose transmitter prediction is unreliable."""

    IGNORE = "ignore"          # keep the neuron, give its outgoing edges weight 0
    EXCITATORY = "excitatory"  # assume excitatory
    EXCLUDE = "exclude"        # drop the neuron from the network entirely


@dataclass
class NeurotransmitterPolicy:
    signs: dict[str, int] = field(default_factory=lambda: dict(NT_POLICY_DEFAULT))
    confidence_threshold: float = 0.5
    low_confidence: LowConfidencePolicy = LowConfidencePolicy.EXCITATORY
    unknown: LowConfidencePolicy = LowConfidencePolicy.EXCITATORY

    def to_dict(self) -> dict:
        return {
            "signs": dict(self.signs),
            "confidence_threshold": self.confidence_threshold,
            "low_confidence": self.low_confidence.value,
            "unknown": self.unknown.value,
            "simplified_transmitters": sorted(SIMPLIFIED_TRANSMITTERS),
            "note": (
                "Dopamine/octopamine/serotonin are treated as excitatory following "
                "Shiu et al. 2024. This is a simplification, not a measurement."
            ),
        }


def assign_signs(
    neurons: pd.DataFrame,
    policy: NeurotransmitterPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each neuron's predicted transmitter to a synaptic sign.

    Parameters
    ----------
    neurons
        Must contain ``body_id``; optionally ``predicted_nt`` and ``confidence``.

    Returns
    -------
    (signs, keep) : arrays aligned with `neurons.index`
        `signs` in {-1, 0, +1}; `keep` is False for neurons the policy excludes.
    """
    policy = policy or NeurotransmitterPolicy()
    n = len(neurons)
    signs = np.zeros(n, dtype=np.int8)
    keep = np.ones(n, dtype=bool)

    nt = (
        neurons["predicted_nt"].astype("string").str.lower().fillna("unknown")
        if "predicted_nt" in neurons.columns
        else pd.Series(["unknown"] * n, index=neurons.index, dtype="string")
    )
    conf = (
        pd.to_numeric(neurons["confidence"], errors="coerce").fillna(0.0).to_numpy()
        if "confidence" in neurons.columns
        else np.zeros(n)
    )

    for i, (transmitter, c) in enumerate(zip(nt.to_numpy(), conf)):
        known = transmitter in policy.signs
        rule = policy.unknown if not known else (
            None if c >= policy.confidence_threshold else policy.low_confidence
        )
        if rule is None:
            signs[i] = policy.signs[transmitter]
        elif rule is LowConfidencePolicy.EXCITATORY:
            signs[i] = +1
        elif rule is LowConfidencePolicy.IGNORE:
            signs[i] = 0
        else:  # EXCLUDE
            signs[i] = 0
            keep[i] = False
    return signs, keep
