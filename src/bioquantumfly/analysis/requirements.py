"""Inverting the feasibility question: what would have to be true?

The forward sweeps ask "given T2, what works?". This module asks the useful
converse -- "to reach a target storage window, what T2 is required?" -- because
that is the number that tells you whether the gap is an engineering problem or
a physical wall.

The closed form. For masked conjugate encoding with uniform bits and bases and
T1 = T2, the per-bit error rate has an exact expression. A Z-encoded 0 relaxes
*toward* |0> and can never be wrong; a Z-encoded 1 fails with probability
1 - e^{-t/T1}; an X-encoded bit fails with probability (1 - e^{-t/T2})/2. So

    BER(t) = (1/4)(1 - e^{-t/T1}) + (1/4)(1 - e^{-t/T2})
           = (1/2)(1 - e^{-t/T2})            when T1 = T2

which inverts to

    t / T2 = -ln(1 - 2 * BER)

verified against the simulator to 3 significant figures in
`tests/test_requirements.py`.

The consequence that matters: the usable window is *linear* in the tolerated
error rate for small BER. Demanding BER <= 1e-3 across a whole message with no
error correction costs a factor of ~111 in storage time compared with tolerating
10% and correcting it. Reporting only the strict threshold therefore overstates
the physical gap by two orders of magnitude, which is why this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Timescales the biology actually operates on, for comparison.
NEURAL_TIMESCALES: dict[str, float] = {
    "optical_readout": 1e-5,      # a fast optical read of a spin state
    "single_spike": 1e-3,
    "synaptic_integration": 5e-3,
    "behavioural": 1e-1,
}

#: Order-of-magnitude coherence times for scale. NOTE the systems: only the two
#: EYFP entries are the ones this project cites as constraints. The NV numbers
#: are DIAMOND, not protein, and are included solely to show what a
#: nuclear-spin memory architecture achieves -- they are not a claim that a
#: protein could do the same. Treat as unverified order-of-magnitude context
#: (see docs/sources.md "Parameters that are deliberately NOT sourced").
REFERENCE_COHERENCE_TIMES: dict[str, tuple[float, str]] = {
    "eyfp_room_temperature": (100e-9, "measured, EYFP, room T (Feder et al. 2025)"),
    "eyfp_cryogenic": (16e-6, "measured, EYFP, cryogenic, CPMG (Feder et al. 2025)"),
    "nv_electron_room_temperature": (1e-3, "context only: NV electron spin in diamond"),
    "nv_nuclear_room_temperature": (1.0, "context only: NV nuclear spin in diamond"),
}


def ber_at(t_over_T2: ArrayLike, t1_over_t2: float = 1.0) -> NDArray[np.float64]:
    """Closed-form bit-error rate at a normalised storage time."""
    x = np.asarray(t_over_T2, dtype=float)
    return 0.25 * (1.0 - np.exp(-x / t1_over_t2)) + 0.25 * (1.0 - np.exp(-x))


def usable_window_over_T2(max_ber: float) -> float:
    """Normalised storage time at which the error rate reaches `max_ber`.

    Inverse of `ber_at` for T1 = T2. Undefined at BER >= 1/2, which is the
    coin-flip floor the channel asymptotes to.
    """
    if not 0.0 < max_ber < 0.5:
        raise ValueError(f"max_ber must lie strictly in (0, 0.5), got {max_ber}")
    return float(-np.log(1.0 - 2.0 * max_ber))


def required_T2(window_seconds: float, max_ber: float) -> float:
    """Coherence time needed to keep the error below `max_ber` for that long."""
    if window_seconds <= 0:
        raise ValueError("window must be positive")
    return window_seconds / usable_window_over_T2(max_ber)


def binary_entropy(p: ArrayLike) -> NDArray[np.float64]:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= 0) | (p >= 1), 0.0, h)


def code_rate(ber: float) -> float:
    """Shannon-limit rate of a classical code correcting a BSC at this error rate.

    The fraction of carriers that survive as reliable bits. Tolerating a higher
    error rate buys storage time but costs rate, and this is the exchange rate.
    """
    return float(max(0.0, 1.0 - binary_entropy(ber)))


@dataclass(frozen=True)
class Requirement:
    """What a given target and error budget demands of the physics."""

    target: str
    window_seconds: float
    max_ber: float
    required_T2_seconds: float
    code_rate: float
    effective_bits_per_carrier: float

    def gap_from(self, t2_seconds: float) -> float:
        """How many times longer the coherence would have to be."""
        return self.required_T2_seconds / t2_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "window_seconds": self.window_seconds,
            "max_ber": self.max_ber,
            "required_T2_seconds": self.required_T2_seconds,
            "code_rate": self.code_rate,
            "effective_bits_per_carrier": self.effective_bits_per_carrier,
        }


def requirement(target: str, window_seconds: float, max_ber: float) -> Requirement:
    rate = code_rate(max_ber)
    return Requirement(
        target=target,
        window_seconds=window_seconds,
        max_ber=max_ber,
        required_T2_seconds=required_T2(window_seconds, max_ber),
        code_rate=rate,
        effective_bits_per_carrier=rate,
    )


def requirement_table(
    targets: dict[str, float] | None = None,
    ber_budgets: tuple[float, ...] = (1e-3, 1e-2, 0.05, 0.10),
) -> list[dict[str, Any]]:
    """Required T2 for every (target window, error budget) pair.

    Each row also records the gap to each reference coherence time, so the
    question "is this an engineering problem or a wall?" is answered by
    inspection rather than by assertion.
    """
    targets = targets or NEURAL_TIMESCALES
    rows: list[dict[str, Any]] = []
    for name, window in targets.items():
        for ber in ber_budgets:
            req = requirement(name, window, ber)
            row = req.to_dict()
            for system, (t2, note) in REFERENCE_COHERENCE_TIMES.items():
                row[f"gap_from_{system}"] = req.gap_from(t2)
                row[f"orders_from_{system}"] = float(np.log10(req.gap_from(t2)))
                row[f"reachable_from_{system}"] = bool(t2 >= req.required_T2_seconds)
            rows.append(row)
    return rows


def error_correction_gain(strict_ber: float = 1e-3, tolerant_ber: float = 0.10) -> dict[str, float]:
    """How much storage time error correction buys, and what it costs.

    This is the single number that decides whether the reported gap is physics
    or bookkeeping.
    """
    strict = usable_window_over_T2(strict_ber)
    tolerant = usable_window_over_T2(tolerant_ber)
    return {
        "strict_ber": strict_ber,
        "tolerant_ber": tolerant_ber,
        "strict_window_over_T2": strict,
        "tolerant_window_over_T2": tolerant,
        "storage_time_gain": tolerant / strict,
        "code_rate_at_tolerant": code_rate(tolerant_ber),
        "net_bit_seconds_gain": (tolerant / strict) * code_rate(tolerant_ber),
    }
