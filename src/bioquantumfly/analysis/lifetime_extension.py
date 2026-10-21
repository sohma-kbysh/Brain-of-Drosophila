"""Ways to extend the storage lifetime, and what each one actually demands.

The feasibility analysis ends with a coherence shortfall. This module asks the
engineering question that follows: which known techniques could close it, by how
much, and on what condition. Each technique is reduced to a factor and a
prerequisite, because "use error correction" is not an answer until one says
what the error correction needs.

Four techniques, in increasing ambition:

1. Dynamical decoupling -- capped at T2 <= 2 T1. Already spent in the 16 us
   cryogenic figure, and bounded above by the population lifetime at room T.
2. Nuclear-spin storage -- move the state from the electron (the interface) to
   a nucleus (the memory). Nuclear moments are ~658x smaller.
3. Quantum error correction -- the only technique that exceeds the physical
   coherence time without changing the qubit. It multiplies an already-good
   qubit; it cannot rescue one that starts above threshold.
4. Redundant (ensemble) storage -- longer-lived in practice, but it destroys the
   cryptographic property outright. Included because it is the obvious move and
   the reason it fails is instructive.

NUMBERS. Gyromagnetic ratios are CODATA constants. The surface-code threshold
and prefactor are the standard circuit-level heuristic from the QEC literature,
not a derivation. Readout times and nuclear coherence in proteins are
order-of-magnitude context, flagged as such at each use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Any

import numpy as np

from bioquantumfly.analysis.requirements import usable_window_over_T2

#: CODATA gyromagnetic ratios, rad s^-1 T^-1.
GAMMA_ELECTRON = 1.76085963e11
GAMMA_PROTON = 2.6752218744e8
GYRO_RATIO = GAMMA_ELECTRON / GAMMA_PROTON          # ~658

#: Standard circuit-level surface-code heuristic: p_L ~ A (p/p_th)^((d+1)/2).
SURFACE_CODE_THRESHOLD = 1e-2
SURFACE_CODE_PREFACTOR = 0.1


# ---------------------------------------------------------------------------
# 1. Dynamical decoupling
# ---------------------------------------------------------------------------
def dd_ceiling(T1: float) -> float:
    """The largest T2 any decoupling sequence can reach: 2 T1.

    Decoupling refocuses dephasing; it cannot undo population relaxation. The
    room-temperature EYFP figure (polarisation gone within ~100 ns) is a
    population lifetime, so if it reflects relaxation or triplet decay it caps
    decoupled coherence near 200 ns -- a factor of two, not an order of
    magnitude.
    """
    if T1 <= 0:
        raise ValueError("T1 must be positive")
    return 2.0 * T1


# ---------------------------------------------------------------------------
# 2. Nuclear-spin storage
# ---------------------------------------------------------------------------
def nuclear_coherence_range(T2_electron: float) -> dict[str, float]:
    """Nuclear T2 IF both spins were limited by the same magnetic noise.

    Dephasing scales with the gyromagnetic ratio in two limiting regimes:

        quasi-static noise:   T2 ~ 1/gamma      -> nucleus gains  gamma_e/gamma_p ~ 658
        fast (narrowed) noise: T2 ~ 1/gamma^2   -> nucleus gains ~658^2 ~ 4.3e5

    A bracket, not a prediction. If the electron's lifetime is set by triplet
    decay or spin-orbit relaxation -- channels a nucleus does not share -- this
    argument does not transfer at all, and the nuclear T2 is set instead by the
    protein's own NMR relaxation (order milliseconds for large proteins in
    solution; context, not a cited measurement).
    """
    return {
        "quasi_static": T2_electron * GYRO_RATIO,
        "motional_narrowing": T2_electron * GYRO_RATIO**2,
        "gyro_ratio": GYRO_RATIO,
    }


# ---------------------------------------------------------------------------
# 3. Quantum error correction
# ---------------------------------------------------------------------------
def idle_error_per_cycle(cycle_time: float, T2: float) -> float:
    """Pauli error accrued while idling for one syndrome cycle.

    The same closed form as the storage BER, (1 - e^{-tau/T2})/2 ~ tau/(2 T2).
    """
    return float(0.5 * (1.0 - np.exp(-cycle_time / T2)))


def repetition_code_logical_error(p: float, d: int) -> float:
    """Exact code-capacity logical error of a distance-d repetition code.

    Majority vote fails when more than half the qubits flip. Protects against
    ONE error type only -- useless on its own for conjugate coding, where half
    the carriers are X-basis and half Z-basis and both must survive. Included
    because it is exact and shows the exponential suppression cleanly.
    """
    if d < 1 or d % 2 == 0:
        raise ValueError("distance must be a positive odd integer")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return float(sum(comb(d, k) * p**k * (1 - p) ** (d - k)
                     for k in range(d // 2 + 1, d + 1)))


def surface_code_logical_error(
    p: float, d: int,
    threshold: float = SURFACE_CODE_THRESHOLD,
    prefactor: float = SURFACE_CODE_PREFACTOR,
) -> float:
    """Heuristic circuit-level surface-code logical error per cycle.

    Protects both error types, which conjugate coding needs. Above threshold
    the formula exceeds the physical rate and the code makes things worse --
    returned uncapped so that regime is visible rather than hidden.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError("surface-code distance must be an odd integer >= 3")
    return float(prefactor * (p / threshold) ** ((d + 1) / 2))


def surface_code_qubits(d: int) -> int:
    """Physical qubits per logical qubit (data + measurement), 2 d^2 - 1."""
    return 2 * d * d - 1


def min_T2_for_threshold(
    cycle_time: float, gate_error: float, threshold: float = SURFACE_CODE_THRESHOLD,
    margin: float = 1.0,
) -> float:
    """Smallest physical T2 at which the code is below threshold.

    Requires idle + gate error < threshold / margin. With margin = 10 the code is
    comfortably below threshold and suppression is useful rather than marginal.
    Returns infinity when gate error alone already exceeds the budget: no
    coherence time rescues bad gates.
    """
    budget = threshold / margin - gate_error
    if budget <= 0:
        return float("inf")
    if budget >= 0.5:
        return 0.0
    return float(-cycle_time / np.log(1.0 - 2.0 * budget))


@dataclass
class QECResult:
    T2_physical: float
    cycle_time: float
    gate_error: float
    distance: int
    physical_error_per_cycle: float
    logical_error_per_cycle: float
    below_threshold: bool
    logical_T2: float
    lifetime_gain: float
    physical_qubits_per_logical: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qec_lifetime(T2: float, cycle_time: float, gate_error: float, distance: int) -> QECResult:
    """Logical coherence under a surface code with periodic syndrome extraction.

    logical_T2 ~ cycle_time / p_L. Compared against the bare T2, the ratio is the
    lifetime gain -- which is BELOW one when the code is above threshold.
    """
    p = idle_error_per_cycle(cycle_time, T2) + gate_error
    p_logical = surface_code_logical_error(p, distance)
    below = p < SURFACE_CODE_THRESHOLD
    logical_T2 = cycle_time / p_logical if p_logical > 0 else float("inf")
    return QECResult(
        T2_physical=T2,
        cycle_time=cycle_time,
        gate_error=gate_error,
        distance=distance,
        physical_error_per_cycle=p,
        logical_error_per_cycle=p_logical,
        below_threshold=below,
        logical_T2=logical_T2,
        lifetime_gain=logical_T2 / T2,
        physical_qubits_per_logical=surface_code_qubits(distance),
    )


# ---------------------------------------------------------------------------
# 4. Redundant storage -- why the obvious fix breaks the cryptography
# ---------------------------------------------------------------------------
def ensemble_identification_probability(n_copies: int) -> float:
    """Probability an adversary identifies an unknown BB84 state from N copies.

    Strategy: measure half the copies in Z and half in X. In the matching basis
    every outcome is identical; in the other, outcomes are uniform, so "all
    identical" happens by chance with probability 2^{-(m-1)} for m copies. She
    fails only when both halves look uniform-or-identical ambiguously and she
    guesses wrong.

    The point: one copy gives the no-cloning protection; N copies give the
    adversary a classical signal. Storing redundantly to extend lifetime --
    the standard move in ensemble NMR -- hands her the state, and she can take
    spare copies without touching the one Bob reads, so the theft that
    Protocol C detects becomes invisible.
    """
    if n_copies < 1:
        raise ValueError("need at least one copy")
    if n_copies == 1:
        # Single copy: the optimal unknown-basis guess (Breidbart bound).
        return float(np.cos(np.pi / 8) ** 2)
    m_z = n_copies // 2
    m_x = n_copies - m_z
    # In the wrong basis, a half of size m looks deterministic with prob 2^{1-m}.
    fake_z = 2.0 ** (1 - m_z) if m_z >= 1 else 1.0
    fake_x = 2.0 ** (1 - m_x) if m_x >= 1 else 1.0
    # True basis Z (prob 1/2): the X half is spurious-deterministic with prob fake_x,
    # creating a tie she breaks at random. Symmetrically for true basis X.
    p_tie = 0.5 * fake_x + 0.5 * fake_z
    return float(1.0 - 0.5 * p_tie)


# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------
def extension_roadmap(
    T2_start: float = 100e-9,
    target_window: float = 1e-3,
    tolerated_ber: float = 0.10,
) -> list[dict[str, Any]]:
    """Every technique, its factor, its prerequisite and whether it reaches the target."""
    needed = target_window / usable_window_over_T2(tolerated_ber)
    nuclear = nuclear_coherence_range(T2_start)
    rows = [
        {"technique": "classical error correction (done)",
         "factor": usable_window_over_T2(tolerated_ber) / usable_window_over_T2(1e-3),
         "prerequisite": "none beyond post-processing", "status": "implemented"},
        {"technique": "dynamical decoupling",
         "factor": 2.0,
         "prerequisite": "microwave control in vivo; capped at 2 T1",
         "status": "cannot exceed the population lifetime"},
        {"technique": "nuclear storage, quasi-static noise",
         "factor": nuclear["quasi_static"] / T2_start,
         "prerequisite": "coherent electron-to-nucleus transfer, single-shot nuclear readout",
         "status": "not demonstrated in vivo"},
        {"technique": "nuclear storage, narrowed noise",
         "factor": nuclear["motional_narrowing"] / T2_start,
         "prerequisite": "as above",
         "status": "not demonstrated in vivo"},
        {"technique": "surface-code QEC",
         "factor": float("nan"),
         "prerequisite": "T2 >> cycle time, gates below 1% error, "
                         "single-shot syndrome readout, ~100 qubits per logical",
         "status": "needs a qubit that is already good; see qec sweep"},
        {"technique": "redundant ensemble storage",
         "factor": float("nan"),
         "prerequisite": "none -- it is easy",
         "status": "BREAKS the cryptography: N copies are a classical signal"},
    ]
    for row in rows:
        f = row["factor"]
        row["T2_after"] = T2_start * f if np.isfinite(f) else float("nan")
        row["reaches_target"] = bool(np.isfinite(f) and T2_start * f >= needed)
        row["required_T2"] = needed
    return rows
