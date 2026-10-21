"""Classical post-processing: error correction and privacy amplification.

The missing layer. The toy protocols in `encoding.py` hand Bob a noisy string
and hand Eve a partially-informative one, and then stop. Every real quantum
protocol continues: reconcile the errors, then compress away whatever the
adversary learned. Adding this layer changes the feasibility conclusion twice
over.

1. It removes the strict-threshold penalty. Demanding BER <= 1e-3 with no
   correction costs a factor ~111 in storage time against tolerating 10% and
   correcting it (see `analysis.requirements.error_correction_gain`).
2. It converts "Eve has 0.15 bits" into "Eve has less than 2^-30 bits", which
   is the difference between a leak and a proof.

Accounting
----------
For a product encoding the adversary's optimal guessing probability factorises,
so the conditional min-entropy is exact rather than bounded:

    P_guess(X|E) = prod_i p_i        =>      H_min(X|E) = -sum_i log2(p_i)

Error correction reveals `leak_EC` bits. The leftover hash lemma then extracts

    ell = H_min(X|E) - leak_EC - 2 log2(1/eps_PA)

bits that are eps_PA-close to uniform and independent of everything Eve holds.

CAVEAT. This is the standard asymptotic-style accounting with a finite-size
correction term, not a composable finite-key security proof. It assumes
collective attacks are captured by the per-carrier guessing probability, which
holds for the independent-carrier adversaries implemented here but not in
general. It is a design tool for locating the feasible region, not a security
certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from bioquantumfly.analysis.metrics import binary_entropy

#: Efficiency of a real reconciliation code relative to the Shannon limit.
#: 1.0 is the unattainable limit; 1.16 is a routinely quoted value for LDPC
#: reconciliation in QKD implementations. Configurable, and recorded in output.
DEFAULT_EC_EFFICIENCY = 1.16

#: Failure probability of the privacy-amplification step.
DEFAULT_EPSILON_PA = 1e-10


def min_entropy_from_guessing(p_guess: ArrayLike) -> float:
    """H_min(X|E) in bits from per-carrier optimal guessing probabilities.

    Exact for a product encoding, where the adversary's best strategy on the
    whole string is her best strategy on each carrier independently.
    """
    p = np.clip(np.asarray(p_guess, dtype=float), 1e-15, 1.0)
    return float(-np.sum(np.log2(p)))


def guessing_from_min_entropy(h_min: float, n: int) -> float:
    """Inverse of the above, as a per-carrier average. For reporting only."""
    return float(2.0 ** (-h_min / max(1, n)))


def ec_leakage(n: int, ber: float, efficiency: float = DEFAULT_EC_EFFICIENCY) -> float:
    """Bits revealed during error reconciliation.

    `efficiency * n * H2(ber)`: the Shannon limit times a real code's overhead.
    Leakage counts against the extractable key whether or not the adversary
    bothers to listen, because reconciliation is public.
    """
    if not 0.0 <= ber <= 1.0:
        raise ValueError(f"ber must lie in [0, 1], got {ber}")
    return float(efficiency * n * binary_entropy(ber))


def privacy_amplification_length(
    h_min: float, leak: float, epsilon: float = DEFAULT_EPSILON_PA
) -> float:
    """Extractable secret length from the leftover hash lemma.

    Returns 0 when the budget is exhausted: a negative length means the
    adversary plus the reconciliation leakage already account for everything
    the carriers held, so no secret survives.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly in (0, 1)")
    return float(max(0.0, h_min - leak - 2.0 * np.log2(1.0 / epsilon)))


def bb84_asymptotic_rate(qber: float) -> float:
    """Asymptotic one-way BB84 key rate, r = 1 - 2 H2(e).

    An independent cross-check on the accounting above, and the origin of the
    famous ~11% threshold: beyond it the rate is zero and no amount of
    post-processing recovers a secret.
    """
    return float(max(0.0, 1.0 - 2.0 * binary_entropy(qber)))


#: The QBER above which one-way BB84 post-processing yields nothing.
BB84_THRESHOLD_QBER = 0.1100


@dataclass
class KeyBudget:
    """The full accounting for one configuration."""

    n_carriers: int
    bob_ber: float
    eve_pguess_per_carrier: float
    h_min_bits: float
    ec_leakage_bits: float
    finite_size_penalty_bits: float
    secret_bits: float
    secret_rate: float
    eve_residual_bits: float
    ec_efficiency: float
    epsilon_pa: float
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def key_budget(
    n_carriers: int,
    bob_ber: float,
    eve_pguess_per_carrier: float,
    *,
    ec_efficiency: float = DEFAULT_EC_EFFICIENCY,
    epsilon: float = DEFAULT_EPSILON_PA,
) -> KeyBudget:
    """Extractable secret length for one (storage time, attack) configuration.

    Parameters
    ----------
    eve_pguess_per_carrier
        The adversary's *optimal* per-carrier guessing probability, as computed
        by the Helstrom bound in `analysis.metrics`. Using the optimum rather
        than a specific measurement keeps the budget from flattering the scheme.
    """
    if n_carriers < 1:
        raise ValueError("n_carriers must be >= 1")
    p = float(np.clip(eve_pguess_per_carrier, 0.5, 1.0))
    h_min = min_entropy_from_guessing(np.full(n_carriers, p))
    leak = ec_leakage(n_carriers, bob_ber, ec_efficiency)
    penalty = 2.0 * np.log2(1.0 / epsilon)
    secret = privacy_amplification_length(h_min, leak, epsilon)
    return KeyBudget(
        n_carriers=n_carriers,
        bob_ber=bob_ber,
        eve_pguess_per_carrier=p,
        h_min_bits=h_min,
        ec_leakage_bits=leak,
        finite_size_penalty_bits=penalty,
        secret_bits=secret,
        secret_rate=secret / n_carriers,
        # After privacy amplification the adversary's information is bounded by
        # epsilon, not by her pre-amplification guessing advantage.
        eve_residual_bits=float(epsilon) if secret > 0 else float(h_min),
        ec_efficiency=ec_efficiency,
        epsilon_pa=epsilon,
        feasible=secret > 0,
    )


def break_even_carriers(
    bob_ber: float,
    eve_pguess_per_carrier: float,
    *,
    ec_efficiency: float = DEFAULT_EC_EFFICIENCY,
    epsilon: float = DEFAULT_EPSILON_PA,
    maximum: int = 1_000_000,
) -> int | None:
    """Smallest carrier count yielding a non-empty secret.

    The finite-size penalty is a fixed cost (about 66 bits at eps = 1e-10), so a
    short message can fail purely on overhead even when the per-carrier rate is
    positive. This says how many carriers pay for that overhead.
    """
    p = float(np.clip(eve_pguess_per_carrier, 0.5, 1.0))
    per_carrier = -np.log2(p) - ec_efficiency * binary_entropy(bob_ber)
    if per_carrier <= 0:
        return None                       # no positive rate at any length
    needed = 2.0 * np.log2(1.0 / epsilon) / per_carrier
    n = int(np.ceil(needed))
    return n if n <= maximum else None
