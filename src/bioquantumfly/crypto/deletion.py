"""Certified deletion: an actual certificate, not a hope that it decayed.

Motivated directly by this project's central negative result. Natural host
decoherence destroys the host's copy and certifies nothing: it produces no
evidence, says nothing about copies the adversary already took, and is timed by
the environment rather than by a protocol participant. Certified deletion, in
the sense of Broadbent and Islam (TCC 2020), fixes exactly that gap by making
deletion an *action* whose result can be checked.

The mechanism implemented here (an operational version, NOT their scheme or its
security proof):

    Alice prepares n carriers in bases theta. Positions with theta_i = 0 carry
    the masked message; positions with theta_i = 1 carry random CHECK bits.

    To delete, the holder measures EVERY carrier in the Hadamard basis and
    returns the outcomes y.

    Alice accepts if y matches the encoded bits on the check positions.

Why it works. A check position was prepared as an X eigenstate, so an X-basis
measurement reproduces it deterministically. A message position was prepared as
a Z eigenstate, so the same measurement yields a uniformly random bit and
destroys the stored value. Honest deletion therefore proves itself and is
irreversible in the same stroke.

Why waiting does not work, and this is the point: after a storage time t the
check positions have decohered, an X measurement returns the prepared value
only with probability (1 + e^{-t/T2})/2, and the certificate stops verifying.
You cannot certify deletion by letting the state rot -- you have to measure it
while it is still coherent. `certification_window` quantifies how long that
leaves.

NOT a security proof. This is a simulation of the mechanism and its operational
statistics. It does not reproduce the Broadbent-Islam analysis and must not be
described as achieving their guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.crypto.encoding import Encoding
from bioquantumfly.quantum.channels import t1_t2_channel
from bioquantumfly.quantum.states import BB84_AXES, encode_bb84, measure_probability

#: Bases: 0 = computational (message), 1 = Hadamard (check).
MESSAGE_BASIS = 0
CHECK_BASIS = 1


@dataclass(frozen=True)
class DeletionCertificate:
    """Measurement outcomes returned by whoever claims to have deleted."""

    outcomes: NDArray[np.int64]
    origin: str = "honest"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.outcomes.size)


@dataclass
class VerificationResult:
    accepted: bool
    mismatch_fraction: float
    n_check_positions: int
    n_mismatches: int
    tolerance: float
    origin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mismatch_fraction": self.mismatch_fraction,
            "n_check_positions": self.n_check_positions,
            "n_mismatches": self.n_mismatches,
            "tolerance": self.tolerance,
            "certificate_origin": self.origin,
        }


def check_positions(encoding: Encoding) -> NDArray[np.int64]:
    """Indices prepared in the Hadamard basis. These carry the certificate."""
    return np.flatnonzero(encoding.bases == CHECK_BASIS)


def message_positions(encoding: Encoding) -> NDArray[np.int64]:
    return np.flatnonzero(encoding.bases == MESSAGE_BASIS)


def hadamard_measurement_probabilities(state: ArrayLike) -> NDArray[np.float64]:
    """Probability of outcome 0 when measuring every carrier along X."""
    return measure_probability(np.asarray(state, dtype=float), BB84_AXES[CHECK_BASIS])


def honest_deletion(
    state: ArrayLike, rng: np.random.Generator
) -> DeletionCertificate:
    """Measure every carrier in the Hadamard basis and return the outcomes.

    This is the deletion action itself: it is what destroys the computational
    values, and the outcomes are the evidence that it happened.
    """
    p_zero = hadamard_measurement_probabilities(state)
    outcomes = (rng.random(p_zero.shape) >= p_zero).astype(np.int64)
    return DeletionCertificate(outcomes, origin="honest")


def cheating_deletion(
    encoding: Encoding, rng: np.random.Generator, *, kept_message: bool = True
) -> DeletionCertificate:
    """A certificate from a holder who kept the message instead of deleting.

    Having measured the carriers in the computational basis to read the message,
    the cheater has destroyed the Hadamard-basis correlations. Any certificate
    they now produce is a coin flip on every check position, so it fails
    verification with probability 1 - 2^{-n_check} for a zero tolerance.
    """
    outcomes = rng.integers(0, 2, size=encoding.n_carriers, dtype=np.int64)
    return DeletionCertificate(
        outcomes, origin="cheating_kept_message",
        metadata={"kept_message": kept_message,
                  "note": "measured in the computational basis, so the Hadamard "
                          "correlations are gone and the certificate is guesswork"},
    )


def decay_certificate(
    encoding: Encoding, storage_time: float, T1: float, T2: float,
    rng: np.random.Generator,
) -> DeletionCertificate:
    """What 'waiting for it to decay' can produce, if asked for a certificate.

    Included to make the contrast concrete: the state is measured in the
    Hadamard basis only AFTER decohering for `storage_time`, so the check
    positions have already lost the correlation the certificate depends on.
    Nothing about the decay itself produces evidence.
    """
    decayed = t1_t2_channel(encoding.initial_state(), storage_time, T1, T2)
    certificate = honest_deletion(decayed, rng)
    return DeletionCertificate(
        certificate.outcomes, origin="natural_decay",
        metadata={"storage_time": storage_time, "T2": T2,
                  "note": "decoherence produces no certificate of its own; this is "
                          "a late measurement of an already-decayed state"},
    )


def verify(
    encoding: Encoding, certificate: DeletionCertificate, *, tolerance: float = 0.0
) -> VerificationResult:
    """Check a certificate against the Hadamard-prepared positions.

    Parameters
    ----------
    tolerance
        Permitted fraction of mismatched check positions. It must exceed the
        honest party's own noise or honest deletions are rejected, and stay far
        enough below 1/2 that a guesser is caught. `certification_window`
        computes when such a tolerance exists.
    """
    idx = check_positions(encoding)
    if idx.size == 0:
        raise ValueError(
            "no Hadamard-basis positions: this encoding carries no check bits, "
            "so deletion cannot be certified"
        )
    expected = encoding.encoded_bits[idx]
    observed = np.asarray(certificate.outcomes, dtype=np.int64)[idx]
    mismatches = int(np.count_nonzero(expected != observed))
    fraction = mismatches / idx.size
    return VerificationResult(
        accepted=fraction <= tolerance + 1e-12,
        mismatch_fraction=fraction,
        n_check_positions=int(idx.size),
        n_mismatches=mismatches,
        tolerance=tolerance,
        origin=certificate.origin,
    )


def honest_mismatch_rate(storage_time: float, T2: float) -> float:
    """Expected check-position error rate for an honest deleter at time t.

    A check qubit prepared along +/- x has Bloch component +/- e^{-t/T2} at
    time t, so the X measurement reproduces it with probability
    (1 + e^{-t/T2})/2 and errs with the complement.
    """
    return float(0.5 * (1.0 - np.exp(-storage_time / T2)))


def partial_cheating_deletion(
    encoding: Encoding, state: ArrayLike, fraction: float, rng: np.random.Generator
) -> DeletionCertificate:
    """A cheater who steals only a FRACTION of the carriers.

    The realistic adversary, and the one that sets the true certification
    window. She measures a fraction `f` of positions in the computational basis
    -- learning those message bits and destroying their Hadamard correlation --
    and deletes the rest honestly. She does not know which positions are checks,
    so her damage lands on the check set in proportion, giving an expected
    mismatch of roughly f/2 instead of the full cheater's 1/2. A tolerance loose
    enough to admit a noisy honest deleter is therefore also loose enough to
    admit her.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction}")
    state = np.asarray(state, dtype=float)
    n = encoding.n_carriers
    stolen = rng.random(n) < fraction

    honest = honest_deletion(state, rng).outcomes
    guessed = rng.integers(0, 2, size=n, dtype=np.int64)
    outcomes = np.where(stolen, guessed, honest)
    return DeletionCertificate(
        outcomes, origin=f"partial_cheat_{fraction:g}",
        metadata={"stolen_fraction": float(fraction),
                  "n_stolen": int(stolen.sum()),
                  "note": "computational-basis measurement on the stolen positions; "
                          "honest Hadamard deletion elsewhere"},
    )


def max_undetected_theft(
    tolerance: float, storage_time: float, T2: float
) -> float:
    """Fraction of carriers a cheater can steal and still pass verification.

    A cheater stealing fraction f has expected check mismatch

        f/2 + (1 - f) q(t),     q(t) = (1 - e^{-t/T2})/2

    so passing a tolerance `tau` requires

        f <= (tau - q) / (1/2 - q)

    Every bit of slack between the tolerance and the honest noise converts
    directly into stolen message. At t = 0 the honest deleter is exact, the
    tolerance can be zero, and no theft survives -- which is why the useful
    certification window is set by how fast q(t) grows, not by how well an
    all-or-nothing cheater can be caught.
    """
    q = honest_mismatch_rate(storage_time, T2)
    slack = tolerance - q
    headroom = 0.5 - q
    if headroom <= 0:
        return 1.0
    return float(np.clip(slack / headroom, 0.0, 1.0))


def certification_window(
    n_check: int,
    T2: float,
    *,
    max_theft_fraction: float = 0.01,
    confidence_sigma: float = 3.0,
    max_t_over_T2: float = 20.0,
    resolution: int = 4000,
) -> dict[str, Any]:
    """How long deletion stays certifiable against a PARTIAL cheater.

    The tolerance cannot be tighter than the honest deleter's own noise plus a
    few standard deviations, or honest deletions get rejected. That mandatory
    slack is exactly what a partial cheater exploits, so the window ends when

        max_undetected_theft(tolerance(t), t) > max_theft_fraction

    Solved numerically because the two effects move in opposite directions: the
    binomial spread shrinks with more check positions, while the honest error
    rate and hence the required slack grow with time.

    An earlier version of this function compared only an honest deleter against
    an all-or-nothing cheater. That comparison is far too generous: it reported
    a window of ~2.6 T2 at 10,000 check positions, where the tolerance needed is
    0.48 and a partial cheater could walk off with most of the message.
    """
    if n_check < 1:
        raise ValueError("need at least one check position")
    if not 0.0 < max_theft_fraction < 1.0:
        raise ValueError("max_theft_fraction must lie strictly in (0, 1)")

    grid = np.linspace(0.0, max_t_over_T2, resolution)
    window = 0.0
    tolerance_at_window = 0.0
    for x in grid:
        q = honest_mismatch_rate(x * T2, T2)
        # Accept an honest deleter: tolerance = mean + k sigma of Binomial(n, q).
        sigma = np.sqrt(max(q * (1.0 - q), 1e-18) / n_check)
        tolerance = q + confidence_sigma * sigma
        if tolerance >= 0.5:
            break
        if max_undetected_theft(tolerance, x * T2, T2) > max_theft_fraction:
            break
        window, tolerance_at_window = float(x), float(tolerance)

    return {
        "certifiable": window > 0.0,
        "n_check": n_check,
        "max_theft_fraction": max_theft_fraction,
        "confidence_sigma": confidence_sigma,
        "window_over_T2": window,
        "window_seconds": window * T2,
        "recommended_tolerance": tolerance_at_window,
        "honest_mismatch_at_window": honest_mismatch_rate(window * T2, T2),
        "note": (
            "Window against a partial cheater, not merely an all-or-nothing one. "
            "The tolerance must admit honest noise, and that slack is what a "
            "partial cheater exploits."
        ),
    }


def residual_message_information(
    encoding: Encoding, deleted: bool
) -> float:
    """Bits about the message still recoverable from the carriers.

    After an honest Hadamard-basis measurement the computational values are
    gone, so a key-holder recovers nothing beyond the pad. Reported as a bit
    count over the message positions.
    """
    n_message = int(message_positions(encoding).size)
    return 0.0 if deleted else float(n_message)
