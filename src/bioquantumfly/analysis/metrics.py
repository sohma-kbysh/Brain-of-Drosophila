"""Quantum-information primitives.

All of these are closed-form for qubits, and all are vectorised over a leading
carrier axis, because the security metrics are evaluated at every point of a
large parameter sweep.

A note on two distinctions this project keeps carefully (spec section 13):

* Holevo information is an *upper bound* on accessible information, not the
  accessible information itself. `holevo_chi` and
  `mutual_information_from_joint` are reported separately and never conflated.
* `fidelity` here is the squared convention F = (Tr|sqrt(rho) sqrt(sigma)|)^2,
  so F = 1 for identical states. QuTiP's `fidelity()` returns the square root
  of this; the tests check the relationship explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = 1e-12


def trace_distance(r1: ArrayLike, r2: ArrayLike) -> NDArray[np.float64]:
    """D(rho, sigma) = ||rho - sigma||_1 / 2.

    For qubits the eigenvalues of rho - sigma are +/- |dr|/2, so the trace
    distance is exactly half the Euclidean Bloch distance.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    return 0.5 * np.linalg.norm(r1 - r2, axis=-1)


def fidelity(r1: ArrayLike, r2: ArrayLike) -> NDArray[np.float64]:
    """Squared state fidelity F(rho, sigma) = (Tr|sqrt(rho)sqrt(sigma)|)^2.

    Closed form on Bloch vectors:
        F = [1 + r1.r2 + sqrt((1 - |r1|^2)(1 - |r2|^2))] / 2
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    n1 = np.sum(r1 * r1, axis=-1)
    n2 = np.sum(r2 * r2, axis=-1)
    dot = np.sum(r1 * r2, axis=-1)
    cross = np.sqrt(np.clip((1.0 - n1) * (1.0 - n2), 0.0, None))
    return np.clip(0.5 * (1.0 + dot + cross), 0.0, 1.0)


def helstrom_guessing_probability(
    r0: ArrayLike, r1: ArrayLike, prior: float = 0.5
) -> NDArray[np.float64]:
    """Optimal probability of correctly distinguishing two qubit states.

    The Helstrom bound for hypotheses (p0, rho0) and (p1, rho1) is

        P = [1 + || p0 rho0 - p1 rho1 ||_1] / 2

    and for qubits the trace norm evaluates in closed form: writing
    a = p0 - p1 and L = |p0 r0 - p1 r1|, the eigenvalues of the difference are
    (a +/- L)/2, so ||.||_1 = max(|a|, L). With equal priors a = 0 and this
    reduces to 1/2 + D(rho0, rho1)/2.

    Reporting the *optimum* rather than the outcome of one particular
    measurement keeps Eve's power from being understated. Because a stored
    classical outcome is a diagonal state, the same formula also gives the
    optimal guess of an intercept-resend adversary.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must lie in [0, 1]")
    r0 = np.asarray(r0, dtype=np.float64)
    r1 = np.asarray(r1, dtype=np.float64)
    p0, p1 = prior, 1.0 - prior
    a = abs(p0 - p1)
    lam = np.linalg.norm(p0 * r0 - p1 * r1, axis=-1)
    return np.clip(0.5 * (1.0 + np.maximum(a, lam)), 0.0, 1.0)


def binary_entropy(p: ArrayLike) -> NDArray[np.float64]:
    """H_2(p) in bits, using the 0 log 0 = 0 convention."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return np.where((p <= _EPS) | (p >= 1.0 - _EPS), 0.0, terms)


def von_neumann_entropy(r: ArrayLike) -> NDArray[np.float64]:
    """S(rho) in bits. Eigenvalues of a qubit are (1 +/- |r|)/2."""
    r = np.asarray(r, dtype=np.float64)
    norm = np.clip(np.linalg.norm(r, axis=-1), 0.0, 1.0)
    return binary_entropy(0.5 * (1.0 + norm))


def holevo_chi(states: ArrayLike, probabilities: ArrayLike | None = None) -> NDArray[np.float64]:
    """Holevo quantity chi = S(sum_m p_m rho_m) - sum_m p_m S(rho_m).

    Parameters
    ----------
    states : (..., n_messages, 3)
        Bloch vectors of the ensemble members, message index second-to-last.
    probabilities : (n_messages,), optional
        Defaults to uniform.

    Returns
    -------
    chi in bits. This UPPER BOUNDS the accessible information; it is not itself
    the information Eve can extract with a concrete measurement.
    """
    states = np.asarray(states, dtype=np.float64)
    n_msg = states.shape[-2]
    p = np.full(n_msg, 1.0 / n_msg) if probabilities is None else np.asarray(probabilities, float)
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("ensemble probabilities must sum to 1")
    avg = np.tensordot(p, states, axes=([0], [states.ndim - 2]))
    s_avg = von_neumann_entropy(avg)
    s_parts = von_neumann_entropy(states)
    return np.clip(s_avg - np.tensordot(p, s_parts, axes=([0], [s_parts.ndim - 1])), 0.0, None)


def mutual_information_from_joint(joint: ArrayLike) -> float:
    """I(X;Y) in bits from a joint probability table.

    This is a *classical* mutual information over actual measurement outcomes,
    i.e. an achieved accessible information for the measurement that produced
    the outcomes -- not a bound.
    """
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2:
        raise ValueError("joint distribution must be 2-D")
    total = joint.sum()
    if not np.isclose(total, 1.0):
        if total <= 0:
            raise ValueError("joint distribution has non-positive mass")
        joint = joint / total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    mask = (joint > _EPS) & (denom > _EPS)
    return float(np.sum(joint[mask] * np.log2(joint[mask] / denom[mask])))


def mutual_information_binary(p_correct: ArrayLike) -> NDArray[np.float64]:
    """I(M;G) in bits for a uniform bit sent through a binary symmetric channel.

    A guess correct with probability p carries I = 1 - H_2(p) bits.
    """
    return 1.0 - binary_entropy(p_correct)


def bit_error_rate(p_correct_per_carrier: ArrayLike) -> float:
    """Mean per-carrier error probability."""
    return float(np.mean(1.0 - np.asarray(p_correct_per_carrier, dtype=np.float64)))


def exact_recovery_probability(p_correct_per_carrier: ArrayLike) -> float:
    """Pr[every carrier correct] = product of per-carrier probabilities.

    Reported alongside BER because for large n exact recovery collapses even
    when the per-bit error rate is small (spec section 13).
    """
    p = np.asarray(p_correct_per_carrier, dtype=np.float64)
    return float(np.exp(np.sum(np.log(np.clip(p, _EPS, 1.0)))))
