"""Analytic, vectorised single-qubit channels (addendum section 10).

Every channel here has a closed form on the Bloch vector, so a whole sweep over
carriers x storage times x coherence times is a few array operations. QuTiP is
reserved for validation (`quantum.lindblad`), general Lindblad models, and the
multi-spin radical-pair system. `tests/test_channels_vs_qutip.py` checks the
two backends agree.

Conventions and the T1/T2 relation
----------------------------------
The Lindblad equation modelled is

    d(rho)/dt = -i[H, rho] + gamma_1 D[sigma_-] rho + gamma_phi D[Z] rho

with D[A]rho = A rho A^dag - {A^dag A, rho}/2, and sigma_- = |0><1| taken in
the quantum-information convention where |0> is the ground state, so amplitude
damping relaxes the qubit toward |0> (Bloch +z). (NB: QuTiP's `sigmam()` uses
the opposite spin convention and relaxes toward -z; see
`quantum.lindblad.lowering_to_ground`.) Then

    rho_11(t)  = rho_11(0) e^{-t/T1}          T1 = 1/gamma_1
    |rho_01(t)| = |rho_01(0)| e^{-t/T2}

The D[Z] term damps coherence at rate 2*gamma_phi, so writing that rate as
1/T_phi gives gamma_phi = 1/(2 T_phi), and the two contributions add:

    1/T2 = 1/(2 T1) + 1/T_phi                 =>  T2 <= 2 T1

On the Bloch vector this is exactly

    x, y  ->  x, y  * exp(-t/T2)
    z     ->  1 - (1 - z) * exp(-t/T1)

No rotating-frame Hamiltonian is applied: the encoding basis is the memory
eigenbasis, so a resonant H would only add a known, invertible rotation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "t_phi_from_t1_t2", "t2_from_t1_t_phi", "validate_t1_t2",
    "amplitude_damping", "pure_dephasing", "depolarizing",
    "t1_t2_channel", "identity_channel", "coherence_factor",
]


def validate_t1_t2(T1: float, T2: float) -> None:
    """Reject the unphysical corner T2 > 2*T1.

    Raising here rather than silently clamping matters: a sweep that quietly
    changed the requested T2 would produce a feasibility boundary at the wrong
    place, which is the one number this project exists to report.
    """
    if T2 <= 0:
        raise ValueError(f"T2 must be positive, got {T2}")
    if T1 <= 0:
        raise ValueError(f"T1 must be positive (use np.inf for no relaxation), got {T1}")
    if np.isfinite(T1) and T2 > 2 * T1 * (1 + 1e-12):
        raise ValueError(
            f"T2 = {T2:g} exceeds the physical bound 2*T1 = {2 * T1:g}; "
            "pure dephasing cannot be negative"
        )


def t_phi_from_t1_t2(T1: float, T2: float) -> float:
    """Pure-dephasing time implied by a (T1, T2) pair."""
    validate_t1_t2(T1, T2)
    inv = 1.0 / T2 - (0.0 if np.isinf(T1) else 1.0 / (2.0 * T1))
    return np.inf if inv <= 0 else 1.0 / inv


def t2_from_t1_t_phi(T1: float, T_phi: float) -> float:
    inv = (0.0 if np.isinf(T1) else 1.0 / (2.0 * T1)) + (0.0 if np.isinf(T_phi) else 1.0 / T_phi)
    return np.inf if inv <= 0 else 1.0 / inv


def _decay(t: ArrayLike, tau: float) -> NDArray[np.float64]:
    """exp(-t/tau), with tau = inf meaning no decay."""
    t = np.asarray(t, dtype=np.float64)
    if np.isinf(tau):
        return np.ones_like(t)
    return np.exp(-t / tau)


def coherence_factor(t: ArrayLike, T2: float) -> NDArray[np.float64]:
    """The factor multiplying off-diagonal coherence: exp(-t/T2)."""
    return _decay(t, T2)


def identity_channel(r: ArrayLike, *_args, **_kwargs) -> NDArray[np.float64]:
    """No-op channel: the infinite-coherence control (spec section 23)."""
    return np.array(np.asarray(r, dtype=np.float64), copy=True)


def amplitude_damping(r: ArrayLike, t: ArrayLike, T1: float) -> NDArray[np.float64]:
    """Relaxation toward |0> with time constant T1.

    Bloch action: transverse components shrink as exp(-t/(2 T1)) -- the factor
    of two is why amplitude damping alone already caps T2 at 2*T1 -- while z
    relaxes toward +1.
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    e = _decay(t, T1)
    out = np.empty(np.broadcast_shapes(r.shape, e.shape), dtype=np.float64)
    out[..., 0] = r[..., 0] * np.sqrt(e[..., 0])
    out[..., 1] = r[..., 1] * np.sqrt(e[..., 0])
    out[..., 2] = 1.0 - (1.0 - r[..., 2]) * e[..., 0]
    return out


def pure_dephasing(r: ArrayLike, t: ArrayLike, T_phi: float) -> NDArray[np.float64]:
    """Coherence loss without energy relaxation: z is untouched."""
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    e = _decay(t, T_phi)
    out = np.empty(np.broadcast_shapes(r.shape, e.shape), dtype=np.float64)
    out[..., 0] = r[..., 0] * e[..., 0]
    out[..., 1] = r[..., 1] * e[..., 0]
    out[..., 2] = np.broadcast_to(r[..., 2], out[..., 2].shape)
    return out


def depolarizing(r: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """rho -> (1 - p) rho + p I/2. Shrinks the Bloch vector uniformly.

    At p = 1 the state is maximally mixed and every measurement is a coin flip,
    which is the BER -> 1/2 sanity check of spec section 22.
    """
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)[..., None]
    if np.any((p < -1e-12) | (p > 1 + 1e-12)):
        raise ValueError("depolarizing probability must lie in [0, 1]")
    return r * (1.0 - p)


def depolarizing_time(r: ArrayLike, t: ArrayLike, T_dep: float) -> NDArray[np.float64]:
    """Depolarisation with an exponential schedule p(t) = 1 - exp(-t/T_dep)."""
    t = np.asarray(t, dtype=np.float64)
    return depolarizing(r, 1.0 - _decay(t, T_dep))


def t1_t2_channel(r: ArrayLike, t: ArrayLike, T1: float, T2: float) -> NDArray[np.float64]:
    """The combined finite-coherence channel used throughout the crypto layer.

    Parameters
    ----------
    r : (..., 3)
        Bloch vectors, typically (n_carriers, 3).
    t : scalar or array
        Storage time(s). Broadcasting `t` against the carrier axis is what makes
        a full time sweep a single call.
    T1, T2
        Relaxation and coherence times; `np.inf` for either gives the
        infinite-coherence control.

    Returns
    -------
    (..., 3) array, broadcast over `r` and `t`.
    """
    validate_t1_t2(T1, T2)
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Broadcast t against the leading axes of r, keeping the Bloch axis last.
    t_b = t[..., None]
    e_coh = _decay(t_b, T2)
    e_pop = _decay(t_b, T1)

    shape = np.broadcast_shapes(r.shape, (*t.shape, 3))
    out = np.empty(shape, dtype=np.float64)
    out[..., 0] = r[..., 0] * e_coh[..., 0]
    out[..., 1] = r[..., 1] * e_coh[..., 0]
    out[..., 2] = 1.0 - (1.0 - r[..., 2]) * e_pop[..., 0]
    return out


def apply_channel_sequence(r: ArrayLike, steps: list[tuple[float, float, float]]) -> NDArray[np.float64]:
    """Apply successive (duration, T1, T2) segments.

    Used when Eve interrupts storage: the host evolves to t_E, is attacked, then
    continues to the read time under the same physics.
    """
    out = np.asarray(r, dtype=np.float64)
    for duration, T1, T2 in steps:
        out = t1_t2_channel(out, duration, T1, T2)
    return out
