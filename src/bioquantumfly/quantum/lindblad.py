"""QuTiP reference solver for the finite-coherence memory.

This module exists to *validate* `quantum.channels`, not to run sweeps. Calling
`mesolve` once per carrier per time point would make the feasibility maps in
addendum section 29 unaffordable, so production code uses the analytic Bloch
backend and the tests check the two agree to numerical tolerance.

Use QuTiP directly for: general Lindblad models, multi-spin Hamiltonians, the
radical-pair system, and any coherent attack that is awkward in Bloch form.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.quantum.states import bloch_to_density, density_to_bloch


def _qutip():
    """Import QuTiP lazily so the analytic path never pays for it."""
    import qutip as qt

    return qt


def lowering_to_ground():
    """The operator |0><1|, which relaxes the qubit toward Bloch +z.

    CONVENTION WARNING. QuTiP's `sigmam()` is [[0,0],[1,0]], the spin-lowering
    operator for a convention in which `basis(2,0)` is the *upper* spin state;
    it therefore relaxes toward `basis(2,1)`, i.e. Bloch -z. This project uses
    the quantum-information convention in which |0> = basis(2,0) is the ground
    state and amplitude damping takes |1> -> |0> (Bloch -z -> +z), matching
    `quantum.channels.amplitude_damping`.

    Building the operator explicitly avoids depending on which convention a
    given QuTiP version attaches to the name `sigmam`.
    """
    qt = _qutip()
    return qt.basis(2, 0) * qt.basis(2, 1).dag()


def lindblad_operators(T1: float, T2: float):
    """Collapse operators reproducing a given (T1, T2) pair.

    Returns `[sqrt(gamma_1) |0><1|, sqrt(gamma_phi) sigma_z]`, dropping terms
    whose rate is zero. The pure-dephasing rate follows from

        1/T2 = 1/(2 T1) + 1/T_phi,      gamma_phi = 1/(2 T_phi)

    because D[Z] damps coherence at twice its own rate.
    """
    from bioquantumfly.quantum.channels import t_phi_from_t1_t2

    qt = _qutip()
    T_phi = t_phi_from_t1_t2(T1, T2)
    c_ops = []
    if np.isfinite(T1):
        c_ops.append(np.sqrt(1.0 / T1) * lowering_to_ground())
    if np.isfinite(T_phi):
        c_ops.append(np.sqrt(1.0 / (2.0 * T_phi)) * qt.sigmaz())
    return c_ops


def evolve_qutip(
    r0: ArrayLike,
    times: ArrayLike,
    T1: float,
    T2: float,
    hamiltonian=None,
) -> NDArray[np.float64]:
    """Evolve one Bloch vector with `mesolve`, returning (len(times), 3).

    Parameters
    ----------
    r0 : (3,)
        Initial Bloch vector.
    hamiltonian
        Optional QuTiP operator. `None` means the encoding basis is the memory
        eigenbasis, matching the analytic backend.
    """
    qt = _qutip()
    r0 = np.asarray(r0, dtype=np.float64)
    if r0.shape != (3,):
        raise ValueError(f"evolve_qutip handles one Bloch vector at a time, got {r0.shape}")
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))

    rho0 = qt.Qobj(bloch_to_density(r0))
    H = hamiltonian if hamiltonian is not None else qt.qzero(2)
    result = qt.mesolve(H, rho0, times, c_ops=lindblad_operators(T1, T2), e_ops=[])
    return np.array([density_to_bloch(state.full()) for state in result.states])


def evolve_qutip_many(
    r0_array: ArrayLike, times: ArrayLike, T1: float, T2: float
) -> NDArray[np.float64]:
    """Loop `evolve_qutip` over carriers. Reference implementation only -- slow."""
    r0_array = np.atleast_2d(np.asarray(r0_array, dtype=np.float64))
    return np.stack([evolve_qutip(r, times, T1, T2) for r in r0_array], axis=1)
