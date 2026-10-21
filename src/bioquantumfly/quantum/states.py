"""Single-qubit states in the Bloch representation.

Why Bloch vectors rather than density matrices everywhere: the cryptographic
encoding is a *product* state over n carriers, so each carrier evolves
independently. Representing n carriers as an (n, 3) real array lets an entire
parameter sweep -- carriers x times x T2 values -- collapse into a handful of
vectorised NumPy operations, which is what makes the feasibility maps in
addendum section 29 affordable. A 2**n statevector is never constructed.

Convention
----------
    rho = (I + r . sigma) / 2,     r = (x, y, z),  |r| <= 1

so |0> is r = (0, 0, +1) and |1> is r = (0, 0, -1).

Density matrices are still available (`bloch_to_density`) for the joint
system-ancilla attacks in `crypto.eve`, where a 4x4 matrix is unavoidable, and
for cross-checks against QuTiP in the tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Pauli matrices, in the order (X, Y, Z) matching the Bloch components.
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)
PAULIS = np.stack([PAULI_X, PAULI_Y, PAULI_Z])

#: Measurement axes for the two BB84 bases. Index by basis bit theta.
#: theta = 0 -> computational (Z); theta = 1 -> Hadamard/diagonal (X).
BB84_AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

#: The Breidbart axis: the equal superposition of the two BB84 measurement
#: directions, (X + Z)/sqrt(2). This is the single measurement that best
#: discriminates the four BB84 states when the basis is unknown.
BREIDBART_AXIS = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)


def encode_bb84(bits: ArrayLike, bases: ArrayLike) -> NDArray[np.float64]:
    """Bloch vectors for BB84 states |psi> = H^theta |bit>.

    Parameters
    ----------
    bits, bases
        Integer arrays of 0/1, broadcast against each other.

    Returns
    -------
    (..., 3) array of unit Bloch vectors.
    """
    bits = np.asarray(bits, dtype=np.int64)
    bases = np.asarray(bases, dtype=np.int64)
    bits, bases = np.broadcast_arrays(bits, bases)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be 0 or 1")
    if not np.all((bases == 0) | (bases == 1)):
        raise ValueError("bases must be 0 or 1")
    # sign = +1 for bit 0, -1 for bit 1; axis chosen by the basis bit.
    sign = 1.0 - 2.0 * bits
    return BB84_AXES[bases] * sign[..., None]


def bloch_to_density(r: ArrayLike) -> NDArray[np.complex128]:
    """(..., 3) Bloch vectors -> (..., 2, 2) density matrices."""
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-1] != 3:
        raise ValueError(f"expected trailing axis of size 3, got {r.shape}")
    return 0.5 * (IDENTITY + np.tensordot(r, PAULIS, axes=([-1], [0])))


def density_to_bloch(rho: ArrayLike) -> NDArray[np.float64]:
    """(..., 2, 2) density matrices -> (..., 3) Bloch vectors."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape[-2:] != (2, 2):
        raise ValueError(f"expected trailing shape (2, 2), got {rho.shape}")
    return np.real(np.einsum("kij,...ji->...k", PAULIS, rho))


def measure_probability(r: ArrayLike, axis: ArrayLike) -> NDArray[np.float64]:
    """Probability of the +1 outcome when measuring along `axis`.

    P(+1) = (1 + n . r) / 2. Outcome +1 is reported as bit 0, matching the
    encoding convention in `encode_bb84`.
    """
    r = np.asarray(r, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    return 0.5 * (1.0 + np.einsum("...i,...i->...", r, axis))


def measure_bb84(r: ArrayLike, bases: ArrayLike) -> NDArray[np.float64]:
    """Probability of reading bit 0, measuring each carrier in its own basis."""
    bases = np.asarray(bases, dtype=np.int64)
    return measure_probability(np.asarray(r, dtype=np.float64), BB84_AXES[bases])


def sample_measurement(
    r: ArrayLike, axis: ArrayLike, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Sample outcomes along `axis` and return (bits, post-measurement Bloch).

    A projective measurement leaves the qubit in the corresponding eigenstate,
    so the post-measurement Bloch vector is +/- the measurement axis. This is
    exactly the state an intercept-resend adversary retransmits.
    """
    r = np.asarray(r, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    p_zero = measure_probability(r, axis)
    bits = (rng.random(p_zero.shape) >= p_zero).astype(np.int64)
    sign = 1.0 - 2.0 * bits
    post = np.broadcast_to(axis, r.shape) * sign[..., None]
    return bits, np.ascontiguousarray(post)


def purity(r: ArrayLike) -> NDArray[np.float64]:
    """Tr(rho^2) = (1 + |r|^2) / 2."""
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * (1.0 + np.sum(r * r, axis=-1))


def bloch_norm(r: ArrayLike) -> NDArray[np.float64]:
    return np.linalg.norm(np.asarray(r, dtype=np.float64), axis=-1)


def is_physical_bloch(r: ArrayLike, tol: float = 1e-9) -> NDArray[np.bool_]:
    """A Bloch vector is physical iff |r| <= 1."""
    return bloch_norm(r) <= 1.0 + tol


def is_valid_density_matrix(rho: ArrayLike, tol: float = 1e-9) -> bool:
    """Unit trace, Hermitian, positive semidefinite -- checked explicitly."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape[-2] != rho.shape[-1]:
        return False
    if not np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0, atol=tol):
        return False
    if not np.allclose(rho, np.conj(np.swapaxes(rho, -2, -1)), atol=tol):
        return False
    evals = np.linalg.eigvalsh(rho)
    return bool(np.all(evals >= -tol))


def maximally_mixed(shape: tuple[int, ...] = ()) -> NDArray[np.float64]:
    """The Bloch vector of I/2: the origin. BER against it is exactly 1/2."""
    return np.zeros((*shape, 3), dtype=np.float64)
