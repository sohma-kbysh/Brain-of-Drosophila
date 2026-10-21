"""Adversary models (spec section 11).

Every attack is expressed as a set of *branches*: each branch has a
probability, a linear map on the host's Bloch vector (what Bob later reads) and
a map giving the state Eve retains. Because density matrices are linear, Bob's
averaged post-attack state is the probability-weighted sum of branch outputs,
while Eve's guessing probability is the probability-weighted average of her
per-branch optimum. That makes every quantity exact -- no Monte Carlo noise --
and keeps a whole feasibility map affordable.

A classical measurement record is represented as the corresponding diagonal
Bloch vector, so the same Helstrom formula scores intercept-resend adversaries
and quantum-memory adversaries on one footing.

The SWAP attack is the required negative control: it establishes that natural
host decoherence is NOT cryptographic deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.quantum.states import (
    BB84_AXES, BREIDBART_AXIS, bloch_to_density, density_to_bloch,
)

BlochMap = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: Bloch vector of |0>, the state a SWAP adversary leaves behind.
_GROUND = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class AttackBranch:
    """One stochastic branch of an attack."""

    probability: float
    host: BlochMap
    eve: BlochMap | None = None


def _project_onto(axis: NDArray[np.float64]) -> BlochMap:
    """Projective measurement along `axis`, averaged over outcomes.

    Measuring along n leaves the qubit in +/- n with probabilities (1 +/- n.r)/2,
    so the mean post-measurement Bloch vector is (n.r) n -- the component along
    the measurement axis survives and everything transverse is destroyed.
    """
    axis = np.asarray(axis, dtype=np.float64)

    def _map(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("...i,i->...", r, axis)[..., None] * axis

    return _map


def _weak_measure(axis: NDArray[np.float64], eta: float) -> tuple[BlochMap, BlochMap]:
    """Weak measurement of strength `eta` along `axis`.

    Kraus operators in the axis eigenbasis:
        M_0 = diag(sqrt((1+eta)/2), sqrt((1-eta)/2)),  M_1 = M_0 flipped

    Averaging over the two outcomes gives a transverse shrink by sqrt(1-eta^2)
    with the axial component untouched, while Eve's record is a classical bit
    biased by eta times the axial component. eta = 0 is no attack, eta = 1 is a
    full projective measurement -- so `eta` interpolates the whole
    information-disturbance curve of Experiment 4.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"attack strength eta must lie in [0, 1], got {eta}")
    shrink = float(np.sqrt(max(0.0, 1.0 - eta**2)))

    def host(r: NDArray[np.float64]) -> NDArray[np.float64]:
        par = np.einsum("...i,i->...", r, axis)[..., None] * axis
        return par + shrink * (r - par)

    def eve(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return eta * np.einsum("...i,i->...", r, axis)[..., None] * axis

    return host, eve


def _controlled_rotation(alpha: float) -> NDArray[np.complex128]:
    """U_QE(alpha) = |0><0|_Q (x) I + |1><1|_Q (x) R(alpha).

    At alpha = 0 the ancilla is untouched; at alpha = pi/2 this is a CNOT, a
    full copy of the computational-basis value into Eve's memory.
    """
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array([[c, -s], [s, c]], dtype=complex)
    U = np.zeros((4, 4), dtype=complex)
    U[0:2, 0:2] = np.eye(2)
    U[2:4, 2:4] = R
    return U


def _partial_traces(joint: NDArray[np.complex128]) -> tuple[NDArray, NDArray]:
    """Reduce an (n, 4, 4) two-qubit state to its (n, 2, 2) marginals."""
    t = joint.reshape(-1, 2, 2, 2, 2)          # [n, q, e, q', e']
    rho_q = np.einsum("nqepe->nqp", t)          # trace out E
    rho_e = np.einsum("nqeqf->nef", t)          # trace out Q
    return rho_q, rho_e


def _coherent_maps(alpha: float) -> tuple[BlochMap, BlochMap]:
    """Host and ancilla maps for the coherent attack, via the joint 4x4 state.

    The joint density matrix is kept explicitly (spec section 11) rather than
    approximated by a Bloch shortcut, so entanglement between host and ancilla
    is represented honestly.
    """
    U = _controlled_rotation(alpha)
    ket0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    def _joint(r: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        rho = bloch_to_density(np.atleast_2d(r))                 # (n, 2, 2)
        joint = np.einsum("nij,kl->nikjl", rho, ket0).reshape(-1, 4, 4)
        evolved = U @ joint @ U.conj().T
        return _partial_traces(evolved)

    def host(r: NDArray[np.float64]) -> NDArray[np.float64]:
        out = density_to_bloch(_joint(r)[0])
        return out.reshape(np.shape(r))

    def eve(r: NDArray[np.float64]) -> NDArray[np.float64]:
        out = density_to_bloch(_joint(r)[1])
        return out.reshape(np.shape(r))

    return host, eve


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attack:
    """An adversary strategy, as a probability-weighted set of branches."""

    name: str
    branches: Sequence[AttackBranch]
    strength: float = 1.0
    keeps_quantum_memory: bool = False
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = sum(b.probability for b in self.branches)
        if not np.isclose(total, 1.0):
            raise ValueError(f"attack {self.name!r} branch probabilities sum to {total}, not 1")

    def apply_to_host(self, r: ArrayLike) -> NDArray[np.float64]:
        """Bob's averaged post-attack state."""
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        for br in self.branches:
            out = out + br.probability * br.host(r)
        return out


def no_attack() -> Attack:
    return Attack(
        "none", [AttackBranch(1.0, lambda r: np.array(r, copy=True), None)],
        strength=0.0, description="Control: Eve does not interact.",
    )


def z_intercept_resend() -> Attack:
    proj = _project_onto(BB84_AXES[0])
    return Attack(
        "z_intercept_resend", [AttackBranch(1.0, proj, proj)],
        description="Eve measures in the computational basis and resends.",
    )


def random_basis_intercept_resend() -> Attack:
    pz, px = _project_onto(BB84_AXES[0]), _project_onto(BB84_AXES[1])
    return Attack(
        "random_basis_intercept_resend",
        [AttackBranch(0.5, pz, pz), AttackBranch(0.5, px, px)],
        description="Eve picks Z or X uniformly per carrier, measures and resends.",
    )


def breidbart() -> Attack:
    """The intermediate basis (X+Z)/sqrt(2), optimal for unknown-basis BB84."""
    proj = _project_onto(BREIDBART_AXIS)
    return Attack(
        "breidbart", [AttackBranch(1.0, proj, proj)],
        description="Breidbart-basis intercept-resend; recovers a raw BB84 bit with "
                    "probability cos^2(pi/8) ~ 0.854 and no key.",
    )


def weak_measurement(eta: float, basis: str = "random") -> Attack:
    """Partial measurement of strength `eta` (spec section 11)."""
    if basis == "z":
        host, eve = _weak_measure(BB84_AXES[0], eta)
        branches = [AttackBranch(1.0, host, eve)]
    elif basis == "breidbart":
        host, eve = _weak_measure(BREIDBART_AXIS, eta)
        branches = [AttackBranch(1.0, host, eve)]
    elif basis == "random":
        hz, ez = _weak_measure(BB84_AXES[0], eta)
        hx, ex = _weak_measure(BB84_AXES[1], eta)
        branches = [AttackBranch(0.5, hz, ez), AttackBranch(0.5, hx, ex)]
    else:
        raise ValueError(f"unknown weak-measurement basis {basis!r}")
    return Attack(
        "weak_measurement", branches, strength=eta,
        description=f"Weak POVM of strength eta={eta:g} in the {basis} basis.",
        metadata={"basis": basis},
    )


def coherent_ancilla(alpha: float) -> Attack:
    """Eve entangles a fresh ancilla with the host via U_QE(alpha).

    She keeps the ancilla as a quantum memory rather than measuring it, so her
    information can be extracted later -- after the key is revealed.
    """
    host, eve = _coherent_maps(alpha)
    return Attack(
        "coherent_ancilla", [AttackBranch(1.0, host, eve)],
        strength=float(np.sin(alpha) ** 2), keeps_quantum_memory=True,
        description=f"Coherent system-ancilla coupling, alpha={alpha:g} rad.",
        metadata={"alpha": float(alpha)},
    )


def swap_theft() -> Attack:
    """NEGATIVE CONTROL (spec sections 11, 30): Eve takes the state outright.

    rho_Q (x) |0><0|_E  ->  |0><0|_Q (x) rho_E

    The host is left in |0> and subsequently decoheres to nothing, while Eve
    holds a perfect copy in her own memory. Any apparent security that rests on
    the host self-erasing evaporates here. This is the difference between
    no-cloning and no-theft, and it is the boundary this project reports rather
    than hides.
    """
    return Attack(
        "swap_theft",
        [AttackBranch(
            1.0,
            lambda r: np.broadcast_to(_GROUND, np.shape(r)).copy(),
            lambda r: np.array(r, copy=True),
        )],
        keeps_quantum_memory=True,
        description="Ideal SWAP: perfect state transfer into Eve's own memory.",
    )


#: Registry of attacks callable by name from configs and the CLI.
ATTACKS: dict[str, Callable[..., Attack]] = {
    "none": no_attack,
    "z_intercept_resend": z_intercept_resend,
    "random_basis_intercept_resend": random_basis_intercept_resend,
    "breidbart": breidbart,
    "weak_measurement": weak_measurement,
    "coherent_ancilla": coherent_ancilla,
    "swap_theft": swap_theft,
}


def get_attack(name: str, **kwargs) -> Attack:
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; choose from {sorted(ATTACKS)}")
    return ATTACKS[name](**kwargs)
