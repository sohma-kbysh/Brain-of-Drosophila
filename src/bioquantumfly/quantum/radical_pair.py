"""Spin-correlated radical-pair dynamics (spec section 19).

SECONDARY model, and a deliberately SEPARATE class hierarchy from
`AbstractEphemeralMemory`. The distinction is not cosmetic. A natural radical
pair is created by photochemistry in a specific initial spin state -- typically
the singlet -- and its subsequent evolution is governed by hyperfine and Zeeman
interactions plus spin-selective recombination. You cannot write an arbitrary
alpha|0> + beta|1> into it. Treating it as a general quantum memory would
smuggle an unjustified capability into the cryptographic claims, so the crypto
layer never touches this module.

Hamiltonian (spec section 19):

    H_RP = H_Z + H_hf + H_exchange + H_dipolar

Recombination uses the Haberkorn form, spin-selective population loss

    d(rho)/dt = -i[H, rho] - (1/2){k_S Q_S + k_T Q_T, rho}

which is NOT of Lindblad D[C] form (it has no repopulation term), so the
Liouvillian is assembled explicitly rather than handed to `mesolve` as c_ops.

The singlet yield then has a closed form. Since rho(t) = e^{Lt} rho(0) and L is
strictly stable whenever any recombination rate is positive,

    Y_S = k_S * int_0^inf Tr[Q_S rho(t)] dt = k_S * Tr[Q_S vec^-1(-L^-1 vec(rho_0))]

so no time integration is needed for the yield itself.

PARAMETER HONESTY. Hyperfine couplings, exchange, dipolar coupling and
recombination rates are molecule-specific. This module ships generic
order-of-magnitude defaults labelled `Status.PROXY` with an explicit note; they
are NOT measured constants for any named cryptochrome or flavin system. Supply
real values from a primary source before drawing any chemical conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bioquantumfly.params import Parameter, Status

#: Electron gyromagnetic ratio, rad s^-1 T^-1 (CODATA; |gamma_e| = 1.76086e11).
GAMMA_E = Parameter(
    1.76085963e11, "gamma_e", Status.MEASURED,
    provenance="CODATA recommended value", regime="free electron",
    unit="rad/s/T", note="Standard physical constant, not a model assumption.",
)

#: Generic hyperfine magnitude. A PROXY: typical isotropic hyperfine couplings
#: in flavin/tryptophan radical pairs are of order 0.1-2 mT, but this is not a
#: measured constant for any specific molecule.
DEFAULT_HYPERFINE_T = Parameter(
    1e-3, "a_hf", Status.PROXY,
    provenance="generic magnitude for organic radical pairs (order-of-magnitude only)",
    regime="unspecified organic radical pair", unit="T",
    note="NOT a measured constant for any named cryptochrome. Replace with a "
         "primary-source value before drawing chemical conclusions.",
)

#: Generic recombination rate, PROXY.
DEFAULT_K_S = Parameter(
    1e6, "k_S", Status.PROXY,
    provenance="generic magnitude for radical-pair recombination (order-of-magnitude only)",
    regime="unspecified", unit="1/s",
    note="Placeholder rate. Not measured for any named system.",
)


def _qutip():
    import qutip as qt

    return qt


@dataclass(frozen=True)
class RadicalPairParameters:
    """Spin Hamiltonian and recombination parameters, all in SI.

    Fields left at ``None`` mean "unknown": the corresponding term is omitted
    from the Hamiltonian rather than filled with an invented number.
    """

    b_field_T: float = 50e-6                 # ~geomagnetic magnitude
    b_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    n_nuclei: int = 1
    hyperfine_T: float | None = 1e-3
    exchange_rad_s: float | None = None      # unknown by default -> omitted
    dipolar_rad_s: float | None = None       # unknown by default -> omitted
    k_singlet: float = 1e6
    k_triplet: float = 1e6
    initial_state: str = "singlet"

    def __post_init__(self) -> None:
        if self.n_nuclei < 0 or self.n_nuclei > 3:
            raise ValueError(
                f"n_nuclei={self.n_nuclei}: keep the Hilbert space small "
                "(spec section 21 forbids large Hilbert-space simulation)"
            )
        if self.initial_state not in ("singlet", "triplet", "mixed"):
            raise ValueError(f"unknown initial state {self.initial_state!r}")
        if self.k_singlet < 0 or self.k_triplet < 0:
            raise ValueError("recombination rates must be non-negative")
        if self.k_singlet == 0 and self.k_triplet == 0:
            raise ValueError(
                "at least one recombination rate must be positive, otherwise the "
                "singlet yield integral does not converge"
            )

    @property
    def hilbert_dim(self) -> int:
        return 4 * 2**self.n_nuclei

    def provenance(self) -> dict[str, Any]:
        return {
            "gamma_e": GAMMA_E.to_dict(),
            "hyperfine": (
                {**DEFAULT_HYPERFINE_T.to_dict(), "value": self.hyperfine_T}
                if self.hyperfine_T is not None else {"status": "unknown", "omitted": True}
            ),
            "exchange": (
                {"value": self.exchange_rad_s, "status": Status.SPECULATIVE.value}
                if self.exchange_rad_s is not None else {"status": "unknown", "omitted": True}
            ),
            "dipolar": (
                {"value": self.dipolar_rad_s, "status": Status.SPECULATIVE.value}
                if self.dipolar_rad_s is not None else {"status": "unknown", "omitted": True}
            ),
            "k_singlet": {**DEFAULT_K_S.to_dict(), "value": self.k_singlet},
            "k_triplet": {**DEFAULT_K_S.to_dict(), "value": self.k_triplet},
            "b_field_T": self.b_field_T,
            "n_nuclei": self.n_nuclei,
            "hilbert_dim": self.hilbert_dim,
            "warning": (
                "Hyperfine, exchange, dipolar and recombination values are generic "
                "proxies unless explicitly overridden from a primary source."
            ),
        }


class RadicalPairMechanism:
    """Two coupled electron spins plus optional nuclei.

    Deliberately NOT an `AbstractEphemeralMemory`: it exposes no arbitrary-write
    interface, only preparation in a photochemically plausible initial state.
    """

    def __init__(self, params: RadicalPairParameters | None = None):
        self.params = params or RadicalPairParameters()
        self._qt = _qutip()
        self._build()

    # --- construction -----------------------------------------------------
    def _spin_ops(self, site: int) -> list:
        """Spin-1/2 operators S = sigma/2 for `site` in the full tensor space."""
        qt = self._qt
        n_sites = 2 + self.params.n_nuclei
        ops = []
        for pauli in (qt.sigmax(), qt.sigmay(), qt.sigmaz()):
            factors = [qt.qeye(2)] * n_sites
            factors[site] = 0.5 * pauli
            ops.append(qt.tensor(factors))
        return ops

    def _build(self) -> None:
        qt = self._qt
        p = self.params
        n_sites = 2 + p.n_nuclei
        self.S1 = self._spin_ops(0)
        self.S2 = self._spin_ops(1)
        self.nuclei = [self._spin_ops(2 + k) for k in range(p.n_nuclei)]

        gamma = float(GAMMA_E)
        b_vec = np.asarray(p.b_direction, dtype=float)
        b_vec = b_vec / np.linalg.norm(b_vec) * p.b_field_T

        # Zeeman: gamma_e B . (S1 + S2)
        H = sum(gamma * b_vec[k] * (self.S1[k] + self.S2[k]) for k in range(3))

        # Hyperfine: isotropic, coupling each nucleus to electron 1 only, which
        # is the standard minimal asymmetry that makes singlet-triplet mixing
        # magnetically sensitive at all.
        if p.hyperfine_T is not None and p.n_nuclei:
            a_rad_s = gamma * p.hyperfine_T
            for I in self.nuclei:
                H = H + a_rad_s * sum(self.S1[k] * I[k] for k in range(3))

        s1_dot_s2 = sum(self.S1[k] * self.S2[k] for k in range(3))
        identity = qt.tensor([qt.qeye(2)] * n_sites)

        # Exchange, convention H_J = -J (1/2 + 2 S1.S2).
        if p.exchange_rad_s is not None:
            H = H - p.exchange_rad_s * (0.5 * identity + 2.0 * s1_dot_s2)

        # Axial dipolar coupling along z, H_D = D (3 S1z S2z - S1.S2).
        if p.dipolar_rad_s is not None:
            H = H + p.dipolar_rad_s * (3.0 * self.S1[2] * self.S2[2] - s1_dot_s2)

        self.H = H
        self.identity = identity

        # Singlet projector on the electron pair, extended over the nuclei.
        singlet = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
                   - qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).unit()
        q_s_electrons = singlet * singlet.dag()
        nuclear_identity = (
            qt.tensor([qt.qeye(2)] * p.n_nuclei) if p.n_nuclei else None
        )
        self.Q_S = (
            qt.tensor(q_s_electrons, nuclear_identity) if nuclear_identity is not None
            else q_s_electrons
        )
        self.Q_T = identity - self.Q_S
        self.K = p.k_singlet * self.Q_S + p.k_triplet * self.Q_T

    # --- states -----------------------------------------------------------
    def initial_density_matrix(self):
        """Photochemically plausible starting state.

        The singlet-born case is normalised over the nuclear manifold, which is
        unpolarised at physiological temperature.
        """
        p = self.params
        dim = self.Q_S.shape[0]
        if p.initial_state == "singlet":
            rho = self.Q_S / self.Q_S.tr()
        elif p.initial_state == "triplet":
            rho = self.Q_T / self.Q_T.tr()
        else:
            rho = self.identity / dim
        return rho

    # --- dynamics ---------------------------------------------------------
    def liouvillian(self) -> NDArray[np.complex128]:
        """Haberkorn Liouvillian as a dense (d^2, d^2) matrix.

        L = -i (H (x) I - I (x) H^T) - (1/2)(K (x) I + I (x) K^T)

        Dense is fine: with at most 3 nuclei the dimension is 32, so L is at
        most 1024 x 1024.
        """
        H = self.H.full()
        K = self.K.full()
        d = H.shape[0]
        eye = np.eye(d, dtype=complex)
        commutator = np.kron(H, eye) - np.kron(eye, H.T)
        anticommutator = np.kron(K, eye) + np.kron(eye, K.T)
        return -1j * commutator - 0.5 * anticommutator

    def singlet_probability(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tr[Q_S rho(t)] on a time grid (survival-weighted, not renormalised)."""
        from scipy.linalg import expm

        times = np.atleast_1d(np.asarray(times, dtype=float))
        L = self.liouvillian()
        rho0 = self.initial_density_matrix().full().reshape(-1)
        q_s = self.Q_S.full()
        out = np.empty(times.size)
        for i, t in enumerate(times):
            rho_t = (expm(L * t) @ rho0).reshape(q_s.shape)
            out[i] = float(np.real(np.trace(q_s @ rho_t)))
        return out

    def singlet_yield(self) -> float:
        """Y_S = k_S * int_0^inf Tr[Q_S rho(t)] dt, evaluated exactly.

        Uses int_0^inf e^{Lt} dt = -L^{-1}, valid because recombination makes L
        strictly stable. Avoids both a time grid and a truncation error.
        """
        L = self.liouvillian()
        rho0 = self.initial_density_matrix().full().reshape(-1)
        integrated = np.linalg.solve(L, -rho0)
        q_s = self.Q_S.full()
        return float(
            self.params.k_singlet * np.real(np.trace(q_s @ integrated.reshape(q_s.shape)))
        )

    def total_yield(self) -> float:
        """Singlet plus triplet yield; must equal 1 if all pairs recombine."""
        L = self.liouvillian()
        rho0 = self.initial_density_matrix().full().reshape(-1)
        integrated = np.linalg.solve(L, -rho0).reshape(self.Q_S.shape)
        return float(
            self.params.k_singlet * np.real(np.trace(self.Q_S.full() @ integrated))
            + self.params.k_triplet * np.real(np.trace(self.Q_T.full() @ integrated))
        )

    def magnetic_field_effect(self, fields_T: NDArray[np.float64]) -> NDArray[np.float64]:
        """Singlet yield versus applied field magnitude.

        The magnetic field effect is the signature observable of radical-pair
        magnetoreception; a flat curve means the configured Hamiltonian has no
        magnetically sensitive singlet-triplet mixing.
        """
        fields_T = np.atleast_1d(np.asarray(fields_T, dtype=float))
        out = np.empty(fields_T.size)
        for i, b in enumerate(fields_T):
            params = RadicalPairParameters(**{**self.params.__dict__, "b_field_T": float(b)})
            out[i] = RadicalPairMechanism(params).singlet_yield()
        return out

    def provenance(self) -> dict[str, Any]:
        return {
            **self.params.provenance(),
            "model": "spin-correlated radical pair, Haberkorn recombination",
            "is_arbitrary_write_memory": False,
            "note": (
                "A radical pair is created in a specific photochemical initial "
                "state; arbitrary quantum states cannot be written into it. This "
                "model is therefore NOT used by the cryptographic layer."
            ),
        }
