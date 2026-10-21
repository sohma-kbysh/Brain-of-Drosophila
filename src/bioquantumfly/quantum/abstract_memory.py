"""Storage model A: an abstract finite-coherence qubit (spec section 8).

This is the primary model for the cryptographic analysis, and it grants the
protocol its strongest capability: an *arbitrary* single-qubit state can be
written into the host and read out later. No biological system is known to
offer that. Keeping the capability in one clearly named class makes the
assumption visible instead of letting it hide inside the protocol code.

Each encoded bit is an independent single-qubit density matrix. An n-bit
message is a product state, so the simulation cost is O(n), not O(2**n).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.params import Status, classify_T2, regime_of_T2
from bioquantumfly.quantum import channels
from bioquantumfly.quantum.states import encode_bb84, measure_bb84, sample_measurement, BB84_AXES


@dataclass(frozen=True)
class MemoryParams:
    """Coherence parameters of one quantum memory, with evidence labelling.

    `T1 = T2 = inf` is the infinite-coherence control of spec section 23: it
    isolates every effect that is *not* decoherence.
    """

    T1: float
    T2: float
    label: str = ""
    status: Status | None = None
    regime: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        channels.validate_t1_t2(self.T1, self.T2)
        if self.status is None:
            object.__setattr__(self, "status", classify_T2(self.T2))
        if self.regime is None:
            object.__setattr__(self, "regime", regime_of_T2(self.T2))

    @classmethod
    def ideal(cls) -> "MemoryParams":
        """No decoherence at all. A control, and Eve's best-case memory."""
        return cls(
            T1=np.inf, T2=np.inf, label="ideal",
            status=Status.CONVENTION, regime="control",
            note="Infinite-coherence control; isolates non-decoherence effects.",
        )

    @classmethod
    def from_T2(cls, T2: float, t1_over_t2: float = 1.0, label: str = "") -> "MemoryParams":
        """Build from T2, choosing T1 = `t1_over_t2` * T2.

        The default T1 = T2 sits inside the physical region T2 <= 2*T1 and keeps
        the sweep one-dimensional; `t1_over_t2 = 0.5` gives the relaxation-
        limited corner T2 = 2*T1.
        """
        return cls(T1=t1_over_t2 * T2, T2=T2, label=label or f"T2={T2:g}s")

    @property
    def T_phi(self) -> float:
        return channels.t_phi_from_t1_t2(self.T1, self.T2)

    @property
    def is_speculative(self) -> bool:
        return self.status is Status.SPECULATIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "T1": self.T1, "T2": self.T2, "T_phi": self.T_phi,
            "label": self.label,
            "status": self.status.value if self.status else None,
            "regime": self.regime, "note": self.note,
        }


@dataclass
class AbstractEphemeralMemory:
    """A bank of independent finite-coherence qubits.

    State is held as an (n_carriers, 3) Bloch array. Every operation is
    vectorised; nothing here builds a 2**n object.
    """

    params: MemoryParams
    state: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    elapsed: float = 0.0

    @classmethod
    def write_bb84(
        cls, bits: ArrayLike, bases: ArrayLike, params: MemoryParams
    ) -> "AbstractEphemeralMemory":
        """Write conjugate-basis states -- the arbitrary-write assumption."""
        return cls(params=params, state=encode_bb84(bits, bases))

    @classmethod
    def write_bloch(cls, r: ArrayLike, params: MemoryParams) -> "AbstractEphemeralMemory":
        state = np.atleast_2d(np.asarray(r, dtype=np.float64))
        return cls(params=params, state=state)

    @property
    def n_carriers(self) -> int:
        return int(self.state.shape[0])

    def evolve(self, duration: float) -> "AbstractEphemeralMemory":
        """Advance the host by `duration` under its own (T1, T2). In place."""
        if duration < 0:
            raise ValueError("duration must be non-negative")
        self.state = channels.t1_t2_channel(self.state, duration, self.params.T1, self.params.T2)
        self.elapsed += duration
        return self

    def state_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """State at many storage times without mutating the memory.

        Returns (len(times), n_carriers, 3) -- one vectorised call, which is
        what a whole time sweep costs.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        return channels.t1_t2_channel(
            self.state[None, :, :], times[:, None], self.params.T1, self.params.T2
        )

    def read_probability(self, bases: ArrayLike) -> NDArray[np.float64]:
        """Probability each carrier reads back as bit 0 in the given basis."""
        return measure_bb84(self.state, bases)

    def read(self, bases: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
        """Sample a destructive read in the given bases."""
        bases = np.asarray(bases, dtype=np.int64)
        bits, _ = sample_measurement(self.state, BB84_AXES[bases], rng)
        return bits

    def copy(self) -> "AbstractEphemeralMemory":
        return AbstractEphemeralMemory(self.params, self.state.copy(), self.elapsed)
