"""EYFP spin-qubit envelope model (spec section 20).

Feder et al., Nature 2025 demonstrated a genetically encodable spin qubit: an
EYFP triplet that is optically addressable and coherently controlled with
microwaves. This module encodes ONLY the coherence envelope reported there,
because that is what can be cited without inventing numbers.

WHAT IS DELIBERATELY ABSENT. No zero-field splitting, g-tensor, hyperfine
tensor or other spectroscopic constant is hard-coded. Those values were not
verified against the primary source while writing this module, and the spec is
explicit that unknown constants must be marked unknown rather than guessed.
`SPECTROSCOPIC_CONSTANTS` therefore records them as `unknown` with a pointer to
the paper; `spin_one_hamiltonian` refuses to run until they are supplied.

WHAT THE NUMBERS ARE AND ARE NOT. Both regimes below are measurements of EYFP,
not of Drosophila. There is no measurement of a quantum memory in a fly, and
this module makes no such claim:

    cryogenic:        T2_CPMG = 16 +/- 2 us  (dynamical decoupling, not free
                                              induction decay -- a decoupled
                                              coherence time, so using it as a
                                              storage lifetime is optimistic)
    room temperature: spin polarisation decays in < 100 ns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bioquantumfly.params import FEDER_2025, Parameter, Status
from bioquantumfly.quantum.abstract_memory import MemoryParams

#: Cryogenic coherence, as reported.
T2_CPMG_CRYOGENIC = Parameter(
    16e-6, "T2_CPMG", Status.MEASURED, provenance=FEDER_2025,
    regime="cryogenic EYFP", unit="s",
    note="16 +/- 2 us under CPMG dynamical decoupling. NOT a free-induction "
         "decay time and NOT a Drosophila measurement.",
)
T2_CPMG_UNCERTAINTY = Parameter(
    2e-6, "T2_CPMG_uncertainty", Status.MEASURED, provenance=FEDER_2025,
    regime="cryogenic EYFP", unit="s",
)
#: Room-temperature envelope: polarisation is gone within 100 ns.
T_POLARISATION_ROOM_TEMPERATURE = Parameter(
    100e-9, "T_polarisation_max", Status.MEASURED, provenance=FEDER_2025,
    regime="room-temperature protein spin", unit="s",
    note="Upper bound: room-temperature spin polarisation disappears within <100 ns.",
)

#: Constants a full spin-1 model needs, none of which are invented here.
SPECTROSCOPIC_CONSTANTS: dict[str, dict[str, Any]] = {
    "zero_field_splitting_D": {
        "value": None, "status": "unknown", "source": FEDER_2025,
        "note": "Not transcribed from the paper; supply from the primary source.",
    },
    "zero_field_splitting_E": {"value": None, "status": "unknown", "source": FEDER_2025},
    "g_factor": {"value": None, "status": "unknown", "source": FEDER_2025},
    "hyperfine_tensor": {"value": None, "status": "unknown", "source": FEDER_2025},
}


@dataclass(frozen=True)
class EYFPEnvelope:
    """Which experimental envelope to use, and what it does and does not say."""

    regime: str = "cryogenic"

    def __post_init__(self) -> None:
        if self.regime not in ("cryogenic", "room_temperature"):
            raise ValueError(
                f"unknown EYFP regime {self.regime!r}; "
                "choose 'cryogenic' or 'room_temperature'"
            )

    @property
    def T2(self) -> float:
        return (
            float(T2_CPMG_CRYOGENIC) if self.regime == "cryogenic"
            else float(T_POLARISATION_ROOM_TEMPERATURE)
        )

    @property
    def parameter(self) -> Parameter:
        return (
            T2_CPMG_CRYOGENIC if self.regime == "cryogenic"
            else T_POLARISATION_ROOM_TEMPERATURE
        )

    def memory_params(self, t1_over_t2: float = 1.0) -> MemoryParams:
        """An `AbstractEphemeralMemory` parameterisation from this envelope.

        NOTE the modelling leap this makes explicit: it treats the reported
        coherence time as the lifetime of an arbitrary-write single-qubit
        memory. Feder et al. demonstrate coherent control, not a general
        addressable memory, so this is a proxy, not a measurement of what the
        protocol needs.
        """
        t2 = self.T2
        return MemoryParams(
            T1=t1_over_t2 * t2, T2=t2, label=f"eyfp_{self.regime}",
            status=Status.PROXY, regime=f"EYFP {self.regime}",
            note=(
                f"Coherence envelope from {FEDER_2025} ({self.parameter.note}). "
                "Used here as a proxy lifetime for an arbitrary-write memory, "
                "which is a stronger capability than the cited experiment shows. "
                "Not a Drosophila measurement."
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "T2": self.T2,
            "parameter": self.parameter.to_dict(),
            "is_drosophila_measurement": False,
            "spectroscopic_constants": SPECTROSCOPIC_CONSTANTS,
        }


def coherence_envelope(times_s: np.ndarray, regime: str = "cryogenic") -> np.ndarray:
    """Simple exponential coherence decay for the chosen envelope."""
    envelope = EYFPEnvelope(regime)
    return np.exp(-np.asarray(times_s, dtype=float) / envelope.T2)


def spin_one_hamiltonian(*_args, **_kwargs):
    """A full spin-1 triplet Hamiltonian -- intentionally not implemented.

    Building it needs D, E and a g-factor. Those constants were not verified
    against Feder et al. while writing this module, and the project rule is to
    mark unknown parameters `unknown` rather than to guess them. Supply the
    values from the primary source and implement here.
    """
    raise NotImplementedError(
        "Spin-1 EYFP Hamiltonian requires zero-field-splitting parameters (D, E) "
        "and a g-factor from Feder et al., Nature 2025 "
        "(DOI:10.1038/s41586-025-09417-w). These are recorded as 'unknown' in "
        "SPECTROSCOPIC_CONSTANTS rather than guessed. Use EYFPEnvelope for the "
        "coherence-envelope model, which is what the cited result supports."
    )
