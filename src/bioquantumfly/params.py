"""Parameters carrying explicit scientific provenance.

The central methodological rule of this project:

    measured data != experimentally constrained model != speculative assumption

Every physical number used anywhere in BioQuantumFly must declare which of
those three it is. `Parameter` makes that machine-readable, so plots, metadata
and reports can label their inputs instead of relying on the reader's memory.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Status(str, enum.Enum):
    """How much experimental support a number actually has."""

    MEASURED = "measured"
    """Directly reported in a cited experiment, in the regime we cite it for."""

    PROXY = "proxy"
    """Order-of-magnitude stand-in inspired by a measurement in a *different*
    system, organism or temperature. Not a measurement of the modelled system."""

    TRANSFERRED = "transferred"
    """A model parameterisation validated on one system and applied to another
    (e.g. Shiu et al. female-FlyWire LIF parameters applied to MaleCNS)."""

    SPECULATIVE = "speculative"
    """No experimental support. Explored to map a feasibility boundary."""

    CONVENTION = "convention"
    """A modelling choice or unit convention, not a physical claim."""


#: Display labels used in plot legends (spec section 28).
STATUS_LABEL = {
    Status.MEASURED: "measured",
    Status.PROXY: "proxy",
    Status.TRANSFERRED: "transferred model",
    Status.SPECULATIVE: "SPECULATIVE",
    Status.CONVENTION: "convention",
}


@dataclass(frozen=True)
class Parameter:
    """A physical number plus the evidence (or lack of it) behind it.

    Examples
    --------
    >>> T2 = Parameter(16e-6, "T2", status=Status.MEASURED,
    ...                provenance="Feder et al., Nature 2025",
    ...                regime="cryogenic EYFP", unit="s")
    >>> float(T2)
    1.6e-05
    """

    value: float
    name: str
    status: Status
    provenance: str | None = None
    regime: str | None = None
    unit: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status in (Status.MEASURED, Status.PROXY, Status.TRANSFERRED) and not self.provenance:
            raise ValueError(
                f"parameter {self.name!r} has status {self.status.value!r} but no provenance; "
                "either cite a source or mark it speculative"
            )
        if self.status is Status.SPECULATIVE and self.provenance:
            raise ValueError(
                f"parameter {self.name!r} is speculative but cites {self.provenance!r}; "
                "a cited number should not be labelled speculative"
            )

    def __float__(self) -> float:
        return float(self.value)

    @property
    def is_speculative(self) -> bool:
        return self.status is Status.SPECULATIVE

    @property
    def label(self) -> str:
        """Short label for plot legends and captions."""
        bits = [f"{self.name}={self.value:g}"]
        if self.unit:
            bits[0] += f" {self.unit}"
        bits.append(STATUS_LABEL[self.status])
        if self.regime:
            bits.append(self.regime)
        return " | ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class ParameterSet:
    """A named bundle of parameters, e.g. one physical regime."""

    name: str
    description: str
    parameters: dict[str, Parameter] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Parameter:
        return self.parameters[key]

    @property
    def contains_speculative(self) -> bool:
        return any(p.is_speculative for p in self.parameters.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "contains_speculative": self.contains_speculative,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
        }


# ---------------------------------------------------------------------------
# Physical regimes (spec section 9).
#
# NOTE ON HONESTY: none of these is a measurement of a quantum memory in
# Drosophila. No such measurement exists. The in-vivo regime is an
# order-of-magnitude proxy; the cryogenic regime cites a real measurement in a
# different system (purified/expressed EYFP, not a fly); the future regime is
# openly speculative.
# ---------------------------------------------------------------------------

BURD_2026 = "Burd et al., Nature 2026, DOI:10.1038/s41586-026-10282-4"
FEDER_2025 = "Feder et al., Nature 2025, DOI:10.1038/s41586-025-09417-w"
SHIU_2024 = "Shiu et al., Nature 2024, DOI:10.1038/s41586-024-07763-9"
BERG_2026 = "Berg et al., Cell 2026 (MaleCNS connectome)"

REGIME_IN_VIVO_CURRENT = ParameterSet(
    name="in_vivo_current",
    description=(
        "Order-of-magnitude proxy for the current in-vivo biological frontier. "
        "Anchored on spin-correlated radical-pair coherence observed in live "
        "C. elegans at room temperature (>4 ns) and on room-temperature "
        "protein-spin polarisation decay (<100 ns). NOT a Drosophila measurement, "
        "and NOT a demonstration of an addressable quantum memory."
    ),
    parameters={
        "T2_min": Parameter(
            4e-9, "T2_min", Status.PROXY, provenance=BURD_2026,
            regime="in vivo, room temperature", unit="s",
            note="Observations in live C. elegans imply coherence >4 ns. Lower bound, not a memory lifetime.",
        ),
        "T2_max": Parameter(
            100e-9, "T2_max", Status.PROXY, provenance=FEDER_2025,
            regime="room temperature protein spin", unit="s",
            note="Room-temperature EYFP spin polarisation disappears within <100 ns.",
        ),
    },
)

REGIME_CRYOGENIC_PROTEIN = ParameterSet(
    name="cryogenic_protein",
    description=(
        "Cryogenic fluorescent-protein spin qubit. This IS a real measurement, "
        "but of an EYFP triplet at cryogenic temperature, not of a fly."
    ),
    parameters={
        "T2_CPMG": Parameter(
            16e-6, "T2_CPMG", Status.MEASURED, provenance=FEDER_2025,
            regime="cryogenic EYFP", unit="s",
            note="T2_CPMG = 16 +/- 2 us. Cryogenic temperature. Dynamical decoupling, not free-induction decay.",
        ),
        "T2_CPMG_uncertainty": Parameter(
            2e-6, "T2_CPMG_uncertainty", Status.MEASURED, provenance=FEDER_2025,
            regime="cryogenic EYFP", unit="s",
        ),
        "T2_min": Parameter(
            1e-6, "T2_min", Status.PROXY, provenance=FEDER_2025,
            regime="cryogenic protein", unit="s",
            note="Rounded lower edge of the sweep range, not a reported value.",
        ),
        "T2_max": Parameter(
            20e-6, "T2_max", Status.PROXY, provenance=FEDER_2025,
            regime="cryogenic protein", unit="s",
        ),
    },
)

REGIME_FUTURE = ParameterSet(
    name="future",
    description=(
        "SPECULATIVE. No experimental support whatsoever. Included only to locate "
        "the coherence time at which the cryptographic toy protocol would become "
        "workable, i.e. to state what would have to be true."
    ),
    parameters={
        "T2_100us": Parameter(100e-6, "T2", Status.SPECULATIVE, regime="future", unit="s"),
        "T2_1ms": Parameter(1e-3, "T2", Status.SPECULATIVE, regime="future", unit="s"),
        "T2_10ms": Parameter(10e-3, "T2", Status.SPECULATIVE, regime="future", unit="s"),
    },
)

REGIMES: dict[str, ParameterSet] = {
    r.name: r for r in (REGIME_IN_VIVO_CURRENT, REGIME_CRYOGENIC_PROTEIN, REGIME_FUTURE)
}

#: Default T2 sweep (spec section 9). Spans measured -> proxy -> speculative.
DEFAULT_T2_SWEEP_S: tuple[float, ...] = (
    4e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3,
)


def classify_T2(t2_seconds: float) -> Status:
    """Which evidence class does a given T2 fall into?

    Used by plotting so that a speculative point is never drawn as if it were
    measured.
    """
    if t2_seconds <= 100e-9:
        return Status.PROXY          # in-vivo frontier, order-of-magnitude only
    if t2_seconds <= 20e-6:
        return Status.MEASURED       # within the cryogenic EYFP envelope
    return Status.SPECULATIVE


def regime_of_T2(t2_seconds: float) -> str:
    if t2_seconds <= 100e-9:
        return "in_vivo_current"
    if t2_seconds <= 20e-6:
        return "cryogenic_protein"
    return "future"
