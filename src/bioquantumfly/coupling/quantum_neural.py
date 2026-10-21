"""Quantum subsystem -> host neuron coupling (spec sections 17, 18).

THE MOST SPECULATIVE LAYER IN THE PROJECT. There is no experimental evidence
that the state of a molecular spin system influences neuronal firing in
Drosophila, and none is claimed. It is `enabled: false` by default, every
symbol carries `speculative` in its name or docstring, and results produced
with it on are flagged in metadata so a figure can never be mistaken for one
produced by the cryptographic layer alone.

The model, as specified:

    y_i(t)      = Tr[M rho_i(t)]                  molecular readout observable
    Delta g_i(t) = kappa * [y_i(t) - y_0]         neural perturbation

Two hybrid modes exist and must never be conflated (spec section 18):

    storage_only -- the quantum memory sits inside a neuron but does not affect
                    neural dynamics. MaleCNS provides host selection,
                    connectivity context and visualisation only. This is the
                    primary mode for every cryptographic question.
    tamper_alarm -- the coupling is on, so a measurement disturbance can change
                    firing. Speculative; explores whether a biological network
                    could act as a tamper-evident host.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.neural.lif import LIFParameters
from bioquantumfly.quantum.states import BB84_AXES, BREIDBART_AXIS

#: Dimensionless gain sweep (spec section 17). There is no empirical
#: calibration for kappa, so the sweep spans four orders of magnitude and its
#: purpose is to locate the gain at which the coupling would matter at all.
COUPLING_GAIN_SWEEP: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: Named measurement operators M for the molecular readout.
READOUT_AXES: dict[str, NDArray[np.float64]] = {
    "z": BB84_AXES[0],
    "x": BB84_AXES[1],
    "breidbart": BREIDBART_AXIS,
}


class HybridMode(str, enum.Enum):
    STORAGE_ONLY = "storage_only"
    TAMPER_ALARM = "tamper_alarm"


@dataclass(frozen=True)
class CouplingConfig:
    """Configuration of the speculative coupling.

    Attributes
    ----------
    speculative_coupling_gain
        The dimensionless kappa. Named to make its status unmissable at every
        call site, as the spec requires.
    """

    enabled: bool = False
    mode: HybridMode = HybridMode.STORAGE_ONLY
    speculative_coupling_gain: float = 0.0
    readout_axis: str = "z"
    baseline_readout: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.readout_axis not in READOUT_AXES:
            raise ValueError(
                f"unknown readout axis {self.readout_axis!r}; "
                f"choose from {sorted(READOUT_AXES)}"
            )
        if self.speculative_coupling_gain < 0:
            raise ValueError("coupling gain must be non-negative")
        if self.mode is HybridMode.TAMPER_ALARM and not self.enabled:
            raise ValueError(
                "tamper_alarm mode requires enabled=True; storage_only is the "
                "mode in which the quantum layer does not touch neural dynamics"
            )

    @property
    def is_active(self) -> bool:
        return self.enabled and self.mode is HybridMode.TAMPER_ALARM

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "speculative_coupling_gain": self.speculative_coupling_gain,
            "readout_axis": self.readout_axis,
            "baseline_readout": self.baseline_readout,
            "status": "speculative",
            "warning": (
                "No empirical calibration exists for the quantum-to-neural "
                "coupling gain. Results with enabled=True are exploratory."
            ),
            **self.metadata,
        }


def molecular_readout(state: ArrayLike, axis: str = "z") -> NDArray[np.float64]:
    """y = Tr[M rho] for M a Pauli observable, i.e. the Bloch projection.

    Returns values in [-1, +1], one per carrier.
    """
    return np.einsum("...i,i->...", np.asarray(state, dtype=np.float64), READOUT_AXES[axis])


def coupling_scale(params: LIFParameters) -> float:
    """Volts of synaptic drive corresponding to kappa = 1.

    Dimensional consistency (spec section 17). The readout y is dimensionless
    in [-1, 1], while `q_drive` in the LIF equations is a voltage. The natural
    scale is `drive_to_fire`: the instantaneous drive that just brings a resting
    neuron to threshold. With that choice kappa = 1 means "a full-swing change
    in the molecular observable is worth exactly one spike's worth of drive",
    which makes the sweep interpretable instead of arbitrary.
    """
    return params.drive_to_fire


def quantum_drive_vector(
    n_neurons: int,
    host_indices: ArrayLike,
    host_states: ArrayLike,
    config: CouplingConfig,
    params: LIFParameters | None = None,
) -> NDArray[np.float64]:
    """Per-neuron additional drive (volts) from the hosted quantum subsystems.

    Returns an all-zero vector unless the coupling is both enabled and in
    `tamper_alarm` mode -- so `storage_only` is guaranteed to be numerically
    identical to no coupling at all, which the tests check.

    Parameters
    ----------
    host_states : (n_hosts, 3) or (n_hosts, n_carriers, 3)
        Bloch vectors of the quantum subsystem in each host neuron. When a host
        holds several carriers their readouts are averaged.
    """
    drive = np.zeros(int(n_neurons), dtype=np.float64)
    if not config.is_active or config.speculative_coupling_gain == 0.0:
        return drive

    params = params or LIFParameters()
    idx = np.asarray(host_indices, dtype=np.int64)
    states = np.asarray(host_states, dtype=np.float64)
    if states.ndim == 3:
        y = molecular_readout(states, config.readout_axis).mean(axis=-1)
    else:
        y = molecular_readout(states, config.readout_axis)
    if y.shape != idx.shape:
        raise ValueError(
            f"host_states gives {y.shape} readouts but {idx.shape} host indices were supplied"
        )
    drive[idx] = config.speculative_coupling_gain * (y - config.baseline_readout) * coupling_scale(params)
    return drive


def readout_shift(
    unattacked_states: ArrayLike, attacked_states: ArrayLike, axis: str = "z"
) -> NDArray[np.float64]:
    """Change in molecular readout caused by an adversary.

    This is the microscopic quantity the tamper-alarm mode proposes to amplify:
    Eve's interaction perturbs rho, which shifts Tr[M rho], which shifts the
    host neuron's drive.
    """
    return molecular_readout(attacked_states, axis) - molecular_readout(unattacked_states, axis)
