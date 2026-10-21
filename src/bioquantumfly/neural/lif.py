"""Leaky integrate-and-fire parameters (spec section 15).

Equations, following Shiu et al. 2024:

    dv_i/dt = [ g_i - (v_i - V_resting) ] / T_mbr
    dg_i/dt = -g_i / tau

with a presynaptic spike from j adding  g_i <- g_i + w_{j,i}, and

    w_{j,i} = N^syn_{j,i} * s_j * W_syn,   s_j = +1 excitatory / -1 inhibitory.

Note that `g` here carries units of voltage, not conductance: in this
formulation the synaptic variable is already scaled into the membrane equation.
The name follows the source model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bioquantumfly.params import SHIU_2024, Parameter, Status


def _shiu(value: float, name: str, unit: str, note: str = "") -> Parameter:
    """A Shiu et al. parameter, marked as a transferred model value."""
    return Parameter(
        value=value, name=name, status=Status.TRANSFERRED, provenance=SHIU_2024,
        regime="Drosophila whole-brain LIF (validated on female FlyWire)",
        unit=unit,
        note=note or (
            "Applied here to MaleCNS connectivity. Model transfer, not a "
            "measurement of MaleCNS dynamics."
        ),
    )


SHIU_PARAMETERS: dict[str, Parameter] = {
    "V_resting": _shiu(-52e-3, "V_resting", "V"),
    "V_reset": _shiu(-52e-3, "V_reset", "V"),
    "V_threshold": _shiu(-45e-3, "V_threshold", "V"),
    "R_mbr": _shiu(10.0, "R_mbr", "kohm*cm^2"),
    "C_mbr": _shiu(2.0, "C_mbr", "uF/cm^2"),
    "T_mbr": _shiu(20e-3, "T_mbr", "s", "R_mbr * C_mbr = 10 kohm cm^2 * 2 uF/cm^2 = 20 ms."),
    "T_refractory": _shiu(2.2e-3, "T_refractory", "s"),
    "tau_synapse": _shiu(5e-3, "tau_synapse", "s"),
    "T_delay": _shiu(1.8e-3, "T_delay", "s"),
    "W_syn": _shiu(0.275e-3, "W_syn", "V", "0.275 mV per synapse. Applied to MaleCNS counts."),
    "baseline_rate": Parameter(
        0.0, "baseline_rate", Status.CONVENTION, regime="model default", unit="Hz",
        note="No spontaneous background drive unless a stimulus is configured.",
    ),
}


@dataclass(frozen=True)
class LIFParameters:
    """Concrete LIF parameters in SI units.

    Defaults reproduce Shiu et al.; every field can be overridden from config,
    and `provenance()` records which values were changed.
    """

    V_resting: float = -52e-3
    V_reset: float = -52e-3
    V_threshold: float = -45e-3
    T_mbr: float = 20e-3
    T_refractory: float = 2.2e-3
    tau_synapse: float = 5e-3
    T_delay: float = 1.8e-3
    W_syn: float = 0.275e-3
    baseline_rate: float = 0.0
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.V_threshold <= self.V_resting:
            raise ValueError(
                f"threshold ({self.V_threshold:g} V) must exceed rest "
                f"({self.V_resting:g} V), otherwise the neuron fires continuously"
            )
        for name in ("T_mbr", "tau_synapse", "T_refractory"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def spike_threshold_gap(self) -> float:
        """How far the membrane must be driven above rest to fire, in volts."""
        return self.V_threshold - self.V_resting

    @property
    def psp_peak_factor(self) -> float:
        """Peak membrane deflection per unit of injected synaptic drive.

        A spike deposits g <- g + w instantaneously, after which g decays with
        tau_syn while v chases it with tau_mbr. Solving the two-exponential
        cascade, an impulse of size G produces

            v(t) - V_rest = G * tau_s/(tau_s - tau_m) * (e^{-t/tau_s} - e^{-t/tau_m})

        peaking at t* = ln(tau_m/tau_s) / (1/tau_s - 1/tau_m).

        With the Shiu time constants (tau_m = 20 ms, tau_s = 5 ms) the factor is
        only ~0.157: because the synaptic variable decays four times faster than
        the membrane integrates, a volley delivers about a sixth of its nominal
        amplitude. Sizing a stimulus without this factor under-drives the
        network by roughly 6x, which is why it is computed rather than guessed.
        """
        import math

        tau_m, tau_s = self.T_mbr, self.tau_synapse
        if math.isclose(tau_m, tau_s, rel_tol=1e-9):
            return 1.0 / math.e        # degenerate case: peak of (t/tau) e^{-t/tau}
        t_peak = math.log(tau_m / tau_s) / (1.0 / tau_s - 1.0 / tau_m)
        return (tau_s / (tau_s - tau_m)) * (
            math.exp(-t_peak / tau_s) - math.exp(-t_peak / tau_m)
        )

    @property
    def min_synapses_to_fire(self) -> float:
        """Simultaneous excitatory synapses needed to reach threshold from rest.

        Accounts for the synaptic/membrane filtering via `psp_peak_factor`, so
        this is the count that actually fires a neuron, not the naive
        gap / W_syn ratio (which understates it ~6x for the Shiu constants).
        """
        return self.spike_threshold_gap / (self.W_syn * self.psp_peak_factor)

    @property
    def drive_to_fire(self) -> float:
        """Instantaneous jump in g needed to just reach threshold from rest."""
        return self.spike_threshold_gap / self.psp_peak_factor

    def provenance(self) -> dict[str, Any]:
        defaults = LIFParameters()
        changed = {
            k: getattr(self, k)
            for k in ("V_resting", "V_reset", "V_threshold", "T_mbr", "T_refractory",
                      "tau_synapse", "T_delay", "W_syn", "baseline_rate")
            if getattr(self, k) != getattr(defaults, k)
        }
        return {
            "source": SHIU_2024,
            "status": Status.TRANSFERRED.value,
            "validated_on": "female FlyWire brain",
            "applied_to": "MaleCNS connectivity",
            "caveat": (
                "Neural dynamics are a Shiu-et-al.-style LIF transfer model applied "
                "to MaleCNS connectivity; this does not constitute experimental "
                "validation of MaleCNS neural dynamics."
            ),
            "modified_from_default": changed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "V_resting": self.V_resting, "V_reset": self.V_reset,
            "V_threshold": self.V_threshold, "T_mbr": self.T_mbr,
            "T_refractory": self.T_refractory, "tau_synapse": self.tau_synapse,
            "T_delay": self.T_delay, "W_syn": self.W_syn,
            "baseline_rate": self.baseline_rate,
        }
