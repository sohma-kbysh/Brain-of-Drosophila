"""Neural layer: a Shiu-et-al.-style LIF network on connectome connectivity.

MODEL TRANSFER CAVEAT (spec section 16). The parameterisation is taken from
Shiu et al., Nature 2024, which was validated on the *female* FlyWire brain.
Applying it to MaleCNS connectivity is a model-transfer assumption, not an
experimental validation of MaleCNS neural dynamics. Every parameter below is
labelled `Status.TRANSFERRED` for exactly this reason.
"""

from bioquantumfly.neural.lif import SHIU_PARAMETERS, LIFParameters
from bioquantumfly.neural.brian_network import (
    NetworkResult, build_network, run_network,
)

__all__ = [
    "SHIU_PARAMETERS", "LIFParameters", "NetworkResult",
    "build_network", "run_network",
]
