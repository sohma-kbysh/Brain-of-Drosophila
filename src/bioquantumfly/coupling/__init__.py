"""Speculative quantum-to-neural coupling. Disabled by default."""

from bioquantumfly.coupling.quantum_neural import (
    COUPLING_GAIN_SWEEP, CouplingConfig, HybridMode, molecular_readout,
    quantum_drive_vector,
)

__all__ = [
    "COUPLING_GAIN_SWEEP", "CouplingConfig", "HybridMode",
    "molecular_readout", "quantum_drive_vector",
]
