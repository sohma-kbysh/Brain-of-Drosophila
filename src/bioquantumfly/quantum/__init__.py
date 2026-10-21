"""Quantum layer.

Two deliberately separate class hierarchies (spec section 19):

* `AbstractEphemeralMemory` -- an arbitrary-write finite-coherence qubit. This
  is the model the cryptographic analysis uses.
* `RadicalPairMechanism` -- spin-correlated radical pair dynamics. A natural
  radical pair is created in a specific photochemical initial state; you cannot
  write an arbitrary alpha|0> + beta|1> into it. Conflating the two would
  smuggle an unjustified capability into the security claims.
"""

from bioquantumfly.quantum.abstract_memory import AbstractEphemeralMemory, MemoryParams
from bioquantumfly.quantum.states import (
    BB84_AXES, bloch_to_density, density_to_bloch, encode_bb84,
    measure_probability, is_valid_density_matrix,
)

__all__ = [
    "AbstractEphemeralMemory", "MemoryParams", "BB84_AXES",
    "bloch_to_density", "density_to_bloch", "encode_bb84",
    "measure_probability", "is_valid_density_matrix",
]
