"""Two conjugate-coding protocols (spec section 10).

Protocol A -- raw BB84-like encoding. The message bit IS the encoded bit; only
the basis is secret. This is deliberately *not* encryption: a Breidbart
measurement recovers the raw bit with probability cos^2(pi/8) ~ 0.854 without
any key, which Experiment 2 demonstrates as a negative result.

Protocol B -- masked conjugate encoding. A uniform one-time pad s masks the
message, x = m XOR s, and x is encoded in bases theta. The key is k = (s, theta).
Before the key is revealed, x is uniform and independent of m, so Eve's state
carries exactly zero information about m regardless of her attack. That
privacy comes from the pad, not from quantum mechanics -- which is precisely
why the SWAP control matters: after the key is revealed, an adversary holding a
perfect copy of the carriers recovers m completely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bioquantumfly.quantum.states import encode_bb84


@dataclass(frozen=True)
class Encoding:
    """The full state of one encoding instance, including the secrets.

    The simulator sees everything; the adversary models are what restrict
    access. Keeping the secrets here (rather than hiding them) lets the metrics
    evaluate optimal-adversary quantities exactly.
    """

    protocol: str
    message: NDArray[np.int64]
    encoded_bits: NDArray[np.int64]
    bases: NDArray[np.int64]
    pad: NDArray[np.int64] | None

    @property
    def n_carriers(self) -> int:
        return int(self.message.size)

    def initial_state(self) -> NDArray[np.float64]:
        """(n, 3) Bloch vectors written into the host at t = 0."""
        return encode_bb84(self.encoded_bits, self.bases)

    def hypothesis_states(self) -> NDArray[np.float64]:
        """(2, n, 3): the state each carrier would hold for encoded bit 0 or 1.

        Used for the post-key-reveal adversary, who knows theta and s and so
        need only distinguish x = 0 from x = 1 in a *known* basis.
        """
        zeros = np.zeros(self.n_carriers, dtype=np.int64)
        return np.stack([encode_bb84(zeros, self.bases), encode_bb84(zeros + 1, self.bases)])

    def decode(self, measured_bits: NDArray[np.int64]) -> NDArray[np.int64]:
        """Recover the message from measured encoded bits."""
        if self.pad is None:
            return np.asarray(measured_bits, dtype=np.int64)
        return np.bitwise_xor(np.asarray(measured_bits, dtype=np.int64), self.pad)


class Protocol:
    """Base class: how a message becomes carrier states, and what Eve knows."""

    name: str = "protocol"
    is_encryption: bool = False

    def encode(self, message: Sequence[int] | NDArray, rng: np.random.Generator) -> Encoding:
        raise NotImplementedError

    def prior_hypotheses(self, message_bit: int) -> list[tuple[float, int, int]]:
        """Eve's view before any key is revealed.

        Returns `(weight, encoded_bit, basis)` triples describing the ensemble
        of carrier states consistent with `message_bit`, given only what Eve
        knows a priori. If the two message values give identical ensembles, Eve
        has zero information before key reveal -- which is exactly the property
        Protocol B has and Protocol A lacks.
        """
        raise NotImplementedError


class RawBB84Encoding(Protocol):
    """Protocol A. Secret basis only. NOT encryption -- see module docstring."""

    name = "raw_bb84"
    is_encryption = False

    def encode(self, message, rng: np.random.Generator) -> Encoding:
        m = np.asarray(message, dtype=np.int64)
        bases = rng.integers(0, 2, size=m.size, dtype=np.int64)
        return Encoding(self.name, m, m.copy(), bases, pad=None)

    def prior_hypotheses(self, message_bit: int) -> list[tuple[float, int, int]]:
        # Eve knows the encoded bit equals the message bit; only theta is hidden.
        return [(0.5, message_bit, 0), (0.5, message_bit, 1)]


class MaskedConjugateEncoding(Protocol):
    """Protocol B. One-time pad plus conjugate coding; key k = (s, theta).

    Still a toy model, not a formal uncloneable-encryption scheme.
    """

    name = "masked_conjugate"
    is_encryption = True

    def encode(self, message, rng: np.random.Generator) -> Encoding:
        m = np.asarray(message, dtype=np.int64)
        pad = rng.integers(0, 2, size=m.size, dtype=np.int64)
        bases = rng.integers(0, 2, size=m.size, dtype=np.int64)
        x = np.bitwise_xor(m, pad)
        return Encoding(self.name, m, x, bases, pad=pad)

    def prior_hypotheses(self, message_bit: int) -> list[tuple[float, int, int]]:
        # x = m XOR s with s uniform, so x is uniform whatever m is, and theta
        # is uniform too: the ensemble is the same for both message values.
        return [(0.25, x, theta) for x in (0, 1) for theta in (0, 1)]


PROTOCOLS: dict[str, type[Protocol]] = {
    RawBB84Encoding.name: RawBB84Encoding,
    MaskedConjugateEncoding.name: MaskedConjugateEncoding,
}


def get_protocol(name: str) -> Protocol:
    if name not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}; choose from {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]()


def random_message(n: int, rng: np.random.Generator) -> NDArray[np.int64]:
    return rng.integers(0, 2, size=n, dtype=np.int64)
