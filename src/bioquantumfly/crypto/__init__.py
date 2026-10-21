"""Cryptographic toy protocols and adversary models.

NOT formal schemes. `RawBB84Encoding` is not encryption, and
`MaskedConjugateEncoding` is not the Broadbent-Lord uncloneable encryption
scheme nor the Broadbent-Islam certified-deletion scheme. Those works are the
conceptual foundation (see docs/sources.md); what is implemented here is a toy
model built to expose an information-disturbance tradeoff and a failure
boundary, and it must not be described as achieving their security guarantees.
"""

from bioquantumfly.crypto.encoding import (
    Encoding, MaskedConjugateEncoding, RawBB84Encoding, get_protocol,
)
from bioquantumfly.crypto.eve import ATTACKS, get_attack

__all__ = [
    "Encoding", "MaskedConjugateEncoding", "RawBB84Encoding", "get_protocol",
    "ATTACKS", "get_attack",
]
