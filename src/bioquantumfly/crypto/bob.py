"""The legitimate receiver.

Bob knows the key, so he measures each carrier in the basis it was written in.
Everything here is a probability, not a sampled bit string: for a product
encoding the per-carrier outcomes are independent, so the bit-error rate and
the exact-recovery probability both have closed forms. Computing them exactly
removes sampling noise from the feasibility boundary, which is the number this
project exists to report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bioquantumfly.analysis.metrics import bit_error_rate, exact_recovery_probability
from bioquantumfly.crypto.encoding import Encoding
from bioquantumfly.quantum.states import BB84_AXES, measure_probability


def correct_read_probability(state: ArrayLike, encoding: Encoding) -> NDArray[np.float64]:
    """Per-carrier probability that Bob reads back the encoded bit.

    Measuring along the carrier's own axis, P(bit 0) = (1 + n.r)/2, so the
    probability of the *encoded* value x is (1 + (-1)^x n.r)/2.
    """
    state = np.asarray(state, dtype=np.float64)
    axes = BB84_AXES[encoding.bases]
    p_zero = measure_probability(state, axes)
    return np.where(encoding.encoded_bits == 0, p_zero, 1.0 - p_zero)


def read_metrics(state: ArrayLike, encoding: Encoding) -> dict[str, float]:
    """Bob's bit-error rate and exact-message-recovery probability.

    Both are reported because they diverge: at n = 128 a per-bit error rate of
    1% still leaves exact recovery at ~28%.
    """
    p = correct_read_probability(state, encoding)
    return {
        "bob_ber": bit_error_rate(p),
        "bob_exact_recovery": exact_recovery_probability(p),
        "bob_mean_correct": float(np.mean(p)),
    }


def sample_read(state: ArrayLike, encoding: Encoding, rng: np.random.Generator) -> NDArray[np.int64]:
    """Draw one concrete readout, for demonstrations rather than metrics."""
    p_zero = measure_probability(np.asarray(state, float), BB84_AXES[encoding.bases])
    measured = (rng.random(p_zero.shape) >= p_zero).astype(np.int64)
    return encoding.decode(measured)
