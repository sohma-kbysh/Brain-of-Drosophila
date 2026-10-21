"""Deterministic seeding (addendum section 24).

The rule: execution parallelism must never change scientific output.
`--workers 1` and `--workers 64` must produce bit-identical results.

That forbids seeding from process id, wall-clock time, or the order in which a
pool happens to schedule work. Instead every run derives its seed from

    (global experiment seed, run identifier)

via a stable hash, so a run's randomness depends only on *what it is*, never on
*when or where* it executed. `tests/test_determinism.py` enforces this.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: 2**64. Seeds are reduced into [0, 2**64) for numpy's SeedSequence.
_MOD = 1 << 64


def stable_hash(text: str) -> int:
    """A stable 64-bit hash of a string.

    Deliberately *not* Python's builtin `hash()`, which is randomised per
    process by PYTHONHASHSEED and would make runs irreproducible across
    invocations.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def derive_seed(global_seed: int, run_id: str, *, stream: str = "") -> int:
    """Derive a run's seed from the experiment seed and its identity.

    `stream` separates independent random streams inside one run (e.g. the
    message bits versus Eve's measurement outcomes), so that changing the
    number of shots in one stream does not shift the other.
    """
    key = f"{int(global_seed)}|{run_id}|{stream}"
    return stable_hash(key) % _MOD


def seed_rng(global_seed: int, run_id: str, *, stream: str = "") -> np.random.Generator:
    """A `numpy.random.Generator` whose stream depends only on run identity."""
    return np.random.default_rng(derive_seed(global_seed, run_id, stream=stream))


def spawn_streams(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split one generator into `n` independent, reproducible sub-streams."""
    return [np.random.default_rng(s) for s in rng.bit_generator.seed_seq.spawn(n)]
