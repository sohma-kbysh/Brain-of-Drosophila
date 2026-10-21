"""Immutable parameter records for embarrassingly parallel runs (addendum 7).

Almost every experiment here is independent over seed, T2, attack strength,
attack time, message length, coupling gain and host selection. Representing
each point as an immutable `RunSpec` with a content-derived id buys us
parallel execution, resume, and deduplication without touching the science.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from bioquantumfly.runtime.seeds import stable_hash


def _canonical(obj: Any) -> Any:
    """Canonicalise for hashing: sorted keys, lists for tuples, plain floats.

    Two specs that mean the same thing must hash the same, so ordering and
    numeric type must not leak into the identifier.
    """
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool):
        return obj          # bool is an int subclass, so it must come first
    if isinstance(obj, (int, float)):
        # n=1 and n=1.0 mean the same run and must hash the same, so an
        # integral value canonicalises to its integer form whatever its type.
        # Large integers (body ids) keep their exact decimal form rather than
        # going through %g, which would lose precision above 2**53.
        if isinstance(obj, int) or (
            float(obj).is_integer() and abs(obj) < 2**53
        ):
            return str(int(obj))
        return f"{float(obj):.17g}"
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


@dataclass(frozen=True)
class RunSpec:
    """One point of a parameter sweep.

    `experiment` names the callable that consumes it; `params` holds every
    scientific input. The pair determines `run_id`, so identical parameters
    always map to the same id -- that is what makes resume and dedup safe.
    """

    experiment: str
    params: dict[str, Any] = field(default_factory=dict)
    global_seed: int = 42
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "experiment": self.experiment,
                "params": _canonical(self.params),
                "global_seed": int(self.global_seed),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def run_id(self) -> str:
        """Stable 16-hex-character identifier derived from the parameters."""
        return f"{stable_hash(self.canonical_json):016x}"

    @property
    def short_id(self) -> str:
        return self.run_id[:8]

    def seed_for(self, stream: str = "") -> int:
        from bioquantumfly.runtime.seeds import derive_seed

        return derive_seed(self.global_seed, self.run_id, stream=stream)

    def rng(self, stream: str = ""):
        from bioquantumfly.runtime.seeds import seed_rng

        return seed_rng(self.global_seed, self.run_id, stream=stream)

    def shared_rng(self, stream: str, keys: tuple[str, ...]):
        """An RNG keyed on only `keys`, so runs differing elsewhere share it.

        Why this exists: `rng()` is seeded from the whole parameter set, which is
        right for independence but wrong when a quantity should be held fixed
        across the swept variable. Sweeping T2 with a different random message
        per point makes the curves incomparable -- their asymptotes differ by
        sampling noise rather than by physics. Keying the message stream on
        (n_carriers, protocol) alone keeps every T2 point on the same message
        while staying fully deterministic.
        """
        from bioquantumfly.runtime.seeds import seed_rng

        selected = {k: _canonical(self.params.get(k)) for k in sorted(keys)}
        key = json.dumps({"experiment": self.experiment, "shared": selected},
                         sort_keys=True, separators=(",", ":"))
        return seed_rng(self.global_seed, key, stream=stream)

    def with_params(self, **updates: Any) -> RunSpec:
        merged = dict(self.params)
        merged.update(updates)
        return RunSpec(self.experiment, merged, self.global_seed, dict(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "global_seed": self.global_seed,
            "tags": dict(self.tags),
            **{f"param_{k}": v for k, v in self.params.items()},
        }


def product_specs(
    experiment: str,
    grid: Mapping[str, Iterable[Any]],
    *,
    global_seed: int = 42,
    tags: Mapping[str, str] | None = None,
) -> list[RunSpec]:
    """Cartesian product of a parameter grid.

    Note the scientific caution in addendum section 9: prefer a coarse
    logarithmic scan plus local refinement over an enormous Cartesian product.
    """
    import itertools

    keys = list(grid)
    values = [list(grid[k]) for k in keys]
    return [
        RunSpec(experiment, dict(zip(keys, combo)), global_seed, dict(tags or {}))
        for combo in itertools.product(*values)
    ]


def deduplicate(specs: Iterable[RunSpec]) -> list[RunSpec]:
    """Drop specs whose parameters are identical, preserving first-seen order."""
    seen: set[str] = set()
    out: list[RunSpec] = []
    for spec in specs:
        if spec.run_id not in seen:
            seen.add(spec.run_id)
            out.append(spec)
    return out


def iter_chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
