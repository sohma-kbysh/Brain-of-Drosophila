"""The nine required experiments plus the two hayate-scale additions.

Each experiment exposes `build_specs(config) -> list[RunSpec]` and
`run(spec) -> list[dict]`, so every one of them can be executed serially or
through the resumable parallel sweep runner with identical results.
"""

from bioquantumfly.experiments.registry import EXPERIMENTS, get_experiment

__all__ = ["EXPERIMENTS", "get_experiment"]
