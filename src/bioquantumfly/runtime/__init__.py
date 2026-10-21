"""Execution-environment concerns: resources, profiles, seeds, sweeps.

Kept strictly separate from the scientific modules. Nothing in here may change
a numerical result -- see `bioquantumfly.runtime.seeds` for the rule that
`workers=1` and `workers=64` must produce identical output.
"""

from bioquantumfly.runtime.profiles import PROFILES, ResourceProfile, resolve_profile
from bioquantumfly.runtime.resources import ResourceReport, detect_resources
from bioquantumfly.runtime.seeds import derive_seed, seed_rng
from bioquantumfly.runtime.runspec import RunSpec

__all__ = [
    "PROFILES", "ResourceProfile", "resolve_profile",
    "ResourceReport", "detect_resources",
    "derive_seed", "seed_rng", "RunSpec",
]
