"""Typed configuration, loaded from YAML and validated with pydantic.

Configuration is the place where a reader should be able to see, in one file,
exactly which assumptions a run made: which coherence regime, whether the
speculative coupling was on, which graph (real or synthetic), and how many CPUs
it was allowed. Those choices are echoed into every results metadata file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bioquantumfly.params import DEFAULT_T2_SWEEP_S


class _Base(BaseModel):
    """Strict base: unknown keys are rejected rather than silently ignored.

    A typo in a config file must not quietly change nothing. `notes` is the
    escape hatch: free-form provenance text that travels with the config into
    results metadata, without loosening validation anywhere else.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)
    notes: dict[str, str] = Field(default_factory=dict)


class RuntimeConfig(_Base):
    profile: Literal["development", "standard", "large", "linalg"] = "development"
    workers: int | None = None
    blas_threads: int | None = None
    max_memory_gb: float | None = None
    scratch_dir: str = "auto"
    use_shm_cache: bool = False
    shm_limit_gb: float = 16.0
    seed: int = 42


class QuantumConfig(_Base):
    """Abstract finite-coherence memory settings."""

    n_carriers: int = 32
    T2_seconds: list[float] = Field(default_factory=lambda: list(DEFAULT_T2_SWEEP_S))
    t1_over_t2: float = 1.0
    # Log-spaced at the short end: the BER = 1e-3 crossing sits near t/T2 ~ 2e-3,
    # so a grid starting at 0.1 cannot resolve the usable window at all and would
    # report it as zero.
    t_over_T2_grid: list[float] = Field(
        default_factory=lambda: [
            0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03,
            0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
        ]
    )
    physical_regime: Literal["in_vivo_current", "cryogenic_protein", "future", "sweep"] = "sweep"

    @field_validator("n_carriers")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_carriers must be >= 1")
        return v

    @field_validator("t1_over_t2")
    @classmethod
    def _physical_t1(cls, v: float) -> float:
        if v < 0.5:
            raise ValueError(
                f"t1_over_t2={v} implies T2 > 2*T1, which is unphysical; use >= 0.5"
            )
        return v


class CryptoConfig(_Base):
    protocol: Literal["raw_bb84", "masked_conjugate"] = "masked_conjugate"
    attacks: list[str] = Field(default_factory=lambda: ["none", "breidbart", "swap_theft"])
    attack_strengths: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    )
    t_attack_over_T2: list[float] = Field(default_factory=lambda: [0.1])
    eve_memory: Literal["ideal", "same_as_host", "better"] = "ideal"
    eve_memory_advantage: float = 100.0


class ConnectomeConfig(_Base):
    source: Literal["neuprint", "synthetic", "processed"] = "synthetic"
    dataset: str = "male-cns:v1.0"
    server: str = "https://neuprint.janelia.org"
    seed_type: str | None = None
    seed_body_ids: list[int] | None = None
    hops: int = 2
    min_weight: int = 5
    max_neurons: int = 20_000
    processed_dir: str = "data/processed/male_cns"
    synthetic_n_neurons: int = 500
    nt_confidence_threshold: float = 0.5
    nt_low_confidence: Literal["ignore", "excitatory", "exclude"] = "excitatory"


class HostConfig(_Base):
    mode: Literal["engineered", "centrality", "curated"] = "engineered"
    strategy: Literal[
        "random", "high_out_degree", "high_in_degree", "high_out_strength",
        "betweenness", "descending", "sensory", "motor", "localized",
    ] = "random"
    n_hosts: int = 32
    curated_path: str | None = None
    seed: int = 42


class NeuralConfig(_Base):
    enabled: bool = True
    duration_s: float = 0.5
    backend: Literal["runtime", "cython", "cpp_standalone"] = "runtime"
    stimulus_rate_hz: float = 100.0
    stimulate_hosts: bool = True
    record_all_spikes: bool = False
    record_voltage: bool = False
    downstream_hops: int = 3
    # Independent network seeds averaged per hybrid data point. A spiking
    # network near threshold is chaotic, so one realisation is noise.
    n_repeats: int = 5


class CouplingConfigModel(_Base):
    """Speculative quantum-to-neural coupling. OFF by default."""

    enabled: bool = False
    mode: Literal["storage_only", "tamper_alarm"] = "storage_only"
    speculative_coupling_gain: float = 0.0
    gain_sweep: list[float] = Field(default_factory=lambda: [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    readout_axis: Literal["z", "x", "breidbart"] = "z"


class RadicalPairConfig(_Base):
    """Radical-pair spin parameters.

    Defaults are generic PROXIES; `exchange_rad_s` and `dipolar_rad_s` default
    to None, meaning UNKNOWN, and the corresponding Hamiltonian terms are
    omitted rather than filled with invented numbers.
    """

    b_field_T: float = 50e-6
    b_direction: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    n_nuclei: int = 1
    hyperfine_T: float | None = 1e-3
    exchange_rad_s: float | None = None
    dipolar_rad_s: float | None = None
    k_singlet: float = 1e6
    k_triplet: float = 1e6
    initial_state: Literal["singlet", "triplet", "mixed"] = "singlet"


class FeasibilityConfig(_Base):
    """Thresholds for the feasibility map (addendum section 29).

    These are CONFIGURABLE MODELLING CHOICES, not universal cryptographic
    standards, and the reports say so wherever the map appears.
    """

    max_bob_ber: float = 1e-3
    max_eve_info: float = 0.01
    min_detection_probability: float = 0.95


class Config(_Base):
    """Top-level configuration."""

    name: str = "default"
    description: str = ""
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    connectome: ConnectomeConfig = Field(default_factory=ConnectomeConfig)
    host_selection: HostConfig = Field(default_factory=HostConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    coupling: CouplingConfigModel = Field(default_factory=CouplingConfigModel)
    radical_pair: RadicalPairConfig = Field(default_factory=RadicalPairConfig)
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    results_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config, or the defaults when `path` is None."""
    data: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config file {path} must contain a YAML mapping")
        data = raw
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return Config.model_validate(data)


def save_config(config: Config, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False))
    return out
