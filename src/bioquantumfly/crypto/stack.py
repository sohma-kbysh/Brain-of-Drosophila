"""Protocol C: the primitives composed into one scheme.

Protocols A and B in `encoding.py` are diagnostic. A (raw conjugate coding)
demonstrates that a secret basis is not encryption; B (one-time-padded conjugate
coding) demonstrates that the pad, not the physics, supplies pre-key privacy.
Neither survives contact with an adversary who can hold quantum state.

Protocol C composes four primitives, each chosen to repair a specific failure
this project measured:

    failure measured                       primitive that repairs it
    ------------------------------------   --------------------------------
    strict BER threshold costs ~111x in    error correction
    storage time
    Eve keeps a fraction of a bit          privacy amplification
    forever
    "it decohered" certifies nothing       certified deletion (an actual
                                           returned certificate)
    an ideal-memory Eve is never           delayed key release under a
    defeated                               noisy-storage assumption

The last is the load-bearing one, and it is an assumption, not a theorem:
against an adversary with a genuinely ideal quantum memory the scheme still
fails, exactly as the SWAP control shows. What the composition buys is a
quantitative statement of *how good* her memory would have to be.

NOT a security proof. Each layer is simulated operationally; none of the formal
analyses (Broadbent-Islam, Damgard-Fehr-Salvail-Schaffner, leftover hash lemma
with composable finite-key terms) is reproduced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from bioquantumfly.analysis.metrics import mutual_information_binary
from bioquantumfly.crypto import deletion as deletion_module
from bioquantumfly.crypto.bob import read_metrics
from bioquantumfly.crypto.encoding import Encoding, MaskedConjugateEncoding
from bioquantumfly.crypto.eve import Attack
from bioquantumfly.crypto.metrics import evaluate
from bioquantumfly.crypto.postprocessing import (
    BB84_THRESHOLD_QBER, DEFAULT_EC_EFFICIENCY, DEFAULT_EPSILON_PA,
    KeyBudget, key_budget,
)
from bioquantumfly.quantum.abstract_memory import MemoryParams
from bioquantumfly.quantum.channels import t1_t2_channel


@dataclass(frozen=True)
class StackConfig:
    """Which layers are switched on, and with what parameters."""

    error_correction: bool = True
    privacy_amplification: bool = True
    certified_deletion: bool = True
    delayed_key_release: bool = True

    ec_efficiency: float = DEFAULT_EC_EFFICIENCY
    epsilon_pa: float = DEFAULT_EPSILON_PA
    #: Key withholding, in units of the ADVERSARY's coherence time. Waiting is
    #: free, and this is the only knob that touches an adversary who already
    #: holds the state.
    key_delay_over_eve_T2: float = 5.0
    #: Fraction of carriers a partial cheater may steal undetected.
    max_theft_fraction: float = 0.01
    deletion_confidence_sigma: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StackResult:
    """End-to-end accounting for one configuration."""

    # --- inputs ---
    n_carriers: int
    T2_host: float
    T2_eve: float
    attack: str
    t_read_over_T2: float
    layers: dict[str, bool]

    # --- physical layer ---
    bob_ber: float
    bob_exact_recovery: float
    eve_pguess_at_attack: float

    # --- delayed key release ---
    key_delay_seconds: float
    eve_pguess_at_key_release: float
    eve_information_bits: float

    # --- post-processing ---
    h_min_bits: float
    ec_leakage_bits: float
    secret_bits: float
    secret_rate: float
    secret_bits_without_delay: float
    requires_storage_assumption: bool

    # --- deletion ---
    deletion_certifiable: bool
    deletion_within_window: bool
    deletion_window_over_T2: float
    deletion_tolerance: float
    honest_certificate_accepted: bool | None
    cheating_certificate_accepted: bool | None
    decay_certificate_accepted: bool | None

    # --- verdict ---
    usable: bool
    verdict: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("metadata", None)
        d["layers"] = dict(self.layers)
        return d


def _verdict(
    secret_bits: float, bob_ber: float, deletion_ok: bool,
    secret_without_delay: float, epsilon: float,
) -> tuple[bool, str]:
    """Decide whether a configuration yields a usable secret, and say why.

    The criterion is `secret_bits > 0`, and only that. Privacy amplification
    bounds the adversary's information about the FINAL key by `epsilon`
    whatever her pre-amplification advantage was, so rejecting on the raw
    guessing probability would double-count the very leakage the layer removes.

    Two earlier mistakes are worth recording, because both flattered or
    maligned the scheme incorrectly. Rejecting whenever the raw Eve information
    exceeded a threshold ignored privacy amplification entirely. Rejecting
    whenever the adversary held an ideal memory over-generalised the SWAP
    control: an ideal memory defeats a *state-theft* adversary, but a weak
    interaction against an ideal-memory adversary is ordinary QKD and is
    handled by post-processing in the normal way.
    """
    if bob_ber >= BB84_THRESHOLD_QBER:
        return False, (
            f"abort: error rate {bob_ber:.3f} exceeds the ~{BB84_THRESHOLD_QBER:.0%} "
            "one-way threshold, so no post-processing yields a secret. This is the "
            "protocol working: the disturbance is detected and the round is discarded."
        )
    if secret_bits <= 0:
        return False, ("fails: reconciliation leakage plus the adversary's min-entropy "
                       "deficit exhaust the budget; no secret survives")
    needed_delay = secret_without_delay <= 0 < secret_bits
    reason = (
        "usable, but ONLY under the noisy-storage assumption: without delayed key "
        f"release the secret is empty. Final key is {epsilon:g}-secure."
        if needed_delay else
        f"usable: secret extracted and {epsilon:g}-secure after privacy amplification."
    )
    if not deletion_ok:
        return True, reason + " Deletion is not certifiable at this storage time."
    return True, reason + " Deletion is certifiable."


def run_stack(
    encoding: Encoding,
    attack: Attack,
    *,
    host: MemoryParams,
    eve_memory: MemoryParams,
    t_read: float,
    t_attack: float = 0.0,
    config: StackConfig | None = None,
    rng: np.random.Generator | None = None,
) -> StackResult:
    """Evaluate the whole stack for one configuration.

    The physical layer is evaluated exactly by `crypto.metrics.evaluate`; each
    subsequent layer is then applied to those numbers, so switching a layer off
    reproduces the corresponding bare protocol.
    """
    config = config or StackConfig()
    rng = rng or np.random.default_rng(0)
    protocol = MaskedConjugateEncoding()

    # --- physical layer -----------------------------------------------------
    base = evaluate(encoding, protocol, attack, host=host, eve_memory=eve_memory,
                    t_attack=t_attack, t_read=t_read, t_reveal=t_read)

    # --- delayed key release ------------------------------------------------
    delay = 0.0
    if config.delayed_key_release and np.isfinite(eve_memory.T2):
        delay = config.key_delay_over_eve_T2 * eve_memory.T2
    advantage = max(0.0, base.eve_pguess_after_key - 0.5)
    decay = 1.0 if not np.isfinite(eve_memory.T2) else float(np.exp(-delay / eve_memory.T2))
    p_eve = 0.5 + advantage * decay
    eve_bits = float(mutual_information_binary(p_eve)) * encoding.n_carriers

    # --- post-processing ----------------------------------------------------
    if config.error_correction or config.privacy_amplification:
        budget: KeyBudget = key_budget(
            encoding.n_carriers, base.bob_ber, p_eve,
            ec_efficiency=config.ec_efficiency if config.error_correction else 0.0,
            epsilon=config.epsilon_pa,
        )
        h_min, leak = budget.h_min_bits, budget.ec_leakage_bits
        secret = budget.secret_bits if config.privacy_amplification else 0.0
        rate = budget.secret_rate if config.privacy_amplification else 0.0
    else:
        h_min = leak = secret = rate = 0.0

    # --- certified deletion -------------------------------------------------
    honest_ok = cheat_ok = decay_ok = None
    window = {"certifiable": False, "window_over_T2": 0.0, "recommended_tolerance": 0.0}
    if config.certified_deletion:
        n_check = int(deletion_module.check_positions(encoding).size)
        if n_check:
            window = deletion_module.certification_window(
                n_check, host.T2,
                max_theft_fraction=config.max_theft_fraction,
                confidence_sigma=config.deletion_confidence_sigma,
            )
            tolerance = window["recommended_tolerance"]
            state = t1_t2_channel(encoding.initial_state(), t_read, host.T1, host.T2)
            honest_ok = deletion_module.verify(
                encoding, deletion_module.honest_deletion(state, rng), tolerance=tolerance
            ).accepted
            cheat_ok = deletion_module.verify(
                encoding, deletion_module.cheating_deletion(encoding, rng), tolerance=tolerance
            ).accepted
            # The contrast that motivates the whole layer: waiting produces no
            # valid certificate once the check correlations have decayed.
            decay_ok = deletion_module.verify(
                encoding,
                deletion_module.decay_certificate(
                    encoding, max(t_read, 5.0 * host.T2), host.T1, host.T2, rng
                ),
                tolerance=tolerance,
            ).accepted

    # Counterfactual: what would the secret be if the key were released at once?
    # The difference is precisely what the storage assumption is buying.
    secret_without_delay = key_budget(
        encoding.n_carriers, base.bob_ber, base.eve_pguess_after_key,
        ec_efficiency=config.ec_efficiency if config.error_correction else 0.0,
        epsilon=config.epsilon_pa,
    ).secret_bits if config.privacy_amplification else 0.0

    # Two different questions: does a certification window exist at this number
    # of check positions at all, and is the current storage time inside it?
    within_window = bool(
        window["certifiable"] and t_read <= window["window_over_T2"] * host.T2 + 1e-18
    )
    deletion_ok = bool(within_window and honest_ok and not cheat_ok)
    usable, verdict = _verdict(secret, base.bob_ber, deletion_ok,
                               secret_without_delay, config.epsilon_pa)

    return StackResult(
        n_carriers=encoding.n_carriers,
        T2_host=host.T2,
        T2_eve=eve_memory.T2,
        attack=attack.name,
        t_read_over_T2=float(t_read / host.T2) if np.isfinite(host.T2) else 0.0,
        layers={
            "error_correction": config.error_correction,
            "privacy_amplification": config.privacy_amplification,
            "certified_deletion": config.certified_deletion,
            "delayed_key_release": config.delayed_key_release,
        },
        bob_ber=base.bob_ber,
        bob_exact_recovery=base.bob_exact_recovery,
        eve_pguess_at_attack=base.eve_pguess_after_key,
        key_delay_seconds=delay,
        eve_pguess_at_key_release=p_eve,
        eve_information_bits=eve_bits,
        h_min_bits=h_min,
        ec_leakage_bits=leak,
        secret_bits=secret,
        secret_rate=rate,
        secret_bits_without_delay=secret_without_delay,
        requires_storage_assumption=bool(secret_without_delay <= 0 < secret),
        deletion_certifiable=bool(window["certifiable"]),
        deletion_within_window=within_window,
        deletion_window_over_T2=float(window["window_over_T2"]),
        deletion_tolerance=float(window["recommended_tolerance"]),
        honest_certificate_accepted=honest_ok,
        cheating_certificate_accepted=cheat_ok,
        decay_certificate_accepted=decay_ok,
        usable=usable,
        verdict=verdict,
    )


#: Layer configurations for the ablation study: which primitive does what.
ABLATIONS: dict[str, StackConfig] = {
    "bare (protocol B)": StackConfig(False, False, False, False),
    "+ error correction": StackConfig(True, False, False, False),
    "+ privacy amplification": StackConfig(True, True, False, False),
    "+ delayed key release": StackConfig(True, True, False, True),
    "+ certified deletion (full C)": StackConfig(True, True, True, True),
}
