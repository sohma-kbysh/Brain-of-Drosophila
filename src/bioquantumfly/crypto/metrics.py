"""Protocol-level security metrics (spec sections 13, 14).

The timeline modelled:

    t = 0        Alice writes the carriers into the host
    t = t_attack Eve interacts (measurement, weak POVM, coherent coupling, SWAP)
    t = t_read   Bob reads, in the basis he knows
    t = t_reveal the key (s, theta) becomes public; Eve extracts from whatever
                 she kept, which has meanwhile decohered in HER memory

Two adversary regimes are reported separately and must not be merged:

  * before key reveal -- Eve holds only her interaction record;
  * after key reveal  -- Eve additionally knows (s, theta).

For Protocol B the first is exactly zero by the one-time pad, and the second is
where the SWAP control does its damage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bioquantumfly.analysis.metrics import (
    fidelity, helstrom_guessing_probability, holevo_chi,
    mutual_information_binary, trace_distance,
)
from bioquantumfly.crypto import bob as bob_module
from bioquantumfly.crypto.encoding import Encoding, Protocol
from bioquantumfly.crypto.eve import Attack
from bioquantumfly.quantum.abstract_memory import MemoryParams
from bioquantumfly.quantum.channels import t1_t2_channel
from bioquantumfly.quantum.states import encode_bb84


@dataclass
class SecurityMetrics:
    """One evaluated protocol instance."""

    # --- Bob ---
    bob_ber: float
    bob_exact_recovery: float
    bob_mean_correct: float
    bob_disturbance_ber: float
    bob_state_fidelity_vs_unattacked: float
    bob_trace_distance_vs_unattacked: float
    # --- Eve ---
    eve_pguess_before_key: float
    eve_pguess_after_key: float
    eve_info_before_key: float
    eve_info_after_key: float
    eve_holevo_before_key: float
    # --- host ---
    # Two DIFFERENT quantities, both reported because they answer different
    # questions. `*_at_read` is what the host says about the message BEFORE the
    # key is revealed -- for the masked protocol that is identically zero by the
    # one-time pad, whatever the physics. `*_after_key` is what someone holding
    # the key could still recover from the host, which is the quantity that
    # actually decays with T2 and the one to compare against Eve's.
    host_info_at_read: float
    host_holevo_at_read: float
    host_pguess_after_key: float
    host_info_after_key: float
    host_bloch_norm: float
    # --- context ---
    attack: str
    attack_strength: float
    protocol: str
    n_carriers: int
    t_attack: float
    t_read: float
    t_reveal: float
    T2_host: float
    T1_host: float
    T2_eve: float
    T1_eve: float
    t_over_T2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evolve(r: NDArray, duration: float, params: MemoryParams) -> NDArray:
    if duration <= 0:
        return np.array(r, copy=True)
    return t1_t2_channel(r, duration, params.T1, params.T2)


def _prior_mixture(
    protocol: Protocol,
    message_bit: int,
    *,
    host: MemoryParams,
    eve_memory: MemoryParams | None,
    attack: Attack,
    t_attack: float,
    t_final: float,
    view: str,
) -> NDArray[np.float64]:
    """The state an observer holds, averaged over their prior uncertainty.

    `view` selects whose state to build: ``"eve"`` follows the branch's ancilla
    map and then Eve's own memory; ``"host"`` follows the host through the
    attack and onward under host decoherence.

    Because Bloch vectors are linear in the density matrix, mixing hypotheses is
    just a weighted sum.
    """
    total = np.zeros(3)
    for weight, x_bit, theta in protocol.prior_hypotheses(message_bit):
        r0 = encode_bb84(np.array([x_bit]), np.array([theta]))[0]
        r_at_attack = _evolve(r0[None, :], t_attack, host)
        for branch in attack.branches:
            if view == "eve":
                if branch.eve is None:
                    continue  # this branch retains nothing
                kept = branch.eve(r_at_attack)
                kept = _evolve(kept, t_final - t_attack, eve_memory or MemoryParams.ideal())
            else:
                kept = _evolve(branch.host(r_at_attack), t_final - t_attack, host)
            total = total + weight * branch.probability * kept[0]
    return total


def evaluate(
    encoding: Encoding,
    protocol: Protocol,
    attack: Attack,
    *,
    host: MemoryParams,
    eve_memory: MemoryParams | None = None,
    t_attack: float = 0.0,
    t_read: float = 0.0,
    t_reveal: float | None = None,
) -> SecurityMetrics:
    """Evaluate one (encoding, attack, timing) configuration exactly."""
    if t_attack < 0 or t_read < 0:
        raise ValueError("times must be non-negative")
    if t_attack > t_read:
        raise ValueError(
            f"t_attack ({t_attack:g}) is after t_read ({t_read:g}); Eve cannot "
            "attack a state Bob has already destructively measured"
        )
    eve_memory = eve_memory or MemoryParams.ideal()
    t_reveal = t_read if t_reveal is None else t_reveal
    if t_reveal < t_attack:
        raise ValueError("key reveal cannot precede the attack")

    # ---- host trajectory -------------------------------------------------
    r_initial = encoding.initial_state()
    r_at_attack = _evolve(r_initial, t_attack, host)
    r_attacked = attack.apply_to_host(r_at_attack)
    r_read = _evolve(r_attacked, t_read - t_attack, host)

    # Counterfactual: the same storage time with no adversary. The difference
    # is the disturbance Eve caused, as opposed to ordinary decoherence.
    r_read_clean = _evolve(r_at_attack, t_read - t_attack, host)

    bob = bob_module.read_metrics(r_read, encoding)
    bob_clean = bob_module.read_metrics(r_read_clean, encoding)

    # ---- Eve, before the key is revealed ---------------------------------
    eve_prior = np.stack([
        _prior_mixture(protocol, m, host=host, eve_memory=eve_memory, attack=attack,
                       t_attack=t_attack, t_final=t_reveal, view="eve")
        for m in (0, 1)
    ])
    p_before = float(helstrom_guessing_probability(eve_prior[0], eve_prior[1]))
    chi_before = float(holevo_chi(eve_prior))

    # ---- Eve, after the key is revealed ----------------------------------
    # She now knows theta and s, so per carrier she need only distinguish
    # encoded bit 0 from encoded bit 1 in a known basis.
    hyp = encoding.hypothesis_states()                       # (2, n, 3)
    p_after = 0.0
    for branch in attack.branches:
        if branch.eve is None:
            p_after += branch.probability * 0.5              # nothing retained
            continue
        kept = []
        for k in (0, 1):
            at_attack = _evolve(hyp[k], t_attack, host)
            kept.append(_evolve(branch.eve(at_attack), t_reveal - t_attack, eve_memory))
        p_after += branch.probability * float(
            np.mean(helstrom_guessing_probability(kept[0], kept[1]))
        )

    # ---- what the host still carries, to someone holding the key ---------
    # Same optimal-discrimination measure as Eve's post-key-reveal number, so
    # the two are directly comparable on one axis in Experiment 6/7.
    host_hyp = []
    for k in (0, 1):
        at_attack = _evolve(hyp[k], t_attack, host)
        host_hyp.append(_evolve(attack.apply_to_host(at_attack), t_read - t_attack, host))
    host_p_after = float(np.mean(helstrom_guessing_probability(host_hyp[0], host_hyp[1])))

    # ---- what the host itself still carries, before key reveal -----------
    host_prior = np.stack([
        _prior_mixture(protocol, m, host=host, eve_memory=eve_memory, attack=attack,
                       t_attack=t_attack, t_final=t_read, view="host")
        for m in (0, 1)
    ])
    host_p = float(helstrom_guessing_probability(host_prior[0], host_prior[1]))

    return SecurityMetrics(
        bob_ber=bob["bob_ber"],
        bob_exact_recovery=bob["bob_exact_recovery"],
        bob_mean_correct=bob["bob_mean_correct"],
        bob_disturbance_ber=bob["bob_ber"] - bob_clean["bob_ber"],
        bob_state_fidelity_vs_unattacked=float(np.mean(fidelity(r_read, r_read_clean))),
        bob_trace_distance_vs_unattacked=float(np.mean(trace_distance(r_read, r_read_clean))),
        eve_pguess_before_key=p_before,
        eve_pguess_after_key=p_after,
        eve_info_before_key=float(mutual_information_binary(p_before)),
        eve_info_after_key=float(mutual_information_binary(p_after)),
        eve_holevo_before_key=chi_before,
        host_info_at_read=float(mutual_information_binary(host_p)),
        host_holevo_at_read=float(holevo_chi(host_prior)),
        host_pguess_after_key=host_p_after,
        host_info_after_key=float(mutual_information_binary(host_p_after)),
        host_bloch_norm=float(np.mean(np.linalg.norm(r_read, axis=-1))),
        attack=attack.name,
        attack_strength=attack.strength,
        protocol=encoding.protocol,
        n_carriers=encoding.n_carriers,
        t_attack=t_attack,
        t_read=t_read,
        t_reveal=t_reveal,
        T2_host=host.T2,
        T1_host=host.T1,
        T2_eve=eve_memory.T2,
        T1_eve=eve_memory.T1,
        t_over_T2=float(t_read / host.T2) if np.isfinite(host.T2) else 0.0,
    )
