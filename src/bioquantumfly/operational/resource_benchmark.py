"""Finite POM metrics and a physical-trial ledger; no automatic NC certification."""

from dataclasses import asdict, dataclass
from itertools import product
import numpy as np
from scipy.optimize import linprog

BITS = np.array(list(product((0, 1), repeat=2)), dtype=int)


def transformed_bits(word):
    result = BITS.copy()
    for gate in word:
        if gate == "F":
            result[:, 0] = 1 - result[:, 0]
        elif gate == "W":
            result = result[:, ::-1].copy()
        else:
            raise ValueError("only F and W are gates")
    return result


def gate_words(max_length):
    return ["".join(w) for n in range(max_length + 1)
            for w in product("FW", repeat=n)]


def metrics(lower, upper, word=""):
    """Intervals for four preparations and two binary readouts.

    Parity gap is over precisely these two effects, not an ontic-TV bound.
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (lo.shape != (4, 2) or hi.shape != lo.shape
            or not np.all(np.isfinite(lo + hi))
            or np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)):
        raise ValueError("expected valid 4 by 2 probability bounds")
    target = transformed_bits(word)
    success_lo = float(np.where(target, lo, 1 - hi).mean())
    success_hi = float(np.where(target, hi, 1 - lo).mean())
    diff_lo = (lo[0] + lo[3] - hi[1] - hi[2]) / 2
    diff_hi = (hi[0] + hi[3] - lo[1] - lo[2]) / 2
    gap_lo = float(np.maximum(0, np.maximum(diff_lo, -diff_hi)).max())
    gap_hi = float(np.maximum(np.abs(diff_lo), np.abs(diff_hi)).max())
    noise = lambda s: max(0., (s - .75) / (s - .5)) if s > .5 else 0.
    return {
        "success_interval": [success_lo, success_hi],
        "parity_gap_interval": [gap_lo, gap_hi],
        "required_ontic_parity_tv_lower": max(0., 4 * success_lo - 3),
        "conditional_witness_noise_interval": [noise(success_lo), noise(success_hi)],
        "finite_interface_candidate_pass": success_lo >= .8 and gap_hi <= .02,
        "exact_noncontextuality_certified": False,
    }


def relaxed_pom_optimum(theta):
    """Independent finite LP for S <= .75 + theta/4, 0 <= theta <= 1."""
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0,1]")
    objective = np.r_[-(BITS[:, None, :] == BITS[None, :, :]).sum(axis=2).ravel()/8,
                      np.zeros(4)]
    eq = np.zeros((4, 20))
    for x in range(4):
        eq[x, 4*x:4*x+4] = 1
    ub = []
    for lam in range(4):
        for sign in (-1, 1):
            row = np.zeros(20)
            for x, c in enumerate((1, -1, -1, 1)):
                row[4*x+lam] = sign*c
            row[16+lam] = -1
            ub.append(row)
    ub.append(np.r_[np.zeros(16), np.ones(4)])
    result = linprog(objective, A_ub=np.array(ub),
                     b_ub=np.r_[np.zeros(8), 4*theta],
                     A_eq=eq, b_eq=np.ones(4), bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def reference_fixtures():
    tables = {
        "one_classical_bit": np.c_[BITS[:, 0], np.full(4, .5)],
        "qubit_reference": .5 + (BITS - .5) / np.sqrt(2),
        "two_bit_shadow": BITS.astype(float),
        "weak_shadow": .5 + .4 * (BITS - .5),
    }
    result = {name: metrics(table, table) for name, table in tables.items()}
    result["ontic_tv_lp"] = [
        {"theta": t, "lp_success": relaxed_pom_optimum(t),
         "analytic_success": .75 + .25*t}
        for t in (0., .02, .05, .1, .2, .5, 1.)
    ]
    # One bit from each of two same-x copies, or both raw bits of one classical
    # two-bit device, determines parity. The resource interface has changed.
    parity = BITS[:, 0] ^ BITS[:, 1]
    result["opened_classical_interface"] = {
        "parity_gap": float(abs(parity[[0, 3]].mean()-parity[[1, 2]].mean())),
        "parity_success": 1., "old_parity_equivalence_valid": False,
        "same_input_two_copy_parity_success": 1.,
        "single_copy_joint_readout_bits": 2,
        "two_copy_selected_readout_bits": 2,
    }
    result["shadow_gate_success"] = {
        word: metrics(transformed_bits(word), transformed_bits(word), word)["success_interval"][0]
        for word in gate_words(4)
    }
    return result


@dataclass
class CostLedger:
    """Additive counts and proxies, with no conversion of activity into joules."""
    max_physical_trials: int
    physical_trials: int = 0
    neuron_time_ms: float = 0.
    emitted_spikes: int = 0
    external_noise_events: int = 0
    absolute_drive_neuron_ms: float = 0.
    readout_bits: int = 0
    simulator_seconds: float = 0.
    reset_operations: int = 0

    def reserve(self, trials):
        if type(trials) is not int or trials <= 0:
            raise ValueError("positive integral trial count required")
        if self.physical_trials + trials > self.max_physical_trials:
            raise RuntimeError("physical-trial budget exhausted before sampling")
        self.physical_trials += trials
        self.reset_operations += trials

    def record(self, *, trials, neurons, duration_ms, spikes, noise_events,
               absolute_drive_neuron_ms, readout_bits_per_trial, simulator_seconds):
        if any(x < 0 for x in (trials, neurons, duration_ms, spikes, noise_events,
                              absolute_drive_neuron_ms, simulator_seconds)):
            raise ValueError("costs cannot be negative")
        self.neuron_time_ms += trials * neurons * duration_ms
        self.emitted_spikes += int(spikes)
        self.external_noise_events += int(noise_events)
        self.absolute_drive_neuron_ms += float(absolute_drive_neuron_ms)
        self.readout_bits += trials * readout_bits_per_trial
        self.simulator_seconds += float(simulator_seconds)

    def to_dict(self):
        return asdict(self)
