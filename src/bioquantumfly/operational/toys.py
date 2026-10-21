"""Exactly specified classical parents, used only for calibration."""

import numpy as np
import sympy as sy


def rational_matrix(array):
    return sy.Matrix([[sy.Rational(str(x)) for x in row] for row in np.asarray(array).tolist()])


def exact_rank(array):
    return int(rational_matrix(array).rank())


def preparation_relations(states):
    """All row dependencies from declared exact states, never noisy data."""
    m = rational_matrix(states)
    basis = m.T.nullspace()
    if not basis:
        return np.empty((0, m.rows))
    return np.array([list(v) for v in basis], dtype=float)


def relative_tomography(states, effects):
    """Oracle audit of the inherited fragment, including the unit effect.

    Data-only minimal factorization cannot establish this parent-space test.
    All toy entries are terminating rational numbers, converted exactly.
    """
    s, e = rational_matrix(states), rational_matrix(effects)
    rs, re, rd = int(s.rank()), int(e.rank()), int((s * e).rank())
    return {"parent_microstate_count": s.cols, "state_span_dimension": rs,
            "effect_span_dimension": re, "pairing_rank": rd,
            "states_separated_by_effects": rd == rs,
            "effects_separated_by_states": rd == re,
            "relative_tomography": rd == rs == re,
            "audit_basis": "known exact parent; not inferred from noisy minimal factorization"}


def compression_toy():
    """Six microstates, three predictive categories, nontrivial hidden dynamics.

    Preparations fix the within-pair conditional distribution to uniform.
    All effects are constant on each pair. Both spans have dimension three,
    even though neither probes the complete six-state classical theory.
    """
    coarse = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0],
                       [.5, 0, .5], [0, .5, .5], [.25, .25, .5], [.125, .375, .5]])
    lift = np.kron(np.eye(3), [[.5, .5]])
    collect = np.kron(np.eye(3), [[1], [1]])
    response = np.array([[.85, .15, .15, .2, .75, .4],
                         [.15, .85, .15, .5, .25, .7],
                         [.15, .15, .85, .8, .5, .3]])
    states = coarse @ lift
    effects = collect @ response
    kv = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
    kernel = np.kron(kv, np.full((2, 2), .5))  # row-stochastic convention
    return {"states": states, "effects": effects, "coarse_states": coarse,
            "coarse_effects": response, "lift": lift, "collect": collect,
            "kernel": kernel, "coarse_kernel": kv, "probabilities": states @ effects}


def shadow_toy(visibility=1.0):
    if not 0 < visibility <= 1:
        raise ValueError("visibility must be in (0,1]")
    # 00,01,10,11; preparations include actual unlabeled classical mixtures.
    states = np.vstack([np.eye(4), [.5, 0, 0, .5], [0, .5, .5, 0], [.25] * 4])
    full = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    full = .5 + visibility * (full - .5)
    return {"states": states, "adequate_effects": full, "hidden_effects": full[:, :2],
            "adequate_probabilities": states @ full, "hidden_probabilities": states @ full[:, :2]}
