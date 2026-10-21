"""Small finite-scenario LP, not a replacement for a general GPT cone solver.

Binary measurement responses are independent coordinates together with unit.
Their full response-polytope vertices are all binary assignments. Callers must
not use this routine if additional effect linear identities are required.
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog


def _constraints(n_preps, vertices, relations):
    nv = len(vertices)
    equalities, rhs = [], []
    for i in range(n_preps):
        a = np.zeros((n_preps, nv)); a[i, :] = 1
        equalities.append(a.ravel()); rhs.append(1.)
    for rel in relations:
        for v in range(nv):
            a = np.zeros((n_preps, nv)); a[:, v] = rel
            equalities.append(a.ravel()); rhs.append(0.)
    evaluation = np.zeros((n_preps * vertices.shape[1], n_preps * nv))
    for i in range(n_preps):
        evaluation[i * vertices.shape[1]:(i + 1) * vertices.shape[1], i * nv:(i + 1) * nv] = vertices.T
    return np.array(equalities), np.array(rhs), evaluation


def embedding_lp(lower, upper, relations, *, tolerance=1e-8):
    """Noncontextual feasibility within probability intervals and fixed identities.

    'infeasible' is a numerical solver verdict, not an independent certificate.
    The runner uses an analytic POM inequality for robust nonembedding claims.
    """
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    relations = np.asarray(relations, float)
    if low.shape != high.shape or low.ndim != 2 or np.any(low > high):
        raise ValueError("invalid intervals")
    if np.any(low < 0) or np.any(high > 1) or not np.all(np.isfinite(low + high)):
        raise ValueError("probability bounds must be finite in [0,1]")
    m, n = low.shape
    if n > 8 or m > 100:
        raise ValueError("this explicit vertex enumerator is for tiny scenarios only")
    if relations.ndim != 2 or relations.shape[1] != m or not np.all(np.isfinite(relations)):
        raise ValueError("invalid preparation identities")
    vertices = np.array(list(product((0., 1.), repeat=n)))
    ae, be, evaluation = _constraints(m, vertices, relations)
    # First check that the declared preparation identities admit any table.
    table_eq = np.kron(relations, np.eye(n)) if len(relations) else None
    consistent = linprog(np.zeros(m*n), A_eq=table_eq,
                         b_eq=np.zeros(len(relations)*n) if len(relations) else None,
                         bounds=list(zip(low.ravel(), high.ravel())), method="highs")
    if not consistent.success:
        return {"status": "scenario_inconsistent" if consistent.status == 2 else "solver_unknown",
                "solver_message": consistent.message}
    result = linprog(np.zeros(m*len(vertices)), A_eq=ae, b_eq=be,
                     A_ub=np.vstack([evaluation, -evaluation]),
                     b_ub=np.r_[high.ravel(), -low.ravel()], bounds=(0, None), method="highs")
    if not result.success:
        return {"status": "infeasible_numeric" if result.status == 2 else "solver_unknown",
                "solver_message": result.message}
    x = result.x
    residual = max(float(np.max(np.abs(ae @ x - be))), float(max(0, -x.min())),
                   float(max(0, np.max(evaluation @ x - high.ravel()))),
                   float(max(0, np.max(low.ravel() - evaluation @ x))))
    return {"status": "feasible" if residual <= tolerance else "residual_failure",
            "max_residual": residual, "ontic_preparations": x.reshape(m, -1).tolist(),
            "ontic_responses": vertices.tolist()}


def replacement_robustness(table, relations):
    """Minimum replacement by p=1/2, with fixed input-scenario identities."""
    d = np.asarray(table)
    m, n = d.shape
    if n > 8 or m > 100:
        raise ValueError("tiny scenarios only")
    vertices = np.array(list(product((0., 1.), repeat=n)))
    ae, be, evaluation = _constraints(m, vertices, relations)
    # mu.v = D + r*(1/2-D)
    equalities = np.vstack([np.column_stack([ae, np.zeros(len(ae))]),
                            np.column_stack([evaluation, (d-.5).ravel()])])
    rhs = np.r_[be, d.ravel()]
    objective = np.zeros(m*len(vertices)+1); objective[-1] = 1
    result = linprog(objective, A_eq=equalities, b_eq=rhs,
                     bounds=[(0, None)]*(len(objective)-1)+[(0, 1)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    residual = float(np.max(np.abs(equalities @ result.x-rhs)))
    if residual > 1e-8:
        raise RuntimeError("robustness residual too large")
    return {"replacement_probability": float(result.x[-1]), "max_residual": residual,
            "noise": "fixed-scenario replacement by uniform preparation"}


def pom_success_bounds(lower, upper):
    """POM success for first four preparations in 00,01,10,11 order."""
    bits = np.array(list(product((0, 1), repeat=2)))
    low = np.asarray(lower)[:4, :2]; high = np.asarray(upper)[:4, :2]
    return float(np.where(bits, low, 1-high).mean()), float(np.where(bits, high, 1-low).mean())
