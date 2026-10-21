"""Finite open-loop predictive models and independent-target evaluation.

Linear coordinates/updates are hypotheses, not a global GPT reconstruction.
No probability clipping is used; validity violations remain visible.
"""

from itertools import product

import numpy as np


def words(length):
    return ["".join(bits) for bits in product("01", repeat=length)]


def words_through(length):
    return [word for size in range(length+1) for word in words(size)]


def fit_predictive(table, futures, core, dimension, ridge):
    """Fit row-state updates from D(h,q) and D(h,u q), using training only.

    dimension excludes the exact unit coordinate. ridge multiplies the sum
    of squared slopes in a mean-squared (per preparation) regression loss.
    """
    table = np.asarray(table, dtype=float)
    if (table.ndim != 2 or table.shape[1] != len(futures)
            or len(set(futures)) != len(futures) or not core or core[0] != ""
            or len(set(core)) != len(core) or not np.isfinite(table).all()
            or np.any((table < 0) | (table > 1))
            or type(dimension) is not int or not 1 <= dimension <= min(len(core), table.shape[0]-1)
            or not np.isfinite(ridge) or ridge < 0):
        raise ValueError("invalid training probabilities, core, dimension or ridge")
    if not set(core + [u+q for u in "01" for q in core]) <= set(futures):
        raise ValueError("training table must contain core and both action-prefixed cores")
    indices = [futures.index(q) for q in core]
    x = table[:, indices]
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x-mean, full_matrices=False)
    basis = vt[:dimension].T
    state = np.c_[np.ones(len(x)), (x-mean)@basis]
    penalty = np.diag([0.]+[ridge*len(x)]*dimension)
    updates = {}
    for u in "01":
        following = table[:, [futures.index(u+q) for q in core]]
        target = (following-mean)@basis
        gram = state.T@state+penalty
        slopes = np.linalg.lstsq(gram, state.T@target, rcond=None)[0]
        action = np.zeros((dimension+1, dimension+1))
        action[0, 0] = 1
        action[:, 1:] = slopes
        updates[u] = action
    return {"dimension": dimension, "linear_dimension": dimension+1, "ridge": ridge,
            "core": list(core), "mean": mean, "basis": basis,
            "updates": updates, "effect": np.r_[mean[0], basis[0]],
            "training_core_singular_values": singular,
            "training_state_condition": float(np.linalg.cond(state))}


def encode(model, anchors):
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != len(model["core"]):
        raise ValueError("anchor columns must match the model core")
    return np.c_[np.ones(len(anchors)), (anchors-np.asarray(model["mean"]))@np.asarray(model["basis"])]


def predict(model, anchors, futures):
    state = encode(model, anchors)
    columns = []
    for future in futures:
        if set(future)-set("01"):
            raise ValueError("future must be a binary string")
        next_state = state.copy()
        for u in future:
            next_state = next_state@np.asarray(model["updates"][u])
        columns.append(next_state@np.asarray(model["effect"]))
    return np.column_stack(columns)


def fit_suffix_baseline(histories, immediate):
    """Last two input bins, not neural spike history."""
    return {suffix: float(np.mean([p for h, p in zip(histories, immediate) if h[-2:] == suffix]))
            for suffix in words(2)}


def predict_suffix(model, histories, futures):
    return np.array([[model[(h+f)[-2:]] for f in futures] for h in histories])


def probability_diagnostics(prediction, tolerance=1e-10):
    p = np.asarray(prediction)
    finite = bool(np.isfinite(p).all())
    return {"valid": finite and bool(np.all((p >= -tolerance) & (p <= 1+tolerance))),
            "minimum": float(p.min()) if finite else None,
            "maximum": float(p.max()) if finite else None,
            "out_of_range_count": int(np.count_nonzero((p < -tolerance) | (p > 1+tolerance))) if finite else p.size}


def choose_candidate(candidates, predictions, validation_counts, shots, *, alpha=.025):
    """Fixed-model paired validation; state anchors independent of labels.

    Choose smallest dimension then strongest ridge among candidates not
    demonstrably worse than the empirical winner. No true rank is supplied.
    """
    if shots <= 0 or not 0 < alpha < 1 or len(candidates) != len(predictions):
        raise ValueError("invalid candidate selection arguments")
    frequency = np.asarray(validation_counts)/shots
    rows = []
    for i, (model, prediction) in enumerate(zip(candidates, predictions)):
        diag = probability_diagnostics(prediction)
        rows.append({"index": i, "dimension": model["dimension"], "ridge": model["ridge"],
                     "probabilities": diag,
                     "validation_mse": float(np.mean((prediction-frequency)**2)) if diag["valid"] else None})
    valid = [row for row in rows if row["probabilities"]["valid"]]
    if not valid:
        return None, rows
    best = min(valid, key=lambda row: row["validation_mse"])
    pairs = max(1, len(valid)*(len(valid)-1)//2)
    cells = frequency.size
    for row in valid:
        difference = predictions[row["index"]]-predictions[best["index"]]
        radius = float(np.sqrt(2*np.log(2*pairs/alpha)*np.sum(difference**2)/(shots*cells**2)))
        excess = row["validation_mse"]-best["validation_mse"]
        row.update({"paired_radius": radius, "excess_mse": excess,
                    "eligible": bool(excess <= radius+1e-15)})
    chosen = min((row for row in valid if row["eligible"]), key=lambda row: (row["dimension"], -row["ridge"]))
    return chosen["index"], rows


def probability_error_bounds(prediction, lo, hi):
    """Bounds on error to true probabilities inside a simultaneous box."""
    p, lo, hi = map(np.asarray, (prediction, lo, hi))
    if p.shape != lo.shape or lo.shape != hi.shape or np.any(lo > hi) or not np.isfinite(p).all():
        raise ValueError("invalid prediction/interval table")
    minimum = np.maximum(np.maximum(lo-p, p-hi), 0)
    maximum = np.maximum(np.abs(p-lo), np.abs(p-hi))
    return {"rmse_lower": float(np.sqrt(np.mean(minimum**2))),
            "rmse_upper": float(np.sqrt(np.mean(maximum**2))),
            "maximum_error_lower": float(minimum.max()),
            "maximum_error_upper": float(maximum.max())}


def brier_improvement_bounds(baseline, candidate, lo, hi):
    """Positive means candidate improves expected Brier loss, same targets.

    E[(b-Y)^2-(c-Y)^2] = b^2-c^2 - 2(b-c)p is affine in p.
    """
    b, c, lo, hi = map(np.asarray, (baseline, candidate, lo, hi))
    if not b.shape == c.shape == lo.shape == hi.shape or np.any(lo > hi):
        raise ValueError("mismatched prediction/interval tables")
    a, d = b*b-c*c-2*(b-c)*lo, b*b-c*c-2*(b-c)*hi
    return [float(np.minimum(a, d).mean()), float(np.maximum(a, d).mean())]
