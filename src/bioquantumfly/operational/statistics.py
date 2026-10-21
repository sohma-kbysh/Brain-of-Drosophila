"""Finite-shot intervals and conservative affine-dimension witnesses."""

from itertools import combinations, product

import numpy as np
from scipy.stats import beta


def simultaneous_binomial(counts, shots, *, alpha=0.05, family_size=None):
    """Bonferroni Clopper–Pearson intervals; dependence across cells is allowed.

    Trials within each cell must be identically distributed and independent.
    family_size includes every inspected shot level/grid in the stated family.
    """
    k = np.asarray(counts)
    if shots <= 0 or not 0 < alpha < 1:
        raise ValueError("positive shots and alpha in (0,1) required")
    if not np.all(np.isfinite(k)) or np.any(k != np.floor(k)) or np.any((k < 0) | (k > shots)):
        raise ValueError("counts must be integers between zero and shots")
    family_size = k.size if family_size is None else family_size
    if family_size < k.size:
        raise ValueError("family_size must cover the entire table")
    tail = alpha / (2 * family_size)
    lo = np.zeros(k.shape, dtype=float)
    hi = np.ones(k.shape, dtype=float)
    positive, nonfull = k > 0, k < shots
    lo[positive] = beta.ppf(tail, k[positive], shots - k[positive] + 1)
    hi[nonfull] = beta.ppf(1 - tail, k[nonfull] + 1, shots - k[nonfull])
    return lo, hi


def affine_rank_witness(lo, hi, *, margin=1e-12):
    """Find an interval-certified nonzero 3x3 minor with a unit column.

    The determinant is multi-affine in the six uncertain entries, so its
    extrema on a box occur at its 64 corners. Such a minor proves rank >= 3,
    without estimating exact preparation equivalences from near-equality.
    Arithmetic uses a conservative margin; this is not a general rank solver.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    if lo.shape != hi.shape or lo.ndim != 2 or np.any(lo > hi):
        raise ValueError("invalid interval table")
    best = None
    for rows in combinations(range(lo.shape[0]), 3):
        for cols in combinations(range(lo.shape[1]), 2):
            lower = lo[np.ix_(rows, cols)].ravel()
            upper = hi[np.ix_(rows, cols)].ravel()
            corners = np.array([np.where(bits, upper, lower) for bits in product((0, 1), repeat=6)])
            a, b, c, d, e, f = corners.T
            determinants = (c - a) * (f - b) - (e - a) * (d - b)
            low, high = float(determinants.min()), float(determinants.max())
            separation = max(low, -high)
            if best is None or separation > best["separation"]:
                best = {"rows": list(rows), "effects": list(cols), "determinant_interval": [low, high],
                        "separation": separation, "certified_rank_at_least_3": separation > margin}
    return best or {"certified_rank_at_least_3": False}


def factor_probabilities(table, dimension):
    """Centered truncated SVD with exact unit coordinate (not PCA geometry).

    Produces an affine-rank approximation; out-of-range probabilities are
    rejected by the caller rather than clipped, which would change the rank.
    """
    f = np.asarray(table, dtype=float)
    if f.ndim != 2 or not 1 <= dimension <= min(f.shape[0], f.shape[1] + 1):
        raise ValueError("invalid table or linear dimension")
    mean = f.mean(axis=0)
    left, singular, right = np.linalg.svd(f - mean, full_matrices=False)
    k = dimension - 1
    states = np.column_stack([np.ones(f.shape[0]), left[:, :k] * singular[:k]])
    effects = np.vstack([mean, right[:k]])
    return states, effects


def select_dimension(train, validation, *, validation_shots=None, alpha=0.05):
    """Independent-table Brier validation, no oracle dimension in the fit.

    With validation_shots, choose the smallest candidate not demonstrably
    worse than the empirical winner. Paired-loss Hoeffding intervals are
    simultaneous over all fitted candidate pairs (conditional on training).
    Validation cells must be independent binomial counts. This is a model
    selection rule, not a confidence certificate of the true dimension.
    """
    scores = []
    predictions = {}
    for d in range(1, min(train.shape[0], train.shape[1] + 1) + 1):
        s, e = factor_probabilities(train, d)
        prediction = s @ e
        valid = bool(np.all((prediction >= 0) & (prediction <= 1)))
        score = float(np.mean((prediction - validation) ** 2)) if valid else None
        scores.append({"dimension": d, "validation_mse": score, "probabilities_valid": valid})
        if valid: predictions[d] = prediction
    eligible = [r for r in scores if r["probabilities_valid"]]
    if not eligible:
        raise RuntimeError("no valid probability reconstruction")
    best = min(eligible, key=lambda r: (r["validation_mse"], r["dimension"]))
    if validation_shots is None:
        return best["dimension"], scores
    if validation_shots <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid validation uncertainty parameters")
    pairs = max(1, len(eligible)*(len(eligible)-1)//2)
    cells = train.size
    for row in eligible:
        diff = predictions[row["dimension"]]-predictions[best["dimension"]]
        radius = float(np.sqrt(2*np.log(2*pairs/alpha)*np.sum(diff**2)/(validation_shots*cells**2)))
        excess = row["validation_mse"]-best["validation_mse"]
        row.update({"raw_minimum_dimension": best["dimension"], "excess_validation_mse": excess,
                    "paired_hoeffding_radius": radius, "not_demonstrably_worse": excess <= radius+1e-15})
    return min(row["dimension"] for row in eligible if row["not_demonstrably_worse"]), scores
