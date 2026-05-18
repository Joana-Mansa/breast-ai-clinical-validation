"""Shared, dependency-light helpers used across the metrics package."""
from __future__ import annotations

import numpy as np
from scipy import stats


def as_float_array(x, name="array"):
    """Return ``x`` as a contiguous 1-D float array."""
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


def check_binary(y, name="y_true"):
    """Validate and return a 0/1 integer label vector.

    Raises if labels are not binary or if one class is missing — both
    conditions make discrimination metrics undefined and are common silent
    failure modes when a sub-cohort is sliced too thin.
    """
    y = np.asarray(y).ravel()
    y_int = y.astype(int)
    if not np.array_equal(y_int, y.astype(float)):
        raise ValueError(f"{name} must be integer 0/1 labels")
    extra = set(np.unique(y_int)) - {0, 1}
    if extra:
        raise ValueError(f"{name} must contain only 0/1; found {sorted(extra)}")
    n_pos = int(y_int.sum())
    if n_pos == 0 or n_pos == len(y_int):
        raise ValueError(
            f"{name} must contain both classes (got {n_pos} positives of {len(y_int)})"
        )
    return y_int


def wilson_ci(successes, n, alpha=0.95):
    """Wilson score confidence interval for a binomial proportion.

    Preferred over the normal-approximation ('Wald') interval because it stays
    inside [0, 1] and remains well behaved for the small counts and extreme
    proportions typical of screening sub-cohorts (e.g. sensitivity estimated
    from a handful of cancers in a single breast-density stratum).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def proportion(successes, n):
    """Safe proportion that returns NaN instead of dividing by zero."""
    return float(successes) / n if n else float("nan")
