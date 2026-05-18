"""Resampling-based confidence intervals and paired model comparisons.

DeLong (see :mod:`mammoval.metrics.delong`) is exact and fast, but only for the
AUC. For *every other* metric a clinical validation report needs a CI for —
sensitivity at a fixed specificity, recall rate, PPV, FROC sensitivity, the
difference between two such metrics — the bootstrap is the general tool.

Two design points that matter for mammography:

* **Stratified resampling.** Cancer is rare. An unstratified bootstrap can draw
  a replicate with zero cancers, making the metric undefined and biasing the
  interval. Resampling within each class fixes the class counts and removes
  that failure mode.
* **Cluster (patient-level) resampling.** A screening exam contributes up to
  four views (L/R x CC/MLO) from one woman; those views are correlated.
  Treating views as independent under-estimates the variance. ``cluster_*``
  functions resample whole patients, which is the statistically honest choice
  when the unit of analysis is the view but the unit of independence is the
  woman.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_ci", "paired_bootstrap_diff", "cluster_bootstrap_ci"]


def _percentile_ci(samples, alpha):
    lo = float(np.nanpercentile(samples, 100 * (1 - alpha) / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - (1 - alpha) / 2)))
    return lo, hi


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, alpha=0.95,
                 stratified=True, seed=12345):
    """Percentile bootstrap CI for an arbitrary scalar metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_score) -> float
        Any metric reducible to a single number, e.g.
        ``lambda yt, ys: sensitivity_at_specificity(yt, ys, 0.9)["sensitivity"]``.
    stratified : bool
        Resample within each class so cancer/non-cancer counts are preserved.

    Returns
    -------
    dict: point estimate, ci_low, ci_high, se (bootstrap SD), n_boot, alpha.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))

    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

    samples = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            samples[b] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            samples[b] = np.nan

    lo, hi = _percentile_ci(samples, alpha)
    return {
        "estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "se": float(np.nanstd(samples, ddof=1)),
        "n_boot": n_boot,
        "alpha": alpha,
        "n_valid": int(np.sum(~np.isnan(samples))),
    }


def paired_bootstrap_diff(y_true, y_score_a, y_score_b, metric_fn,
                          n_boot=2000, alpha=0.95, stratified=True, seed=12345):
    """Bootstrap CI and two-sided p-value for ``metric(A) - metric(B)``.

    The *same* resampled indices are applied to both models on every replicate,
    preserving the case-level pairing — the resampling analogue of the paired
    DeLong test, but usable for any metric (sensitivity at fixed specificity,
    PPV, recall rate, ...).

    The p-value is the bootstrap two-sided test of H0: difference = 0,
    p = 2 * min(Pr(diff* <= 0), Pr(diff* >= 0)).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    ya = np.asarray(y_score_a, dtype=float)
    yb = np.asarray(y_score_b, dtype=float)
    n = len(y_true)
    point = float(metric_fn(y_true, ya)) - float(metric_fn(y_true, yb))

    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            diffs[b] = metric_fn(y_true[idx], ya[idx]) - metric_fn(y_true[idx], yb[idx])
        except Exception:
            diffs[b] = np.nan

    lo, hi = _percentile_ci(diffs, alpha)
    valid = diffs[~np.isnan(diffs)]
    frac_le0 = float(np.mean(valid <= 0)) if valid.size else np.nan
    frac_ge0 = float(np.mean(valid >= 0)) if valid.size else np.nan
    p = float(min(1.0, 2.0 * min(frac_le0, frac_ge0)))
    return {
        "diff": point,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def cluster_bootstrap_ci(cluster_ids, y_true, y_score, metric_fn,
                         n_boot=2000, alpha=0.95, seed=12345):
    """Cluster (patient-level) bootstrap CI.

    Resamples whole clusters — patients — with replacement, keeping every view
    belonging to a drawn patient together. This is the correct variance
    estimator when rows are views/images but independence holds at the woman
    level. Intervals are typically *wider* than a naive per-image bootstrap;
    that extra width is real uncertainty, not noise.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    point = float(metric_fn(y_true, y_score))

    unique = np.unique(cluster_ids)
    members = {c: np.where(cluster_ids == c)[0] for c in unique}

    samples = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[c] for c in drawn])
        try:
            samples[b] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            samples[b] = np.nan

    lo, hi = _percentile_ci(samples, alpha)
    return {
        "estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "se": float(np.nanstd(samples, ddof=1)),
        "n_clusters": int(len(unique)),
        "n_boot": n_boot,
        "alpha": alpha,
    }
