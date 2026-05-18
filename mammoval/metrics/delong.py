"""DeLong's method for AUC variance, comparison and non-inferiority testing.

Implements the fast algorithm of Sun & Xu (2014), *Fast Implementation of
DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves*, IEEE Signal Processing Letters 21(11):1389-1393.

Why this matters for clinical validation
----------------------------------------
DeLong's method gives a **closed-form, fully non-parametric** estimate of the
variance of an empirical ROC AUC, and of the *covariance* between two AUCs
measured on the **same set of cases** (a paired / correlated design).

That paired design is exactly the standalone-validation setting used in
mammography-AI regulatory dossiers: every case is read by AI *and* has a
reference standard (and often a radiologist score), so the AUCs are correlated
and an unpaired comparison would be statistically wrong. DeLong is the de-facto
standard for:

* putting a 95% confidence interval on a single AI AUC;
* testing AI-vs-radiologist or AI-model-A-vs-B differences in discrimination;
* powering **non-inferiority** claims ("AI standalone is not worse than the
  radiologist by more than a pre-specified margin").
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import check_binary

__all__ = ["auc_ci", "delong_test", "auc_noninferiority"]


def _compute_midrank(x):
    """Mid-ranks of ``x`` (ties receive the average of the tied ranks)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(scores_pos_first, n_pos):
    """Core fast-DeLong computation.

    Parameters
    ----------
    scores_pos_first : ndarray, shape (k, n)
        ``k`` score vectors over the SAME ``n`` cases, with the ``n_pos``
        positive cases in the leading columns.
    n_pos : int
        Number of positive cases.

    Returns
    -------
    aucs : ndarray, shape (k,)
    cov : ndarray, shape (k, k)
        DeLong covariance matrix of the AUC estimators.
    """
    m = n_pos
    n = scores_pos_first.shape[1] - m
    k = scores_pos_first.shape[0]
    if m == 0 or n == 0:
        raise ValueError("DeLong needs at least one positive and one negative case")
    pos = scores_pos_first[:, :m]
    neg = scores_pos_first[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(scores_pos_first[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    # np.cov over rows (each row = one model); ddof=1.
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def _order_pos_first(y_true):
    """Index order that places positive cases first (stable)."""
    return np.argsort(-y_true, kind="mergesort")


def auc_ci(y_true, y_score, alpha=0.95):
    """Empirical ROC AUC with a DeLong confidence interval.

    Parameters
    ----------
    y_true : array of {0, 1}
        Reference standard (1 = cancer / positive).
    y_score : array of float
        Continuous AI score; higher = more suspicious.
    alpha : float
        Two-sided confidence level (0.95 -> 95% CI).

    Returns
    -------
    dict with keys: auc, se, ci_low, ci_high, alpha, n_pos, n_neg.
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    order = _order_pos_first(y_true)
    n_pos = int(y_true.sum())
    aucs, cov = _fast_delong(y_score[order][np.newaxis, :], n_pos)
    auc = float(aucs[0])
    var = float(cov[0, 0])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return {
        "auc": auc,
        "se": se,
        "ci_low": float(max(0.0, auc - z * se)),
        "ci_high": float(min(1.0, auc + z * se)),
        "alpha": alpha,
        "n_pos": n_pos,
        "n_neg": int(len(y_true) - n_pos),
    }


def delong_test(y_true, y_score_a, y_score_b):
    """Paired DeLong test for the difference between two AUCs.

    Both score vectors must be measured on the **same** cases (paired design),
    e.g. AI model vs radiologist BI-RADS, or AI version A vs version B.

    Returns
    -------
    dict: auc_a, auc_b, auc_diff (a - b), se_diff, z, p_value, cov.
    A small p_value indicates the two readers/models differ in discrimination.
    """
    y_true = check_binary(y_true)
    sa = np.asarray(y_score_a, dtype=float)
    sb = np.asarray(y_score_b, dtype=float)
    order = _order_pos_first(y_true)
    n_pos = int(y_true.sum())
    scores = np.vstack([sa[order], sb[order]])
    aucs, cov = _fast_delong(scores, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    if se == 0:
        z, p = 0.0, 1.0
    else:
        z = diff / se
        p = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "se_diff": se,
        "z": float(z),
        "p_value": p,
        "cov": cov.tolist(),
    }


def auc_noninferiority(y_true, y_score_new, y_score_ref, margin=0.05, alpha=0.95):
    """One-sided non-inferiority test for AUC (new vs reference, paired).

    The standalone-AI regulatory question is rarely "is AI better?" but "is AI
    **not meaningfully worse** than the radiologist?". Non-inferiority is
    declared when the lower one-sided ``alpha`` confidence bound on
    ``AUC_new - AUC_ref`` lies above ``-margin``.

    Parameters
    ----------
    margin : float
        Pre-specified non-inferiority margin on the AUC scale (e.g. 0.05).
        Must be justified clinically and fixed *before* looking at the data.

    Returns
    -------
    dict: auc_new, auc_ref, auc_diff, se_diff, margin, lower_bound,
    one-sided p_value against H0 (diff <= -margin), and `non_inferior` flag.
    """
    res = delong_test(y_true, y_score_new, y_score_ref)
    diff, se = res["auc_diff"], res["se_diff"]
    z_crit = stats.norm.ppf(alpha)  # one-sided
    lower = diff - z_crit * se
    # H0: diff <= -margin ; reject (=> non-inferior) when (diff + margin)/se large
    z_ni = (diff + margin) / se if se > 0 else np.inf
    p = float(stats.norm.sf(z_ni))
    return {
        "auc_new": res["auc_a"],
        "auc_ref": res["auc_b"],
        "auc_diff": diff,
        "se_diff": se,
        "margin": margin,
        "alpha": alpha,
        "lower_bound": float(lower),
        "p_value": p,
        "non_inferior": bool(lower > -margin),
    }
