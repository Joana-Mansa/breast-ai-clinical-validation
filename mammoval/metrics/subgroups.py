"""Subgroup (effect-modifier) analysis.

A single pooled AUC can hide clinically critical failure modes. The headline
question for any mammography AI is whether performance holds in **dense
breasts**, where masking lowers sensitivity for radiologists and AI alike and
where the clinical need is greatest. Regulators and reader studies therefore
expect performance broken down by:

* breast density (BI-RADS A-D / categories 1-4),
* lesion type (mass vs calcification),
* image view, and where available age, vendor and site.

This module estimates a metric within each subgroup *with* a confidence
interval, and tests whether the subgroups genuinely differ — Cochran's Q
heterogeneity test on the per-group AUCs, using their DeLong variances. A
significant Q is evidence that the variable is an **effect modifier**: the
device should then be reported (and possibly operating-point-tuned) per
stratum, not pooled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .delong import auc_ci

__all__ = ["subgroup_auc", "subgroup_metric"]


def subgroup_auc(df, label_col, score_col, group_col, alpha=0.95, min_per_group=10):
    """Per-subgroup AUC with DeLong CIs plus a heterogeneity test.

    Parameters
    ----------
    df : pandas.DataFrame
    label_col, score_col, group_col : str
        Columns holding the binary label, the AI score and the subgroup key.
    min_per_group : int
        Subgroups with fewer cases, or lacking both classes, are skipped and
        listed under ``skipped`` rather than producing an unstable estimate.

    Returns
    -------
    dict with ``groups`` (list of per-group results), ``skipped``, and a
    ``heterogeneity`` block (Cochran's Q, df, p-value, I-squared).
    """
    groups, skipped = [], []
    for key, sub in df.groupby(group_col, dropna=True):
        y = sub[label_col].to_numpy()
        s = sub[score_col].to_numpy()
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if len(sub) < min_per_group or n_pos == 0 or n_neg == 0:
            skipped.append({"group": _key(key), "n": int(len(sub)),
                            "n_pos": n_pos, "n_neg": n_neg,
                            "reason": "too few cases or single class"})
            continue
        res = auc_ci(y, s, alpha=alpha)
        res["group"] = _key(key)
        res["n"] = int(len(sub))
        groups.append(res)

    return {
        "group_col": group_col,
        "groups": groups,
        "skipped": skipped,
        "heterogeneity": _cochran_q([g["auc"] for g in groups],
                                    [g["se"] for g in groups]),
    }


def subgroup_metric(df, group_col, metric_fn, min_per_group=10):
    """Apply an arbitrary ``metric_fn(sub_df) -> dict`` to each subgroup.

    A general escape hatch for stratifying any metric (recall rate, FROC
    sensitivity, calibration slope, ...) that does not fit :func:`subgroup_auc`.
    """
    rows = []
    for key, sub in df.groupby(group_col, dropna=True):
        if len(sub) < min_per_group:
            continue
        try:
            res = dict(metric_fn(sub))
        except Exception as exc:  # keep the report building even if one slice fails
            res = {"error": str(exc)}
        res["group"] = _key(key)
        res["n"] = int(len(sub))
        rows.append(res)
    return {"group_col": group_col, "groups": rows}


def _key(key):
    """Normalise a groupby key to a plain JSON-friendly scalar."""
    if isinstance(key, (np.integer,)):
        return int(key)
    if isinstance(key, (np.floating,)):
        return float(key)
    return key if pd.notna(key) else "missing"


def _cochran_q(aucs, ses):
    """Fixed-effect heterogeneity test across independent subgroup AUCs.

    Q = sum w_i (auc_i - auc_bar)^2 with weights w_i = 1/se_i^2; under the null
    of a common AUC, Q ~ chi-square with (k - 1) df. I-squared is the share of
    total variation attributable to genuine between-group differences rather
    than sampling error.
    """
    aucs = np.asarray(aucs, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(aucs)
    if k < 2 or np.any(ses <= 0):
        return {"applicable": False,
                "note": "needs >=2 subgroups with positive standard errors"}
    w = 1.0 / ses ** 2
    auc_bar = float(np.sum(w * aucs) / np.sum(w))
    q = float(np.sum(w * (aucs - auc_bar) ** 2))
    dfree = k - 1
    p = float(stats.chi2.sf(q, dfree))
    i2 = float(max(0.0, (q - dfree) / q)) if q > 0 else 0.0
    return {
        "applicable": True,
        "pooled_auc": auc_bar,
        "Q": q,
        "df": dfree,
        "p_value": p,
        "I_squared": i2,
        "effect_modifier": bool(p < 0.05),
    }
