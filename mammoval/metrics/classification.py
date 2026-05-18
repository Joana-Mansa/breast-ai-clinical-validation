"""Case-level discrimination metrics: ROC, operating points, partial AUC.

These quantify how well a continuous AI score separates cancer from non-cancer
at the *exam* level — the "standalone classification" view of a mammography AI
device. Localisation quality (does the mark land on the lesion?) is handled
separately in :mod:`mammoval.metrics.localization`.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve as _sk_roc_curve
from sklearn.metrics import precision_recall_curve as _sk_pr_curve
from sklearn.metrics import average_precision_score as _sk_ap

from ._common import check_binary, wilson_ci, proportion
from .delong import auc_ci

__all__ = [
    "roc_analysis",
    "partial_auc",
    "operating_point",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity",
    "youden_point",
    "precision_recall",
    "threshold_for_flag_rate",
]


def roc_analysis(y_true, y_score, alpha=0.95):
    """Full ROC analysis: curve points plus AUC with a DeLong CI.

    Returns a dict containing the ``fpr``/``tpr``/``thresholds`` arrays (as
    lists, for serialisation) and the AUC summary from :func:`delong.auc_ci`.
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thr = _sk_roc_curve(y_true, y_score)
    out = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
    out.update(auc_ci(y_true, y_score, alpha=alpha))
    return out


def partial_auc(y_true, y_score, spec_low=0.80, spec_high=1.00, standardized=True):
    """Partial AUC restricted to a high-specificity region of the ROC.

    Screening mammography operates at high specificity (a radiologist's recall
    rate is only a few percent). The global AUC can be dominated by low-
    specificity behaviour that is clinically irrelevant, so regulators and
    reader studies often look at the **partial AUC** over a clinically
    operating range — e.g. specificity 0.80-1.00.

    With ``standardized=True`` the partial AUC is rescaled by McClish's method
    onto [0.5, 1.0], so it reads on the same intuitive scale as a full AUC
    (0.5 = chance within the region, 1.0 = perfect).
    """
    y_true = check_binary(y_true)
    fpr, tpr, _ = _sk_roc_curve(y_true, np.asarray(y_score, dtype=float))
    fpr_lo, fpr_hi = 1.0 - spec_high, 1.0 - spec_low
    grid = np.linspace(fpr_lo, fpr_hi, 1024)
    tpr_grid = np.interp(grid, fpr, tpr)
    pauc = float(np.trapz(tpr_grid, grid))
    width = fpr_hi - fpr_lo
    result = {"pauc": pauc, "spec_low": spec_low, "spec_high": spec_high,
              "fpr_range": [fpr_lo, fpr_hi]}
    if standardized and width > 0:
        pauc_min = 0.5 * (fpr_hi ** 2 - fpr_lo ** 2)  # area under chance line
        pauc_max = width                              # area under tpr == 1
        result["pauc_standardized"] = float(
            0.5 * (1.0 + (pauc - pauc_min) / (pauc_max - pauc_min))
        )
    return result


def operating_point(y_true, y_score, threshold, alpha=0.95):
    """Confusion-matrix metrics at a fixed decision ``threshold``.

    A score >= ``threshold`` is treated as a positive AI call ("recall" /
    "flag for biopsy", depending on context). Every proportion is reported
    with a Wilson confidence interval.
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    n_pos, n_neg = tp + fn, fp + tn

    sens = proportion(tp, n_pos)
    spec = proportion(tn, n_neg)
    ppv = proportion(tp, tp + fp)
    npv = proportion(tn, tn + fn)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "sensitivity_ci": wilson_ci(tp, n_pos, alpha),
        "specificity": spec, "specificity_ci": wilson_ci(tn, n_neg, alpha),
        "ppv": ppv, "ppv_ci": wilson_ci(tp, tp + fp, alpha),
        "npv": npv, "npv_ci": wilson_ci(tn, tn + fn, alpha),
        "accuracy": proportion(tp + tn, len(y_true)),
        "f1": proportion(2 * tp, 2 * tp + fp + fn),
        "youden_j": (sens + spec - 1.0) if not (np.isnan(sens) or np.isnan(spec)) else float("nan"),
        "flag_rate": proportion(tp + fp, len(y_true)),
    }


def sensitivity_at_specificity(y_true, y_score, specificity, alpha=0.95):
    """Sensitivity (and the threshold) at the operating point meeting a
    target specificity.

    Picks the highest-sensitivity ROC point whose specificity is still
    >= ``specificity`` — i.e. the constraint is honoured, never overshot.
    """
    y_true = check_binary(y_true)
    fpr, tpr, thr = _sk_roc_curve(y_true, np.asarray(y_score, dtype=float))
    target_fpr = 1.0 - specificity
    feasible = np.where(fpr <= target_fpr + 1e-12)[0]
    idx = feasible[np.argmax(tpr[feasible])]
    n_pos = int(y_true.sum())
    sens = float(tpr[idx])
    return {
        "target_specificity": specificity,
        "sensitivity": sens,
        "sensitivity_ci": wilson_ci(int(round(sens * n_pos)), n_pos, alpha),
        "specificity_achieved": float(1.0 - fpr[idx]),
        "threshold": float(thr[idx]),
    }


def specificity_at_sensitivity(y_true, y_score, sensitivity, alpha=0.95):
    """Specificity (and threshold) at the operating point meeting a target
    sensitivity — the dual of :func:`sensitivity_at_specificity`.

    Useful for "match the radiologist's sensitivity, then compare recall rate".
    """
    y_true = check_binary(y_true)
    fpr, tpr, thr = _sk_roc_curve(y_true, np.asarray(y_score, dtype=float))
    feasible = np.where(tpr >= sensitivity - 1e-12)[0]
    idx = feasible[np.argmin(fpr[feasible])]
    n_neg = int(len(y_true) - y_true.sum())
    spec = float(1.0 - fpr[idx])
    return {
        "target_sensitivity": sensitivity,
        "specificity": spec,
        "specificity_ci": wilson_ci(int(round(spec * n_neg)), n_neg, alpha),
        "sensitivity_achieved": float(tpr[idx]),
        "threshold": float(thr[idx]),
    }


def youden_point(y_true, y_score):
    """Threshold maximising Youden's J = sensitivity + specificity - 1.

    A data-driven 'balanced' operating point. Useful as a reference, but a
    deployed screening device is tuned to a target recall rate, not to J.
    """
    y_true = check_binary(y_true)
    fpr, tpr, thr = _sk_roc_curve(y_true, np.asarray(y_score, dtype=float))
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "threshold": float(thr[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1.0 - fpr[idx]),
        "youden_j": float(j[idx]),
    }


def precision_recall(y_true, y_score):
    """Precision-recall curve and average precision.

    PR-space is more informative than ROC-space when positives are rare, which
    is the screening reality (~5 cancers per 1000 exams).
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, thr = _sk_pr_curve(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thr.tolist(),
        "average_precision": float(_sk_ap(y_true, y_score)),
        "prevalence": float(np.mean(y_true)),
    }


def threshold_for_flag_rate(y_score, flag_rate):
    """Score threshold that flags (recalls) a target fraction of exams.

    Lets you compare an AI device at the *same workload* as the radiologist:
    set ``flag_rate`` to the programme recall rate, then read off sensitivity.
    """
    y_score = np.asarray(y_score, dtype=float)
    return float(np.quantile(y_score, 1.0 - flag_rate))
