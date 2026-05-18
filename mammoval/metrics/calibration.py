"""Probability calibration and clinical-utility (decision-curve) analysis.

Discrimination (AUC) asks "are cancers ranked above non-cancers?". Calibration
asks the orthogonal question "when the model says 0.10, do ~10 % of those women
actually have cancer?". A model can discriminate perfectly yet be badly
calibrated — and miscalibration directly corrupts any downstream decision that
treats the score as a probability (rule-out thresholds, risk banding, the net
benefit computed below).

Provided here:

* a binned **reliability curve** and the **Brier score**;
* **Expected Calibration Error** (ECE);
* the **calibration intercept and slope** — calibration-in-the-large and the
  slope of observed-vs-predicted log-odds (1.0 = ideal; <1 = over-confident);
* **decision-curve analysis** (net benefit), which expresses usefulness in
  units a clinician understands: true positives gained per patient, after
  charging for false positives at the rate implied by a decision threshold.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import check_binary

__all__ = [
    "reliability_curve",
    "brier_score",
    "expected_calibration_error",
    "calibration_intercept_slope",
    "decision_curve",
]

_EPS = 1e-7


def reliability_curve(y_true, y_prob, n_bins=10, strategy="quantile"):
    """Binned observed-vs-predicted frequencies (the reliability diagram).

    ``strategy='quantile'`` puts an equal number of exams in each bin (robust
    when scores cluster); ``'uniform'`` uses equal-width probability bins.
    """
    y_true = check_binary(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b + 1,
            "predicted_mean": float(np.mean(y_prob[mask])),
            "observed_rate": float(np.mean(y_true[mask])),
            "count": int(mask.sum()),
        })
    return rows


def brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and outcome.

    Lower is better; it is a *proper scoring rule*, jointly sensitive to
    calibration and discrimination.
    """
    y_true = check_binary(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile"):
    """Count-weighted mean gap between observed and predicted frequency."""
    rows = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total = sum(r["count"] for r in rows)
    if total == 0:
        return float("nan")
    return float(
        sum(r["count"] * abs(r["observed_rate"] - r["predicted_mean"]) for r in rows)
        / total
    )


def calibration_intercept_slope(y_true, y_prob):
    """Calibration intercept and slope via logistic recalibration.

    Fits ``logit(P(cancer)) = a + b * logit(y_prob)``:

    * **slope b** — 1.0 is ideal; b < 1 flags over-confident scores (extreme
      probabilities too extreme), b > 1 flags under-confident scores.
    * **intercept a** (calibration-in-the-large, evaluated at mean logit) —
      detects a systematic over- or under-estimation of overall risk, the
      classic symptom of deploying a model on a population whose prevalence
      differs from the training set.

    Uses a Newton-Raphson logistic fit so the only dependency is SciPy/NumPy.
    """
    y_true = check_binary(y_true).astype(float)
    p = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p))

    X = np.column_stack([np.ones_like(logit), logit])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), _EPS, None)
        grad = X.T @ (y_true - mu)
        hess = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y_true)),
    }


def decision_curve(y_true, y_prob, thresholds=None):
    """Decision-curve analysis: net benefit across decision thresholds.

    Net benefit at threshold probability ``pt`` is

        NB = TP/n - FP/n * (pt / (1 - pt))

    where a case is acted on when ``y_prob >= pt``. The ``pt/(1-pt)`` factor is
    the exchange rate between a false positive and a true positive implied by
    that threshold. The AI model is clinically useful over the ``pt`` range
    where its curve sits above both reference strategies — **treat all** and
    **treat none** (net benefit 0).
    """
    y_true = check_binary(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for pt in thresholds:
        act = y_prob >= pt
        tp = int(np.sum(act & (y_true == 1)))
        fp = int(np.sum(act & (y_true == 0)))
        w = pt / (1.0 - pt)
        nb_model = tp / n - fp / n * w
        nb_all = prevalence - (1.0 - prevalence) * w
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })
    return rows
