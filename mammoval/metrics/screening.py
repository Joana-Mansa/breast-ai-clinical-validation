"""Screening-programme metrics and AI triage / rule-out simulation.

ROC AUC measures discrimination in the abstract. A screening programme is run
at *one* operating point and is judged on the numbers a radiology department
and a regulator actually report:

* **Cancer detection rate (CDR)** — screen-detected cancers per 1,000 exams.
* **Recall / abnormal-interpretation rate** — fraction of women called back.
* **PPV of recall (PPV1)** — of those recalled, the fraction with cancer.
* **Sensitivity / specificity** against the reference standard.

This module also simulates the two clinical uses ScreenPoint's Transpara is
positioned for, both expressed against the score distribution:

* **Rule-out / triage** — exams with a very low AI score are deprioritised or
  auto-classified as normal, reducing radiologist workload. The key safety
  question is how many cancers fall into the ruled-out band.
* **Risk banding** — mapping the continuous score onto ordinal categories
  (à la the 1-10 Transpara score) and checking the cancer rate rises
  monotonically across bands.

Cohort caveat: CBIS-DDSM is a lesion-enriched, non-consecutive dataset, so its
CDR and recall rate are *not* programme estimates — they are reported here to
exercise the metric and must be read as illustrative, not epidemiological.
"""
from __future__ import annotations

import numpy as np

from ._common import check_binary, wilson_ci, proportion

__all__ = [
    "screening_summary",
    "screening_summary_from_score",
    "triage_simulation",
    "risk_band_table",
]


def screening_summary(y_true, y_recall, alpha=0.95):
    """Programme-style metrics from a binary recall decision.

    Parameters
    ----------
    y_true : array {0, 1}
        Cancer outcome (reference standard).
    y_recall : array {0, 1}
        Recall / positive decision per exam.
    """
    y_true = check_binary(y_true)
    y_recall = np.asarray(y_recall).astype(int)
    n = len(y_true)
    tp = int(np.sum((y_recall == 1) & (y_true == 1)))
    fp = int(np.sum((y_recall == 1) & (y_true == 0)))
    tn = int(np.sum((y_recall == 0) & (y_true == 0)))
    fn = int(np.sum((y_recall == 0) & (y_true == 1)))
    n_recalled = tp + fp
    n_cancer = tp + fn

    return {
        "n_exams": n,
        "n_cancers": n_cancer,
        "n_recalled": n_recalled,
        "recall_rate": proportion(n_recalled, n),
        "recall_rate_ci": wilson_ci(n_recalled, n, alpha),
        "cancer_detection_rate_per_1000": proportion(tp, n) * 1000.0,
        "sensitivity": proportion(tp, n_cancer),
        "sensitivity_ci": wilson_ci(tp, n_cancer, alpha),
        "specificity": proportion(tn, tn + fp),
        "specificity_ci": wilson_ci(tn, tn + fp, alpha),
        "ppv1_recall": proportion(tp, n_recalled),
        "ppv1_recall_ci": wilson_ci(tp, n_recalled, alpha),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def screening_summary_from_score(y_true, y_score, threshold, alpha=0.95):
    """As :func:`screening_summary`, deriving the recall decision from an AI
    score (recall when ``score >= threshold``)."""
    y_score = np.asarray(y_score, dtype=float)
    return screening_summary(y_true, (y_score >= threshold).astype(int), alpha=alpha)


def triage_simulation(y_true, y_score, n_steps=50):
    """Simulate AI rule-out: deprioritise exams below a low score threshold.

    For a sweep of rule-out thresholds the function reports, for the band the
    AI would set aside as 'normal':

    * ``workload_reduction`` — fraction of all exams ruled out (the efficiency
      gain — fewer exams for the radiologist to read);
    * ``missed_cancers`` and ``sensitivity_retained`` — the safety cost: how
      many cancers sit in the ruled-out band, and the sensitivity kept among
      the exams still read.

    The clinically defensible operating point is the largest workload
    reduction that keeps ``sensitivity_retained`` above a pre-agreed floor —
    this is exactly the trade-off curve behind AI workload-reduction claims.

    Returns
    -------
    list of dict, one row per threshold, ascending in workload reduction.
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    total_cancers = int(y_true.sum())

    thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, n_steps))
    rows = []
    for t in np.unique(thresholds):
        ruled_out = y_score < t
        n_out = int(ruled_out.sum())
        missed = int(np.sum(y_true[ruled_out] == 1))
        read = ~ruled_out
        cancers_read = int(np.sum(y_true[read] == 1))
        rows.append({
            "threshold": float(t),
            "workload_reduction": proportion(n_out, n),
            "n_ruled_out": n_out,
            "missed_cancers": missed,
            "sensitivity_retained": proportion(cancers_read, total_cancers),
            "ruled_out_cancer_rate": proportion(missed, n_out) if n_out else 0.0,
        })
    return rows


def risk_band_table(y_true, y_score, edges=None, alpha=0.95):
    """Cancer rate per ordinal risk band.

    Bins the continuous score into ordinal categories (default deciles, the
    spirit of the 1-10 Transpara score) and reports the observed cancer rate
    per band with a Wilson CI. A well-behaved device shows a monotonically
    rising cancer rate — evidence the score is a genuine risk stratifier, not
    just a classifier.
    """
    y_true = check_binary(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if edges is None:
        edges = np.quantile(y_score, np.linspace(0.0, 1.0, 11))
        edges = np.unique(edges)
    band = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = band == b
        n_b = int(mask.sum())
        k_b = int(np.sum(y_true[mask] == 1))
        rows.append({
            "band": b + 1,
            "score_low": float(edges[b]),
            "score_high": float(edges[b + 1]),
            "n_exams": n_b,
            "n_cancers": k_b,
            "cancer_rate": proportion(k_b, n_b),
            "cancer_rate_ci": wilson_ci(k_b, n_b, alpha),
        })
    return rows
