"""Clinical validation metrics for mammography AI.

Grouped by the question each set of metrics answers:

* :mod:`~mammoval.metrics.delong`         -- AUC, its CI, comparison and
                                             non-inferiority testing (paired).
* :mod:`~mammoval.metrics.classification` -- ROC, operating points, partial AUC.
* :mod:`~mammoval.metrics.localization`   -- FROC for lesion marks (2D and 3D).
* :mod:`~mammoval.metrics.screening`      -- CDR, recall rate, triage/rule-out.
* :mod:`~mammoval.metrics.calibration`    -- reliability, Brier, decision curve.
* :mod:`~mammoval.metrics.bootstrap`      -- resampling CIs (incl. patient-level).
* :mod:`~mammoval.metrics.subgroups`      -- effect-modifier / stratified analysis.
"""
from .delong import auc_ci, delong_test, auc_noninferiority
from .classification import (
    roc_analysis,
    partial_auc,
    operating_point,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
    youden_point,
    precision_recall,
    threshold_for_flag_rate,
)
from .localization import (
    build_froc,
    sensitivity_at_fp,
    froc_summary,
    bbox_hit,
    duke_dbt_hit,
)
from .screening import (
    screening_summary,
    screening_summary_from_score,
    triage_simulation,
    risk_band_table,
)
from .calibration import (
    reliability_curve,
    brier_score,
    expected_calibration_error,
    calibration_intercept_slope,
    decision_curve,
)
from .bootstrap import bootstrap_ci, paired_bootstrap_diff, cluster_bootstrap_ci
from .subgroups import subgroup_auc, subgroup_metric

__all__ = [
    "auc_ci", "delong_test", "auc_noninferiority",
    "roc_analysis", "partial_auc", "operating_point",
    "sensitivity_at_specificity", "specificity_at_sensitivity",
    "youden_point", "precision_recall", "threshold_for_flag_rate",
    "build_froc", "sensitivity_at_fp", "froc_summary", "bbox_hit", "duke_dbt_hit",
    "screening_summary", "screening_summary_from_score",
    "triage_simulation", "risk_band_table",
    "reliability_curve", "brier_score", "expected_calibration_error",
    "calibration_intercept_slope", "decision_curve",
    "bootstrap_ci", "paired_bootstrap_diff", "cluster_bootstrap_ci",
    "subgroup_auc", "subgroup_metric",
]
