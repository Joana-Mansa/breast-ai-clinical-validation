"""Validation orchestration.

Two entry points turn a predictions table into a complete, structured
validation result:

* :func:`run_classification_validation` -- the standalone exam-level study:
  discrimination, operating points, calibration, screening behaviour, subgroup
  analysis and (optionally) an AI-vs-reader comparison.
* :func:`run_localization_validation` -- the FROC study: do the marks land on
  the lesions?

Both return a plain nested ``dict`` (JSON-serialisable) — :mod:`mammoval.report`
renders it, but it is equally usable from a notebook or a test. Optional
sections are wrapped so that one failing slice never sinks the whole report;
the failure is recorded in the result instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .metrics import (
    roc_analysis, partial_auc, operating_point, sensitivity_at_specificity,
    youden_point, precision_recall, threshold_for_flag_rate,
    delong_test, auc_noninferiority,
    screening_summary_from_score, triage_simulation, risk_band_table,
    reliability_curve, brier_score, expected_calibration_error,
    calibration_intercept_slope, decision_curve,
    bootstrap_ci, cluster_bootstrap_ci, subgroup_auc,
    build_froc, froc_summary,
)

__all__ = ["ClassificationConfig", "run_classification_validation",
           "run_localization_validation"]


@dataclass
class ClassificationConfig:
    """Knobs for a standalone classification study.

    The defaults encode sensible screening choices; override per study.
    """
    score_col: str = "y_score"
    label_col: str = "y_true"
    patient_col: str = "patient_id"
    reader_col: str | None = None          # e.g. BI-RADS assessment as a reader proxy
    subgroup_cols: tuple = ()              # e.g. ("breast_density_cat", "abnormality_type")
    target_specificities: tuple = (0.90, 0.96)
    ni_margin: float = 0.05                # AUC non-inferiority margin (pre-specified)
    n_boot: int = 2000
    is_probability: bool = True            # are scores calibrated probabilities?
    alpha: float = 0.95
    dataset_name: str = "dataset"


def _safe(section, fn):
    """Run an optional report section; capture failure instead of aborting."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad for robustness
        return {"error": f"{type(exc).__name__}: {exc}", "_section": section}


def run_classification_validation(df, config=None, **overrides):
    """Full standalone classification validation from a predictions table.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the label, score and patient columns named in ``config``.
    config : ClassificationConfig, optional
    **overrides
        Convenience keyword overrides applied on top of ``config``.

    Returns
    -------
    dict
        Nested results keyed by section: ``meta``, ``discrimination``,
        ``operating_points``, ``calibration``, ``screening``, ``subgroups``,
        ``reader_comparison``.
    """
    config = config or ClassificationConfig()
    if overrides:
        config = ClassificationConfig(**{**asdict(config), **overrides})

    # ---- clean: keep complete label/score rows -----------------------------
    work = df.copy()
    work = work[[c for c in work.columns]]
    n_raw = len(work)
    work = work.dropna(subset=[config.label_col, config.score_col])
    work[config.label_col] = work[config.label_col].astype(int)
    y = work[config.label_col].to_numpy()
    s = work[config.score_col].to_numpy(dtype=float)
    n_dropped = n_raw - len(work)

    results = {"meta": {
        "dataset": config.dataset_name,
        "n_cases": int(len(work)),
        "n_dropped_missing": int(n_dropped),
        "n_malignant": int(np.sum(y == 1)),
        "n_non_malignant": int(np.sum(y == 0)),
        "prevalence": float(np.mean(y)),
        "n_patients": int(work[config.patient_col].nunique())
        if config.patient_col in work else None,
        "config": asdict(config),
    }}

    # ---- discrimination ----------------------------------------------------
    roc = roc_analysis(y, s, alpha=config.alpha)
    discrimination = {
        "auc": roc["auc"], "auc_ci": [roc["ci_low"], roc["ci_high"]],
        "auc_se": roc["se"], "roc_curve": {"fpr": roc["fpr"], "tpr": roc["tpr"]},
        "partial_auc_high_spec": partial_auc(y, s, spec_low=0.80, spec_high=1.0),
        "average_precision": precision_recall(y, s)["average_precision"],
    }
    if config.patient_col in work:
        discrimination["auc_patient_cluster_ci"] = _safe(
            "auc_patient_cluster_ci",
            lambda: cluster_bootstrap_ci(
                work[config.patient_col].to_numpy(), y, s,
                lambda yt, ys: roc_analysis(yt, ys)["auc"],
                n_boot=config.n_boot, alpha=config.alpha),
        )
    results["discrimination"] = discrimination

    # ---- operating points --------------------------------------------------
    ops = {"youden": youden_point(y, s), "by_target_specificity": []}
    for spec in config.target_specificities:
        sat = sensitivity_at_specificity(y, s, spec, alpha=config.alpha)
        sat["sensitivity_bootstrap_ci"] = _safe(
            f"sens@spec{spec}",
            lambda spec=spec: bootstrap_ci(
                y, s,
                lambda yt, ys: sensitivity_at_specificity(yt, ys, spec)["sensitivity"],
                n_boot=config.n_boot, alpha=config.alpha),
        )
        sat["confusion"] = operating_point(y, s, sat["threshold"], alpha=config.alpha)
        ops["by_target_specificity"].append(sat)
    results["operating_points"] = ops

    # primary operating point = first requested specificity
    primary_threshold = ops["by_target_specificity"][0]["threshold"]

    # ---- calibration (only meaningful for probability-like scores) ---------
    if config.is_probability:
        results["calibration"] = _safe("calibration", lambda: {
            "reliability_curve": reliability_curve(y, s, n_bins=10),
            "brier_score": brier_score(y, s),
            "expected_calibration_error": expected_calibration_error(y, s, n_bins=10),
            "intercept_slope": calibration_intercept_slope(y, s),
            "decision_curve": decision_curve(y, s),
        })
    else:
        results["calibration"] = {"skipped": "scores are not probabilities"}

    # ---- screening behaviour ----------------------------------------------
    results["screening"] = _safe("screening", lambda: {
        "at_primary_operating_point": screening_summary_from_score(
            y, s, primary_threshold, alpha=config.alpha),
        "primary_threshold": float(primary_threshold),
        "triage_simulation": triage_simulation(y, s, n_steps=40),
        "risk_bands": risk_band_table(y, s, alpha=config.alpha),
        "cohort_caveat": "Screening rates are dataset-dependent; on an enriched "
                         "cohort (e.g. CBIS-DDSM) they are illustrative, not "
                         "programme estimates.",
    })

    # ---- subgroup / effect-modifier analysis ------------------------------
    subgroups = {}
    for col in config.subgroup_cols:
        if col in work.columns:
            subgroups[col] = _safe(f"subgroup:{col}", lambda col=col: subgroup_auc(
                work, config.label_col, config.score_col, col, alpha=config.alpha))
    results["subgroups"] = subgroups

    # ---- AI vs reference reader -------------------------------------------
    if config.reader_col and config.reader_col in work.columns:
        results["reader_comparison"] = _safe("reader_comparison", lambda: _reader_block(
            work, y, s, config))
    else:
        results["reader_comparison"] = {"skipped": "no reader column supplied"}

    return results


def _reader_block(work, y, s, config):
    """AI-vs-reader paired comparison, on cases where the reader score exists."""
    sub = work.dropna(subset=[config.reader_col])
    yr = sub[config.label_col].to_numpy().astype(int)
    s_ai = sub[config.score_col].to_numpy(dtype=float)
    s_rd = sub[config.reader_col].to_numpy(dtype=float)
    dl = delong_test(yr, s_ai, s_rd)
    ni = auc_noninferiority(yr, s_ai, s_rd, margin=config.ni_margin, alpha=config.alpha)
    return {
        "reader_col": config.reader_col,
        "n_cases_with_reader": int(len(sub)),
        "auc_ai": dl["auc_a"],
        "auc_reader": dl["auc_b"],
        "delong_test": dl,
        "noninferiority": ni,
        "interpretation": _interpret_reader(dl, ni),
    }


def _interpret_reader(dl, ni):
    direction = "higher" if dl["auc_diff"] > 0 else "lower"
    sig = "significant" if dl["p_value"] < 0.05 else "not significant"
    ni_txt = ("non-inferior" if ni["non_inferior"]
              else "NOT shown to be non-inferior")
    return (f"AI AUC is {direction} than the reference reader "
            f"(diff {dl['auc_diff']:+.3f}, DeLong p={dl['p_value']:.3g}, {sig}); "
            f"against a {ni['margin']:.02f} margin AI is {ni_txt}.")


# --------------------------------------------------------------------------
# Localisation (FROC)
# --------------------------------------------------------------------------
def run_localization_validation(cases, hit_fn, fp_points=(1.0, 2.0, 3.0, 4.0),
                                n_boot=1000, alpha=0.95, seed=12345,
                                dataset_name="dataset"):
    """FROC validation from per-image detections and ground-truth lesions.

    Parameters
    ----------
    cases : list of dict
        ``{case_id, lesions, detections}`` per image/volume — see
        :func:`mammoval.metrics.build_froc`.
    hit_fn : callable
        Geometric TP criterion (e.g. :func:`mammoval.metrics.duke_dbt_hit`).
    fp_points : tuple
        False-marks-per-image operating points. The default reproduces the
        official Duke BCS-DBT ranking metric.

    Returns
    -------
    dict with the FROC curve, sensitivity at each operating point, the mean
    sensitivity, and a case-level bootstrap CI on that mean.
    """
    summary = froc_summary(cases, hit_fn, fp_points=fp_points)

    rng = np.random.default_rng(seed)
    n = len(cases)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [cases[i] for i in idx]
        try:
            boot[b] = froc_summary(resampled, hit_fn, fp_points=fp_points)["mean_sensitivity"]
        except Exception:
            boot[b] = np.nan
    lo = float(np.nanpercentile(boot, 100 * (1 - alpha) / 2))
    hi = float(np.nanpercentile(boot, 100 * (1 - (1 - alpha) / 2)))

    return {
        "meta": {"dataset": dataset_name, "n_cases": summary["n_cases"],
                 "n_lesions": summary["n_lesions"], "fp_points": list(fp_points)},
        "froc_curve": summary["froc_curve"],
        "sensitivity_at_fp": summary["sensitivity_at_fp"],
        "mean_sensitivity": summary["mean_sensitivity"],
        "mean_sensitivity_ci": [lo, hi],
        "alpha": alpha,
    }
