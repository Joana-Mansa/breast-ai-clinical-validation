"""Unit tests for the discrimination, calibration and screening metrics.

Each test pins a metric to a value known by construction, so a regression in
the validation engine fails loudly rather than silently shifting a report.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mammoval import metrics as M
from mammoval.models import simulate_scores


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 1500)
    s = simulate_scores(y, auc=0.85, seed=1)
    return y, s


# --------------------------------------------------------------- DeLong / AUC
def test_delong_auc_matches_sklearn(cohort):
    y, s = cohort
    assert M.auc_ci(y, s)["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_delong_ci_brackets_target_auc(cohort):
    y, s = cohort
    res = M.auc_ci(y, s)
    assert res["ci_low"] < res["auc"] < res["ci_high"]
    assert res["ci_low"] < 0.85 < res["ci_high"]  # target AUC recovered


def test_perfect_and_random_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert M.auc_ci(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))["auc"] == 1.0
    assert M.auc_ci(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))["auc"] == 0.0


def test_delong_test_identical_scores_is_null(cohort):
    y, s = cohort
    res = M.delong_test(y, s, s.copy())
    assert res["auc_diff"] == pytest.approx(0.0, abs=1e-9)
    assert res["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_delong_test_detects_real_difference(cohort):
    y, _ = cohort
    strong = simulate_scores(y, auc=0.92, seed=2)
    weak = simulate_scores(y, auc=0.72, seed=3)
    res = M.delong_test(y, strong, weak)
    assert res["auc_diff"] > 0
    assert res["p_value"] < 0.05


def test_noninferiority_flags_clear_cases(cohort):
    y, _ = cohort
    a = simulate_scores(y, auc=0.88, seed=4)
    b = simulate_scores(y, auc=0.875, seed=5)
    assert M.auc_noninferiority(y, a, b, margin=0.05)["non_inferior"] is True
    far_worse = simulate_scores(y, auc=0.62, seed=6)
    assert M.auc_noninferiority(y, far_worse, a, margin=0.05)["non_inferior"] is False


def test_check_binary_rejects_single_class():
    with pytest.raises(ValueError):
        M.auc_ci(np.zeros(10), np.arange(10.0))


# ----------------------------------------------------- operating points / ROC
def test_operating_point_confusion_consistency(cohort):
    y, s = cohort
    op = M.operating_point(y, s, threshold=0.5)
    assert op["tp"] + op["fp"] + op["tn"] + op["fn"] == len(y)
    assert op["tp"] + op["fn"] == int(y.sum())
    assert 0.0 <= op["sensitivity"] <= 1.0
    assert op["sensitivity_ci"][0] <= op["sensitivity"] <= op["sensitivity_ci"][1]


def test_sensitivity_at_specificity_honours_constraint(cohort):
    y, s = cohort
    for target in (0.80, 0.90, 0.95):
        res = M.sensitivity_at_specificity(y, s, target)
        assert res["specificity_achieved"] >= target - 1e-9


def test_partial_auc_within_bounds(cohort):
    y, s = cohort
    res = M.partial_auc(y, s, spec_low=0.8, spec_high=1.0)
    assert 0.5 <= res["pauc_standardized"] <= 1.0


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_ci_brackets_point_estimate(cohort):
    y, s = cohort
    res = M.bootstrap_ci(y, s, lambda yt, ys: M.auc_ci(yt, ys)["auc"], n_boot=400)
    assert res["ci_low"] <= res["estimate"] <= res["ci_high"]


def test_cluster_bootstrap_widens_ci(cohort):
    # duplicating every case as a 2-view 'patient' must not shrink the CI
    y, s = cohort
    y2 = np.concatenate([y, y])
    s2 = np.concatenate([s, s])
    clusters = np.concatenate([np.arange(len(y)), np.arange(len(y))])
    naive = M.bootstrap_ci(y2, s2, lambda yt, ys: M.auc_ci(yt, ys)["auc"], n_boot=400)
    clust = M.cluster_bootstrap_ci(clusters, y2, s2,
                                   lambda yt, ys: M.auc_ci(yt, ys)["auc"], n_boot=400)
    naive_w = naive["ci_high"] - naive["ci_low"]
    clust_w = clust["ci_high"] - clust["ci_low"]
    assert clust_w > naive_w  # honest variance is wider


# ---------------------------------------------------------------- calibration
def test_brier_of_perfect_probabilities_is_zero():
    y = np.array([0, 1, 0, 1, 1])
    assert M.brier_score(y, y.astype(float)) == pytest.approx(0.0)


def test_calibration_slope_near_one_for_calibrated_scores():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, 4000)
    p = simulate_scores(y, auc=0.85, calibration="good", seed=8)
    cs = M.calibration_intercept_slope(y, p)
    assert cs["slope"] == pytest.approx(1.0, abs=0.25)
    assert abs(cs["intercept"]) < 0.4


# ------------------------------------------------------------------ screening
def test_screening_summary_counts():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 cancers in 10
    recall = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = M.screening_summary(y, recall)
    assert s["n_recalled"] == 2
    assert s["cancer_detection_rate_per_1000"] == pytest.approx(100.0)
    assert s["sensitivity"] == pytest.approx(0.5)
    assert s["ppv1_recall"] == pytest.approx(0.5)


def test_triage_monotonic_in_workload(cohort):
    y, s = cohort
    rows = M.triage_simulation(y, s, n_steps=20)
    wl = [r["workload_reduction"] for r in rows]
    sens = [r["sensitivity_retained"] for r in rows]
    assert wl == sorted(wl)               # workload non-decreasing
    assert sens == sorted(sens, reverse=True)  # retained sensitivity non-increasing


# ------------------------------------------------------------------ subgroups
def test_subgroup_auc_detects_effect_modifier():
    import pandas as pd
    rng = np.random.default_rng(11)
    frames = []
    for grp, auc in [("easy", 0.95), ("hard", 0.70)]:
        y = rng.binomial(1, 0.3, 1200)
        frames.append(pd.DataFrame({
            "y_true": y, "y_score": simulate_scores(y, auc=auc, seed=hash(grp) % 99),
            "grp": grp}))
    df = pd.concat(frames, ignore_index=True)
    res = M.subgroup_auc(df, "y_true", "y_score", "grp")
    assert len(res["groups"]) == 2
    assert res["heterogeneity"]["effect_modifier"] is True
