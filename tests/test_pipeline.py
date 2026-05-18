"""Integration tests: predictions table -> validation result -> HTML report."""
import os

import numpy as np
import pandas as pd
import pytest

from mammoval.models import simulate_scores, simulate_detections
from mammoval.metrics import bbox_hit
from mammoval.pipeline import (
    ClassificationConfig, run_classification_validation,
    run_localization_validation,
)
from mammoval.report import build_report


@pytest.fixture
def predictions():
    rng = np.random.default_rng(3)
    n = 1200
    y = rng.binomial(1, 0.3, n)
    return pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)],
        "patient_id": [f"p{i // 2}" for i in range(n)],   # 2 views per patient
        "y_true": y,
        "y_score": simulate_scores(y, auc=0.87, seed=1),
        "birads_assessment": np.clip(np.round(simulate_scores(y, auc=0.82, seed=2) * 5), 0, 5),
        "density": rng.choice(["fatty", "dense"], n),
    })


def test_classification_pipeline_sections(predictions):
    res = run_classification_validation(
        predictions,
        ClassificationConfig(reader_col="birads_assessment",
                             subgroup_cols=("density",), n_boot=200))
    for section in ("meta", "discrimination", "operating_points",
                    "calibration", "screening", "subgroups", "reader_comparison"):
        assert section in res
    assert 0.5 < res["discrimination"]["auc"] < 1.0
    lo, hi = res["discrimination"]["auc_ci"]
    assert lo < res["discrimination"]["auc"] < hi
    # reader comparison ran on the paired cases
    assert res["reader_comparison"]["n_cases_with_reader"] == len(predictions)


def test_pipeline_handles_missing_scores(predictions):
    dirty = predictions.copy()
    dirty.loc[:50, "y_score"] = np.nan
    res = run_classification_validation(dirty, ClassificationConfig(n_boot=100))
    assert res["meta"]["n_dropped_missing"] == 51
    assert res["meta"]["n_cases"] == len(predictions) - 51


def test_localization_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    cases = []
    for i in range(200):
        lesions = [{"x": 500.0, "y": 500.0, "width": 150.0, "height": 150.0}] \
            if rng.random() < 0.5 else []
        cases.append({"case_id": f"c{i}", "lesions": lesions,
                      "width": 2000, "height": 2000})
    cases = simulate_detections(cases, hit_sensitivity=0.8, fp_per_image=1.0, seed=6)
    res = run_localization_validation(cases, bbox_hit, fp_points=(1, 2, 4), n_boot=150)
    assert 0.0 <= res["mean_sensitivity"] <= 1.0
    lo, hi = res["mean_sensitivity_ci"]
    assert lo <= res["mean_sensitivity"] <= hi


def test_report_is_written(predictions, tmp_path):
    cls = run_classification_validation(
        predictions, ClassificationConfig(subgroup_cols=("density",), n_boot=100))
    out = tmp_path / "report.html"
    path = build_report(cls, output_path=str(out))
    assert os.path.exists(path)
    html = out.read_text(encoding="utf-8")
    assert "Clinical Validation Report" in html
    assert "data:image/png;base64," in html      # plots embedded
    assert 'class="err' not in html              # no section errored
