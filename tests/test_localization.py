"""Unit tests for the FROC localisation engine."""
import math

import numpy as np
import pytest

from mammoval.metrics import build_froc, sensitivity_at_fp, froc_summary
from mammoval.metrics import bbox_hit, duke_dbt_hit


def _lesion(x, y, w=100, h=100, **extra):
    return {"x": x, "y": y, "width": w, "height": h, **extra}


# --------------------------------------------------------------- hit criteria
def test_bbox_hit_centre_semantics():
    les = _lesion(500, 500, w=200, h=200)          # box spans 400..600
    assert bbox_hit({"x": 560, "y": 460}, les) is True
    assert bbox_hit({"x": 650, "y": 500}, les) is False


def test_duke_hit_distance_and_slice_tolerance():
    les = _lesion(1000, 1000, w=120, h=160, slice=40, volume_slices=80)
    # within the 100 px floor and within +-20 slices -> hit
    assert duke_dbt_hit({"x": 1050, "y": 1000, "slice": 50}, les) is True
    # right distance, but slice too far (|10-40| = 30 > 80/4 = 20) -> miss
    assert duke_dbt_hit({"x": 1010, "y": 1000, "slice": 10}, les) is False
    # in-plane distance too large -> miss
    assert duke_dbt_hit({"x": 1400, "y": 1000, "slice": 40}, les) is False


# ----------------------------------------------------------------------- FROC
def test_perfect_detector_reaches_full_sensitivity():
    cases = [{"case_id": f"c{i}",
              "lesions": [_lesion(500, 500)],
              "detections": [{"score": 0.99, "x": 500, "y": 500}]}
             for i in range(20)]
    summary = froc_summary(cases, bbox_hit, fp_points=(1, 2, 4))
    assert summary["mean_sensitivity"] == pytest.approx(1.0)
    assert summary["n_lesions"] == 20


def test_froc_curve_is_monotone():
    rng = np.random.default_rng(0)
    cases = []
    for i in range(60):
        lesions = [_lesion(rng.uniform(200, 800), rng.uniform(200, 800))] \
            if rng.random() < 0.5 else []
        dets = [{"score": float(rng.random()),
                 "x": float(rng.uniform(0, 1000)),
                 "y": float(rng.uniform(0, 1000))}
                for _ in range(rng.integers(0, 4))]
        cases.append({"case_id": f"c{i}", "lesions": lesions, "detections": dets})
    froc = build_froc(cases, bbox_hit)
    sens = froc["sensitivity"]
    fppi = froc["fp_per_image"]
    assert sens == sorted(sens)        # sensitivity non-decreasing
    assert fppi == sorted(fppi)        # FP per image non-decreasing
    assert all(0.0 <= v <= 1.0 for v in sens)


def test_false_positives_only_on_negative_images():
    # 2 images, no lesions anywhere, one false mark each -> 1 FP/image, sens 0
    cases = [{"case_id": f"n{i}", "lesions": [],
              "detections": [{"score": 0.8, "x": 10, "y": 10}]}
             for i in range(2)]
    froc = build_froc(cases, bbox_hit)
    assert froc["n_lesions"] == 0
    assert froc["fp_per_image"][-1] == pytest.approx(1.0)


def test_missed_lesion_lowers_sensitivity():
    cases = [
        {"case_id": "hit", "lesions": [_lesion(500, 500)],
         "detections": [{"score": 0.9, "x": 500, "y": 500}]},
        {"case_id": "miss", "lesions": [_lesion(500, 500)],
         "detections": [{"score": 0.9, "x": 50, "y": 50}]},  # far from lesion
    ]
    froc = build_froc(cases, bbox_hit)
    assert max(froc["sensitivity"]) == pytest.approx(0.5)  # 1 of 2 lesions found


def test_sensitivity_at_fp_interpolates():
    cases = [{"case_id": f"c{i}", "lesions": [_lesion(500, 500)],
              "detections": [{"score": 0.9, "x": 500, "y": 500}]}
             for i in range(10)]
    froc = build_froc(cases, bbox_hit)
    pts = sensitivity_at_fp(froc, [0.5, 1.0, 2.0])
    assert all(0.0 <= v <= 1.0 for v in pts.values())
