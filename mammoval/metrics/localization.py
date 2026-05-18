"""Free-response ROC (FROC) analysis for lesion localisation.

A mammography AI device does not just emit an exam score — it places **marks**
on suspicious regions, and a mark only helps the radiologist if it lands on the
actual lesion. Pure ROC/AUC ignores location: a model can get the exam-level
call right for the wrong reason. FROC fixes that by scoring each *mark*:

* a mark that lands on a true lesion (per a geometric hit criterion) is a
  true-positive localisation;
* every other mark is a false positive.

The FROC curve plots **lesion-localisation sensitivity** against the **mean
number of false marks per image**. Unlike ROC it is unbounded on the x-axis
(an image can carry arbitrarily many false marks), so performance is reported
as sensitivity at clinically tolerable mark rates.

This module is metric-agnostic about geometry: pass any ``hit_fn``. Two are
provided — a generic 2D bounding-box test and the **official Duke BCS-DBT**
criterion — so the same FROC engine serves both the 2D and 3D pipelines.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "build_froc",
    "sensitivity_at_fp",
    "froc_summary",
    "bbox_hit",
    "duke_dbt_hit",
]


def build_froc(cases, hit_fn):
    """Construct an FROC curve from per-image detections and ground truth.

    Parameters
    ----------
    cases : list of dict
        One entry per image/volume, each with:
          ``lesions``    : list of ground-truth lesion dicts (geometry only);
          ``detections`` : list of detection dicts, each with a ``score`` key
                           plus whatever geometry ``hit_fn`` consumes.
        Normal images simply carry an empty ``lesions`` list — they still
        contribute their false marks to the per-image FP rate.
    hit_fn : callable(detection, lesion) -> bool
        Geometric true-positive criterion.

    Matching rule
    -------------
    Within an image, detections are matched greedily in descending score
    order; each ground-truth lesion can be claimed once. A detection that
    claims a lesion is a TP mark, otherwise an FP mark. This is the standard
    FROC accounting and matches the Duke BCS-DBT evaluator.

    Returns
    -------
    dict with ``thresholds``, ``sensitivity`` and ``fp_per_image`` arrays
    (monotone, ready to plot), plus ``n_cases`` and ``n_lesions``.
    """
    n_cases = len(cases)
    total_lesions = sum(len(c.get("lesions", [])) for c in cases)

    marks = []  # (score, is_true_positive)
    for c in cases:
        lesions = list(c.get("lesions", []))
        claimed = [False] * len(lesions)
        dets = sorted(c.get("detections", []), key=lambda d: -float(d["score"]))
        for d in dets:
            hit_idx = None
            for li, lesion in enumerate(lesions):
                if not claimed[li] and hit_fn(d, lesion):
                    hit_idx = li
                    break
            if hit_idx is None:
                marks.append((float(d["score"]), False))
            else:
                claimed[hit_idx] = True
                marks.append((float(d["score"]), True))

    marks.sort(key=lambda x: -x[0])

    thresholds = [math.inf]
    tp_counts, fp_counts = [0], [0]
    tp = fp = 0
    i, m = 0, len(marks)
    while i < m:
        score = marks[i][0]
        j = i
        while j < m and marks[j][0] == score:  # consume all ties together
            if marks[j][1]:
                tp += 1
            else:
                fp += 1
            j += 1
        thresholds.append(score)
        tp_counts.append(tp)
        fp_counts.append(fp)
        i = j

    tp_counts = np.asarray(tp_counts, dtype=float)
    fp_counts = np.asarray(fp_counts, dtype=float)
    sensitivity = tp_counts / total_lesions if total_lesions else np.zeros_like(tp_counts)
    fp_per_image = fp_counts / n_cases if n_cases else np.zeros_like(fp_counts)
    return {
        "thresholds": thresholds,
        "sensitivity": sensitivity.tolist(),
        "fp_per_image": fp_per_image.tolist(),
        "n_cases": n_cases,
        "n_lesions": total_lesions,
    }


def sensitivity_at_fp(froc, fp_points):
    """Interpolate localisation sensitivity at given false-marks-per-image.

    Linear interpolation on the (monotone non-decreasing) FROC curve. If a
    requested operating point exceeds the highest FP rate the model reaches,
    the plateau sensitivity is returned.
    """
    fppi = np.asarray(froc["fp_per_image"], dtype=float)
    sens = np.asarray(froc["sensitivity"], dtype=float)
    out = {}
    for fp in fp_points:
        if fp >= fppi[-1]:
            out[float(fp)] = float(sens[-1])
        else:
            out[float(fp)] = float(np.interp(fp, fppi, sens))
    return out


def froc_summary(cases, hit_fn, fp_points=(1.0, 2.0, 3.0, 4.0)):
    """End-to-end FROC summary.

    ``mean_sensitivity`` averages sensitivity over ``fp_points``. With the
    default points (1, 2, 3, 4 FP/volume) this reproduces the **official
    Duke BCS-DBT / DBTex challenge ranking metric**.
    """
    froc = build_froc(cases, hit_fn)
    sens = sensitivity_at_fp(froc, fp_points)
    return {
        "froc_curve": froc,
        "sensitivity_at_fp": {f"fp_{fp:g}": v for fp, v in sens.items()},
        "mean_sensitivity": float(np.mean(list(sens.values()))),
        "fp_points": [float(x) for x in fp_points],
        "n_cases": froc["n_cases"],
        "n_lesions": froc["n_lesions"],
    }


# --------------------------------------------------------------------------
# Hit criteria
# --------------------------------------------------------------------------
def bbox_hit(detection, lesion):
    """Generic 2D hit test: the detection point lies inside the lesion box.

    Coordinate convention (shared with :func:`duke_dbt_hit`): ``lesion`` ``x``/
    ``y`` are the **box centre**, with ``width``/``height`` the box extent;
    ``detection`` needs the predicted point ``x``/``y``. Used for the CBIS-DDSM
    2D localisation check, where ground-truth ROI masks give a box per
    abnormality.
    """
    half_w, half_h = lesion["width"] / 2.0, lesion["height"] / 2.0
    return (abs(detection["x"] - lesion["x"]) <= half_w
            and abs(detection["y"] - lesion["y"]) <= half_h)


def duke_dbt_hit(detection, lesion):
    """Official Duke BCS-DBT true-positive criterion.

    A predicted box centre is a hit when both hold:

    * its in-plane distance to the ground-truth box centre is at most
      ``max(sqrt(width**2 + height**2) / 2, 100)`` pixels — half the GT box
      diagonal, with a 100 px floor for small lesions;
    * its centre slice is within ``volume_slices / 4`` of the GT centre slice
      (±25 % of the reconstructed volume depth).

    ``detection`` keys: ``x``, ``y`` (box centre), ``slice``.
    ``lesion`` keys: ``x``, ``y`` (box centre), ``width``, ``height``,
    ``slice``, ``volume_slices``.

    Mirrors ``_is_tp`` in ``mazurowski-lab/duke-dbt-data/duke_dbt_data.py`` so
    pipeline numbers are directly comparable to the published baseline.
    """
    dist_threshold = max(math.hypot(lesion["width"], lesion["height"]) / 2.0, 100.0)
    distance = math.hypot(detection["x"] - lesion["x"], detection["y"] - lesion["y"])
    slice_tolerance = lesion["volume_slices"] / 4.0
    slice_ok = abs(detection["slice"] - lesion["slice"]) <= slice_tolerance
    return (distance <= dist_threshold) and slice_ok
