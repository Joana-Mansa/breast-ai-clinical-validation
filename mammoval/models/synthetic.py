"""Synthetic predictions — the pipeline's reproducible self-test.

No public dataset and no pretrained checkpoint are needed to demonstrate that
the validation engine is correct end to end: this module fabricates
predictions with *known, controllable* properties so every metric can be
checked against a value we already know.

It serves three jobs:

* **Guaranteed-runnable demo.** ``examples/demo_synthetic.py`` builds a full
  HTML report from these scores with zero downloads.
* **Unit-test ground truth.** A score set generated for a target AUC must be
  recovered (within sampling error) by :func:`mammoval.metrics.auc_ci`.
* **Operating-point sandbox.** Calibration can be deliberately broken, or two
  "readers" of different skill simulated, to exercise comparison metrics.

Construction guarantee
----------------------
Scores are the Bayes posterior probability of a Gaussian signal-detection
model. The posterior is a strictly monotone transform of the latent score, so
the ROC AUC is **exactly preserved** — ``simulate_scores(y, auc=0.88)`` yields
scores whose population AUC is 0.88 — while the output is, by construction, a
*calibrated probability*. The ``calibration`` knob then distorts it on purpose.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["simulate_scores", "simulate_detections"]


def simulate_scores(y_true, auc=0.88, calibration="good", seed=0):
    """Fabricate calibrated-probability scores with a known target AUC.

    Parameters
    ----------
    y_true : array of {0, 1}
        Reference labels; their empirical prevalence sets the posterior.
    auc : float
        Target ROC AUC (the population AUC the scores will exhibit).
    calibration : {'good', 'overconfident', 'underconfident'}
        ``good`` returns the exact Bayes posterior. The other two apply a
        monotone distortion — AUC is unchanged, calibration is degraded — so
        the calibration metrics have something to detect.
    seed : int

    Returns
    -------
    ndarray of float in (0, 1).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    prevalence = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))

    # Gaussian signal-detection model: positives ~ N(d, 1), negatives ~ N(0, 1).
    # AUC = Phi(d / sqrt(2))  =>  d = sqrt(2) * Phi^-1(AUC).
    d = np.sqrt(2.0) * stats.norm.ppf(np.clip(auc, 0.5 + 1e-6, 1 - 1e-6))
    latent = rng.normal(d * y, 1.0)

    # Bayes posterior P(y=1 | latent): a monotone function of latent,
    # so AUC is preserved exactly and the score is a true probability.
    likelihood_ratio = np.exp(d * latent - 0.5 * d * d)
    prob = prevalence * likelihood_ratio / (
        prevalence * likelihood_ratio + (1.0 - prevalence)
    )

    if calibration == "overconfident":
        prob = _temperature(prob, 1.7)
    elif calibration == "underconfident":
        prob = _temperature(prob, 0.6)
    return np.clip(prob, 1e-4, 1 - 1e-4)


def _temperature(prob, gamma):
    """Monotone logit-power distortion (gamma>1 sharpens, <1 flattens)."""
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    pg = prob ** gamma
    return pg / (pg + (1.0 - prob) ** gamma)


def simulate_detections(cases, hit_sensitivity=0.82, fp_per_image=1.4, seed=0):
    """Fabricate per-image detections for the FROC engine.

    Each ground-truth lesion gets, with probability ``hit_sensitivity``, a
    detection at its centre carrying a high score; every image additionally
    gets ``Poisson(fp_per_image)`` false detections carrying lower scores. The
    score overlap is what gives :func:`mammoval.metrics.build_froc` a realistic,
    non-degenerate curve to trace.

    Parameters
    ----------
    cases : list of dict
        Each with a ``lesions`` list (lesion dicts carrying ``x``/``y`` box
        centres, and ``slice``/``volume_slices`` for 3D) and an image extent
        under ``width``/``height`` (defaults 3000x3000 if absent).

    Returns
    -------
    A new list of cases with a ``detections`` key added to each.
    """
    rng = np.random.default_rng(seed)
    out = []
    for c in cases:
        width = c.get("width", 3000)
        height = c.get("height", 3000)
        is_3d = any("volume_slices" in les for les in c.get("lesions", []))
        detections = []

        for lesion in c.get("lesions", []):
            if rng.random() < hit_sensitivity:
                det = {
                    "score": float(rng.beta(5.0, 1.8)),  # skewed high
                    "x": lesion["x"] + rng.normal(0, 8),
                    "y": lesion["y"] + rng.normal(0, 8),
                }
                if is_3d:
                    det["slice"] = lesion["slice"] + rng.normal(0, 3)
                detections.append(det)

        for _ in range(rng.poisson(fp_per_image)):
            det = {
                "score": float(rng.beta(1.8, 4.5)),  # skewed low
                "x": float(rng.uniform(0, width)),
                "y": float(rng.uniform(0, height)),
            }
            if is_3d:
                vs = c["lesions"][0]["volume_slices"] if c.get("lesions") else 60
                det["slice"] = float(rng.uniform(0, vs))
            detections.append(det)

        out.append({**c, "detections": detections})
    return out
