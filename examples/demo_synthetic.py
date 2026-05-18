"""End-to-end demo on synthetic data — the pipeline's runnability guarantee.

This script needs **no dataset download and no model weights**: it fabricates a
realistic predictions table with known properties, runs the entire validation
pipeline, and writes the HTML report. If this runs, the validation engine,
orchestration and report are all working — which is exactly what a reviewer
wants to confirm before pointing the pipeline at real data.

The synthetic cohort is deliberately built with a *known* flaw — AUC degrades
in dense breasts — so the subgroup analysis has a real effect modifier to find.

    python examples/demo_synthetic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mammoval.models import simulate_scores, simulate_detections
from mammoval.metrics import bbox_hit
from mammoval.pipeline import (
    ClassificationConfig, run_classification_validation,
    run_localization_validation,
)
from mammoval.report import build_report

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def make_cohort(n=2600, seed=42):
    """A CBIS-DDSM-shaped synthetic cohort with a built-in density effect."""
    rng = np.random.default_rng(seed)
    density = rng.choice(
        ["A (fatty)", "B (scattered)", "C (heterog. dense)", "D (extremely dense)"],
        size=n, p=[0.10, 0.40, 0.40, 0.10])
    lesion_type = rng.choice(["mass", "calcification"], size=n, p=[0.55, 0.45])
    y_true = rng.binomial(1, 0.27, size=n)            # enriched, CBIS-like
    patient_id = rng.integers(0, 1700, size=n)        # ~1.5 views per patient

    # AI discrimination genuinely degrades in dense breasts (the masking effect)
    auc_by_density = {"A (fatty)": 0.93, "B (scattered)": 0.90,
                      "C (heterog. dense)": 0.85, "D (extremely dense)": 0.80}
    score = np.zeros(n)
    for i, (dens, auc) in enumerate(auc_by_density.items()):
        mask = density == dens
        if mask.any():
            score[mask] = simulate_scores(
                y_true[mask], auc=auc, calibration="overconfident", seed=100 + i)

    # reference 'reader': an ordinal BI-RADS-style assessment (0-5), AUC ~0.86
    reader_latent = simulate_scores(y_true, auc=0.86, seed=7)
    birads = np.clip(np.round(reader_latent * 5.0), 0, 5)

    return pd.DataFrame({
        "case_id": [f"SYN-{i:05d}" for i in range(n)],
        "patient_id": [f"P{p:04d}" for p in patient_id],
        "y_true": y_true,
        "y_score": score,
        "birads_assessment": birads,
        "breast_density_cat": density,
        "abnormality_type": lesion_type,
    })


def make_froc_cases(n=900, seed=1):
    """Synthetic per-image detections for the FROC (localisation) study."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        lesions = []
        if rng.random() < 0.5:  # half the images carry a lesion
            for _ in range(rng.choice([1, 1, 1, 2])):
                lesions.append({
                    "x": float(rng.uniform(600, 2400)),   # box centre
                    "y": float(rng.uniform(600, 2400)),
                    "width": float(rng.uniform(120, 380)),
                    "height": float(rng.uniform(120, 380)),
                })
        cases.append({"case_id": f"IMG-{i:04d}", "lesions": lesions,
                      "width": 3000, "height": 3000})
    return simulate_detections(cases, hit_sensitivity=0.85,
                               fp_per_image=1.3, seed=seed + 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Building synthetic cohort ...")
    df = make_cohort()

    config = ClassificationConfig(
        reader_col="birads_assessment",
        subgroup_cols=("breast_density_cat", "abnormality_type"),
        target_specificities=(0.90, 0.96),
        ni_margin=0.05,
        n_boot=800,
        is_probability=True,
        dataset_name="SYNTHETIC demo cohort (no real patient data)",
    )
    print("Running classification validation ...")
    cls = run_classification_validation(df, config)
    print(f"  AUC = {cls['discrimination']['auc']:.3f} "
          f"{tuple(round(x, 3) for x in cls['discrimination']['auc_ci'])}")

    print("Running localisation (FROC) validation ...")
    loc = run_localization_validation(
        make_froc_cases(), bbox_hit, fp_points=(0.5, 1.0, 2.0, 4.0),
        n_boot=400, dataset_name="SYNTHETIC demo (localisation)")
    print(f"  FROC mean sensitivity = {loc['mean_sensitivity']:.3f}")

    out = os.path.abspath(os.path.join(OUT_DIR, "demo_validation_report.html"))
    build_report(
        cls, loc, output_path=out,
        title="Mammography AI - Clinical Validation Report (Synthetic Demo)",
        extra_limitations=[
            "This report is built from SYNTHETIC data for demonstration only - "
            "no real model and no patient images were used.",
        ])
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
