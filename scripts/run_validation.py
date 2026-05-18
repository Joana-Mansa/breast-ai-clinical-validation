"""Run the classification validation pipeline on a predictions CSV — headless.

This is the command-line face of the pipeline: hand it any CSV with a label
column and a score column and it writes the full HTML validation report. It
keeps the model and the metrics fully decoupled — the CSV could come from this
project's adapters, a vendor device, or an external study.

Usage
-----
    python scripts/run_validation.py preds.csv \
        --label y_true --score y_score --patient patient_id \
        --reader birads_assessment \
        --subgroups breast_density_cat,abnormality_type \
        --report outputs/report.html

The CSV needs at least the label and score columns; everything else is
optional and enables extra report sections when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mammoval.pipeline import ClassificationConfig, run_classification_validation
from mammoval.report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="predictions CSV")
    parser.add_argument("--label", default="y_true", help="label column (0/1)")
    parser.add_argument("--score", default="y_score", help="AI score column")
    parser.add_argument("--patient", default="patient_id", help="patient id column")
    parser.add_argument("--reader", default=None,
                        help="optional reader-reference score column")
    parser.add_argument("--subgroups", default="",
                        help="comma-separated subgroup columns")
    parser.add_argument("--specificities", default="0.90,0.96",
                        help="comma-separated target specificities")
    parser.add_argument("--ni-margin", type=float, default=0.05,
                        help="AUC non-inferiority margin")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap replicates")
    parser.add_argument("--not-probability", action="store_true",
                        help="scores are not calibrated probabilities (skip calibration)")
    parser.add_argument("--name", default="dataset", help="dataset name for the report")
    parser.add_argument("--report", default="outputs/validation_report.html",
                        help="output HTML path")
    parser.add_argument("--json", default=None,
                        help="also dump the raw results dict to this JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in (args.label, args.score):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {args.csv}; have {list(df.columns)}")

    config = ClassificationConfig(
        label_col=args.label,
        score_col=args.score,
        patient_col=args.patient,
        reader_col=args.reader,
        subgroup_cols=tuple(c for c in args.subgroups.split(",") if c),
        target_specificities=tuple(float(s) for s in args.specificities.split(",")),
        ni_margin=args.ni_margin,
        n_boot=args.n_boot,
        is_probability=not args.not_probability,
        dataset_name=args.name,
    )
    print(f"Running validation on {len(df)} rows from {args.csv} ...")
    results = run_classification_validation(df, config)

    d = results["discrimination"]
    print(f"  AUC = {d['auc']:.3f}  95% CI {d['auc_ci'][0]:.3f}-{d['auc_ci'][1]:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    build_report(results, output_path=args.report, title=f"Clinical Validation — {args.name}")
    print(f"  report -> {args.report}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  results -> {args.json}")


if __name__ == "__main__":
    main()
