# Metrics Reference

Metric definitions and their Python functions.
Conceptual background is in [`methodology.md`](methodology.md).

Confidence-interval conventions used throughout:

- **DeLong**: closed-form non-parametric CI for ROC AUC and AUC differences.
- **Wilson**: score interval for binomial proportions (sensitivity, specificity,
  PPV, cancer rates); reliable for small counts and extreme proportions.
- **Bootstrap**: percentile interval (2,000 replicates by default); stratified
  by class, with an optional patient-level cluster variant.

---

## 1 · Discrimination: `mammoval.metrics.delong`, `mammoval.metrics.classification`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| ROC AUC | Does the score rank cancers above non-cancers? Computed across score thresholds. | DeLong | `delong.auc_ci` |
| AUC, patient-cluster CI | AUC uncertainty accounting for correlated images from the same patient. | Cluster bootstrap | `bootstrap.cluster_bootstrap_ci` |
| Paired AUC comparison | Do two models / a model and a reader differ in discrimination on shared cases? | DeLong (correlated) | `delong.delong_test` |
| AUC non-inferiority | Is the new model *not worse* than a reference by more than a pre-set margin? | One-sided DeLong | `delong.auc_noninferiority` |
| Standardised partial AUC | Discrimination restricted to the high-specificity region screening operates in. | Not reported | `classification.partial_auc` |
| Average precision (PR-AUC) | Discrimination in PR-space: more informative than ROC when cancers are rare. | Not reported | `classification.precision_recall` |

---

## 2 · Operating points: `mammoval.metrics.classification`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| Sensitivity @ fixed specificity | Sensitivity at a clinically chosen false-positive rate. | Bootstrap (sens), Wilson (confusion) | `sensitivity_at_specificity` |
| Specificity @ fixed sensitivity | Specificity at a target sensitivity. | Wilson | `specificity_at_sensitivity` |
| Operating point | Full confusion matrix + sensitivity, specificity, PPV, NPV, F1, Youden J at a threshold. | Wilson for sensitivity, specificity, PPV and NPV | `operating_point` |
| Youden point | Threshold maximising sensitivity + specificity − 1 (reference only). | Not reported | `youden_point` |
| Threshold for flag rate | Threshold that recalls a target fraction of exams: compare AI at equal workload. | Not reported | `threshold_for_flag_rate` |

---

## 3 · Lesion localisation (FROC): `mammoval.metrics.localization`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| FROC curve | Localisation sensitivity vs mean false marks per image. | Not reported | `build_froc` |
| Sensitivity @ N FP/image | Localisation sensitivity at tolerable mark rates. | Not reported | `sensitivity_at_fp` |
| Mean FROC sensitivity | Mean over 1/2/3/4 FP/volume = official Duke BCS-DBT ranking metric. | Case bootstrap | `froc_summary`, `pipeline.run_localization_validation` |
| Hit criteria | TP geometry: 2D box-centre test, or the official Duke 3D distance + slice rule. | Not reported | `bbox_hit`, `duke_dbt_hit` |

---

## 4 · Calibration & clinical utility: `mammoval.metrics.calibration`

| Metric | What it answers | Function |
|---|---|---|
| Reliability curve | Does "score 0.1" mean a ~10% cancer rate? | `reliability_curve` |
| Brier score | Proper scoring rule: joint calibration + discrimination error. | `brier_score` |
| Expected calibration error | Count-weighted mean gap between predicted and observed frequency. | `expected_calibration_error` |
| Calibration slope / intercept | Slope and intercept summarise disagreement between predicted probabilities and observed outcomes. | `calibration_intercept_slope` |
| Decision curve (net benefit) | Net true positives per patient after charging for false positives. | `decision_curve` |

---

## 5 · Screening & triage: `mammoval.metrics.screening`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| Recall rate | Fraction of women called back. | Wilson | `screening_summary` |
| Cancer detection rate | Screen-detected cancers per 1,000 exams. | Not reported | `screening_summary` |
| PPV1 (PPV of recall) | Of those recalled, the fraction with cancer. | Wilson | `screening_summary` |
| Triage / rule-out simulation | Workload reduction vs sensitivity retained vs missed cancers. | Not reported | `triage_simulation` |
| Risk-band table | Observed cancer rate per ordinal score band. | Wilson per band | `risk_band_table` |

---

## 6 · Uncertainty & subgroups: `mammoval.metrics.bootstrap`, `mammoval.metrics.subgroups`

| Metric | What it answers | Function |
|---|---|---|
| Bootstrap CI | CI for any scalar metric; stratified by class. | `bootstrap.bootstrap_ci` |
| Paired bootstrap difference | CI + p-value for `metric(A) − metric(B)` on shared cases. | `bootstrap.paired_bootstrap_diff` |
| Cluster bootstrap CI | Patient-level resampling to account for repeated views. | `bootstrap.cluster_bootstrap_ci` |
| Subgroup AUC | Per-stratum AUC with DeLong CIs. | `subgroups.subgroup_auc` |
| Cochran's Q heterogeneity | Do subgroup AUC estimates show heterogeneity? | `subgroups.subgroup_auc` |

---

## Metric selection summary

| Validation question | Primary metric(s) |
|---|---|
| Does the device discriminate cancer? | ROC AUC + DeLong CI; partial AUC |
| How does AUC compare with the supplied reference scores? | Paired DeLong test; AUC non-inferiority |
| Where do we set the threshold? | Sensitivity @ fixed specificity; operating-point confusion matrix |
| Do the marks land on lesions? | FROC; mean sensitivity @ 1–4 FP/image |
| Are the scores trustworthy probabilities? | Calibration slope/intercept; Brier; ECE |
| What net benefit follows from the assumed thresholds? | Decision-curve net benefit |
| What workload and missed-case trade-offs appear in simulation? | Triage simulation; risk-band table |
| Does it fail for some women? | Subgroup AUC; Cochran's Q (esp. breast density) |
