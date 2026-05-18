# Metrics Reference

Every metric in the validation engine, what it answers, and where it lives.
Conceptual background is in [`methodology.md`](methodology.md).

Confidence-interval conventions used throughout:

- **DeLong** — closed-form non-parametric CI for ROC AUC and AUC differences.
- **Wilson** — score interval for binomial proportions (sensitivity, specificity,
  PPV, cancer rates); reliable for small counts and extreme proportions.
- **Bootstrap** — percentile interval (2,000 replicates by default); stratified
  by class, with an optional patient-level cluster variant.

---

## 1 · Discrimination — `mammoval.metrics.delong`, `…classification`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| ROC AUC | Does the score rank cancers above non-cancers? Threshold- and prevalence-free. | DeLong | `delong.auc_ci` |
| AUC, patient-cluster CI | Same, but honest about correlated multi-view exams. | Cluster bootstrap | `bootstrap.cluster_bootstrap_ci` |
| Paired AUC comparison | Do two models / a model and a reader differ in discrimination on shared cases? | DeLong (correlated) | `delong.delong_test` |
| AUC non-inferiority | Is the new model *not worse* than a reference by more than a pre-set margin? | One-sided DeLong | `delong.auc_noninferiority` |
| Standardised partial AUC | Discrimination restricted to the high-specificity region screening operates in. | — | `classification.partial_auc` |
| Average precision (PR-AUC) | Discrimination in PR-space — more informative than ROC when cancers are rare. | — | `classification.precision_recall` |

**Why for a breast-AI company:** AUC with a DeLong CI is the headline number of
every standalone validation; the non-inferiority test is the statistical form
of a "comparable to radiologists" claim; partial AUC reflects that screening
lives at high specificity.

---

## 2 · Operating points — `mammoval.metrics.classification`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| Sensitivity @ fixed specificity | Sensitivity at a clinically chosen false-positive rate. | Bootstrap (sens), Wilson (confusion) | `sensitivity_at_specificity` |
| Specificity @ fixed sensitivity | The dual — "match the radiologist's sensitivity, compare recall". | Wilson | `specificity_at_sensitivity` |
| Operating point | Full confusion matrix + sensitivity, specificity, PPV, NPV, F1, Youden J at a threshold. | Wilson on every proportion | `operating_point` |
| Youden point | Threshold maximising sensitivity + specificity − 1 (reference only). | — | `youden_point` |
| Threshold for flag rate | Threshold that recalls a target fraction of exams — compare AI at equal workload. | — | `threshold_for_flag_rate` |

**Why:** a deployed device runs at one operating point; a screening device is
constrained by specificity, so sensitivity-at-fixed-specificity is the
decision-relevant number.

---

## 3 · Lesion localisation (FROC) — `mammoval.metrics.localization`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| FROC curve | Localisation sensitivity vs mean false marks per image. | — | `build_froc` |
| Sensitivity @ N FP/image | Localisation sensitivity at tolerable mark rates. | — | `sensitivity_at_fp` |
| Mean FROC sensitivity | Mean over 1/2/3/4 FP/volume = official Duke BCS-DBT ranking metric. | Case bootstrap | `froc_summary`, `pipeline.run_localization_validation` |
| Hit criteria | TP geometry: 2D box-centre test, or the official Duke 3D distance + slice rule. | — | `bbox_hit`, `duke_dbt_hit` |

**Why:** a CADe device draws regions — exam-level AUC cannot tell whether marks
land on lesions. FROC is the metric that scores the actual product.

---

## 4 · Calibration & clinical utility — `mammoval.metrics.calibration`

| Metric | What it answers | Function |
|---|---|---|
| Reliability curve | Does "score 0.1" mean a ~10% cancer rate? | `reliability_curve` |
| Brier score | Proper scoring rule — joint calibration + discrimination error. | `brier_score` |
| Expected calibration error | Count-weighted mean gap between predicted and observed frequency. | `expected_calibration_error` |
| Calibration slope / intercept | Slope < 1 = over-confident; non-zero intercept = prevalence-shift drift. | `calibration_intercept_slope` |
| Decision curve (net benefit) | Net true positives per patient after charging for false positives. | `decision_curve` |

**Why:** rule-out thresholds and risk bands treat the score as a probability —
miscalibration silently breaks them. Decision-curve analysis states usefulness
in clinical, not statistical, units.

---

## 5 · Screening & triage — `mammoval.metrics.screening`

| Metric | What it answers | CI | Function |
|---|---|---|---|
| Recall rate | Fraction of women called back. | Wilson | `screening_summary` |
| Cancer detection rate | Screen-detected cancers per 1,000 exams. | — | `screening_summary` |
| PPV1 (PPV of recall) | Of those recalled, the fraction with cancer. | Wilson | `screening_summary` |
| Triage / rule-out simulation | Workload reduction vs sensitivity retained vs missed cancers. | — | `triage_simulation` |
| Risk-band table | Observed cancer rate per ordinal score band (Transpara-style). | Wilson per band | `risk_band_table` |

**Why:** these are the numbers a radiology department and a regulator quote.
The triage simulation is the direct analogue of an AI workload-reduction claim;
it keeps the safety cost (missed cancers) visible next to the efficiency gain.

---

## 6 · Uncertainty & subgroups — `mammoval.metrics.bootstrap`, `…subgroups`

| Metric | What it answers | Function |
|---|---|---|
| Bootstrap CI | CI for any scalar metric; stratified by class. | `bootstrap.bootstrap_ci` |
| Paired bootstrap difference | CI + p-value for `metric(A) − metric(B)` on shared cases. | `bootstrap.paired_bootstrap_diff` |
| Cluster bootstrap CI | Patient-level resampling — correct variance for multi-view exams. | `bootstrap.cluster_bootstrap_ci` |
| Subgroup AUC | Per-stratum AUC with DeLong CIs. | `subgroups.subgroup_auc` |
| Cochran's Q heterogeneity | Is a variable (e.g. breast density) an effect modifier? | `subgroups.subgroup_auc` |

**Why:** breast density is the headline effect modifier in mammography AI;
detecting heterogeneity is what turns a pooled number into a safe one. Cluster
resampling stops correlated views from understating uncertainty.

---

## Metric selection summary

| Validation question | Primary metric(s) |
|---|---|
| Does the device discriminate cancer? | ROC AUC + DeLong CI; partial AUC |
| Is it comparable to a radiologist? | Paired DeLong test; AUC non-inferiority |
| Where do we set the threshold? | Sensitivity @ fixed specificity; operating-point confusion matrix |
| Do the marks land on lesions? | FROC; mean sensitivity @ 1–4 FP/image |
| Are the scores trustworthy probabilities? | Calibration slope/intercept; Brier; ECE |
| Is it clinically useful? | Decision-curve net benefit |
| Can it cut workload safely? | Triage simulation; risk-band table |
| Does it fail for some women? | Subgroup AUC; Cochran's Q (esp. breast density) |
