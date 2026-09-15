# Evaluation methodology

The pipeline evaluates saved model scores or lesion detections against dataset labels. Its purpose is to describe retrospective performance, uncertainty and differences between subgroups.

[Project overview](../README.md) · [Metrics reference](metrics_reference.md)

## Study scope

Keep model training, model selection and final evaluation separate. Record the dataset version, split, label definition, exclusions and unit of analysis with each result. When multiple images belong to one patient, preserve the patient identifier.

The included classification workflow trains on the CBIS-DDSM training split and evaluates on its test split. Several operating points are estimated from the evaluation cohort. These describe the observed ROC curve; validating a fixed decision threshold requires selecting it separately and evaluating it on held-out data.

The project reports standalone performance. It does not run a prospective trial or a multi-reader multi-case study, and it does not establish benefit from AI-assisted reading or regulatory compliance.

## Discrimination and uncertainty

- **ROC AUC** measures how well scores rank positive cases above negative cases.
- **Partial AUC** restricts evaluation to a configured false-positive-rate range.
- **DeLong intervals** estimate uncertainty in AUC. Paired DeLong comparisons account for two scores measured on the same cases.
- **Patient-cluster bootstrap** resamples patients together with their images to account for repeated observations. It addresses dependence within this sample, not differences between institutions or populations.

AUC depends on the distribution of scores in each class. It should not be interpreted as a dataset-independent property of a model.

## Operating points

The classification report estimates sensitivity at target specificities, with defaults of 0.90 and 0.96. It also reports confusion matrices and a Youden operating point.

A threshold selected from the same cohort being evaluated can give optimistic performance estimates. The reported thresholds are exploratory unless they were fixed independently. Target specificity alone does not establish an appropriate clinical operating point.

## Lesion localisation

Free-response ROC (FROC) measures lesion sensitivity against false detections per image or volume. A geometric matching rule determines whether a detection corresponds to an annotated lesion.

The Duke path uses `duke_dbt_hit`: the predicted centre must lie within `max(sqrt(W² + H²) / 2, 100)` pixels of the annotated centre and within `VolumeSlices / 4` slices. Detections are matched in descending score order. The summary averages sensitivity at 1, 2, 3 and 4 false positives per volume.

See [the 3D guide](3d_pipeline.md) for coordinate handling, data assembly and limitations of the annotations.

## Calibration and decision curves

Calibration compares predicted probabilities with observed outcome frequencies. The report includes reliability curves, Brier score, expected calibration error, and calibration intercept and slope.

These estimates depend on the evaluation cohort. A nonzero calibration intercept can indicate systematic bias; it does not identify a unique cause. Scores that are not probabilities should be evaluated with calibration disabled.

Decision curves calculate net benefit over assumed decision thresholds. They illustrate the consequences of those assumptions and the sampled outcomes; they do not by themselves establish clinical usefulness.

## Screening and triage summaries

The report computes cancer-detection rate, recall rate and positive predictive value, and simulates rule-out thresholds and score bands. These quantities depend on the cohort and its prevalence.

CBIS-DDSM is lesion-enriched and non-consecutive. Its screening and workload summaries are illustrations of the calculations, not estimates for a screening programme. Simulated triage results do not establish that reducing workload would be safe in practice.

## Subgroups

The pipeline reports AUC and uncertainty by available subgroup, such as breast density or lesion type, and uses Cochran's Q to summarise heterogeneity across strata.

Interpret subgroup results with sample sizes, class counts and repeated-patient structure. Multiple comparisons are exploratory. A significant difference does not establish its cause, and a nonsignificant difference does not establish equivalent performance.

## Reference-reader comparison

The classification workflow can compare AI scores with a reference-score column using paired DeLong and AUC non-inferiority calculations. The non-inferiority margin is configurable and must be justified independently for any confirmatory interpretation.

CBIS-DDSM BI-RADS assessment is an ordinal proxy. Category 0 denotes an incomplete assessment and does not fit a simple malignancy ranking. This comparison demonstrates the calculation; it is not an independent reader study or a measurement of AI-assisted radiologist performance.

## Limitations

- Historical public datasets may differ from the population, equipment and acquisition practices of an intended application.
- Missing images and excluded rows can alter the evaluated cohort. Review reported exclusions.
- Confidence intervals describe sampling uncertainty under their assumptions. They do not account for all label errors, verification bias or dataset shift.
- Geometric lesion matching does not measure clinical relevance or reading behaviour.
- The available results do not establish clinical deployment readiness.

## References

- DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*.
- Sun X, Xu W (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*.
- Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*.
- Buda M et al. (2021). A data set and deep learning algorithm for the detection of masses and architectural distortions in digital breast tomosynthesis. *JAMA Network Open*.
