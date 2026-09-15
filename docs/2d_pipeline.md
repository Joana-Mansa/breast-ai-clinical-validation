# 2D mammography validation with CBIS-DDSM

This workflow loads mammograms, trains a classifier on the training split, scores the test split, and generates a classification report.

[Project overview](../README.md) · [Dataset guide](datasets.md) · [Metrics reference](metrics_reference.md)

## Data flow

```text
CBIS-DDSM CSVs and JPEG images
    -> CBISDDSMDataset
    -> FineTunedClassifier.fit(training split)
    -> score_dataset(test split)
    -> predictions table
    -> run_classification_validation
    -> build_report
```

The metrics consume a predictions table independently of the model. Model training uses the training split. Operating-point thresholds estimated from the test cohort describe that cohort; they are not independently validated deployment thresholds.

## Load the data

`CBISDDSMDataset` in `mammoval/data/cbis_ddsm.py` reads the Kaggle JPEG mirror. It reconstructs study identifiers from the case CSV fields and joins them to `dicom_info.csv` to resolve JPEG paths. It prints a count of unresolved cases.

Several abnormalities may refer to one image. The loader aggregates them to one case, setting `y_true` to 1 if any abnormality is malignant.

| Column | Meaning |
|---|---|
| `case_id` | Image-level study identifier |
| `patient_id` | Identifier for patient-level resampling |
| `y_true` | Binary malignancy label |
| `image_path` | Resolved JPEG path |
| `view`, `laterality` | CC/MLO and left/right breast |
| `abnormality_type` | Mass, calcification or mixed |
| `breast_density_cat` | Density category |
| `birads_assessment` | Ordinal reference score used for the reader comparison |

`load(row)` opens an image when needed. See [dataset files and labels](datasets.md#cbis-ddsm).

## Model adapters

| Adapter | Implementation |
|---|---|
| `FineTunedClassifier` | ImageNet backbone with a two-class head, trained on CBIS-DDSM |
| `LinearProbeClassifier` | Frozen ImageNet features with a fitted logistic-regression head |
| `HFImageClassifier` | Adapter for a compatible Hugging Face image-classification checkpoint |

The shared interface is `predict_proba(image)`, returning a malignancy score between 0 and 1. A compatible adapter does not by itself establish that the score is calibrated.

The fine-tuned route defaults to ResNet-50, 320-pixel inputs, Adam with learning rate `1e-4`, and class-weighted cross-entropy. An internal validation hold-out selects the best epoch. Training includes random horizontal flips; evaluation uses deterministic preprocessing. Review patient overlap when defining or extending splits.

`score_dataset` appends `y_score` to the case table. Image-loading failures produce missing scores that are reported and excluded from metric calculations.

## Run the notebook

Open [the 2D notebook](../notebooks/01_validation_2d_cbis_ddsm.ipynb).

- **Kaggle:** add `awsaf49/cbis-ddsm-breast-cancer-image-dataset` as an input, enable Internet and a GPU, then run the notebook.
- **Colab:** enable a GPU and configure the Kaggle credentials requested by the download step.

The notebook loads both splits, trains the model, scores the test images and writes `outputs/cbis_ddsm_validation_report.html`. Runtime depends on hardware and data access. Re-import the notebook to receive changes to notebook cells; updating the cloned package only updates imported Python code.

For existing predictions, use the [CSV command](../README.md#evaluate-a-predictions-csv). The [synthetic example](../examples/demo_synthetic.py) checks report generation without downloading CBIS-DDSM.

## Report configuration

```python
ClassificationConfig(
    reader_col="birads_assessment",
    subgroup_cols=("breast_density_cat", "abnormality_type", "view"),
    target_specificities=(0.90, 0.96),
    ni_margin=0.05,
    is_probability=True,
    dataset_name="CBIS-DDSM test split",
)
```

The report includes discrimination, operating points, calibration, simulated screening metrics, subgroup AUCs and an optional paired reader-score comparison. The example non-inferiority margin is a configuration value, not evidence that a clinical margin has been justified.

## Recorded results

The project records these results on the official test split, approximately 645 image-level cases:

| Model | ROC AUC | 95% DeLong CI |
|---|---:|---|
| Frozen ResNet-50 with linear probe | 0.590 | 0.534 to 0.647 |
| Fine-tuned ResNet-50, 320-pixel inputs | 0.772 | 0.726 to 0.818 |

The fine-tuned model scored higher in these recorded runs. They do not establish a general performance guarantee for other architectures or datasets. The committed [classification report](cbis_ddsm_validation_report.html) contains the fine-tuned result. Automated tests do not repeat this training experiment.

## Limitations

- CBIS-DDSM contains digitised-film images and an enriched lesion cohort. Absolute screening-rate calculations are illustrative.
- BI-RADS assessment is an ordinal proxy, including an incomplete-assessment category. It is not an independent prospective reader score.
- Multiple views from a patient are correlated. Patient-level resampling is available, but does not address dataset shift or label bias.
- The analysis measures standalone model performance. It does not measure radiologist performance with AI assistance.

See [methodology](methodology.md) for interpretation and [verification instructions](validation.md) for test coverage.
