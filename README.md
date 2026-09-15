# 🩺 Breast Cancer AI: Clinical Validation Pipeline

Evaluate breast-cancer model predictions on **2D mammography** and **3D digital breast tomosynthesis (DBT)**. The Python package, `mammoval`, computes performance metrics and produces a self-contained HTML report.

The project includes a synthetic demonstration, a CBIS-DDSM classification notebook, and a Duke BCS-DBT lesion-localisation notebook. It is a retrospective research tool; its reports do not establish clinical readiness.

## 🚀 Quick start

```bash
git clone https://github.com/Joana-Mansa/breast-ai-clinical-validation.git
cd breast-ai-clinical-validation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/demo_synthetic.py
```

Use Python 3.9 or later. On Windows, activate the environment with `.venv\Scripts\activate`.

Open `outputs/demo_validation_report.html` in a browser. The demo uses generated predictions with deliberately different subgroup performance. It requires no dataset or model downloads after installation.

## 📂 Workflows

| Workflow | Inputs | Guide |
|---|---|---|
| Synthetic demonstration | Generated labels, scores and detections | [Example](examples/demo_synthetic.py) |
| 2D classification | CBIS-DDSM JPEG mirror and a fine-tuned CNN | [Notebook](notebooks/01_validation_2d_cbis_ddsm.ipynb) · [Guide](docs/2d_pipeline.md) |
| 3D lesion localisation | Duke metadata and published DBTex detector predictions | [Notebook](notebooks/02_validation_3d_duke_dbt.ipynb) · [Guide](docs/3d_pipeline.md) |
| Evaluate your predictions | CSV containing labels and model scores | [Command-line script](scripts/run_validation.py) |

The 2D notebook trains a model and is intended for a GPU runtime. The 3D notebook evaluates saved detections on CPU using metadata, without downloading the image volumes. Exam-level 3D classification is an extension and has no verified result in this repository.

For model inference, install `requirements-models.txt`. To import `mammoval` outside the checkout, run `pip install -e .`.

## 📊 Saved results

These are results from the committed real-data reports. The synthetic demo and automated checks do not rerun these experiments.

| Analysis | Recorded result | Report |
|---|---|---|
| CBIS-DDSM test split, fine-tuned ResNet-50 | ROC AUC **0.772**, 95% CI **0.726 to 0.818** | [HTML file](docs/cbis_ddsm_validation_report.html) · [Browser preview](https://htmlpreview.github.io/?https://github.com/Joana-Mansa/breast-ai-clinical-validation/blob/main/docs/cbis_ddsm_validation_report.html) |
| Duke validation split, DBTex `nyu_bteam` predictions | Mean FROC sensitivity **0.987**, 95% CI **0.955 to 1.000** | [HTML file](docs/duke_dbt_froc_report.html) · [Browser preview](https://htmlpreview.github.io/?https://github.com/Joana-Mansa/breast-ai-clinical-validation/blob/main/docs/duke_dbt_froc_report.html) |

Download an HTML report and open it locally if the preview service is unavailable. The DBTex detector predictions belong to the challenge teams; this project evaluates them.

## What the report contains

- ROC AUC, partial AUC and confidence intervals.
- Sensitivity, specificity and confusion matrices at selected operating points.
- Calibration, decision curves and simulated screening or triage metrics.
- Subgroup comparisons and an optional comparison with reference-reader scores.
- Lesion localisation through free-response ROC (FROC).
- Cohort information and limitations.

Sections depend on the supplied columns and whether classification or localisation results are provided. See the [metrics reference](docs/metrics_reference.md) for functions and uncertainty estimates.

## Evaluate a predictions CSV

The required columns are `y_true` (0 or 1) and `y_score`. Include `patient_id` for patient-level resampling. Reader and subgroup columns are optional.

```bash
python scripts/run_validation.py preds.csv \
    --reader birads_assessment \
    --subgroups breast_density_cat,abnormality_type \
    --report outputs/report.html
```

Omit `--reader` or `--subgroups` when those fields are absent. Use `--not-probability` for scores that should not be interpreted as probabilities. Run `python scripts/run_validation.py --help` for all options.

## 📚 Documentation

| Guide | Contents |
|---|---|
| [Methodology](docs/methodology.md) | Study scope, metric interpretation and limitations |
| [Datasets](docs/datasets.md) | Required files, labels and access |
| [2D pipeline](docs/2d_pipeline.md) | Loading images, model training and classification reports |
| [3D pipeline](docs/3d_pipeline.md) | DBTex predictions and FROC evaluation |
| [Metrics reference](docs/metrics_reference.md) | Metric definitions and Python functions |
| [Running the checks](docs/validation.md) | Tests, synthetic example and CI coverage |

## Project structure

```text
mammoval/       Dataset loaders, model adapters, metrics and report generation
notebooks/      2D classification and 3D localisation workflows
scripts/        Dataset download and CSV evaluation commands
examples/       Synthetic demonstration
tests/          Metric, loader, model and pipeline checks
docs/           Guides and saved HTML reports
```

## Verification and limitations

The full local suite passed **38 tests** with optional model dependencies installed. GitHub CI checks the core suite and synthetic demo. See [validation instructions](docs/validation.md) for the commands and scope.

CBIS-DDSM is a lesion-enriched, digitised-film dataset. Its screening-rate calculations should not be interpreted as population estimates. BI-RADS assessment is a proxy reader score, not an independent reader study. Confidence intervals describe sampling uncertainty and do not establish performance on new sites or populations. Neither workflow measures the effect of AI assistance on radiologists.

## License and attribution

Code: [MIT License](LICENSE). Dataset and model licences remain with their respective owners. See [dataset citations](docs/datasets.md) and [method references](docs/methodology.md#references).
