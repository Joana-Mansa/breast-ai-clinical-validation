# Breast Cancer AI — Clinical Validation Pipeline

A reproducible pipeline for the **clinical validation** of breast-cancer
detection AI on **2D mammography** and **3D digital breast tomosynthesis
(DBT)**, built with publicly available datasets and pretrained models.

It is designed around the workflow a clinical validation function owns at a
commercial breast-AI company — the kind of standalone, regulator-grade
evidence that supports a CADe / CADt / decision-support mammography device
(e.g. products such as Transpara). The emphasis is deliberately **not** on
building a better model: it is on *measuring* a model rigorously, treating it
as a sealed black box, and reporting where it works and where it fails.

> **The validation question:** *given this device exactly as it ships, does the
> evidence support its intended use — and where does it fail?*

---

## What it does

- Treats any breast-cancer model as a **black box** behind a one-method adapter
  — HuggingFace checkpoint, ONNX export, vendor API or synthetic stand-in.
- Runs the full standalone validation battery: **discrimination, operating
  points, localisation (FROC), calibration, screening behaviour, subgroup
  analysis, and an AI-vs-radiologist comparison**.
- Reports every estimate with a **confidence interval** — DeLong for AUC,
  Wilson for proportions, bootstrap (incl. patient-level cluster) elsewhere.
- Emits a single self-contained **HTML validation report** with embedded plots.
- Is **unit-tested** (28 tests pinning metrics to known-by-construction values)
  and runs **end-to-end with zero downloads** via a synthetic demo.

---

## What is real vs. what needs your input

Honesty is part of a validation tool. Here is exactly what runs out of the box:

| Component | Status |
|---|---|
| Validation metrics engine | ✅ Real, unit-tested, verified against `sklearn` and known values |
| Pipeline + HTML report | ✅ Real, runs end-to-end |
| Synthetic demo (no downloads) | ✅ Runs in ~30 s — the runnability guarantee |
| 2D path: CBIS-DDSM loader + HuggingFace model | ✅ Real, runnable in Colab (needs a free Kaggle account) |
| 3D path: Duke BCS-DBT loader + slice-aggregated classifier | ✅ Real, runnable on a downloaded subset (TCIA) |
| 3D localisation (FROC) | ✅ Engine real & tested to the **official Duke metric**; needs a detector's `predictions.csv` — no DBT detector has public weights, so the notebook supplies a clearly-labelled simulated one |
| Default models | ⚠️ Verified-runnable **baselines**, not state-of-the-art — the pipeline *quantifies* their shortfall rather than hiding it |

See [`docs/methodology.md`](docs/methodology.md) for the reasoning behind every
design choice.

---

## Quickstart — synthetic demo (no downloads)

```bash
pip install -r requirements.txt
python examples/demo_synthetic.py
```

This fabricates a realistic predictions table — with a *deliberately built-in
flaw* (AUC degrades in dense breasts) — runs the entire pipeline, and writes
`outputs/demo_validation_report.html`. If this runs, the validation engine,
orchestration and report are all working.

---

## The two real pipelines (Google Colab or Kaggle)

Both notebooks run on **Colab or Kaggle** — they detect the platform, get the
data, run a pretrained model, and produce a validation report. On Kaggle,
enable a GPU accelerator and turn Internet on; the 2D pipeline is easiest there
since CBIS-DDSM is a native Kaggle dataset (just *+ Add Input*).

| Notebook | Modality | Dataset | Model |
|---|---|---|---|
| [`notebooks/01_validation_2d_cbis_ddsm.ipynb`](notebooks/01_validation_2d_cbis_ddsm.ipynb) | 2D mammography | CBIS-DDSM (Kaggle JPEG mirror, ~6 GB) | HuggingFace Swin-V2 breast classifier |
| [`notebooks/02_validation_3d_duke_dbt.ipynb`](notebooks/02_validation_3d_duke_dbt.ipynb) | 3D tomosynthesis | Duke BCS-DBT (TCIA subset) | 2D model aggregated over slices |

Dataset access and caveats: [`docs/datasets.md`](docs/datasets.md). Full 2D
pipeline walkthrough — components, model, results, engineering log:
[`docs/2d_pipeline.md`](docs/2d_pipeline.md).

You can also run the pipeline headless on any predictions CSV:

```bash
python scripts/run_validation.py preds.csv \
    --reader birads_assessment \
    --subgroups breast_density_cat,abnormality_type \
    --report outputs/report.html
```

---

## The validation report

A **real example**, generated on the CBIS-DDSM test split (fine-tuned ResNet-50,
AUC 0.772), is committed at
[`docs/cbis_ddsm_validation_report.html`](docs/cbis_ddsm_validation_report.html)
— [**rendered view**](https://htmlpreview.github.io/?https://github.com/Joana-Mansa/breast-ai-clinical-validation/blob/main/docs/cbis_ddsm_validation_report.html).

A single HTML file with ten sections:

1. **Executive summary** — headline AUC, operating point, reader verdict
2. **Validation cohort** — prevalence, patient count, subgroups
3. **Discrimination** — ROC, AUC + DeLong CI, partial AUC, patient-cluster CI
4. **Operating points** — sensitivity at fixed specificities, confusion matrices
5. **Calibration & clinical utility** — reliability, Brier, slope, decision curve
6. **Screening behaviour & AI triage** — CDR, recall rate, rule-out trade-off
7. **Subgroup analysis** — per-stratum AUC + Cochran's Q effect-modifier test
8. **AI vs reference reader** — paired DeLong test + non-inferiority test
9. **Lesion localisation (FROC)** — official Duke operating points
10. **Limitations** — surfaced in every report, by design

---

## Project structure

```
mammoval/
  metrics/        validation engine — discrimination, FROC, screening,
                  calibration, bootstrap, subgroups   (the core deliverable)
  data/           dataset loaders — CBIS-DDSM (2D), Duke BCS-DBT (3D)
  models/         black-box model adapters + synthetic predictions
  pipeline.py     orchestration: predictions table -> structured result
  report.py       structured result -> standalone HTML report
  plotting.py     ROC / FROC / calibration / forest / decision-curve plots
notebooks/        the two Colab validation notebooks
scripts/          dataset download + headless CLI + notebook generator
tests/            28 unit + integration tests
docs/             methodology, metrics reference, dataset notes
examples/         synthetic end-to-end demo
```

---

## Metrics at a glance

Discrimination (ROC AUC, partial AUC, non-inferiority) · Operating points
(sensitivity at fixed specificity) · Localisation (FROC, official Duke metric)
· Calibration (reliability, Brier, slope/intercept, decision curve) · Screening
(cancer detection rate, recall rate, PPV1, triage/rule-out simulation, risk
bands) · Uncertainty (DeLong, Wilson, stratified & cluster bootstrap) ·
Subgroups (per-stratum AUC, Cochran's Q effect-modifier test).

Full reference: [`docs/metrics_reference.md`](docs/metrics_reference.md).

---

## Installation

```bash
git clone <your-fork-url> breast-ai-clinical-validation
cd breast-ai-clinical-validation
pip install -r requirements.txt          # validation engine + demo
pip install -r requirements-models.txt   # only for real model inference
pip install -e .                         # optional: import mammoval anywhere
```

Python ≥ 3.9. `torch` / `torchvision` are pre-installed in Colab.

## Testing

```bash
python -m pytest -q          # 28 tests, ~30 s
```

The tests pin each metric to a value known by construction — DeLong AUC against
`sklearn`, FROC against hand-checked detection sets, calibration slope to ~1.0
for calibrated scores, and so on.

---

## Limitations

This is a **retrospective standalone** validation pipeline. It does not run a
prospective trial or a multi-reader multi-case (MRMC) reader study, and it does
not establish how radiologists perform *with* the AI. Public datasets diverge
from a live screening population; the default models are runnable baselines,
not cleared devices. Confidence intervals cover sampling error only, not
distribution shift. It is an educational / methodological project — not a
regulatory submission and not a medical device. Full discussion in
[`docs/methodology.md`](docs/methodology.md) §11.

---

## Datasets & acknowledgements

- **CBIS-DDSM** — Lee RS et al., *Scientific Data* 2017; via TCIA / the Kaggle
  JPEG mirror.
- **Duke BCS-DBT** — Buda M et al., *JAMA Network Open* 2021; via TCIA
  (DOI 10.7937/E4WT-CD02).
- Pretrained models are loaded from their original authors on the HuggingFace
  Hub; licences are the authors'.

## License

MIT — see [`LICENSE`](LICENSE). Dataset and model licences are held by their
respective owners and are **not** redistributed here.
