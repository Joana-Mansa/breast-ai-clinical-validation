# The 2D Validation Pipeline — CBIS-DDSM

Complete documentation of the 2D mammography arm of the project: what it does,
how every component works, the engineering decisions behind it, how to run it,
and the problems solved along the way.

For the clinical reasoning behind the metrics see
[`methodology.md`](methodology.md); for the metric definitions see
[`metrics_reference.md`](metrics_reference.md); for dataset specifics see
[`datasets.md`](datasets.md).

---

## 1. Objective

Perform a **standalone (black-box) clinical validation** of a 2D mammography
breast-cancer AI model against a biopsy-proven reference standard, using a
public dataset, and produce a regulator-style validation report.

The emphasis is the *validation* — the model is treated as a sealed device and
only ever measured, never tuned on the evaluation data. The pipeline answers:
*given this model, does the evidence support its use, and where does it fail?*

---

## 2. Architecture & data flow

```
 CBIS-DDSM  (Kaggle JPEG mirror, ~6 GB)
      │
      ▼
 CBISDDSMDataset                     mammoval/data/cbis_ddsm.py
   · joins case CSVs → dicom_info.csv (JPEG paths)
   · aggregates abnormality rows → one image-level case
      │
      ├─ split='train' ─►  FineTunedClassifier.fit()      mammoval/models/finetune.py
      │                      fine-tunes an ImageNet CNN
      │                      end-to-end on the train split
      │
      └─ split='test'  ─►  score_dataset(model, ds_test)  mammoval/models/inference.py
                              │
                              ▼
                     predictions table  (case table + y_score column)
                              │
                              ▼
                  run_classification_validation()         mammoval/pipeline.py
                   · discrimination · operating points
                   · calibration   · screening behaviour
                   · subgroups     · AI-vs-reader
                              │
                              ▼
                        results dict  ──►  build_report()  ──►  HTML report
                                            mammoval/report.py
```

Every stage is decoupled: the metrics see only the predictions table, never the
model or the pixels — which is what makes the validation reproducible and the
model genuinely a black box.

---

## 3. Dataset — CBIS-DDSM

**What it is.** The Curated Breast Imaging Subset of DDSM: a re-curated subset
of the Digital Database for Screening Mammography, with biopsy-proven pathology,
lesion segmentations, BI-RADS assessment, breast density and lesion type.
~3,100 mammography studies, ~6,775 annotated abnormalities, 1,566 patients.

**Source.** The pipeline targets the Kaggle JPEG mirror
`awsaf49/cbis-ddsm-breast-cancer-image-dataset` (~6 GB) rather than the ~160 GB
raw DICOM collection on TCIA.

**On-disk layout.**

```
csv/
  mass_case_description_train_set.csv    calc_case_description_train_set.csv
  mass_case_description_test_set.csv     calc_case_description_test_set.csv
  dicom_info.csv         # maps each image to a SeriesDescription + JPEG path
jpeg/<study-hash>/*.jpg
```

**Labels.** `pathology` ∈ {`MALIGNANT`, `BENIGN`, `BENIGN_WITHOUT_CALLBACK`}.
The pipeline's binary label is `y_true = 1` when **any** abnormality on an image
is `MALIGNANT`, else 0.

**Split.** CBIS-DDSM ships a fixed official train/test split (~2,458 / ~645
image-level cases). The pipeline uses it unchanged so results are comparable to
published work — the **train** split fine-tunes the model, the **test** split
is validated.

**Caveats** (surfaced in every report): CBIS-DDSM is *digitised film*, not
full-field digital mammography — a real domain shift from a modern screening
device; and it is *lesion-enriched and non-consecutive*, so its prevalence,
cancer detection rate and recall rate are illustrative, not epidemiological.

---

## 4. The data loader — `CBISDDSMDataset`

`mammoval/data/cbis_ddsm.py`. Turns the dataset into the project's standard
**case table** (a pandas DataFrame).

**The path-join problem.** The case-description CSVs label abnormalities, but
their `image file path` column holds *original DICOM* paths, not the JPEG
mirror's paths. The loader resolves this with a CBIS-DDSM quirk: every DICOM's
`PatientID` tag equals the descriptive study string, e.g.
`Mass-Training_P_00001_LEFT_CC`. The loader **reconstructs that string** from
the CSV fields (`{Mass|Calc}-{Training|Test}_{patient_id}_{LEFT|RIGHT}_{CC|MLO}`)
and joins it to `dicom_info.csv`, which carries the JPEG path. Cases that fail
to resolve are dropped with a printed count, never silently.

**Abnormality → image aggregation.** A mammogram can carry several abnormality
rows; they are aggregated to one image-level case: `y_true` = malignant if any
abnormality is malignant, BI-RADS = the most suspicious assessment, subtlety =
the most subtle, lesion type = `mixed` when both mass and calcification occur.

**Case table columns.**

| Column | Meaning |
|---|---|
| `case_id` | unique image id (the study string) |
| `patient_id` | patient — the unit of statistical independence |
| `y_true` | 1 = biopsy-proven malignant |
| `image_path` | resolved JPEG path |
| `view` | CC / MLO |
| `laterality` | LEFT / RIGHT |
| `abnormality_type` | mass / calcification / mixed |
| `breast_density` / `breast_density_cat` | density 1–4 / labelled A–D |
| `birads_assessment` | BI-RADS 0–5 — used as the radiologist-reference signal |
| `subtlety` | 1–5 |
| `n_abnormalities` | abnormality count on the image |

`load(row)` lazily returns the mammogram as a PIL image.

---

## 5. The model

### 5.1 The black-box contract

Any model enters the pipeline through one interface,
`ImageClassifier` (`mammoval/models/base.py`): a single method
`predict_proba(image) -> P(malignant) ∈ [0, 1]`. The metrics never see anything
else. Three concrete adapters were built; see §11 for why the model choice
evolved.

### 5.2 `FineTunedClassifier` — the pipeline's 2D model

`mammoval/models/finetune.py`. An ImageNet-pretrained CNN fine-tuned end-to-end
on the CBIS-DDSM training split.

| Aspect | Choice |
|---|---|
| Backbone | torchvision ImageNet model — `resnet50` (default), `resnet18`, `densenet121`, `efficientnet_b0` |
| Head | the 1000-class ImageNet head replaced with a fresh 2-class linear layer |
| Input | square, `image_size` (default 320 — larger preserves small-lesion detail) |
| Training | end-to-end (backbone weights updated), Adam, lr 1e-4 |
| Loss | class-weighted cross-entropy (robust to malignant/benign imbalance) |
| Augmentation | random horizontal flip (safe — laterality is not a cancer cue) |
| Model selection | internal 15 % validation hold-out; best-epoch weights kept |
| Efficiency | training images cached in memory after the first read, so epochs do not re-decode JPEGs |

Runtime: ~10–15 min on a Kaggle GPU for 6 epochs. The backbone is fine-tuned
only on **training** data; the test split is then scored as a black box, so the
model/validation separation is preserved.

### 5.3 `LinearProbeClassifier` — the no-training baseline

`mammoval/models/linear_probe.py`. A frozen ImageNet backbone used purely as a
feature extractor, with a logistic-regression head fit on the training split.
Fast and fully reliable, but limited — ImageNet features transfer poorly to
grayscale mammography (it scored AUC ≈ 0.59 on CBIS-DDSM, see §10). Retained as
a documented baseline and a reliability fallback.

### 5.4 `HFImageClassifier` — HuggingFace adapter

`mammoval/models/adapters_2d.py`. Wraps any HuggingFace
`AutoModelForImageClassification` checkpoint. Retained for the case where a
trustworthy fine-tuned checkpoint is available, and hardened against the common
community-upload defects (see §11). It is **not** the default — public
ready-made mammography checkpoints proved unreliable.

---

## 6. Inference — `score_dataset`

`mammoval/models/inference.py`. Runs a fitted model over a dataset and returns
the case table with an added `y_score` column — the **predictions table**.
This is the single bridge between the model and the metrics; everything
downstream consumes only this table. Cases whose image fails to load receive
NaN and are reported.

---

## 7. The validation pipeline — `run_classification_validation`

`mammoval/pipeline.py`. One call turns the predictions table into a structured
result. Configured via `ClassificationConfig`. Sections produced:

| Section | What it computes |
|---|---|
| **Cohort** | n cases, patients, prevalence, dropped rows |
| **Discrimination** | ROC AUC with a **DeLong** CI; standardised partial AUC (specificity 0.80–1.00); a **patient-level cluster-bootstrap** CI (correlated multi-view exams); average precision |
| **Operating points** | sensitivity at fixed specificities (0.90, 0.96) with bootstrap CIs and full confusion matrices; the Youden point |
| **Calibration** | reliability curve, Brier score, expected calibration error, calibration slope/intercept, decision-curve net benefit |
| **Screening** | cancer detection rate, recall rate, PPV1 at the primary operating point; an **AI triage / rule-out simulation**; a risk-band table |
| **Subgroups** | per-stratum AUC with DeLong CIs plus a **Cochran's Q** effect-modifier test, for breast density, lesion type and view |
| **AI vs reader** | a paired **DeLong test** and an **AUC non-inferiority test** of the model against the BI-RADS `assessment` as a radiologist-reference signal |

For the 2D run the config is:

```python
ClassificationConfig(
    reader_col='birads_assessment',
    subgroup_cols=('breast_density_cat', 'abnormality_type', 'view'),
    target_specificities=(0.90, 0.96),
    ni_margin=0.05,
    is_probability=True,
    dataset_name='CBIS-DDSM test split',
)
```

---

## 8. The report — `build_report`

`mammoval/report.py`. Renders the results into a single self-contained HTML
file with embedded plots (ROC, calibration, decision curve, triage trade-off,
risk bands, subgroup forest plot). Ten sections, ending with a Limitations
section that is included in *every* report by design. Output:
`outputs/cbis_ddsm_validation_report.html`.

---

## 9. Running the 2D pipeline

### On Kaggle (recommended — CBIS-DDSM is a native Kaggle dataset)

1. **Import** `notebooks/01_validation_2d_cbis_ddsm.ipynb`
   (File → Import Notebook → Upload).
2. **+ Add Input** → *CBIS-DDSM Breast Cancer Image Dataset* (awsaf49).
3. Notebook options → **GPU** on, **Internet** on.
4. **Run All.** The notebook clones the package, loads both splits, fine-tunes
   the model, scores the test split, runs the validation and renders the
   report inline (~15–20 min total).

### On Google Colab

The same notebook detects Colab and downloads the Kaggle mirror via the Kaggle
API (`kaggle.json`).

### Headless / locally

```bash
python scripts/download_cbis_ddsm.py --out data/cbis-ddsm
# … produce a predictions CSV …
python scripts/run_validation.py preds.csv \
    --reader birads_assessment \
    --subgroups breast_density_cat,abnormality_type \
    --report outputs/report.html
```

The synthetic demo (`python examples/demo_synthetic.py`) runs the whole
pipeline with no downloads — the runnability guarantee.

---

## 10. Results

Measured on the **official CBIS-DDSM test split** (~645 image-level cases).
AUC is the empirical ROC AUC with a DeLong 95% confidence interval.

| Model | Test-split AUC (95% CI) | Notes |
|---|---|---|
| `LinearProbeClassifier` (resnet50, frozen backbone) | 0.590 (0.534–0.647) | frozen ImageNet features — barely above chance |
| `FineTunedClassifier` (resnet50, fine-tuned, 320 px) | **0.772 (0.726–0.818)** | end-to-end fine-tune on the training split |

Fine-tuning lifts the AUC by ~0.18 over the frozen-backbone linear probe — and
that gap is the whole transfer-learning story here. A generic ImageNet
representation does not encode mass margins or calcification morphology, so a
linear head on top of it barely beats chance (0.590). Once the convolutional
weights are updated on mammograms, the network learns the relevant features and
reaches 0.772.

0.772 is a credible, honest baseline for *whole-image* CBIS-DDSM malignancy
classification. The task is genuinely hard — the lesion is a small fraction of
a digitised-film mammogram, and CBIS-DDSM is a domain shift from modern digital
mammography. It is not a state-of-the-art device (the report says so); a
patch/ROI-based model or higher input resolution would score higher.

What matters for this project: the validation pipeline now characterises a
model that genuinely discriminates. The ROC, operating points, calibration,
screening behaviour, breast-density subgroup analysis and BI-RADS comparison
are all measured on a working model, with confidence intervals throughout. That
rigorous measurement — not the model — is the deliverable.

The actual generated report from this run is committed at
[`cbis_ddsm_validation_report.html`](cbis_ddsm_validation_report.html) (a
[rendered view](https://htmlpreview.github.io/?https://github.com/Joana-Mansa/breast-ai-clinical-validation/blob/main/docs/cbis_ddsm_validation_report.html)).

---

## 11. Engineering problems solved

Real public data and public models are messy. Each issue below was diagnosed
and fixed defensively — this log is part of the project's value.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: mammoval.data` after cloning | `.gitignore` rule `data/` (unanchored) also matched the `mammoval/data/` source package, so it was never committed | anchored the rule to `/data/`; recovered and committed the package |
| 2 | `AssertionError` on the Kaggle dataset path | the dataset mount folder name depends on the dataset slug, which was hard-coded | auto-detect the mount: scan `/kaggle/input/*` for a folder holding `csv/` + `jpeg/` |
| 3 | `OSError: Can't load image processor` | the HuggingFace checkpoint shipped no `preprocessor_config.json` | `HFImageClassifier` loads preprocessing in 3 tiers — checkpoint, base backbone, then a manual transform |
| 4 | `ValueError: …does not recognize this architecture` | the checkpoint's `config.json` had `model_type` set to the base-model *name* instead of `swinv2` | `_load_model` recovers the architecture from the `architectures` field / the `model_type` string and reloads with a corrected config |
| 5 | `RuntimeError: mismatched_keys` | the same checkpoint's `config.json` did not match its own weight shapes — an irredeemably broken upload | abandoned that checkpoint; moved to torchvision backbones, which always load cleanly |
| 6 | AUC only 0.590 | a frozen-backbone linear probe has a low ceiling on mammography | added `FineTunedClassifier` — end-to-end fine-tuning of the backbone |
| — | notebook kept running old cells after fixes | refreshing the git clone updates the *package*, not the imported *notebook cells* | documented that notebook changes require a re-import |

**Lesson carried into the code:** every adapter and loader now fails loudly
with a diagnostic, degrades gracefully, or repairs the input — because a
validation tool that hides a problem is worse than useless.

---

## 12. Limitations

- **Retrospective standalone.** It measures the model in isolation; it does not
  run a prospective trial or a multi-reader multi-case (MRMC) reader study, and
  does not establish radiologist performance *with* the AI.
- **Dataset.** CBIS-DDSM is digitised film, lesion-enriched and
  non-consecutive — not a screening population; absolute screening rates are
  illustrative.
- **Reference reader.** BI-RADS `assessment` is an ordinal proxy, not an
  independent prospective radiologist read (category 0 is not strictly
  ordered).
- **Model.** The fine-tuned CNN is a reasonable baseline, not a cleared device;
  the report says so.
- **Confidence intervals** cover sampling error only — not distribution shift
  across vendor, site, population or acquisition era. External validation is
  the key extension.

---

## 13. File map (2D-relevant)

```
mammoval/data/cbis_ddsm.py        CBISDDSMDataset — loader + path-join
mammoval/data/base.py             MammoDataset interface, case-table contract
mammoval/models/base.py           ImageClassifier black-box interface
mammoval/models/finetune.py       FineTunedClassifier — the 2D model
mammoval/models/linear_probe.py   LinearProbeClassifier — no-training baseline
mammoval/models/adapters_2d.py    HFImageClassifier — HuggingFace adapter
mammoval/models/inference.py      score_dataset — model → predictions table
mammoval/pipeline.py              run_classification_validation, ClassificationConfig
mammoval/metrics/                 the validation engine (see metrics_reference.md)
mammoval/report.py                build_report — HTML report
mammoval/plotting.py              ROC / calibration / subgroup / … plots
notebooks/01_validation_2d_cbis_ddsm.ipynb   the runnable 2D notebook
scripts/download_cbis_ddsm.py     Kaggle download helper
scripts/run_validation.py         headless validation CLI
tests/test_linear_probe.py        linear-probe end-to-end test
tests/test_finetune.py            fine-tuning end-to-end test
```
