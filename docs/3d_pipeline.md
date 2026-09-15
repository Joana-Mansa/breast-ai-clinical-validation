# The 3D Validation Pipeline: Duke BCS-DBT (FROC)

Evaluate breast-cancer **lesion localisation** in digital breast tomosynthesis (DBT), using the Duke
*Breast-Cancer-Screening-DBT* dataset and the official FROC metric.

For interpretation see [`methodology.md`](methodology.md); for metric
definitions see [`metrics_reference.md`](metrics_reference.md); for dataset
specifics see [`datasets.md`](datasets.md).

---

## 1. Objective

FROC measures how often detector marks match annotated lesions at selected false-positive rates. This workflow uses the Duke BCS-DBT / DBTex matching rule.

The pipeline evaluates published DBTex detector predictions using metadata files. It does not run detector inference or download the image volumes.

---

## 2. Architecture & data flow

```
 Duke BCS-DBT metadata CSVs  +  DBTex team predictions   (a few MB, from TCIA)
      │
      ▼
 DukeDBTDataset                          mammoval/data/duke_dbt.py
   · reads file-paths / labels / boxes CSVs
   · builds the case table + ground-truth lesion boxes
      │
      ├─ read_dbtex_predictions(zip, team) ─►  detector predictions table
      │                                         (case_id, score, x, y, slice)
      ▼
 assemble_froc_cases(predictions)  ─►  per-volume {lesions, detections}
      │
      ▼
 run_localization_validation(cases, duke_dbt_hit)        mammoval/pipeline.py
   · matches marks to lesions under the official Duke criterion
   · FROC curve + sensitivity at 1/2/3/4 FP/volume + bootstrap CI
      │
      ▼
 build_report(localization_results=...)  ─►  standalone HTML FROC report
```

The workflow runs on CSV files and requires no model weights or GPU.

---

## 3. Dataset: Duke BCS-DBT

**What it is.** The Duke *Breast-Cancer-Screening-DBT* collection (TCIA, DOI
`10.7937/E4WT-CD02`): a real screening-population tomosynthesis dataset:
5,060 patients, 22,032 reconstructed DBT volumes. Each view (LCC/RCC/LMLO/RMLO)
is a multi-frame DICOM, a 3D stack of reconstructed slices.

**Required downloads.** This workflow uses the following metadata and prediction files:

| File (validation split) | Size | Content |
|---|---|---|
| `BCS-DBT-file-paths-validation-v2.csv` | ~380 KB | per-view DICOM paths |
| `BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv` | ~40 KB | per-view one-hot `Normal/Actionable/Benign/Cancer` |
| `BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv` | ~4.6 KB | ground-truth lesion boxes (biopsied cases) |
| `team_predictions_bothphases.zip` | ~7 MB | DBTex challenge team detector submissions |

All from `https://www.cancerimagingarchive.net/wp-content/uploads/`. FROC needs
the metadata and saved predictions, without image volumes.

**Labels.** One four-way label per view; for cancer-detection the positive
class is `Cancer`. The validation split: 1,163 views, 280 patients, **37
cancers**, **75 ground-truth lesion boxes** (biopsied benign + cancer).

**The DBTex challenge predictions.** `team_predictions_bothphases.zip` holds the
DBTex1/DBTex2 challenge submissions: `phase{1,2}/{validation,test}/<team>.csv`,
each with `StudyUID, View, X, Y, Width, Height, Z, Score`. The phase-2
validation set has three teams: `nyu_bteam`, `vicorob`, `zedus`. These are saved detections produced by the challenge teams.

---

## 4. The data loader: `DukeDBTDataset`

`mammoval/data/duke_dbt.py`.

- **CSV discovery by glob.** Filenames vary across TCIA snapshots (`-v2`,
  `-PHASE-2`, `-Jan-2024` suffixes), so the loader locates the three CSVs by
  keyword glob rather than hard-coded names.
- **Case table.** Merges file-paths + labels into one per-view case
  (`case_id`, `patient_id`, `view`, `laterality`, `label_class`, `y_true`).
  `View` is lower-case in the CSVs: the loader upper-cases it for display and
  derives laterality from it.
- **`require_local=False`.** For a FROC run no DICOMs are needed; the loader
  builds the table from CSVs alone. `require_local=True` (for the classification
  extension) keeps only views whose DICOM is on disk.
- **`lesion_boxes(case_id)`.** Ground-truth lesions as **centre-coordinate**
  boxes (`x, y, width, height, slice, volume_slices`): converted from the
  corner coordinates in the CSV.
- **`read_dbtex_predictions(zip, team)`.** Loads a DBTex team's submission for
  the split and converts corner boxes to the centre-coordinate predictions
  table the FROC engine consumes.
- **`assemble_froc_cases(predictions)`.** Joins ground-truth lesions and
  detector predictions into the per-volume structure `build_froc` consumes:
  including cancer-negative views, so they contribute their false marks to the
  FP-per-volume denominator.

---

## 5. FROC localisation validation

`mammoval/metrics/localization.py` + `mammoval/pipeline.py:run_localization_validation`.

**The official Duke true-positive criterion** (`duke_dbt_hit`): a predicted box
centre is a hit when

* its in-plane distance to a ground-truth box centre is at most
  `max(√(Width² + Height²)/2, 100)` pixels, **and**
* its centre slice is within `VolumeSlices/4` slices of the ground-truth slice.

Within each volume, detections are matched to lesions greedily in descending
score order. The FROC curve plots lesion-localisation sensitivity against mean
false marks per volume; **mean sensitivity at 1/2/3/4 FP/volume** is the
official DBTex ranking metric. A case-level bootstrap gives its CI.

This implementation is unit-tested against the criterion definition
(`tests/test_localization.py`, `tests/test_duke_dbt.py`).

---

## 6. The report: `build_report`

`mammoval/report.py`. Called as `build_report(localization_results=...)`, it
renders a standalone **localisation-only** HTML report: cohort, the FROC curve,
sensitivity at each operating point, the mean with its CI, the method, and
limitations. (The same function renders a full exam-level report when given
`classification_results`.)

A saved example is committed at
[`docs/duke_dbt_froc_report.html`](duke_dbt_froc_report.html).

---

## 7. Running the 3D pipeline

Notebook: [`notebooks/02_validation_3d_duke_dbt.ipynb`](../notebooks/02_validation_3d_duke_dbt.ipynb)
runs on Colab or Kaggle, **Internet on, no GPU needed**.

1. Setup: clones the package.
2. Downloads the validation CSVs + the DBTex predictions zip (a few MB).
3. `DukeDBTDataset(split='validation', require_local=False)`.
4. `read_dbtex_predictions` for a chosen team.
5. `run_localization_validation` → FROC + the official metric.
6. Compares all three challenge teams.
7. `build_report` → HTML FROC report.

No GPU is required. Runtime depends on downloads and bootstrap settings.

---

## 8. Recorded results

FROC on the Duke BCS-DBT **validation split** (1,163 views, 75 ground-truth
lesion boxes), scoring the real DBTex phase-2 challenge submissions:

| Team | Sens @1 FP | @2 FP | @4 FP | Mean sensitivity (95% CI) |
|---|---|---|---|---|
| `nyu_bteam` | 0.99 | 0.99 | 0.99 | **0.987** (0.955–1.000) |
| `zedus` | 0.96 | 0.96 | 0.96 | **0.960** (0.914–1.000) |
| `vicorob` | 0.81 | 0.87 | 0.91 | **0.867** (0.792–0.935) |

Mean sensitivity averages the results at 1, 2, 3 and 4 false positives per volume. The intervals use case-level bootstrap resampling. These are recorded project results; the automated tests do not rerun the complete challenge comparison.

## 9. Scope

The implemented notebook evaluates saved detector predictions. It does not train a detector or perform exam-level cancer classification on DICOM volumes.

`SliceAggregatorClassifier` in `mammoval/models/adapters_3d.py` provides a starting point for an exam-level extension. That route requires local images, a suitable model and separate validation. No verified exam-level 3D result is included here.

## 10. Limitations

- Localisation is scored against ground-truth boxes for **biopsied** lesions
  only; other findings are not in the lesion set.
- The true-positive criterion is **geometric**, not a radiologist's judgement
  of clinical relevance.
- Confidence intervals are case-level bootstrap: sampling error only, not
  distribution shift across vendor, site or era.
- The detectors scored are published **challenge submissions**, not live
  cleared devices.
- Standalone retrospective localisation: it does not establish how radiologists
  perform with AI assistance.

---

## 11. File map (3D-relevant)

```
mammoval/data/duke_dbt.py          DukeDBTDataset: loader, boxes, DBTex predictions
mammoval/metrics/localization.py   FROC engine + duke_dbt_hit criterion
mammoval/pipeline.py               run_localization_validation
mammoval/report.py                 build_report: localisation-only report
notebooks/02_validation_3d_duke_dbt.ipynb   the runnable 3D notebook
tests/test_localization.py         FROC engine tests
tests/test_duke_dbt.py             Duke loader + DBTex-prediction tests
docs/duke_dbt_froc_report.html     saved FROC report
```
