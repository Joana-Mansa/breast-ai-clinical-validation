# Datasets

The pipeline validates against two public, ethically released datasets — one
2D, one 3D. Both are normalised by their loader into the single case table
described in `mammoval/data/base.py`.

---

## CBIS-DDSM — 2D scanned-film mammography

**What it is.** The *Curated Breast Imaging Subset of DDSM* — a cleaned,
re-curated subset of the Digital Database for Screening Mammography, with
verified lesion segmentations and biopsy-proven pathology. ~3,100 mammography
studies, ~6,775 annotated abnormalities, 1,566 patients.

**Why it suits clinical validation.** It carries the fields a stratified
validation needs: biopsy-confirmed pathology, BI-RADS *assessment*, breast
density (1–4), lesion type (mass vs calcification), subtlety (1–5), and a
fixed, published train/test split.

**Access.** The pipeline targets the Kaggle JPEG mirror
`awsaf49/cbis-ddsm-breast-cancer-image-dataset` (~6 GB) instead of the ~160 GB
raw DICOM collection on TCIA. Download with `scripts/download_cbis_ddsm.py` or
the 2D notebook.

**Layout (Kaggle mirror).**

```
csv/
  mass_case_description_train_set.csv   calc_case_description_train_set.csv
  mass_case_description_test_set.csv    calc_case_description_test_set.csv
  dicom_info.csv      # maps each image to a SeriesDescription + JPEG path
jpeg/<study-hash>/*.jpg
```

**Key columns** (case-description CSVs): `patient_id`, `breast_density`
(1–4; header spacing varies), `left or right breast`, `image view` (CC/MLO),
`abnormality type`, `assessment` (BI-RADS 0–5), `pathology`
(`MALIGNANT` / `BENIGN` / `BENIGN_WITHOUT_CALLBACK`), `subtlety` (1–5).

**The path-join gotcha.** The case CSVs' `image file path` still holds *original
DICOM* paths, not the mirror's JPEG paths. `CBISDDSMDataset` solves this with a
CBIS-DDSM quirk: each DICOM's `PatientID` tag equals the descriptive study
string (e.g. `Mass-Training_P_00001_LEFT_CC`). The loader reconstructs that
string from the CSV fields and joins it to `dicom_info.csv`, which carries the
JPEG path. Unresolved cases are dropped with a printed count rather than
failing silently.

**Validation caveats.**

- **Digitised film, not FFDM.** A genuine domain shift from a modern digital
  screening device — relevant when interpreting absolute numbers.
- **Lesion-enriched, non-consecutive.** Prevalence is a curation artefact;
  cancer detection rate and recall rate are illustrative, not epidemiological.
- `BENIGN_WITHOUT_CALLBACK` means a benign finding not recalled at screening —
  it is grouped as non-malignant (`y_true = 0`).

**Citation.** Lee RS et al. (2017). A curated mammography data set for use in
computer-aided detection and diagnosis research. *Scientific Data* 4:170177.

---

## Duke BCS-DBT — 3D digital breast tomosynthesis

**What it is.** The Duke *Breast-Cancer-Screening-DBT* collection — a real
screening-population tomosynthesis dataset: 5,060 patients, 5,610 studies,
22,032 reconstructed volumes. Each view (LCC/RCC/LMLO/RMLO) is a multi-frame
DICOM — a 3D stack of reconstructed slices.

**Labels.** One four-way label per view:

| Class | Meaning |
|---|---|
| `Normal` | BI-RADS 1, no findings |
| `Actionable` | Additional imaging requested, no biopsy |
| `Benign` | Biopsied lesion, pathology negative |
| `Cancer` | Biopsied lesion, pathology positive |

For binary cancer detection the positive class is `Cancer`; `Normal`,
`Actionable` and `Benign` form the non-cancer class. For biopsied lesions
(`Benign` / `Cancer`) ground-truth **bounding boxes with a centre slice** are
provided — this is what enables FROC.

**Access.** TCIA collection `Breast-Cancer-Screening-DBT`, DOI
`10.7937/E4WT-CD02`. Download via the NBIA Data Retriever (manifest files) or
the `tcia_utils` Python API. **The full collection is ~1.5 TB** — the pipeline
is subset-first: pull the ~79 GB validation split, or a few hundred views via a
custom selection (see the 3D notebook).

**Metadata CSVs** (filenames vary across TCIA snapshots — `-v2`, `-PHASE-2`
suffixes — so `DukeDBTDataset` locates them by glob):

- `BCS-DBT-file-paths-{split}*.csv` — `PatientID, StudyUID, View,
  descriptive_path, classic_path`
- `BCS-DBT-labels-{split}*.csv` — `StudyUID, View` + one-hot
  `Normal, Actionable, Benign, Cancer`
- `BCS-DBT-boxes-{split}*.csv` — `PatientID, StudyUID, View, X, Y, Width,
  Height, Slice, Class, AD, VolumeSlices`

**Splits.** By views/scans: train 19,148 / validation 1,163 / test 1,721.

**Official FROC metric.** A predicted box centre is a true positive when its
in-plane distance to a ground-truth box centre is at most
`max(√(Width²+Height²)/2, 100)` pixels **and** its centre slice is within
`VolumeSlices/4` of the ground-truth centre slice. Reported operating points:
sensitivity at 1, 2, 3, 4 false positives per volume; their mean is the
challenge ranking metric. Implemented exactly in
`mammoval/metrics/localization.py: duke_dbt_hit`.

**No pretrained detector.** The Duke BCS-DBT / DBTex baselines
(`mateuszbuda/duke-dbt-detection`, `mazurowski-lab/DBTex-baseline`) publish
training code but **no downloadable weights**. Consequently:

- *Classification* is run end-to-end with a `SliceAggregatorClassifier` (a
  validated 2D model applied over the slice stack — how DBT triage products
  operate in practice).
- *Localisation (FROC)* is fully implemented to the official spec and consumes
  a detector's `predictions.csv`. Until a real detector is trained, the 3D
  notebook supplies a clearly-labelled simulated prediction file so the FROC
  pipeline still runs end-to-end.

**DICOM note.** `DukeDBTDataset.load` uses a plain `pydicom` read (compressed
transfer syntaxes need `pylibjpeg`). For localisation work that must align with
ground-truth boxes, use the official `dcmread_image` from
`mazurowski-lab/duke-dbt-data`, which also corrects laterality flips.

**Citation.** Buda M et al. (2021). A data set and deep-learning algorithm for
the detection of masses and architectural distortions in digital breast
tomosynthesis. *JAMA Network Open* 4(8):e2119100.

---

## Reference standard

Both datasets anchor positives to **biopsy/pathology**. This is the appropriate
reference standard for a cancer-detection device, but two caveats carry into
every report:

- **Verification bias** — cases sent to biopsy are not a random sample of the
  screening population; metrics computed on a biopsy-enriched cohort do not
  transfer unchanged to consecutive screening.
- **Interval cancers** — cancers surfacing *between* screening rounds are the
  hardest test of a device and are largely absent from these datasets; true
  programme sensitivity must account for them.

Neither dataset is a substitute for a prospective, consecutively-recruited
screening cohort — see `methodology.md` §11.
