"""CBIS-DDSM loader (2D scanned-film mammography).

CBIS-DDSM (Curated Breast Imaging Subset of DDSM) provides biopsy-proven
pathology, BI-RADS assessment, breast density and lesion type — exactly the
fields a stratified validation needs. This loader targets the widely used
Kaggle JPEG mirror ``awsaf49/cbis-ddsm-breast-cancer-image-dataset`` (~6 GB),
so the pipeline runs in Colab without the ~160 GB raw DICOM collection.

The one real complication is path joining. The case-description CSVs label
abnormalities but their ``image file path`` column still holds *original DICOM*
paths, not the mirror's JPEG paths. The fix used here exploits a CBIS-DDSM
quirk: each DICOM's ``PatientID`` tag is the descriptive study string
(e.g. ``Mass-Training_P_00001_LEFT_CC``). That string is reconstructable from
the CSV fields and appears in ``dicom_info.csv`` alongside the JPEG path, so we
rebuild it and join on it.

Validation-relevant caveats (surfaced in the report, not hidden):

* CBIS-DDSM is **digitised film**, not full-field digital mammography — a
  genuine domain shift from a modern screening device's input.
* It is **lesion-enriched and non-consecutive**: prevalence here is an artefact
  of curation, so recall rate / cancer detection rate are illustrative only.
* The unit is the single-view image; ``patient_id`` still drives cluster CIs.
"""
from __future__ import annotations

import os

import pandas as pd

from .base import MammoDataset

_CASE_CSVS = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calc", "train"): "calc_case_description_train_set.csv",
    ("calc", "test"): "calc_case_description_test_set.csv",
}


def _norm_cols(df):
    """Lower-case, trim and underscore column names (CSV headers are messy:
    'breast density' vs 'breast_density', 'image file path', ...)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _resolve_jpeg(image_path, root):
    """Map a ``dicom_info.csv`` image path onto this install's ``jpeg/`` tree."""
    p = str(image_path).replace("\\", "/")
    rel = p[p.index("jpeg/"):] if "jpeg/" in p else "jpeg/" + p.lstrip("/")
    return os.path.join(root, *rel.split("/"))


class CBISDDSMDataset(MammoDataset):
    """CBIS-DDSM as an image-level case table.

    Parameters
    ----------
    root : str
        Folder of the extracted Kaggle dataset (contains ``csv/`` and
        ``jpeg/``).
    split : {'test', 'train'}
        CBIS-DDSM ships a fixed official split — use ``'test'`` for validation
        so results are comparable to published work.
    abnormality : {'both', 'mass', 'calc'}
        Restrict to one lesion type, or keep both.
    csv_dir, jpeg_dir : str, optional
        Override sub-folders if the mirror's layout differs.

    Notes
    -----
    A study/image can carry several abnormality rows; rows are aggregated to
    one image-level case: ``y_true`` = malignant if *any* abnormality is
    malignant, BI-RADS = the most suspicious assessment, subtlety = the most
    subtle, lesion type = ``'mixed'`` when both occur.
    """

    name = "CBIS-DDSM"
    modality = "2D"

    def __init__(self, root, split="test", abnormality="both",
                 csv_dir=None, jpeg_dir=None):
        self.root = root
        self.split = split
        self.csv_dir = csv_dir or os.path.join(root, "csv")
        self.jpeg_root = jpeg_dir or root
        self._jpeg_index = self._build_jpeg_index()
        self.cases = self._build_cases(split, abnormality)
        self.validate()

    # ------------------------------------------------------------- jpeg join
    def _build_jpeg_index(self):
        """study_id -> full-mammogram JPEG path, via ``dicom_info.csv``."""
        dicom_info = _norm_cols(pd.read_csv(os.path.join(self.csv_dir, "dicom_info.csv")))
        full = dicom_info[
            dicom_info["seriesdescription"].astype(str).str.contains(
                "full mammogram", case=False, na=False)
        ]
        index = {}
        for _, r in full.iterrows():
            study = str(r.get("patientid", "")).strip()
            if study and study not in index:
                index[study] = _resolve_jpeg(r["image_path"], self.jpeg_root)
        if not index:
            raise RuntimeError(
                "no 'full mammogram images' rows found in dicom_info.csv — "
                "check the Kaggle mirror layout")
        return index

    # ------------------------------------------------------------ case table
    def _build_cases(self, split, abnormality):
        wanted = (["mass", "calc"] if abnormality == "both" else [abnormality])
        frames = []
        for abn in wanted:
            path = os.path.join(self.csv_dir, _CASE_CSVS[(abn, split)])
            df = _norm_cols(pd.read_csv(path))
            df["abnormality_type"] = abn
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)

        # density column header varies across the calc/mass files
        density_col = next((c for c in ("breast_density", "breast density",
                                        "density") if c in raw.columns), None)

        records = []
        for _, r in raw.iterrows():
            phase = "Training" if split == "train" else "Test"
            abn_tag = "Mass" if r["abnormality_type"] == "mass" else "Calc"
            side = str(r["left_or_right_breast"]).strip().upper()
            view = str(r["image_view"]).strip().upper()
            patient = str(r["patient_id"]).strip()
            study_id = f"{abn_tag}-{phase}_{patient}_{side}_{view}"
            records.append({
                "study_id": study_id,
                "patient_id": patient,
                "view": view,
                "laterality": side,
                "abnormality_type": r["abnormality_type"],
                "pathology": str(r["pathology"]).strip().upper(),
                "is_malignant": int("MALIGNANT" in str(r["pathology"]).upper()),
                "breast_density": (pd.to_numeric(r.get(density_col), errors="coerce")
                                   if density_col else None),
                "birads_assessment": pd.to_numeric(r.get("assessment"), errors="coerce"),
                "subtlety": pd.to_numeric(r.get("subtlety"), errors="coerce"),
            })
        ab = pd.DataFrame.from_records(records)

        # aggregate abnormality rows -> one image-level case
        grouped = ab.groupby("study_id")
        cases = grouped.agg(
            patient_id=("patient_id", "first"),
            view=("view", "first"),
            laterality=("laterality", "first"),
            y_true=("is_malignant", "max"),
            breast_density=("breast_density", "first"),
            birads_assessment=("birads_assessment", "max"),
            subtlety=("subtlety", "min"),
            n_abnormalities=("is_malignant", "size"),
        ).reset_index().rename(columns={"study_id": "case_id"})

        cases["abnormality_type"] = grouped["abnormality_type"].apply(
            lambda s: "mixed" if s.nunique() > 1 else s.iloc[0]).values
        cases["breast_density_cat"] = cases["breast_density"].map(
            {1: "A (fatty)", 2: "B (scattered)", 3: "C (heterog. dense)",
             4: "D (extremely dense)"}).fillna("unknown")
        cases["image_path"] = cases["case_id"].map(self._jpeg_index)
        cases["modality"] = "2D"

        missing = int(cases["image_path"].isna().sum())
        if missing:
            print(f"[CBIS-DDSM] {missing}/{len(cases)} cases had no JPEG match "
                  f"and were dropped (path-join miss).")
        cases = cases.dropna(subset=["image_path"]).reset_index(drop=True)
        return cases

    # ------------------------------------------------------------------ load
    def load(self, row):
        """Return the mammogram for a case as a PIL image."""
        from PIL import Image
        return Image.open(row["image_path"])
