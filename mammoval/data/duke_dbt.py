"""Duke BCS-DBT loader (3D digital breast tomosynthesis).

The Duke *Breast-Cancer-Screening-DBT* collection (TCIA, DOI 10.7937/E4WT-CD02)
is a real screening-population tomosynthesis set: per view it provides a
four-way label — ``Normal`` / ``Actionable`` / ``Benign`` / ``Cancer`` — and,
for biopsied lesions, ground-truth bounding boxes with a centre slice. That
makes it usable for both exam-level classification *and* localisation (FROC).

The full collection is ~1.5 TB, so this loader is **subset-first**: point it at
whatever portion has been downloaded (the ~79 GB validation split, or a custom
manifest of a few hundred views) and it builds the case table from whatever
DICOMs are actually present.

Label/CSV filenames vary across TCIA snapshots (``-v2``, ``-PHASE-2`` suffixes),
so the three CSVs are located by glob rather than hard-coded names.

For binary cancer detection the positive class is ``Cancer``; ``Normal``,
``Actionable`` and ``Benign`` (biopsy-negative) form the non-cancer class.
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from .base import MammoDataset

_LABEL_CLASSES = ["Normal", "Actionable", "Benign", "Cancer"]


def _find_csv(root, *keywords):
    """First CSV under ``root`` whose name contains all ``keywords``."""
    for path in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        name = os.path.basename(path).lower()
        if all(k.lower() in name for k in keywords):
            return path
    raise FileNotFoundError(
        f"no CSV matching {keywords} under {root} — check the download")


class DukeDBTDataset(MammoDataset):
    """Duke BCS-DBT as a per-view case table.

    Parameters
    ----------
    root : str
        Folder holding the downloaded DICOM tree and the three metadata CSVs.
    split : {'validation', 'test', 'train'}
        Selects the CSV trio. ``'validation'`` (~79 GB) is the practical
        choice for a Colab validation run.
    require_local : bool
        When True (default) keep only views whose DICOM file is present on
        disk — the right behaviour when working from a partial download.
    """

    name = "Duke-BCS-DBT"
    modality = "3D"

    def __init__(self, root, split="validation", require_local=True):
        self.root = root
        self.split = split
        self.paths_csv = _find_csv(root, "file-paths", split)
        self.labels_csv = _find_csv(root, "labels", split)
        try:
            self.boxes_csv = _find_csv(root, "boxes", split)
        except FileNotFoundError:
            self.boxes_csv = None  # boxes are optional (classification still works)
        self.cases = self._build_cases(require_local)
        self._boxes = self._load_boxes()
        self.validate()

    # ------------------------------------------------------------ case table
    def _build_cases(self, require_local):
        paths = pd.read_csv(self.paths_csv)
        labels = pd.read_csv(self.labels_csv)
        df = paths.merge(labels, on=["StudyUID", "View"], how="inner",
                         suffixes=("", "_lbl"))

        present = [c for c in _LABEL_CLASSES if c in df.columns]
        df["label_class"] = df[present].astype(float).idxmax(axis=1)
        df["y_true"] = (df["label_class"] == "Cancer").astype(int)

        path_col = "descriptive_path" if "descriptive_path" in df.columns else "classic_path"
        df["volume_path"] = df[path_col].apply(lambda p: os.path.join(self.root, str(p)))
        df["case_id"] = df["StudyUID"].astype(str) + "::" + df["View"].astype(str)
        df["patient_id"] = df["PatientID"].astype(str)
        # `View` is lower-case in the CSVs (e.g. "lcc", "rmlo"); upper-case it
        # for display and derive laterality/projection from the upper form.
        df["view"] = df["View"].astype(str).str.upper()
        df["laterality"] = df["view"].str[0].map({"L": "Left", "R": "Right"})
        df["projection"] = df["view"].str[1:]
        df["modality"] = "3D"

        cases = df[["case_id", "patient_id", "StudyUID", "view", "laterality",
                    "projection", "label_class", "y_true", "volume_path"]].copy()

        if require_local:
            exists = cases["volume_path"].apply(os.path.exists)
            n_missing = int((~exists).sum())
            if n_missing:
                print(f"[Duke-BCS-DBT] {n_missing}/{len(cases)} views not found "
                      f"on disk (partial download) — excluded.")
            cases = cases[exists].reset_index(drop=True)
        return cases.reset_index(drop=True)

    # ----------------------------------------------------------------- boxes
    def _load_boxes(self):
        """case_id -> list of ground-truth lesion dicts (centre coordinates)."""
        if not self.boxes_csv:
            return {}
        boxes = pd.read_csv(self.boxes_csv)
        out = {}
        for _, r in boxes.iterrows():
            case_id = f"{r['StudyUID']}::{r['View']}"
            out.setdefault(case_id, []).append({
                "x": float(r["X"]) + float(r["Width"]) / 2.0,   # box centre
                "y": float(r["Y"]) + float(r["Height"]) / 2.0,
                "width": float(r["Width"]),
                "height": float(r["Height"]),
                "slice": float(r["Slice"]),
                "volume_slices": float(r.get("VolumeSlices", np.nan)),
            })
        return out

    def lesion_boxes(self, case_id):
        """Ground-truth lesion boxes for a case (empty list if none)."""
        return self._boxes.get(case_id, [])

    def assemble_froc_cases(self, predictions):
        """Build the case list :func:`mammoval.metrics.build_froc` consumes.

        Parameters
        ----------
        predictions : pandas.DataFrame
            One row per detection, columns ``case_id, score, x, y, slice``
            (``x``/``y`` = predicted box centre, in image pixels). This is the
            output a trained DBT detector would emit; the 3D notebook also
            shows how to synthesise it for an illustrative end-to-end run.

        Returns
        -------
        list of dict with ``case_id``, ``lesions`` and ``detections`` — covers
        every case in the dataset, so cancer-negative views correctly
        contribute their false marks to the FP-per-volume denominator.
        """
        by_case = {cid: [] for cid in self.cases["case_id"]}
        for _, r in predictions.iterrows():
            if r["case_id"] in by_case:
                by_case[r["case_id"]].append({
                    "score": float(r["score"]), "x": float(r["x"]),
                    "y": float(r["y"]), "slice": float(r["slice"]),
                })
        return [
            {"case_id": cid,
             "lesions": self.lesion_boxes(cid),
             "detections": by_case[cid]}
            for cid in self.cases["case_id"]
        ]

    def read_dbtex_predictions(self, source, team, phase="phase2"):
        """Load a DBTex-challenge team's detector predictions for this split.

        The Duke group published the DBTex1/DBTex2 challenge submissions in
        ``team_predictions_bothphases.zip`` (on the TCIA collection page). This
        reads one team's predictions for this dataset's split and converts the
        corner-coordinate boxes into the centre-coordinate table that
        :meth:`assemble_froc_cases` consumes — so the FROC engine can score a
        *real* competition detector, not a simulated one.

        Parameters
        ----------
        source : str
            Path to ``team_predictions_bothphases.zip`` (or an extracted dir).
        team : str
            Team name, e.g. ``'nyu_bteam'``, ``'vicorob'``, ``'zedus'``.
        phase : {'phase2', 'phase1'}
            Challenge phase; ``phase2`` matches the PHASE-2 ground-truth boxes.

        Returns
        -------
        pandas.DataFrame with columns ``case_id, score, x, y, slice``.
        """
        rel = f"{phase}/{self.split}/{team}.csv"
        if str(source).lower().endswith(".zip"):
            with zipfile.ZipFile(source) as zf:
                match = next(
                    (n for n in zf.namelist()
                     if n.endswith(rel) and not n.split("/")[-1].startswith("._")),
                    None)
                if match is None:
                    raise FileNotFoundError(f"{rel} not found inside {source}")
                raw = pd.read_csv(io.BytesIO(zf.read(match)))
        else:
            raw = pd.read_csv(os.path.join(source, phase, self.split, f"{team}.csv"))

        return pd.DataFrame({
            "case_id": raw["StudyUID"].astype(str) + "::" + raw["View"].astype(str),
            "score": raw["Score"].astype(float),
            "x": raw["X"].astype(float) + raw["Width"].astype(float) / 2.0,
            "y": raw["Y"].astype(float) + raw["Height"].astype(float) / 2.0,
            "slice": raw["Z"].astype(float),
        })

    # ------------------------------------------------------------------ load
    def load(self, row):
        """Return a DBT volume as a ``(n_slices, H, W)`` NumPy array.

        Uses a plain ``pydicom`` read. For localisation work that must align
        with the ground-truth boxes, prefer the official ``dcmread_image`` from
        ``mazurowski-lab/duke-dbt-data`` (it also corrects laterality flips).
        Compressed transfer syntaxes need ``pylibjpeg`` installed.
        """
        import pydicom
        ds = pydicom.dcmread(row["volume_path"])
        arr = ds.pixel_array
        if arr.ndim == 2:           # single frame -> add slice axis
            arr = arr[np.newaxis, ...]
        return np.asarray(arr)
