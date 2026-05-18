"""Unit tests for the Duke BCS-DBT loader and its FROC helpers.

Uses tiny synthetic CSV fixtures that mimic the real BCS-DBT metadata schema
(including the lower-case ``View`` values), so the tests need no download and
no pixel data.
"""
import io
import zipfile

import pandas as pd
import pytest

from mammoval.data import DukeDBTDataset
from mammoval.metrics import duke_dbt_hit, froc_summary


def _write_fixtures(root):
    """Write minimal BCS-DBT-style CSVs for the 'validation' split into root."""
    pd.DataFrame({
        "PatientID": ["DBT-P1", "DBT-P1", "DBT-P2", "DBT-P2"],
        "StudyUID": ["S1", "S1", "S2", "S2"],
        "View": ["lcc", "rmlo", "lcc", "rmlo"],          # lower-case, as in TCIA
        "descriptive_path": ["a/1.dcm", "a/2.dcm", "b/1.dcm", "b/2.dcm"],
        "classic_path": ["a/1.dcm", "a/2.dcm", "b/1.dcm", "b/2.dcm"],
    }).to_csv(root / "BCS-DBT-file-paths-validation-v2.csv", index=False)

    pd.DataFrame({
        "PatientID": ["DBT-P1", "DBT-P1", "DBT-P2", "DBT-P2"],
        "StudyUID": ["S1", "S1", "S2", "S2"],
        "View": ["lcc", "rmlo", "lcc", "rmlo"],
        "Normal": [1, 0, 0, 1], "Actionable": [0, 0, 0, 0],
        "Benign": [0, 0, 1, 0], "Cancer": [0, 1, 0, 0],
    }).to_csv(root / "BCS-DBT-labels-validation-PHASE-2.csv", index=False)

    pd.DataFrame({
        "PatientID": ["DBT-P1", "DBT-P2"],
        "StudyUID": ["S1", "S2"], "View": ["rmlo", "lcc"],
        "Subject": [0, 0], "Slice": [25, 30],
        "X": [100, 200], "Y": [150, 250],
        "Width": [80, 120], "Height": [90, 110],
        "Class": ["cancer", "benign"], "AD": [0, 0],
        "VolumeSlices": [64, 70],
    }).to_csv(root / "BCS-DBT-boxes-validation-v2.csv", index=False)


def test_loader_builds_case_table(tmp_path):
    _write_fixtures(tmp_path)
    ds = DukeDBTDataset(root=str(tmp_path), split="validation", require_local=False)
    assert len(ds.cases) == 4
    assert ds.cases["y_true"].sum() == 1                  # one Cancer view
    assert set(ds.cases["view"]) == {"LCC", "RMLO"}       # upper-cased for display
    rmlo = ds.cases[ds.cases["view"] == "RMLO"].iloc[0]
    assert rmlo["laterality"] == "Right"                  # derived correctly


def test_lesion_boxes_use_centre_coordinates(tmp_path):
    _write_fixtures(tmp_path)
    ds = DukeDBTDataset(root=str(tmp_path), split="validation", require_local=False)
    boxes = ds.lesion_boxes("S1::rmlo")
    assert len(boxes) == 1
    assert boxes[0]["x"] == 100 + 80 / 2                  # corner -> centre
    assert boxes[0]["y"] == 150 + 90 / 2
    assert boxes[0]["volume_slices"] == 64


def test_assemble_froc_cases_and_perfect_detector(tmp_path):
    _write_fixtures(tmp_path)
    ds = DukeDBTDataset(root=str(tmp_path), split="validation", require_local=False)
    # a perfect detector: one detection exactly on each lesion centre
    preds = pd.DataFrame([
        {"case_id": "S1::rmlo", "score": 0.9, "x": 140, "y": 195, "slice": 25},
        {"case_id": "S2::lcc", "score": 0.9, "x": 260, "y": 305, "slice": 30},
    ])
    cases = ds.assemble_froc_cases(preds)
    assert len(cases) == 4                                # negatives included
    summary = froc_summary(cases, duke_dbt_hit, fp_points=(1, 2))
    assert summary["n_lesions"] == 2
    assert summary["mean_sensitivity"] == pytest.approx(1.0)


def test_read_dbtex_predictions(tmp_path):
    _write_fixtures(tmp_path)
    ds = DukeDBTDataset(root=str(tmp_path), split="validation", require_local=False)
    # a synthetic team-predictions zip in the DBTex archive layout
    team = pd.DataFrame({
        "PatientID": ["DBT-P1"], "StudyUID": ["S1"], "View": ["rmlo"],
        "X": [120], "Width": [40], "Y": [170], "Height": [50],
        "Z": [25], "Depth": [0], "Score": [0.8],
    })
    zpath = tmp_path / "team_predictions_bothphases.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        buf = io.StringIO()
        team.to_csv(buf, index=False)
        zf.writestr("team_predictions_bothphases/phase2/validation/myteam.csv",
                    buf.getvalue())
    preds = ds.read_dbtex_predictions(str(zpath), team="myteam")
    assert list(preds.columns) == ["case_id", "score", "x", "y", "slice"]
    assert preds.iloc[0]["case_id"] == "S1::rmlo"
    assert preds.iloc[0]["x"] == 120 + 40 / 2             # corner -> centre


def test_read_dbtex_predictions_missing_team(tmp_path):
    _write_fixtures(tmp_path)
    ds = DukeDBTDataset(root=str(tmp_path), split="validation", require_local=False)
    zpath = tmp_path / "team_predictions_bothphases.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("team_predictions_bothphases/phase2/validation/other.csv", "x\n1\n")
    with pytest.raises(FileNotFoundError):
        ds.read_dbtex_predictions(str(zpath), team="nonexistent")
