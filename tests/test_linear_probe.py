"""Integration test for the linear-probe classifier.

Skipped automatically where torch / torchvision are not installed (the metrics
engine and the rest of the suite do not need them). Where they are available,
this checks the full path: frozen backbone -> feature extraction -> logistic
head -> black-box scoring of a held-out split.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from mammoval.models import LinearProbeClassifier, score_dataset
from mammoval.metrics import auc_ci


class _FakeDataset:
    """Minimal MammoDataset stand-in: a case table plus in-memory images."""

    name = "fake"
    modality = "2D"

    def __init__(self, cases, images):
        self.cases = cases
        self._images = images

    def load(self, row):
        return self._images[row["case_id"]]


def _make_image(label, rng, size=160):
    """A learnable signal: malignant cases carry a bright circular blob."""
    img = rng.integers(40, 95, (size, size, 3), dtype=np.uint8)
    if label == 1:
        yy, xx = np.ogrid[:size, :size]
        cx = size // 2 + int(rng.integers(-18, 18))
        cy = size // 2 + int(rng.integers(-18, 18))
        blob = (xx - cx) ** 2 + (yy - cy) ** 2 < (size // 5) ** 2
        img[blob] = 235
    return img


def _build_dataset(n, rng, prefix):
    labels = rng.integers(0, 2, n)
    case_ids = [f"{prefix}-{i:03d}" for i in range(n)]
    images = {cid: _make_image(lab, rng) for cid, lab in zip(case_ids, labels)}
    cases = pd.DataFrame({
        "case_id": case_ids,
        "patient_id": [f"p{i}" for i in range(n)],
        "y_true": labels,
    })
    return _FakeDataset(cases, images)


def test_linear_probe_fits_and_scores_a_holdout():
    rng = np.random.default_rng(0)
    train = _build_dataset(64, rng, "tr")
    test = _build_dataset(48, rng, "te")

    probe = LinearProbeClassifier(backbone="resnet18")
    probe.fit(train, progress=False)
    assert probe.n_train == 64
    assert probe.feature_dim == 512  # resnet18 penultimate width

    preds = score_dataset(probe, test, progress=False)
    assert preds["y_score"].between(0, 1).all()

    # the blob signal is trivially separable -> the probe should score well
    auc = auc_ci(preds["y_true"], preds["y_score"])["auc"]
    assert auc > 0.75


def test_predict_proba_single_matches_batch():
    rng = np.random.default_rng(1)
    train = _build_dataset(48, rng, "tr")
    probe = LinearProbeClassifier(backbone="resnet18").fit(train, progress=False)

    imgs = [_make_image(i % 2, rng) for i in range(6)]
    batch = probe.predict_proba_batch(imgs)
    single = [probe.predict_proba(im) for im in imgs]
    assert np.allclose(batch, single, atol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in batch)


def test_predict_before_fit_raises():
    probe = LinearProbeClassifier(backbone="resnet18")
    with pytest.raises(RuntimeError):
        probe.predict_proba(np.zeros((160, 160, 3), dtype=np.uint8))
