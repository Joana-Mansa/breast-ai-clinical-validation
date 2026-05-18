"""Integration test for the fine-tuned CNN classifier.

Skipped where torch / torchvision are absent. Where present, it checks that the
training loop actually learns: on a trivially separable synthetic signal a
couple of fine-tuning epochs must beat chance by a wide margin.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from mammoval.models import FineTunedClassifier, score_dataset
from mammoval.metrics import auc_ci


class _FakeDataset:
    name = "fake"
    modality = "2D"

    def __init__(self, cases, images):
        self.cases = cases
        self._images = images

    def load(self, row):
        return self._images[row["case_id"]]


def _make_image(label, rng, size=120):
    """Malignant cases carry a bright blob; benign cases do not."""
    img = rng.integers(40, 95, (size, size, 3), dtype=np.uint8)
    if label == 1:
        yy, xx = np.ogrid[:size, :size]
        cx = size // 2 + int(rng.integers(-15, 15))
        cy = size // 2 + int(rng.integers(-15, 15))
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < (size // 5) ** 2] = 235
    return img


def _build(n, rng, prefix):
    labels = rng.integers(0, 2, n)
    ids = [f"{prefix}-{i:03d}" for i in range(n)]
    images = {cid: _make_image(lab, rng) for cid, lab in zip(ids, labels)}
    cases = pd.DataFrame({"case_id": ids,
                          "patient_id": [f"p{i}" for i in range(n)],
                          "y_true": labels})
    return _FakeDataset(cases, images)


def test_finetune_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    train = _build(80, rng, "tr")
    test = _build(48, rng, "te")

    model = FineTunedClassifier(backbone="resnet18", image_size=96)
    model.fit(train, epochs=2, batch_size=16, val_fraction=0.2, progress=False)

    assert model.fitted is True
    assert 0.0 <= model.val_auc <= 1.0
    assert model.n_train > 0

    preds = score_dataset(model, test, progress=False)
    assert preds["y_score"].between(0, 1).all()
    auc = auc_ci(preds["y_true"], preds["y_score"])["auc"]
    assert auc > 0.70  # fine-tuning must clearly learn the blob


def test_predict_before_fit_raises():
    model = FineTunedClassifier(backbone="resnet18", image_size=96)
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((96, 96, 3), dtype=np.uint8))
