"""Fine-tuned CNN classifier — an ImageNet backbone trained on the target task.

A frozen-backbone linear probe (see :mod:`mammoval.models.linear_probe`) is
reliable but weak on mammography: ImageNet features transfer poorly to
grayscale breast tissue, so the cancer signal is largely not linearly
separable in them. Fine-tuning lifts that ceiling — the backbone's
convolutional weights are *updated* on the target training split, so the
network learns mammography-specific features.

This is a genuine (short) training step. It is kept deliberately simple and
reproducible:

* a torchvision ImageNet-pretrained backbone with a fresh 2-class head;
* end-to-end training with Adam and a class-weighted cross-entropy loss;
* horizontal-flip augmentation (safe for mammography — laterality is not a
  cancer cue);
* an internal validation split, with the best-epoch weights kept.

Crucially, the **validation pipeline is unchanged**: this class only produces a
better model; the test split is still scored and validated as a black box.
Training images are cached in memory after the first read so multiple epochs do
not re-decode the JPEGs.
"""
from __future__ import annotations

import numpy as np

from .base import ImageClassifier

__all__ = ["FineTunedClassifier"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FineTunedClassifier(ImageClassifier):
    """ImageNet backbone fine-tuned end-to-end for breast-cancer classification.

    Workflow::

        model = FineTunedClassifier(backbone="resnet50", image_size=320)
        model.fit(train_dataset, epochs=6)
        predictions = score_dataset(model, test_dataset)

    Parameters
    ----------
    backbone : str
        torchvision model with ImageNet weights: ``resnet18``, ``resnet50``,
        ``densenet121`` or ``efficientnet_b0``.
    image_size : int
        Square input size. Larger preserves more lesion detail (a small mass or
        calcification survives the resize) at the cost of speed/memory.
    device : str, optional
        ``'cuda'`` / ``'cpu'``; auto-detected when omitted.
    """

    _BACKBONES = ("resnet18", "resnet50", "densenet121", "efficientnet_b0")

    def __init__(self, backbone="resnet50", image_size=320, device=None):
        import torch
        import torch.nn as nn
        from torchvision import models

        if backbone not in self._BACKBONES:
            raise ValueError(f"backbone must be one of {self._BACKBONES}")

        self._torch = torch
        self.backbone_name = backbone
        self.image_size = int(image_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights_attr = {"resnet18": "ResNet18_Weights", "resnet50": "ResNet50_Weights",
                        "densenet121": "DenseNet121_Weights",
                        "efficientnet_b0": "EfficientNet_B0_Weights"}[backbone]
        weights = getattr(models, weights_attr).DEFAULT
        net = getattr(models, backbone)(weights=weights)

        # Replace the 1000-class ImageNet head with a fresh 2-class head.
        if backbone.startswith("resnet"):
            net.fc = nn.Linear(net.fc.in_features, 2)
        elif backbone == "densenet121":
            net.classifier = nn.Linear(net.classifier.in_features, 2)
        else:  # efficientnet_b0: classifier is Sequential(Dropout, Linear)
            net.classifier = nn.Sequential(nn.Dropout(0.2),
                                           nn.Linear(net.classifier[1].in_features, 2))
        self._net = net.to(self.device)

        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.fitted = False
        self.val_auc = float("nan")
        self.n_train = 0
        self.name = f"finetuned[{backbone}@{self.image_size}]"

    # ----------------------------------------------------------- image utils
    @staticmethod
    def _to_pil(image):
        from PIL import Image

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                lo, hi = float(np.min(arr)), float(np.max(arr))
                arr = (np.zeros_like(arr, np.uint8) if hi <= lo
                       else (255.0 * (arr - lo) / (hi - lo)).astype(np.uint8))
            image = Image.fromarray(arr)
        return image.convert("RGB")

    def _load_resize(self, image):
        """Image -> a square uint8 ``(H, W, 3)`` array at ``image_size``."""
        pil = self._to_pil(image).resize((self.image_size, self.image_size))
        return np.array(pil, dtype=np.uint8)  # np.array copies -> writable

    def _to_batch(self, imgs_uint8, train):
        """uint8 ``(B, H, W, 3)`` -> normalised float tensor on the device."""
        torch = self._torch
        t = torch.from_numpy(np.ascontiguousarray(imgs_uint8)).float()
        t = t.permute(0, 3, 1, 2) / 255.0
        if train:
            flip = torch.rand(t.shape[0]) < 0.5  # random horizontal flip
            if flip.any():
                t[flip] = torch.flip(t[flip], dims=[3])
        t = (t - self._mean) / self._std
        return t.to(self.device)

    # ----------------------------------------------------------------- fit
    def fit(self, dataset, epochs=6, batch_size=24, lr=1e-4, val_fraction=0.15,
            limit=None, seed=0, progress=True):
        """Fine-tune the network on a dataset's training split.

        An internal ``val_fraction`` hold-out tracks generalisation each epoch;
        the weights from the best-AUC epoch are restored at the end.
        """
        torch = self._torch
        import torch.nn as nn
        from ..metrics import auc_ci

        cases = dataset.cases.iloc[:int(limit)] if limit else dataset.cases
        try:
            from tqdm.auto import tqdm
            rows = tqdm(cases.iterrows(), total=len(cases), disable=not progress,
                        desc=f"loading[{self.backbone_name}@{self.image_size}]")
        except Exception:
            rows = cases.iterrows()

        images, labels = [], []
        for _, row in rows:
            try:
                images.append(self._load_resize(dataset.load(row)))
            except Exception:
                continue
            labels.append(int(row["y_true"]))
        if not images:
            raise RuntimeError("no training images could be loaded")
        X = np.stack(images)                       # (N, H, W, 3) uint8, cached
        y = np.asarray(labels, dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise RuntimeError("training split has only one class")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        n_val = max(1, int(len(y) * val_fraction))
        val_idx, train_idx = order[:n_val], order[n_val:]

        # class-weighted loss — robust to malignant/benign imbalance
        counts = np.bincount(y[train_idx], minlength=2).astype(np.float32)
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                              dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)

        best_auc, best_state = -1.0, None
        for epoch in range(epochs):
            self._net.train()
            epoch_order = rng.permutation(train_idx)   # reshuffle each epoch
            for start in range(0, len(epoch_order), batch_size):
                bidx = epoch_order[start:start + batch_size]
                xb = self._to_batch(X[bidx], train=True)
                yb = torch.tensor(y[bidx], device=self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

            val_scores = self._predict_array(X[val_idx], batch_size)
            try:
                val_auc = auc_ci(y[val_idx], val_scores)["auc"]
            except ValueError:
                val_auc = 0.5
            if progress:
                print(f"  epoch {epoch + 1}/{epochs}  internal val AUC = {val_auc:.3f}")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self._net.state_dict().items()}

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self.fitted = True
        self.val_auc = float(best_auc)
        self.n_train = int(len(train_idx))
        return self

    # ------------------------------------------------------------- predict
    def _predict_array(self, imgs_uint8, batch_size):
        """Malignancy probabilities for a uint8 image array."""
        torch = self._torch
        self._net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(imgs_uint8), batch_size):
                xb = self._to_batch(imgs_uint8[start:start + batch_size], train=False)
                probs = torch.softmax(self._net(xb), dim=1)[:, 1]
                out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, image):
        """Malignancy probability for one image (path / PIL / ndarray)."""
        if not self.fitted:
            raise RuntimeError("call fit(train_dataset) before predict_proba()")
        return float(self._predict_array(self._load_resize(image)[None], 1)[0])

    def predict_proba_batch(self, images, batch_size=64):
        """Batched malignancy probabilities."""
        if not self.fitted:
            raise RuntimeError("call fit(train_dataset) before predict_proba()")
        arr = np.stack([self._load_resize(im) for im in images])
        return [float(p) for p in self._predict_array(arr, batch_size)]
