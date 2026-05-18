"""Linear-probe classifier — a frozen pretrained backbone plus a logistic head.

Reliable, ready-to-use breast-cancer classifiers are scarce on public model
hubs, and community uploads are frequently broken (malformed configs, missing
preprocessing, weights that do not match their own config). When no trustworthy
checkpoint is available, the standard, reproducible alternative is a **linear
probe**: freeze a backbone pretrained on a large dataset, use it purely as a
fixed feature extractor, and fit a simple linear classifier on the target
task's *training* split.

This keeps the model/validation separation honest:

* the backbone is genuinely pretrained (ImageNet) and is **never updated** —
  every parameter is frozen;
* only a logistic-regression head is fit, and only on the dataset's
  **training** split;
* the fitted classifier is then frozen and the **test** split is validated as
  a black box, exactly like any other model.

torchvision backbones are used deliberately: they always load cleanly, so the
pipeline cannot be derailed by a bad checkpoint. A linear probe is also a
recognised evaluation protocol in its own right — it measures how linearly
separable the cancer signal is in a generic visual representation, a sensible
baseline a validation report can state and defend.
"""
from __future__ import annotations

import numpy as np

from .base import ImageClassifier

__all__ = ["LinearProbeClassifier"]

# ImageNet normalisation — what every torchvision ImageNet backbone expects.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class LinearProbeClassifier(ImageClassifier):
    """Frozen torchvision backbone + logistic-regression malignancy head.

    Workflow::

        probe = LinearProbeClassifier(backbone="resnet50")
        probe.fit(train_dataset)                 # fits the linear head only
        predictions = score_dataset(probe, test_dataset)

    Parameters
    ----------
    backbone : str
        torchvision model with ImageNet weights: ``resnet18``, ``resnet50``,
        ``densenet121`` or ``efficientnet_b0``.
    image_size : int
        Square input size for the backbone (224 for ImageNet models).
    device : str, optional
        ``'cuda'`` / ``'cpu'``; auto-detected when omitted.

    Notes
    -----
    ``torch`` / ``torchvision`` are imported lazily inside ``__init__`` so the
    metrics engine and synthetic demo run without the deep-learning stack.
    """

    # backbone name -> torchvision weights-enum attribute name
    _BACKBONES = {
        "resnet18": "ResNet18_Weights",
        "resnet50": "ResNet50_Weights",
        "densenet121": "DenseNet121_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
    }

    def __init__(self, backbone="resnet50", image_size=224, device=None):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        if backbone not in self._BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(self._BACKBONES)}")

        self._torch = torch
        self.backbone_name = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = getattr(models, self._BACKBONES[backbone]).DEFAULT
        net = getattr(models, backbone)(weights=weights)
        # Replace the classification head with identity -> pure feature extractor.
        if backbone.startswith("resnet"):
            net.fc = nn.Identity()
        else:  # densenet121 / efficientnet_b0
            net.classifier = nn.Identity()
        for param in net.parameters():       # freeze every backbone weight
            param.requires_grad_(False)
        self._net = net.to(self.device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self.head = None             # sklearn pipeline, set by fit()
        self._malignant_col = 1
        self.feature_dim = None
        self.n_train = 0
        self.name = f"linear-probe[{backbone}]"

    # ----------------------------------------------------------- image utils
    @staticmethod
    def _to_pil(image):
        """Coerce a path / PIL image / ndarray to an 8-bit RGB PIL image."""
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

    def extract_features(self, images, batch_size=32):
        """Frozen-backbone feature vectors for an iterable of images."""
        torch = self._torch
        pil = [self._to_pil(im) for im in images]
        chunks = []
        for i in range(0, len(pil), batch_size):
            batch = torch.stack([self._transform(im)
                                 for im in pil[i:i + batch_size]]).to(self.device)
            with torch.no_grad():
                feats = self._net(batch)
            chunks.append(feats.detach().cpu().float().numpy())
        if not chunks:
            return np.zeros((0, self.feature_dim or 1))
        feats = np.vstack(chunks)
        return feats.reshape(feats.shape[0], -1)

    # ----------------------------------------------------------------- fit
    def fit(self, dataset, limit=None, batch_size=32, C=1.0, progress=True):
        """Fit the logistic-regression head on a dataset's training split.

        Parameters
        ----------
        dataset : mammoval.data.base.MammoDataset
            Training-split dataset; its ``y_true`` column supplies the labels.
        limit : int, optional
            Use only the first ``limit`` cases (a faster pass).
        C : float
            Inverse L2 regularisation strength for the logistic head.

        Returns ``self`` so calls can be chained.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cases = dataset.cases.iloc[:int(limit)] if limit else dataset.cases
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(cases.iterrows(), total=len(cases),
                            disable=not progress,
                            desc=f"features[{self.backbone_name}]")
        except Exception:
            iterator = cases.iterrows()

        feature_chunks, labels, buffer = [], [], []
        for _, row in iterator:
            try:
                buffer.append(self._to_pil(dataset.load(row)))
            except Exception:
                continue  # skip unreadable images
            labels.append(int(row["y_true"]))
            if len(buffer) >= batch_size:
                feature_chunks.append(self.extract_features(buffer, batch_size))
                buffer = []
        if buffer:
            feature_chunks.append(self.extract_features(buffer, batch_size))

        if not feature_chunks:
            raise RuntimeError("no training images could be loaded")
        X = np.vstack(feature_chunks)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            raise RuntimeError("training split has only one class")

        self.feature_dim = X.shape[1]
        self.head = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
        )
        self.head.fit(X, y)
        self._malignant_col = list(self.head.classes_).index(1)
        self.n_train = len(y)
        return self

    # ------------------------------------------------------------- predict
    def predict_proba(self, image):
        """Malignancy probability for one image (path / PIL / ndarray)."""
        if self.head is None:
            raise RuntimeError("call fit(train_dataset) before predict_proba()")
        feats = self.extract_features([image])
        return float(self.head.predict_proba(feats)[0, self._malignant_col])

    def predict_proba_batch(self, images, batch_size=32):
        """Batched malignancy probabilities — faster on a GPU."""
        if self.head is None:
            raise RuntimeError("call fit(train_dataset) before predict_proba()")
        feats = self.extract_features(images, batch_size)
        probs = self.head.predict_proba(feats)[:, self._malignant_col]
        return [float(p) for p in probs]
