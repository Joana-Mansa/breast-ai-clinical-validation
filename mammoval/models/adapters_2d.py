"""2D mammography model adapters.

The default adapter wraps a HuggingFace image-classification checkpoint. The
verified default is ``keanteng/swin-v2-large-ft-breast-cancer-classification``
-- it loads in two lines and runs in Colab without a training step, which is
why it is the pipeline's out-of-the-box model.

It is, however, a *baseline*: a Swin-V2 head fine-tuned on Mini-DDSM-derived
images at 192x192. The point of this project is not the model -- it is the
validation. Swapping in a stronger model (the RSNA-2023 winning ConvNeXt
ensemble, a Mammo-CLIP linear probe, a vendor device) means writing one
``ImageClassifier`` subclass; the metrics, report and notebooks are unchanged.
The pipeline will, in fact, *quantify* exactly how much the baseline falls
short -- which is the honest thing for a validation tool to do.
"""
from __future__ import annotations

import numpy as np

from .base import ImageClassifier

__all__ = ["HFImageClassifier"]


class HFImageClassifier(ImageClassifier):
    """Adapter over a HuggingFace ``AutoModelForImageClassification``.

    Parameters
    ----------
    model_id : str
        HuggingFace hub id. Default: the verified Swin-V2 breast classifier.
    device : str, optional
        ``'cuda'`` / ``'cpu'``; auto-detected when omitted.
    malignant_keywords : tuple of str
        The malignant class is found by scanning the checkpoint's ``id2label``
        for these substrings (case-insensitive). The default Swin model labels
        its classes ``Has_Cancer`` / ``Normal``, so ``'cancer'`` selects the
        right index automatically rather than hard-coding 0 vs 1.

    Notes
    -----
    ``torch`` / ``transformers`` are imported lazily inside ``__init__`` so the
    metrics engine and the synthetic demo run in an environment without the
    deep-learning stack installed.
    """

    DEFAULT_MODEL = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"

    def __init__(self, model_id=DEFAULT_MODEL, device=None,
                 malignant_keywords=("cancer", "malign", "abnormal", "pos")):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        self._torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForImageClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        id2label = self.model.config.id2label
        self.malignant_index = None
        for idx, label in id2label.items():
            if any(k in str(label).lower() for k in malignant_keywords):
                self.malignant_index = int(idx)
                break
        if self.malignant_index is None:  # 2-class fallback: assume index 0
            self.malignant_index = 0
        self.id2label = id2label
        self.name = f"hf:{model_id}"

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _to_pil(image):
        from PIL import Image

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:  # rescale to 8-bit for the processor
                lo, hi = float(np.min(arr)), float(np.max(arr))
                arr = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else \
                    (255.0 * (arr - lo) / (hi - lo)).astype(np.uint8)
            image = Image.fromarray(arr)
        return image.convert("RGB")  # 3-channel input for the backbone

    # --------------------------------------------------------------- predict
    def predict_proba(self, image):
        """Malignancy probability for one image (path / PIL / ndarray)."""
        pil = self._to_pil(image)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        return float(probs[self.malignant_index].item())

    def predict_proba_batch(self, images, batch_size=16):
        """Batched inference — materially faster on a GPU."""
        pil = [self._to_pil(im) for im in images]
        out = []
        for start in range(0, len(pil), batch_size):
            chunk = pil[start:start + batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                logits = self.model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[:, self.malignant_index]
            out.extend(float(p) for p in probs.cpu().numpy())
        return out
