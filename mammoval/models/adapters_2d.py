"""2D mammography model adapters.

The default adapter wraps a HuggingFace image-classification checkpoint. The
default is ``keanteng/swin-v2-large-ft-breast-cancer-classification`` -- it
loads with no training step, which is why it is the pipeline's out-of-the-box
model.

It is, however, a *baseline*: a Swin-V2 head fine-tuned on Mini-DDSM-derived
images at 192x192. The point of this project is not the model -- it is the
validation. Swapping in a stronger model (the RSNA-2023 winning ConvNeXt
ensemble, a Mammo-CLIP linear probe, a vendor device) means writing one
``ImageClassifier`` subclass; the metrics, report and notebooks are unchanged.
The pipeline will, in fact, *quantify* exactly how much the baseline falls
short -- which is the honest thing for a validation tool to do.

Robustness to malformed checkpoints
-----------------------------------
Community checkpoints are frequently published with defective metadata. This
adapter defends against the two common defects so the pipeline still runs:

* **Bad ``model_type``.** Some configs set ``model_type`` to the full
  checkpoint name instead of the architecture key (e.g. ``swinv2``), which
  breaks ``AutoModel``. ``_load_model`` recovers the true type from the
  ``architectures`` field and reloads with a corrected config.
* **Missing ``preprocessor_config.json``.** Image preprocessing is loaded in
  three tiers -- the checkpoint's own processor, then the base backbone's
  processor, then a manual resize + ImageNet-normalise transform built from the
  model config. ``preprocess_source`` records which tier was used.
"""
from __future__ import annotations

import numpy as np

from .base import ImageClassifier

__all__ = ["HFImageClassifier"]

# ImageNet statistics — the standard normalisation for Swin/SwinV2 backbones,
# used by the tier-3 manual transform fallback.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class HFImageClassifier(ImageClassifier):
    """Adapter over a HuggingFace ``AutoModelForImageClassification``.

    Parameters
    ----------
    model_id : str
        HuggingFace hub id. Default: the Swin-V2 breast classifier.
    device : str, optional
        ``'cuda'`` / ``'cpu'``; auto-detected when omitted.
    malignant_keywords : tuple of str
        The malignant class is found by scanning the checkpoint's ``id2label``
        for these substrings (case-insensitive). The default Swin model labels
        its classes ``Has_Cancer`` / ``Normal``, so ``'cancer'`` selects the
        right index automatically rather than hard-coding 0 vs 1.
    fallback_processors : tuple of str
        Image-processor sources tried when the checkpoint itself ships none.

    Notes
    -----
    ``torch`` / ``transformers`` are imported lazily inside ``__init__`` so the
    metrics engine and the synthetic demo run in an environment without the
    deep-learning stack installed.
    """

    DEFAULT_MODEL = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"
    FALLBACK_PROCESSORS = ("microsoft/swinv2-large-patch4-window12-192-22k",)

    def __init__(self, model_id=DEFAULT_MODEL, device=None,
                 malignant_keywords=("cancer", "malign", "abnormal", "pos"),
                 fallback_processors=FALLBACK_PROCESSORS):
        from transformers import AutoImageProcessor
        import torch

        self._torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_id).to(self.device).eval()

        # --- image preprocessing: robust 3-tier loading --------------------
        self.processor = None
        self._transform = None
        self.preprocess_source = None
        for src in (model_id, *fallback_processors):
            try:
                self.processor = AutoImageProcessor.from_pretrained(src)
                self.preprocess_source = f"processor:{src}"
                break
            except Exception:  # noqa: BLE001 - try the next source
                continue
        if self.processor is None:
            from torchvision import transforms
            size = self._infer_image_size()
            self._transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
            self.preprocess_source = f"manual-transform:{size}px-imagenet"

        # --- locate the malignant class index -----------------------------
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

    # ------------------------------------------------------------ model load
    # Backbone inference from a malformed ``model_type`` string. Ordered so
    # that more specific keys (swinv2, convnextv2) are tested before their
    # prefixes (swin, convnext).
    _ARCH_HINTS = (
        ("swinv2", "Swinv2ForImageClassification"),
        ("swin", "SwinForImageClassification"),
        ("convnextv2", "ConvNextV2ForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("deit", "DeiTForImageClassification"),
        ("vit", "ViTForImageClassification"),
    )

    @staticmethod
    def _load_model(model_id):
        """Load the classifier, repairing a malformed ``config.json`` if needed.

        Tries ``AutoModelForImageClassification`` first. If the config has a
        ``model_type`` Transformers does not recognise (a common community-
        upload defect), the real architecture is recovered -- from the
        ``architectures`` field, or by inferring the backbone from the bad
        ``model_type`` string -- and the model is reloaded with a corrected
        config object.
        """
        from transformers import AutoModelForImageClassification
        try:
            return AutoModelForImageClassification.from_pretrained(model_id)
        except (ValueError, KeyError):
            pass  # fall through to the repair path

        import json
        import transformers
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_id, "config.json")) as fh:
            cfg = json.load(fh)

        # 1) trust the architectures field when it names a known class
        arch = (cfg.get("architectures") or [None])[0]
        model_cls = getattr(transformers, arch, None) if arch else None

        # 2) otherwise infer the backbone from the malformed model_type string
        if model_cls is None:
            model_type = str(cfg.get("model_type", "")).lower()
            for hint, cls_name in HFImageClassifier._ARCH_HINTS:
                if hint in model_type:
                    model_cls = getattr(transformers, cls_name, None)
                    if model_cls is not None:
                        break

        if model_cls is None:
            raise ValueError(
                f"Cannot repair config.json for '{model_id}': unrecognised "
                f"architecture (architectures={arch!r}, "
                f"model_type={cfg.get('model_type')!r}).")

        config_cls = model_cls.config_class
        cfg.pop("model_type", None)            # drop the bad value
        config = config_cls.from_dict(cfg)     # falls back to the class default
        print(f"[HFImageClassifier] repaired malformed config.json -> "
              f"{model_cls.__name__} (model_type '{config.model_type}')")
        return model_cls.from_pretrained(model_id, config=config)

    # ----------------------------------------------------------------- utils
    def _infer_image_size(self):
        """Square input size from the model config (default 192)."""
        size = getattr(self.model.config, "image_size", 192)
        if isinstance(size, (list, tuple)):
            size = size[0]
        return int(size) if size else 192

    @staticmethod
    def _to_pil(image):
        """Coerce a path / PIL image / ndarray to an 8-bit RGB PIL image."""
        from PIL import Image

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:  # rescale to 8-bit
                lo, hi = float(np.min(arr)), float(np.max(arr))
                arr = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else \
                    (255.0 * (arr - lo) / (hi - lo)).astype(np.uint8)
            image = Image.fromarray(arr)
        return image.convert("RGB")  # 3-channel input for the backbone

    def _logits(self, pil_images):
        """Forward a list of PIL images and return the raw logits tensor."""
        torch = self._torch
        if self.processor is not None:
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                return self.model(**inputs).logits
        batch = torch.stack([self._transform(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            return self.model(pixel_values=batch).logits

    # --------------------------------------------------------------- predict
    def predict_proba(self, image):
        """Malignancy probability for one image (path / PIL / ndarray)."""
        logits = self._logits([self._to_pil(image)])
        probs = self._torch.softmax(logits, dim=-1)[0]
        return float(probs[self.malignant_index].item())

    def predict_proba_batch(self, images, batch_size=16):
        """Batched inference — materially faster on a GPU."""
        pil = [self._to_pil(im) for im in images]
        out = []
        for start in range(0, len(pil), batch_size):
            logits = self._logits(pil[start:start + batch_size])
            probs = self._torch.softmax(logits, dim=-1)[:, self.malignant_index]
            out.extend(float(p) for p in probs.detach().cpu().numpy())
        return out
