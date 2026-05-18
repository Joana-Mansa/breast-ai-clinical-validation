"""Model adapter interfaces.

Clinical validation treats the AI as a sealed device: the pipeline measures it,
it never trains or alters it. Every model therefore enters the pipeline through
one of two tiny interfaces — a 2D exam classifier or a 3D volume classifier —
that expose exactly one thing: a malignancy probability for an image/volume.

Anything that satisfies the interface (a HuggingFace checkpoint, an ONNX
export, a vendor REST API, a synthetic stand-in) can be validated without
touching the metrics code. That separation is the whole point: it is what lets
the same report compare model versions, or compare AI against a radiologist.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ImageClassifier(ABC):
    """Black-box 2D exam-level classifier: an image -> P(malignant) in [0, 1]."""

    name = "abstract-image-classifier"

    @abstractmethod
    def predict_proba(self, image):
        """Return the malignancy probability for a single 2D image.

        ``image`` may be a file path, a PIL image or a NumPy array; concrete
        adapters document what they accept.
        """

    def predict_proba_batch(self, images):
        """Score an iterable of images. Override for true batched inference."""
        return [self.predict_proba(im) for im in images]

    def __call__(self, image):
        return self.predict_proba(image)


class VolumeClassifier(ABC):
    """Black-box 3D classifier: a tomosynthesis volume -> P(malignant)."""

    name = "abstract-volume-classifier"

    @abstractmethod
    def predict_proba(self, volume):
        """Return the malignancy probability for a single DBT volume.

        ``volume`` is a 3D array shaped ``(n_slices, height, width)``.
        """

    def __call__(self, volume):
        return self.predict_proba(volume)
