"""Model adapters — turn any breast-cancer model into a black box the
validation pipeline can measure.

* :class:`~mammoval.models.base.ImageClassifier` /
  :class:`~mammoval.models.base.VolumeClassifier` -- the two adapter contracts.
* :class:`~mammoval.models.adapters_2d.HFImageClassifier` -- HuggingFace 2D
  checkpoint adapter (verified default model for the CBIS-DDSM pipeline).
* :class:`~mammoval.models.adapters_3d.SliceAggregatorClassifier` -- lifts a 2D
  classifier to a DBT volume classifier.
* :mod:`~mammoval.models.synthetic` -- fabricated predictions for the
  guaranteed-runnable demo and the unit tests.
* :func:`~mammoval.models.inference.score_dataset` -- run a model over a
  dataset to get the predictions table the metrics consume.

``HFImageClassifier`` is imported lazily: importing this package does not pull
in ``torch``/``transformers``, so the metrics engine stays light.
"""
from .base import ImageClassifier, VolumeClassifier
from .synthetic import simulate_scores, simulate_detections
from .linear_probe import LinearProbeClassifier
from .adapters_3d import SliceAggregatorClassifier
from .inference import score_dataset

__all__ = [
    "ImageClassifier",
    "VolumeClassifier",
    "LinearProbeClassifier",
    "SliceAggregatorClassifier",
    "simulate_scores",
    "simulate_detections",
    "score_dataset",
    "HFImageClassifier",
]


def __getattr__(name):
    """Lazily expose HFImageClassifier so `torch` is only imported on demand."""
    if name == "HFImageClassifier":
        from .adapters_2d import HFImageClassifier
        return HFImageClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
