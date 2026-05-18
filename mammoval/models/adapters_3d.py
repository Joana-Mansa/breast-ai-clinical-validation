"""3D digital breast tomosynthesis (DBT) model adapters.

No DBT detection or classification model with publicly downloadable weights
could be verified (the Duke BCS-DBT baselines publish training code only). So
the 3D classifier here is built the way DBT triage products actually work in
practice: a tomosynthesis exam is a stack of reconstructed slices, and the
exam-level decision is an aggregation over slice-level evidence (vendors
similarly run on the synthesized-2D image derived from that same stack).

:class:`SliceAggregatorClassifier` therefore lifts *any* validated 2D
:class:`ImageClassifier` to a volume classifier. This makes the 3D
classification pipeline genuinely runnable end-to-end today, and keeps it
honest: when a real 3D detector becomes available it drops in as a new
``VolumeClassifier`` with no change to the metrics or the report.
"""
from __future__ import annotations

import numpy as np

from .base import VolumeClassifier

__all__ = ["SliceAggregatorClassifier"]


class SliceAggregatorClassifier(VolumeClassifier):
    """Lift a 2D :class:`ImageClassifier` to a DBT volume classifier.

    A subset of slices is scored by the 2D model and the per-slice malignancy
    probabilities are pooled into one exam-level probability.

    Parameters
    ----------
    base_classifier : ImageClassifier
        Any validated 2D classifier.
    n_slices : int
        Number of evenly spaced slices to score per volume. Sampling rather
        than scoring every slice keeps inference tractable in Colab.
    aggregate : {'top_k_mean', 'max', 'mean'}
        How per-slice scores combine. ``max`` mirrors "the exam is suspicious
        if any slice is"; ``top_k_mean`` (default, k = ``top_k``) is steadier
        because it is not driven by a single noisy slice.
    top_k : int
        k for ``top_k_mean``.
    slice_fraction : (float, float)
        Keep only the central fraction of the stack — the first/last slices of
        a reconstruction are noisy and rarely carry the finding.

    Notes
    -----
    Aggregation choice is itself a validation variable: the pipeline can
    compare ``max`` vs ``top_k_mean`` with a paired DeLong test and report
    which pooling rule discriminates better.
    """

    def __init__(self, base_classifier, n_slices=16, aggregate="top_k_mean",
                 top_k=3, slice_fraction=(0.15, 0.85)):
        self.base = base_classifier
        self.n_slices = int(n_slices)
        self.aggregate = aggregate
        self.top_k = int(top_k)
        self.slice_fraction = slice_fraction
        self.name = f"slice-agg[{aggregate}]({getattr(base_classifier, 'name', '2d')})"

    def _slice_indices(self, n_total):
        lo = int(n_total * self.slice_fraction[0])
        hi = max(lo + 1, int(n_total * self.slice_fraction[1]))
        count = min(self.n_slices, hi - lo)
        return np.unique(np.linspace(lo, hi - 1, count).astype(int))

    def predict_proba(self, volume):
        """Malignancy probability for a DBT volume ``(n_slices, H, W)``."""
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

        idx = self._slice_indices(volume.shape[0])
        slice_scores = np.asarray(
            self.base.predict_proba_batch([volume[i] for i in idx]), dtype=float
        )
        return float(self._pool(slice_scores))

    def predict_with_slices(self, volume):
        """Like :meth:`predict_proba` but also return the per-slice scores and
        the index of the peak slice — a crude localisation cue, and the input
        a slice-level FROC would consume."""
        volume = np.asarray(volume)
        idx = self._slice_indices(volume.shape[0])
        scores = np.asarray(
            self.base.predict_proba_batch([volume[i] for i in idx]), dtype=float
        )
        return {
            "score": float(self._pool(scores)),
            "slice_indices": idx.tolist(),
            "slice_scores": scores.tolist(),
            "peak_slice": int(idx[int(np.argmax(scores))]),
        }

    def _pool(self, scores):
        if scores.size == 0:
            return 0.0
        if self.aggregate == "max":
            return np.max(scores)
        if self.aggregate == "mean":
            return np.mean(scores)
        if self.aggregate == "top_k_mean":
            k = min(self.top_k, scores.size)
            return np.mean(np.sort(scores)[-k:])
        raise ValueError(f"unknown aggregate '{self.aggregate}'")
