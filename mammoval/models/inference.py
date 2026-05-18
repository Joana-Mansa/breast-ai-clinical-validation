"""Run a black-box model over a dataset to produce a predictions table.

This is the single bridge between :mod:`mammoval.models` and
:mod:`mammoval.metrics`. Everything downstream — every metric, plot and report
— consumes only the resulting predictions ``DataFrame``; it never sees the
model or the pixels. That is what keeps the validation reproducible and the
model genuinely a black box.
"""
from __future__ import annotations

__all__ = ["score_dataset"]


def score_dataset(classifier, dataset, limit=None, progress=True):
    """Score every case in ``dataset`` with ``classifier``.

    Parameters
    ----------
    classifier : ImageClassifier or VolumeClassifier
        Anything exposing ``predict_proba``.
    dataset : mammoval.data.base.MammoDataset
        Provides a ``cases`` table and a ``load(row)`` method.
    limit : int, optional
        Score only the first ``limit`` cases — useful for a quick Colab pass
        or when working from a download subset.

    Returns
    -------
    pandas.DataFrame
        A copy of ``dataset.cases`` with a float ``y_score`` column added.
        Cases whose image fails to load receive NaN and are reported; drop
        them before computing metrics.
    """
    cases = (dataset.cases.iloc[:int(limit)] if limit is not None
             else dataset.cases).copy()

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(cases.iterrows(), total=len(cases),
                        disable=not progress, desc=f"scoring {dataset.name}")
    except Exception:
        iterator = cases.iterrows()

    scores, n_failed = [], 0
    for _, row in iterator:
        try:
            scores.append(float(classifier.predict_proba(dataset.load(row))))
        except Exception as exc:
            scores.append(float("nan"))
            n_failed += 1
            if n_failed <= 5:
                print(f"  [warn] case {row.get('case_id', '?')} failed: {exc}")

    cases["y_score"] = scores
    if n_failed:
        print(f"[score_dataset] {n_failed}/{len(cases)} cases could not be "
              f"scored (NaN); drop these before computing metrics.")
    return cases.reset_index(drop=True)
