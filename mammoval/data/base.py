"""Dataset interface — a standardised case table for any mammography source.

Every dataset, whatever its on-disk mess, is normalised to one ``cases`` table
so the pipeline, metrics and report are dataset-agnostic. The contract is
deliberately small:

Required columns
    ``case_id``    unique string id for the analysis unit (an image or a view);
    ``patient_id`` the **unit of statistical independence** — used for
                   patient-level (cluster) bootstrap CIs;
    ``y_true``     int {0, 1}, 1 = biopsy-proven malignant (the reference
                   standard).

Anything else (breast density, lesion type, view, BI-RADS, ...) is an optional
subgroup column the report will stratify on automatically when present.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class MammoDataset(ABC):
    """Base class for a validation dataset.

    Subclasses populate ``self.cases`` (a pandas DataFrame) and implement
    :meth:`load`, which lazily returns the pixels for one case.
    """

    name = "abstract-dataset"
    modality = "2D"  # "2D" or "3D"
    REQUIRED_COLUMNS = ("case_id", "patient_id", "y_true")

    @abstractmethod
    def load(self, row):
        """Return the image (2D ndarray/PIL) or volume (3D ndarray) for a case.

        ``row`` is a row of ``self.cases`` (a pandas Series / mapping).
        """

    # ---------------------------------------------------------------- checks
    def validate(self):
        """Sanity-check the case table; raise on a malformed dataset."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.cases.columns]
        if missing:
            raise ValueError(f"{self.name}: case table missing columns {missing}")
        if self.cases["case_id"].duplicated().any():
            n = int(self.cases["case_id"].duplicated().sum())
            raise ValueError(f"{self.name}: {n} duplicate case_id values")
        labels = set(self.cases["y_true"].dropna().unique())
        if not labels <= {0, 1}:
            raise ValueError(f"{self.name}: y_true must be 0/1, found {labels}")
        return self

    # --------------------------------------------------------------- summary
    def summary(self):
        """Compact cohort description — prevalence, patient count, subgroups."""
        df = self.cases
        n = len(df)
        n_pos = int((df["y_true"] == 1).sum())
        out = {
            "dataset": self.name,
            "modality": self.modality,
            "n_cases": n,
            "n_patients": int(df["patient_id"].nunique()),
            "n_malignant": n_pos,
            "n_non_malignant": n - n_pos,
            "prevalence": n_pos / n if n else float("nan"),
        }
        for col in df.columns:
            if col in ("case_id", "patient_id", "y_true") or df[col].dtype == float:
                continue
            if 1 < df[col].nunique() <= 12:
                out.setdefault("subgroups", {})[col] = (
                    df[col].value_counts(dropna=False).to_dict()
                )
        return out

    def __len__(self):
        return len(self.cases)

    def __repr__(self):
        return (f"<{self.__class__.__name__} '{self.name}' "
                f"{self.modality} n={len(self)}>")
