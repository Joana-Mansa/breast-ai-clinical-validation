"""Dataset loaders — normalise public mammography datasets into one case table.

* :class:`~mammoval.data.cbis_ddsm.CBISDDSMDataset` -- 2D scanned-film
  mammography with biopsy-proven pathology, BI-RADS, density and lesion type.
* :class:`~mammoval.data.duke_dbt.DukeDBTDataset` -- 3D digital breast
  tomosynthesis with four-way labels and ground-truth lesion boxes.

Both expose the :class:`~mammoval.data.base.MammoDataset` interface, so the
pipeline does not care which one it is handed.
"""
from .base import MammoDataset
from .cbis_ddsm import CBISDDSMDataset
from .duke_dbt import DukeDBTDataset

__all__ = ["MammoDataset", "CBISDDSMDataset", "DukeDBTDataset"]
