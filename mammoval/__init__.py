"""mammoval -- a clinical validation pipeline for 2D and 3D mammography AI.

The package separates the three concerns of a validation study:

* ``mammoval.data``    -- load public datasets (CBIS-DDSM 2D, Duke BCS-DBT 3D)
                          into a uniform case table.
* ``mammoval.models``  -- thin adapters that turn any breast-cancer model into
                          a callable producing case-level scores / detections.
* ``mammoval.metrics`` -- the validation engine: discrimination, localisation,
                          screening, calibration and subgroup analysis.

``mammoval.pipeline`` wires these together and ``mammoval.report`` renders the
result as a standalone HTML validation report.

The model is always treated as a **black box**: the pipeline never trains or
modifies it, it only measures it against a reference standard -- which is what
clinical validation, as opposed to model development, actually is.
"""
__version__ = "0.1.0"
