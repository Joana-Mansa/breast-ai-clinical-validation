# Running the checks

From the repository root:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
python examples/demo_synthetic.py
```

To include model-adapter tests, install the optional dependencies first:

```bash
python -m pip install -r requirements-models.txt
python -m pytest -q
```

## Coverage

Core tests check metrics against known values, dataset loading, and report generation on small fixtures. Model tests are skipped when their optional dependencies are absent.

The full local suite passed **38 tests** on 15 September 2026. The synthetic example generated `outputs/demo_validation_report.html`. GitHub CI runs the core suite and synthetic example on Python 3.12.

These checks do not download the complete medical datasets or repeat the training and detector-comparison experiments. The saved reports are described in the [2D guide](2d_pipeline.md#recorded-results) and [3D guide](3d_pipeline.md#8-recorded-results).
