# Running the checks

From the repository root:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
python examples/demo_synthetic.py
```

Core tests exercise metrics and the report/pipeline on small fixtures. Model-related tests are skipped if torch/torchvision are absent; install the optional model dependencies to exercise them. In the maintenance environment, all 38 tests passed with torch/torchvision available. The synthetic demo completed and generated `outputs/demo_validation_report.html`.

CI runs the core suite and synthetic demo on Python 3.12. It does not download large medical datasets or reproduce the real-data training runs. The two committed real-data reports remain historical artifacts and are described separately in the pipeline documentation.
