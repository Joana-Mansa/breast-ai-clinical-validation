"""Generate the two Colab validation notebooks.

The notebooks are thin: every non-trivial step lives in the ``mammoval``
package and is unit-tested, so the notebooks stay readable and cannot silently
drift from the library. Run this script to (re)build them:

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
import os

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


_SETUP_PACKAGE = """\
# Make the `mammoval` package importable. Auto-clones the project repo if it
# is not already present (Internet must be ON for the clone on Kaggle).
import os, sys, glob, subprocess

REPO_URL = 'https://github.com/Joana-Mansa/breast-ai-clinical-validation.git'

def _find_mammoval():
    cands = ['.', 'breast-ai-clinical-validation',
             '/content/breast-ai-clinical-validation']
    cands += glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/*/*')
    cands += glob.glob('/kaggle/working/*')
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'mammoval')):
            return os.path.abspath(c)
    return None

loc = _find_mammoval()
if loc is None:
    print('cloning', REPO_URL, '...')
    subprocess.run(['git', 'clone', '-q', REPO_URL], check=False)
    loc = _find_mammoval()
if loc is None:
    raise RuntimeError('mammoval not found and clone failed - turn Internet ON, '
                       'or add the project folder as a Kaggle dataset (+ Add Input).')
sys.path.insert(0, loc)
import mammoval
print('mammoval', mammoval.__version__, 'ready (from ' + loc + ')')"""


# ==========================================================================
# 2D notebook — CBIS-DDSM
# ==========================================================================
def build_2d():
    cells = [
        md("# Clinical Validation of 2D Mammography AI — CBIS-DDSM\n"
           "\n"
           "A reproducible **standalone (black-box) validation** of a pretrained "
           "breast-cancer classifier on the CBIS-DDSM test split.\n"
           "\n"
           "The pipeline never trains or modifies the model — it *measures* it "
           "against a biopsy-proven reference standard, which is what clinical "
           "validation (as opposed to model development) is.\n"
           "\n"
           "**Dataset:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM), Kaggle "
           "JPEG mirror — biopsy-proven pathology, BI-RADS assessment, breast "
           "density, lesion type.\n"
           "**Model:** an ImageNet-pretrained CNN fine-tuned on the CBIS-DDSM "
           "training split (swap in any model by writing one "
           "`ImageClassifier`).\n"
           "**Output:** an HTML validation report — ROC/AUC with DeLong CIs, "
           "operating points, calibration, screening behaviour, breast-density "
           "subgroup analysis, and an AI-vs-radiologist (BI-RADS) comparison."),

        md("## 1 · Environment setup\n"
           "Runs on **Google Colab** or **Kaggle**.\n"
           "\n"
           "**On Kaggle:** in the *Notebook options* panel set *Accelerator* to "
           "a **GPU** and turn *Internet* **On** — both are required. Internet "
           "lets the notebook install packages, fetch the model, and clone the "
           "project code.\n"
           "\n"
           "The next cell installs dependencies; the one after it auto-clones "
           "the `mammoval` package if it is not already present."),
        code("!pip install -q transformers timm pillow\n"
             "# torch / torchvision / pydicom are pre-installed on Colab and Kaggle."),
        code(_SETUP_PACKAGE),

        md("## 2 · Get the CBIS-DDSM dataset\n"
           "**On Kaggle (easiest):** click **+ Add Input**, search "
           "*\"CBIS-DDSM Breast Cancer Image Dataset\"* (by *awsaf49*) and add it "
           "— it mounts read-only, no download or credentials needed.\n"
           "\n"
           "**On Colab:** the cell downloads the ~6 GB JPEG mirror; it will "
           "prompt you to upload `kaggle.json` (Kaggle → Settings → Create New "
           "API Token)."),
        code("import os, glob\n"
             "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
             "\n"
             "if ON_KAGGLE:\n"
             "    # The mount folder name depends on the dataset slug, so detect\n"
             "    # it: any /kaggle/input/* folder holding both csv/ and jpeg/.\n"
             "    hits = [d for d in sorted(glob.glob('/kaggle/input/*'))\n"
             "            if os.path.isdir(os.path.join(d, 'csv'))\n"
             "            and os.path.isdir(os.path.join(d, 'jpeg'))]\n"
             "    assert hits, ('No CBIS-DDSM dataset (csv/ + jpeg/) found - use '\n"
             "                  '+ Add Input to attach the awsaf49 CBIS-DDSM dataset.')\n"
             "    CBIS_ROOT = hits[0]\n"
             "else:\n"
             "    from google.colab import files\n"
             "    print('Upload your kaggle.json:'); files.upload()\n"
             "    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
             "    !pip install -q kaggle\n"
             "    !kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset\n"
             "    !unzip -q -o cbis-ddsm-breast-cancer-image-dataset.zip -d data/cbis-ddsm\n"
             "    CBIS_ROOT = 'data/cbis-ddsm'\n"
             "\n"
             "print('CBIS-DDSM root:', CBIS_ROOT)\n"
             "print('csv/ :', sorted(os.listdir(os.path.join(CBIS_ROOT, 'csv'))))"),

        md("## 3 · Load the dataset (training + test splits)\n"
           "`CBISDDSMDataset` joins the case-description CSVs to the JPEG mirror "
           "and aggregates abnormality rows to one image-level case "
           "(`y_true` = malignant if any abnormality is malignant). CBIS-DDSM "
           "ships a fixed official split: the **training** split fits the "
           "model's linear head, the **test** split is what gets validated."),
        code("from mammoval.data import CBISDDSMDataset\n"
             "import json\n"
             "\n"
             "ds_train = CBISDDSMDataset(root=CBIS_ROOT, split='train')\n"
             "ds_test  = CBISDDSMDataset(root=CBIS_ROOT, split='test')\n"
             "print('train:', json.dumps(ds_train.summary(), default=str))\n"
             "print('test :', json.dumps(ds_test.summary(),  default=str))\n"
             "ds_test.cases.head()"),

        md("## 4 · Build the model — fine-tuned CNN\n"
           "A frozen-backbone linear probe is reliable but weak on mammography "
           "— ImageNet features transfer poorly to grayscale breast tissue. "
           "Here we **fine-tune** an ImageNet-pretrained CNN end-to-end on the "
           "CBIS-DDSM **training** split: the convolutional weights are "
           "updated, so the network learns mammography-specific features.\n"
           "\n"
           "Training holds out an internal validation split and keeps the "
           "best-epoch weights. This is a genuine (short) training step "
           "(~10–15 min on a Kaggle GPU); the pipeline that scores the **test** "
           "split afterwards is unchanged. `LinearProbeClassifier` remains "
           "available as the faster, no-training baseline."),
        code("from mammoval.models import FineTunedClassifier\n"
             "\n"
             "model = FineTunedClassifier(backbone='resnet50', image_size=320)\n"
             "# Fine-tune end-to-end on the TRAINING split. Lower image_size or\n"
             "# epochs for a faster pass; raise them for a stronger model.\n"
             "model.fit(ds_train, epochs=6, batch_size=24)\n"
             "print('fine-tuned:', model.name,\n"
             "      '| trained on', model.n_train, 'images',\n"
             "      '| best internal val AUC = %.3f' % model.val_auc)"),

        md("## 5 · Run inference on the test split → predictions table\n"
           "`score_dataset` runs the fitted model over every test case and "
           "returns the case table with a `y_score` column. Set `LIMIT=None` "
           "for the full official test split."),
        code("from mammoval.models import score_dataset\n"
             "\n"
             "LIMIT = None         # or an integer for a quick pass\n"
             "preds = score_dataset(model, ds_test, limit=LIMIT)\n"
             "preds = preds.dropna(subset=['y_score'])\n"
             "preds[['case_id','y_true','y_score','breast_density_cat',\n"
             "       'abnormality_type','birads_assessment']].head()"),

        md("## 6 · Run the clinical validation pipeline\n"
           "One call produces the full result: discrimination, operating points, "
           "calibration, screening behaviour, subgroup analysis, and the paired "
           "AI-vs-reader comparison (BI-RADS assessment is used as the radiologist "
           "reference)."),
        code("from mammoval.pipeline import ClassificationConfig, run_classification_validation\n"
             "\n"
             "cfg = ClassificationConfig(\n"
             "    reader_col='birads_assessment',\n"
             "    subgroup_cols=('breast_density_cat', 'abnormality_type', 'view'),\n"
             "    target_specificities=(0.90, 0.96),\n"
             "    ni_margin=0.05,\n"
             "    is_probability=True,\n"
             "    dataset_name='CBIS-DDSM test split',\n"
             ")\n"
             "results = run_classification_validation(preds, cfg)\n"
             "d = results['discrimination']\n"
             "print('AUC = %.3f   95%% CI %.3f-%.3f' % (d['auc'], *d['auc_ci']))"),

        md("## 7 · Generate the HTML validation report"),
        code("from mammoval.report import build_report\n"
             "from IPython.display import HTML\n"
             "\n"
             "os.makedirs('outputs', exist_ok=True)\n"
             "path = build_report(\n"
             "    results,\n"
             "    output_path='outputs/cbis_ddsm_validation_report.html',\n"
             "    title='2D Mammography AI — Clinical Validation (CBIS-DDSM)',\n"
             "    extra_limitations=[\n"
             "        'CBIS-DDSM is digitised film, not full-field digital mammography — '\n"
             "        'a real domain shift from a modern screening device input.',\n"
             "        'The model is an ImageNet CNN fine-tuned briefly on CBIS-DDSM - '\n"
             "        'a reasonable baseline, not a state-of-the-art device; the '\n"
             "        'validation methodology is the deliverable.',\n"
             "    ])\n"
             "HTML(open(path, encoding='utf-8').read())"),

        md("## Notes & next steps\n"
           "* **Stronger model.** Raise `image_size` / `epochs`, try another "
           "backbone, or write one `ImageClassifier` subclass for a different "
           "model — the validation is identical.\n"
           "* **Reader caveat.** BI-RADS *assessment* is an ordinal proxy reader, "
           "not an independent radiologist read; category 0 ('incomplete') is not "
           "strictly ordered. A true reader study is MRMC (see `docs/methodology.md`).\n"
           "* **External validation.** Re-run on VinDr-Mammo or RSNA to probe "
           "generalisation across vendor / population — the report's CIs cover "
           "sampling error only, not distribution shift."),
    ]
    return notebook(cells)


# ==========================================================================
# 3D notebook — Duke BCS-DBT
# ==========================================================================
def build_3d():
    cells = [
        md("# Clinical Validation of 3D Mammography AI — Duke BCS-DBT (FROC)\n"
           "\n"
           "Standalone **lesion-localisation** validation on digital breast "
           "tomosynthesis (DBT) — 3D mammography — using the Duke "
           "*Breast-Cancer-Screening-DBT* dataset.\n"
           "\n"
           "The headline 3D metric is **FROC** (free-response ROC): it scores "
           "every region *mark* a detector emits — a mark counts only when it "
           "lands on a real lesion under the official geometric criterion — and "
           "reports sensitivity at 1/2/3/4 false marks per volume. Their mean is "
           "the official Duke BCS-DBT / DBTex challenge ranking metric.\n"
           "\n"
           "This notebook runs entirely on **small metadata files** (a few MB — "
           "no 1.5 TB DICOM download): the ground-truth lesion boxes, and the "
           "**real DBTex-challenge team submissions**. So it validates *actual* "
           "competition detectors, end-to-end, against the official metric.\n"
           "\n"
           "*(Exam-level volume classification — running a model on the DICOM "
           "pixels — is a heavier separate effort; see the closing notes.)*"),

        md("## 1 · Environment setup\n"
           "Runs on **Colab** or **Kaggle**. Turn *Internet* **On** (to clone "
           "the project code and download the metadata). No GPU needed — FROC "
           "is pure metric computation on small CSVs."),
        code(_SETUP_PACKAGE),

        md("## 2 · Download the Duke metadata and DBTex predictions\n"
           "All public, no login, a few MB total: the validation-split "
           "ground-truth CSVs, and the DBTex challenge team predictions "
           "(`team_predictions_bothphases.zip`, ~7 MB)."),
        code("import os, urllib.request\n"
             "os.makedirs('data/duke-dbt', exist_ok=True)\n"
             "BASE = 'https://www.cancerimagingarchive.net/wp-content/uploads/'\n"
             "FILES = [\n"
             "    'BCS-DBT-file-paths-validation-v2.csv',\n"
             "    'BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv',\n"
             "    'BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv',\n"
             "    'team_predictions_bothphases.zip',\n"
             "]\n"
             "for name in FILES:\n"
             "    dst = os.path.join('data/duke-dbt', name)\n"
             "    if not os.path.exists(dst):\n"
             "        urllib.request.urlretrieve(BASE + name, dst)\n"
             "    print('%-46s %8.1f KB' % (name, os.path.getsize(dst) / 1024))"),

        md("## 3 · Load the dataset\n"
           "`DukeDBTDataset` reads the metadata CSVs into the standard case "
           "table. `require_local=False` because this run needs no DICOM pixels "
           "— only the per-view labels and the ground-truth lesion boxes."),
        code("from mammoval.data import DukeDBTDataset\n"
             "import json\n"
             "\n"
             "ds = DukeDBTDataset(root='data/duke-dbt', split='validation',\n"
             "                    require_local=False)\n"
             "print(json.dumps(ds.summary(), indent=2, default=str))\n"
             "n_boxes = sum(len(ds.lesion_boxes(c)) for c in ds.cases['case_id'])\n"
             "print('ground-truth lesion boxes:', n_boxes)"),

        md("## 4 · Detector predictions — DBTex challenge teams\n"
           "The Duke group published the DBTex1/DBTex2 challenge submissions. "
           "`read_dbtex_predictions` loads one team's detections for this split "
           "and converts the boxes to the format the FROC engine consumes. The "
           "phase-2 validation set has three teams: `nyu_bteam`, `vicorob`, "
           "`zedus`."),
        code("ZIP = 'data/duke-dbt/team_predictions_bothphases.zip'\n"
             "TEAM = 'nyu_bteam'        # or 'vicorob', 'zedus'\n"
             "\n"
             "preds = ds.read_dbtex_predictions(ZIP, team=TEAM)\n"
             "print(TEAM, '->', len(preds), 'detections over',\n"
             "      preds['case_id'].nunique(), 'volumes')\n"
             "preds.head()"),

        md("## 5 · FROC localisation validation\n"
           "`run_localization_validation` matches each detection to a "
           "ground-truth lesion under the **official Duke criterion** "
           "(`duke_dbt_hit`: centre within `max(√(W²+H²)/2, 100)` px and within "
           "`VolumeSlices/4` slices), builds the FROC curve, and reports "
           "sensitivity at 1/2/3/4 FP per volume with a case-level bootstrap CI."),
        code("from mammoval.metrics import duke_dbt_hit\n"
             "from mammoval.pipeline import run_localization_validation\n"
             "\n"
             "loc = run_localization_validation(\n"
             "    ds.assemble_froc_cases(preds), duke_dbt_hit,\n"
             "    fp_points=(1, 2, 3, 4),\n"
             "    dataset_name='Duke BCS-DBT validation split')\n"
             "print('sensitivity @ FP/volume:',\n"
             "      {k: round(v, 3) for k, v in loc['sensitivity_at_fp'].items()})\n"
             "print('mean sensitivity = %.3f  (95%% CI %.3f-%.3f)  [official Duke metric]'\n"
             "      % (loc['mean_sensitivity'], *loc['mean_sensitivity_ci']))"),

        md("### Compare all three challenge teams\n"
           "The same engine ranks every team — exactly what a validation "
           "function does when comparing candidate detectors."),
        code("import pandas as pd\n"
             "\n"
             "rows = []\n"
             "for team in ['nyu_bteam', 'vicorob', 'zedus']:\n"
             "    p = ds.read_dbtex_predictions(ZIP, team=team)\n"
             "    r = run_localization_validation(ds.assemble_froc_cases(p),\n"
             "                                    duke_dbt_hit, fp_points=(1, 2, 3, 4),\n"
             "                                    n_boot=500)\n"
             "    rows.append({'team': team,\n"
             "                 'mean_sensitivity': round(r['mean_sensitivity'], 3),\n"
             "                 'CI_low': round(r['mean_sensitivity_ci'][0], 3),\n"
             "                 'CI_high': round(r['mean_sensitivity_ci'][1], 3),\n"
             "                 **{k: round(v, 3) for k, v in r['sensitivity_at_fp'].items()}})\n"
             "pd.DataFrame(rows).sort_values('mean_sensitivity', ascending=False)"),

        md("## 6 · Validation report"),
        code("from mammoval.report import build_report\n"
             "from IPython.display import HTML\n"
             "\n"
             "os.makedirs('outputs', exist_ok=True)\n"
             "path = build_report(\n"
             "    localization_results=loc,\n"
             "    output_path='outputs/duke_dbt_froc_report.html',\n"
             "    title='3D Mammography AI - FROC Localisation (Duke BCS-DBT, '\n"
             "          + TEAM + ')',\n"
             "    extra_limitations=[\n"
             "        'Detections are a published DBTex-challenge submission, not a '\n"
             "        'live device. The validation split has ~1,163 views and 75 '\n"
             "        'biopsied-lesion boxes.',\n"
             "    ])\n"
             "HTML(open(path, encoding='utf-8').read())"),

        md("## 7 · Notes & next steps\n"
           "* **Validate your own detector.** Any DBT detector that emits a CSV "
           "with `StudyUID, View, X, Y, Width, Height, Z, Score` plugs straight "
           "in — convert it to `case_id, score, x, y, slice` (box corner → "
           "centre) and pass it to `ds.assemble_froc_cases(...)`. The FROC engine "
           "and the Duke hit criterion are unit-tested.\n"
           "* **Held-out test set.** Use `split='test'` with the matching "
           "`BCS-DBT-*-test-*` CSVs and the `phase2/test` team predictions.\n"
           "* **Exam-level classification.** Scoring the DICOM volumes themselves "
           "(cancer vs not) needs the ~1.5 TB image set: download a subset with "
           "`tcia_utils`, read volumes with the official `dcmread_image`, and "
           "wrap a 2D classifier in `SliceAggregatorClassifier`. That is a "
           "heavier, separate effort — the localisation validation above is a "
           "complete study on its own."),
    ]
    return notebook(cells)


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    for fname, nb in [("01_validation_2d_cbis_ddsm.ipynb", build_2d()),
                      ("02_validation_3d_duke_dbt.ipynb", build_3d())]:
        path = os.path.join(NB_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
