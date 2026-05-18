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
# Locate the mammoval package. Works on Colab and Kaggle, whether you
# git-cloned the repo or added it as a Kaggle dataset / uploaded the folder.
import os, sys, glob
cands = ['.', 'breast-ai-clinical-validation', '/content/breast-ai-clinical-validation']
cands += glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/*/*')
cands += glob.glob('/kaggle/working/*')
for c in cands:
    if c and os.path.isdir(os.path.join(c, 'mammoval')):
        sys.path.insert(0, os.path.abspath(c)); print('found mammoval in', c); break
else:
    raise RuntimeError(
        'mammoval package not found. Either (a) enable Internet and run '
        '!git clone <your-repo-url>, or (b) add the project folder as a '
        'Kaggle dataset via "+ Add Input".')
import mammoval
print('mammoval', mammoval.__version__, 'ready')"""


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
           "**Model:** a HuggingFace Swin-V2 breast classifier (a *baseline* — "
           "swap in any stronger model by writing one `ImageClassifier`).\n"
           "**Output:** an HTML validation report — ROC/AUC with DeLong CIs, "
           "operating points, calibration, screening behaviour, breast-density "
           "subgroup analysis, and an AI-vs-radiologist (BI-RADS) comparison."),

        md("## 1 · Environment setup\n"
           "Runs on **Google Colab** or **Kaggle**.\n"
           "\n"
           "**On Kaggle:** in the right-hand panel set *Accelerator* to a GPU "
           "and turn *Internet* **On** (needed to `pip install transformers` and "
           "download the model from the HuggingFace Hub).\n"
           "\n"
           "Get the `mammoval` package onto the machine first — either enable "
           "Internet and `!git clone` your repo, or add the project folder as a "
           "dataset via *+ Add Input*. Then run the cells below."),
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
        code("import os\n"
             "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
             "\n"
             "if ON_KAGGLE:\n"
             "    CBIS_ROOT = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'\n"
             "    assert os.path.isdir(CBIS_ROOT), (\n"
             "        'Add the awsaf49 \"CBIS-DDSM Breast Cancer Image Dataset\" '\n"
             "        'via + Add Input first.')\n"
             "else:\n"
             "    from google.colab import files\n"
             "    print('Upload your kaggle.json:'); files.upload()\n"
             "    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
             "    !pip install -q kaggle\n"
             "    !kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset\n"
             "    !unzip -q -o cbis-ddsm-breast-cancer-image-dataset.zip -d data/cbis-ddsm\n"
             "    CBIS_ROOT = 'data/cbis-ddsm'\n"
             "print('CBIS-DDSM root:', CBIS_ROOT)"),

        md("## 3 · Load the dataset into a standard case table\n"
           "`CBISDDSMDataset` joins the case-description CSVs to the JPEG mirror "
           "and aggregates abnormality rows to one image-level case "
           "(`y_true` = malignant if any abnormality is malignant)."),
        code("from mammoval.data import CBISDDSMDataset\n"
             "import json\n"
             "\n"
             "ds = CBISDDSMDataset(root=CBIS_ROOT, split='test')\n"
             "print(json.dumps(ds.summary(), indent=2, default=str))\n"
             "ds.cases.head()"),

        md("## 4 · Load the pretrained model — treated as a black box\n"
           "`HFImageClassifier` wraps a HuggingFace checkpoint and auto-detects "
           "which class index is malignant. The default is a verified Swin-V2 "
           "breast classifier. **This is a baseline**: it was fine-tuned on "
           "Mini-DDSM-derived images, so expect domain shift on CBIS-DDSM — the "
           "report will quantify exactly that."),
        code("from mammoval.models import HFImageClassifier\n"
             "\n"
             "model = HFImageClassifier()    # or HFImageClassifier('your/model-id')\n"
             "print('model :', model.name)\n"
             "print('labels:', model.id2label, '| malignant index =', model.malignant_index)"),

        md("## 5 · Run inference → predictions table\n"
           "`score_dataset` runs the model over every case and returns the case "
           "table with a `y_score` column. Set `LIMIT=None` for the full test "
           "split; a few hundred cases is enough for a quick pass."),
        code("from mammoval.models import score_dataset\n"
             "\n"
             "LIMIT = 400          # set to None for the whole test split\n"
             "preds = score_dataset(model, ds, limit=LIMIT)\n"
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
             "        'The default Swin-V2 model is a baseline trained on Mini-DDSM-'\n"
             "        'derived images; treat its absolute numbers as a lower bound.',\n"
             "    ])\n"
             "HTML(open(path, encoding='utf-8').read())"),

        md("## Notes & next steps\n"
           "* **Stronger model.** Replace `HFImageClassifier()` with the RSNA-2023 "
           "winning ensemble or a Mammo-CLIP linear probe — only the adapter "
           "changes; the validation is identical.\n"
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
        md("# Clinical Validation of 3D Mammography AI — Duke BCS-DBT\n"
           "\n"
           "Standalone validation on **digital breast tomosynthesis (DBT)** — 3D "
           "mammography — using the Duke *Breast-Cancer-Screening-DBT* dataset.\n"
           "\n"
           "Two studies:\n"
           "1. **Exam-level classification** — a `SliceAggregatorClassifier` lifts "
           "a validated 2D model onto the tomosynthesis slice stack (how DBT "
           "triage products actually operate). Fully runnable end-to-end.\n"
           "2. **Lesion localisation (FROC)** — implemented to the *official Duke "
           "BCS-DBT* true-positive criterion and 1/2/3/4 FP-per-volume operating "
           "points. No DBT detector has public weights, so this notebook scores "
           "the FROC engine on a detector's `predictions.csv` (a simulated, "
           "clearly-labelled stand-in here — drop in a real one to get real "
           "numbers).\n"
           "\n"
           "> **Heads-up:** the full collection is ~1.5 TB. Work from a *subset* — "
           "the cells below pull a small number of validation-split volumes."),

        md("## 1 · Environment setup\n"
           "Runs on **Colab** or **Kaggle**. On Kaggle, set *Accelerator* to a "
           "GPU and turn *Internet* **On** (needed for the installs, the TCIA "
           "download and the model). Get the `mammoval` package onto the "
           "machine first — `!git clone` your repo, or add it as a dataset."),
        code("!pip install -q transformers timm pydicom pylibjpeg pylibjpeg-libjpeg "
             "pylibjpeg-openjpeg tcia_utils\n"
             "# torch / torchvision are pre-installed on Colab and Kaggle."),
        code(_SETUP_PACKAGE),

        md("## 2 · Download a subset of Duke BCS-DBT\n"
           "The dataset lives on TCIA: collection **`Breast-Cancer-Screening-DBT`** "
           "(DOI 10.7937/E4WT-CD02). You need two things:\n"
           "\n"
           "1. **The three metadata CSVs** — `BCS-DBT-file-paths-*.csv`, "
           "`BCS-DBT-labels-*.csv`, `BCS-DBT-boxes-*.csv` — from the TCIA "
           "collection page (small files). Place them under `data/duke-dbt/`.\n"
           "2. **A subset of DICOM volumes.** The cell below uses `tcia_utils` to "
           "fetch a handful of validation-split series. Downloading all of it is "
           "neither necessary nor practical in Colab.\n"
           "\n"
           "Edit `N_SERIES` to control how much you pull."),
        code("import os, glob, shutil\n"
             "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
             "os.makedirs('data/duke-dbt', exist_ok=True)\n"
             "\n"
             "# --- DICOM subset via the TCIA API (needs Internet ON) ---------------\n"
             "from tcia_utils import nbia\n"
             "N_SERIES = 40        # number of DBT volumes to pull (keep small)\n"
             "series = nbia.getSeries(collection='Breast-Cancer-Screening-DBT')\n"
             "nbia.downloadSeries(series[:N_SERIES], path='data/duke-dbt')\n"
             "print('downloaded', N_SERIES, 'series')\n"
             "\n"
             "# --- metadata CSVs: BCS-DBT-{file-paths,labels,boxes}-*.csv ----------\n"
             "# Get these small files from the TCIA collection page.\n"
             "if ON_KAGGLE:\n"
             "    # Add the CSVs as a Kaggle dataset (+ Add Input), then this copies\n"
             "    # them next to the DICOMs so the loader sees one root:\n"
             "    for src in glob.glob('/kaggle/input/*/BCS-DBT-*.csv'):\n"
             "        shutil.copy(src, 'data/duke-dbt')\n"
             "    print('metadata CSVs:', sorted(os.listdir('data/duke-dbt')))\n"
             "else:\n"
             "    from google.colab import files\n"
             "    print('Upload the three BCS-DBT-*.csv metadata files:')\n"
             "    for name in files.upload():\n"
             "        os.replace(name, os.path.join('data/duke-dbt', name))"),

        md("## 3 · Load the dataset\n"
           "`DukeDBTDataset` locates the CSVs by glob (filenames vary across TCIA "
           "snapshots) and keeps only the views whose DICOM is actually on disk — "
           "the right behaviour for a partial download."),
        code("from mammoval.data import DukeDBTDataset\n"
             "import json\n"
             "\n"
             "ds = DukeDBTDataset(root='data/duke-dbt', split='validation',\n"
             "                    require_local=True)\n"
             "print(json.dumps(ds.summary(), indent=2, default=str))\n"
             "ds.cases.head()"),

        md("## 4 · Build the 3D classifier\n"
           "A DBT exam is a stack of reconstructed slices. "
           "`SliceAggregatorClassifier` scores a sample of slices with a 2D model "
           "and pools them (top-k mean). The 2D base model is the same verified "
           "classifier validated in the 2D notebook."),
        code("from mammoval.models import HFImageClassifier, SliceAggregatorClassifier\n"
             "\n"
             "base_2d = HFImageClassifier()\n"
             "model = SliceAggregatorClassifier(base_2d, n_slices=12,\n"
             "                                  aggregate='top_k_mean', top_k=3)\n"
             "print('volume model:', model.name)"),

        md("## 5 · Run inference → predictions table\n"
           "DBT volumes are large; keep `LIMIT` modest in Colab."),
        code("from mammoval.models import score_dataset\n"
             "\n"
             "LIMIT = 40\n"
             "preds = score_dataset(model, ds, limit=LIMIT).dropna(subset=['y_score'])\n"
             "preds[['case_id','y_true','y_score','view','label_class']].head()"),

        md("## 6 · Exam-level classification validation"),
        code("from mammoval.pipeline import ClassificationConfig, run_classification_validation\n"
             "\n"
             "cfg = ClassificationConfig(\n"
             "    subgroup_cols=('view', 'laterality'),\n"
             "    target_specificities=(0.90, 0.96),\n"
             "    is_probability=True,\n"
             "    dataset_name='Duke BCS-DBT (validation subset)',\n"
             ")\n"
             "results = run_classification_validation(preds, cfg)\n"
             "d = results['discrimination']\n"
             "print('AUC = %.3f   95%% CI %.3f-%.3f' % (d['auc'], *d['auc_ci']))"),

        md("## 7 · Lesion localisation — FROC (official Duke metric)\n"
           "FROC scores each *mark*: a mark is a true positive only if its centre "
           "is within `max(√(W²+H²)/2, 100)` px of a ground-truth lesion **and** "
           "within `VolumeSlices/4` slices of it — the exact `duke_dbt_hit` "
           "criterion. The mean sensitivity at 1/2/3/4 FP/volume is the official "
           "DBTex ranking metric.\n"
           "\n"
           "A real DBT detector emits `predictions.csv` with columns "
           "`case_id, score, x, y, slice`. No such model has public weights, so "
           "the cell below **simulates** that file from the ground-truth boxes — "
           "clearly an illustrative stand-in. Replace `predictions` with a real "
           "detector's output to get real localisation numbers."),
        code("import numpy as np, pandas as pd\n"
             "\n"
             "# ---- ILLUSTRATIVE simulated detector output (replace with real CSV) --\n"
             "rng = np.random.default_rng(0)\n"
             "rows = []\n"
             "for cid in ds.cases['case_id']:\n"
             "    for les in ds.lesion_boxes(cid):          # GT lesions -> noisy hits\n"
             "        if rng.random() < 0.7:\n"
             "            rows.append(dict(case_id=cid, score=float(rng.beta(5,2)),\n"
             "                             x=les['x']+rng.normal(0,40),\n"
             "                             y=les['y']+rng.normal(0,40),\n"
             "                             slice=les['slice']+rng.normal(0,4)))\n"
             "    for _ in range(rng.poisson(1.2)):          # false marks\n"
             "        rows.append(dict(case_id=cid, score=float(rng.beta(2,5)),\n"
             "                         x=rng.uniform(0,2000), y=rng.uniform(0,2000),\n"
             "                         slice=rng.uniform(0,60)))\n"
             "predictions = pd.DataFrame(rows)\n"
             "# ----------------------------------------------------------------------\n"
             "\n"
             "from mammoval.metrics import duke_dbt_hit\n"
             "from mammoval.pipeline import run_localization_validation\n"
             "\n"
             "froc_cases = ds.assemble_froc_cases(predictions)\n"
             "loc = run_localization_validation(froc_cases, duke_dbt_hit,\n"
             "                                  fp_points=(1,2,3,4),\n"
             "                                  dataset_name='Duke BCS-DBT (localisation)')\n"
             "print('FROC mean sensitivity = %.3f  (official Duke metric)'\n"
             "      % loc['mean_sensitivity'])"),

        md("## 8 · Generate the HTML validation report"),
        code("from mammoval.report import build_report\n"
             "from IPython.display import HTML\n"
             "\n"
             "os.makedirs('outputs', exist_ok=True)\n"
             "path = build_report(\n"
             "    results, localization_results=loc,\n"
             "    output_path='outputs/duke_dbt_validation_report.html',\n"
             "    title='3D Mammography AI — Clinical Validation (Duke BCS-DBT)',\n"
             "    extra_limitations=[\n"
             "        'Exam scores come from a 2D model aggregated over slices, not '\n"
             "        'a native 3D model — a deliberate, documented stand-in.',\n"
             "        'FROC here is computed on a SIMULATED detector output; the '\n"
             "        'curve is illustrative until a real predictions.csv is supplied.',\n"
             "        'Only a small validation subset was downloaded — widen it for '\n"
             "        'a stable estimate.',\n"
             "    ])\n"
             "HTML(open(path, encoding='utf-8').read())"),

        md("## Notes & next steps\n"
           "* **Real localisation numbers** need a trained DBT detector "
           "(`mateuszbuda/duke-dbt-detection` or `mazurowski-lab/DBTex-baseline` "
           "provide training code). Export its marks as `predictions.csv` and "
           "re-run section 7 — nothing else changes.\n"
           "* **DICOM fidelity.** `DukeDBTDataset.load` uses a plain pydicom read. "
           "For localisation that must align with the ground-truth boxes, use the "
           "official `dcmread_image` (laterality correction) from "
           "`mazurowski-lab/duke-dbt-data`.\n"
           "* The FROC engine and `duke_dbt_hit` criterion are unit-tested against "
           "the published metric definition — see `tests/test_localization.py`."),
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
