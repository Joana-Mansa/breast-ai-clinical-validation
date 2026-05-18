"""Download the CBIS-DDSM JPEG mirror from Kaggle (for local / non-Colab runs).

Prerequisites
-------------
* a free Kaggle account;
* ``~/.kaggle/kaggle.json`` credentials (Kaggle -> Settings -> Create New API
  Token);
* ``pip install kaggle``.

Usage
-----
    python scripts/download_cbis_ddsm.py --out data/cbis-ddsm

The JPEG mirror (``awsaf49/cbis-ddsm-breast-cancer-image-dataset``) is ~6 GB,
versus ~160 GB for the raw DICOM collection.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

SLUG = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/cbis-ddsm",
                        help="destination directory")
    parser.add_argument("--keep-zip", action="store_true",
                        help="keep the downloaded .zip after extraction")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {SLUG} -> {args.out}")
    try:
        subprocess.check_call([sys.executable, "-m", "kaggle", "datasets",
                               "download", "-d", SLUG, "-p", args.out])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(f"Kaggle download failed ({exc}). Check that the `kaggle` "
                 f"package is installed and ~/.kaggle/kaggle.json exists.")

    for fname in os.listdir(args.out):
        if fname.endswith(".zip"):
            zpath = os.path.join(args.out, fname)
            print(f"Extracting {fname} ...")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(args.out)
            if not args.keep_zip:
                os.remove(zpath)

    print(f"Done. Point CBISDDSMDataset(root='{args.out}') at this folder.")


if __name__ == "__main__":
    main()
