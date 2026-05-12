"""
download_data.py
Download competition data via kagglehub.
Usage:
    export KAGGLE_API_TOKEN=KGAT_xxxx
    python3 src/download_data.py
"""
import os
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "data-mining-2026-final-project"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print(f"Downloading competition: {COMPETITION}")
path = kagglehub.competition_download(COMPETITION)
print(f"Downloaded to cache: {path}")

# Copy files to data/
src = Path(path)
for csv_file in src.rglob("*.csv"):
    dest = DATA_DIR / csv_file.name
    shutil.copy2(csv_file, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"  → {dest.name}  ({size_mb:.1f} MB)")

print("\nDone! Files in data/:")
for f in sorted(DATA_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
