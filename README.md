# Natural Disaster Severity Prediction
**NYCU Data Mining Spring 2026 — Group 5**

Kaggle Competition: [data-mining-2026-final-project](https://www.kaggle.com/competitions/data-mining-2026-final-project)

---

## Task
Predict drought severity scores (0–5) for **5 future weeks** per geographic region,
using 91 days of historical meteorological data.  
Metric: **MAE** (lower = better).

## Project Structure

```
disaster-severity-prediction/
├── data/                      # ← NOT in git (large files, download below)
│   ├── train.csv              #   1.1 GB
│   ├── test.csv               #   18 MB
│   └── sample_submission.csv  #   (committed as reference)
├── notebooks/                 # EDA & experiments
├── src/
│   ├── features.py            # Feature engineering
│   ├── experiment_utils.py    # Versioned experiment tracking helpers
│   ├── train.py               # Train LightGBM model
│   └── predict.py             # Generate submission file
├── experiments/               # Run metadata; large artifacts ignored
├── models/                    # Saved model files (not in git)
├── submissions/               # Generated CSVs (not in git)
├── report/
│   └── main.tex               # IEEE report (Overleaf)
├── pyproject.toml
└── README.md
```

---

## 1. Environment Setup

```bash
# Clone the repo
git clone <repo-url>
cd disaster-severity-prediction

# Install uv (if not already installed)
# macOS / Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment & install dependencies
uv sync
```

---

## 2. Download the Dataset

You need a **Kaggle API Token**. Get one from https://www.kaggle.com/settings → API → Create New Token.

### Option A — kagglehub (recommended)
```bash
# Set your Kaggle token as an env variable
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxx

# Then run the download script
uv run python src/download_data.py
```

### Option B — kaggle CLI
```bash
# Put your kaggle.json in ~/.kaggle/ (chmod 600)
kaggle competitions download -c data-mining-2026-final-project -p data/
unzip data/data-mining-2026-final-project.zip -d data/
```

After downloading, your `data/` folder should contain:
- `train.csv` (~1.1 GB)
- `test.csv` (~18 MB)
- `sample_submission.csv`

---

## 3. Run the Pipeline

```bash
# Step 1: Feature engineering + Training
uv run python src/train.py

# Optional: name the experiment run
uv run python src/train.py --experiment-name lgbm_v1

# Step 2: Generate predictions
uv run python src/predict.py

# Optional: predict with a specific saved run
uv run python src/predict.py --run-dir experiments/<run_id>

# Output:
# - submissions/submission_<timestamp>_<run_id>.csv
# - experiments/<run_id>/config.json
# - experiments/<run_id>/metrics.json
# - experiments/<run_id>/submission_metadata.json
# → Upload this file to Kaggle
```

`src/train.py` saves every run under `experiments/<run_id>/` and also updates
the legacy latest-model files under `models/` for compatibility. `src/predict.py`
defaults to the latest experiment recorded in `experiments/latest.txt`.

---

## 4. Dataset Overview

| Property | Train | Test |
|---|---|---|
| Days per region | 5,480 | 91 |
| Number of regions | 2,248 | 2,248 |
| Features | 14 meteorological | 14 meteorological |
| Target `score` | 0–5 (weekly, rest NaN) | — |

**Meteorological features:** `prec`, `surf_pre`, `humidity`, `tmp`, `dp_tmp`,
`wb_tmp`, `tmp_max`, `tmp_min`, `tmp_range`, `surf_tmp`, `wind`, `wind_max`,
`wind_min`, `wind_range`

---

## 5. Method Overview

We use **LightGBM** with rich temporal feature engineering:
- Lag features (7, 14, 21, 28, 35 days)
- Rolling mean/std over multiple windows
- Score history (last 1–5 weekly scores)
- Calendar features (week, month, season)

See `src/features.py` for details.

## 6. Iterating on Algorithms

The repository is organized so the current LightGBM implementation can be kept
as a reproducible baseline while future methods are added.

Recommended workflow:

```bash
# Current baseline
uv run python src/train.py \
  --model-family lightgbm_two_stage \
  --experiment-name lgbm_v1

# Future algorithm example
uv run python src/train.py \
  --model-family xgboost_direct \
  --experiment-name xgb_v1
```

For each algorithm version, keep the same artifact contract:

- `experiments/<run_id>/config.json`: parameters, feature groups, validation split
- `experiments/<run_id>/metrics.json`: local validation metrics
- `experiments/<run_id>/submission_metadata.json`: prediction statistics and output paths
- `experiments/<run_id>/models/`: model files, ignored by git
- `experiments/<run_id>/submissions/`: generated Kaggle CSVs, ignored by git

When a future method changes feature engineering, validation, ensembling, or
model family, create a new `--experiment-name` instead of overwriting the
previous run. The report can then be updated by comparing the JSON summaries.

---

## 7. Kaggle Submission Rules
- Team name must be **Team 5**
- Max **3 submissions per day** (resets 8 AM Taiwan time)
- Deadline: **June 10, 2026 11:55 PM**
- Your best 2 public LB submissions are auto-selected for private LB

---

## 8. Report
Report is in `report/main.tex` (IEEE format).  
Upload to [Overleaf](https://www.overleaf.com) → New Project → Upload `.tex` file.  
Submit PDF via E3 as `DM_project_Group_5.pdf` by June 10.

---

## Contact
TA email: nycu.dm.ta@gmail.com
