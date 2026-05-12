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
│   ├── train.py               # Train LightGBM model
│   └── predict.py             # Generate submission file
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

# Step 2: Generate predictions
uv run python src/predict.py

# Output: submissions/submission_<timestamp>.csv
# → Upload this file to Kaggle
```

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

---

## 6. Kaggle Submission Rules
- Team name must be **Team 5**
- Max **3 submissions per day** (resets 8 AM Taiwan time)
- Deadline: **June 10, 2026 11:55 PM**
- Your best 2 public LB submissions are auto-selected for private LB

---

## 7. Report
Report is in `report/main.tex` (IEEE format).  
Upload to [Overleaf](https://www.overleaf.com) → New Project → Upload `.tex` file.  
Submit PDF via E3 as `DM_project_Group_5.pdf` by June 10.

---

## Contact
TA email: nycu.dm.ta@gmail.com
