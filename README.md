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
│   ├── train_xgb.py           # Train XGBoost model
│   ├── train_catboost.py      # Train CatBoost model
│   ├── train_group3_ar_gru.py # Reproduce Group 3 autoregressive GRU baseline
│   └── predict.py             # Generate submission file
├── experiments/               # Run metadata; large artifacts ignored
├── models/                    # Saved model files (not in git)
├── submissions/               # Generated CSVs (not in git)
├── docs/                      # Structured project docs and experiment readouts
│   ├── status/                # Current state, leaderboard, submission plans
│   ├── experiments/           # Model-family and Kaggle probe readouts
│   ├── validation/            # Blind backtests, drift, distribution reports
│   ├── presentations/         # Marp slides and exported PDFs
│   ├── project/               # Assignment notes and planning references
│   └── engineering/           # Implementation notes
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
# Train a LightGBM direct-horizon baseline
uv run python src/train.py

# Train an XGBoost direct-horizon diversity model
uv run python src/train_xgb.py --experiment-name xgb_v1

# Train the CatBoost robustness model used in the best current public blend
uv run python src/train_catboost.py \
  --experiment-name catboost_lean_tail2737_regularized_500 \
  --feature-profile lean \
  --train-tail-days 2737 \
  --validation-mode rolling_origin \
  --rolling-folds 3 \
  --regularized \
  --recency-half-life-days 1095 \
  --iterations 500

# Reproduce the Group 3 presentation's autoregressive GRU architecture
# This is an experimental neural baseline, not the current submission path.
uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_smoke \
  --max-rows 60000 \
  --max-regions 8 \
  --train-tail-days 730 \
  --epochs 1 \
  --batch-size 64

# Run a Kaggle-like blind-window backtest against a saved run
uv run python scripts/run_blind_backtest.py \
  --run-dir experiments/<run_id> \
  --origins "5,13,26" \
  --history-tail-days 1100

# Poison blind-window scores and verify final-row features do not change
uv run python scripts/leakage_sentinel.py \
  --origin 5 \
  --history-tail-days 1100 \
  --feature-profile micro

# Compare train-tail candidates against the current test-window distribution
uv run python scripts/drift_report.py \
  --tail-days "1095,1825,2737,3650,0" \
  --out docs/validation/drift_report_20260521.md

# Generate predictions from a specific saved run
uv run python src/predict.py --run-dir experiments/<run_id>

# Blend LGB/XGB/CatBoost submissions
uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --cat submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv \
  --out submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  --weights 'lgb=0.35;xgb=0.35;cat=0.30'

# Fit validation-driven constrained blend weights from OOF/blind predictions
uv run python scripts/fit_blend_weights.py \
  --preds lgb=validation/lgb.csv,xgb=validation/xgb.csv,cat=validation/cat.csv \
  --target validation/target.csv \
  --anchor 'lgb=0.35;xgb=0.35;cat=0.30' \
  --lambda-reg 0.05 \
  --bootstrap 100

# Generate an ablation command matrix without launching training
uv run python scripts/feature_ablation.py \
  --model-family lightgbm \
  --groups score_history,climatology,rolling,ewm,long_drought_proxy \
  --feature-profile micro \
  --out docs/experiments/feature_ablation_20260521_commands.md

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

Current model-selection state is tracked in `docs/status/current_state.json`. The
current best legal public submission is
`submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` with public MAE
`0.8124`.

The full document map is in `docs/README.md`.

Before updating the report, slides, or experiment notes, run:

```bash
uv run python scripts/check_current_state.py
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

The current implementation treats drought prediction as a leakage-aware
direct multi-horizon panel-regression problem:

- One independent model is trained for each forecast week, week 1 through week 5.
- Validation is separated from the saved final model. Trainers support
  `--final-train-mode last_fold`, `fold_ensemble`, and `refit_full`; the default
  is `refit_full`, which uses CV to choose a stable tree count and then retrains
  on all legal rows available to that horizon.
- Feature profiles (`micro`, `lean`, `full`) control memory use and feature count.
- Features include meteorological lags, rolling statistics, EWMA signals,
  long-window drought proxies, calendar encodings, climatology anomalies, and
  optional 91-day-gapped score-history features.
- Model families currently implemented: LightGBM, XGBoost, and CatBoost.
- `src/train_group3_ar_gru.py` reproduces Group 3's AR-GRU presentation baseline:
  a 91-day weather sequence, region/date static context, a 2-layer hidden-64
  GRU encoder, and a five-step GRUCell decoder with previous-score feedback and
  teacher forcing. It is kept separate from `src/predict.py` because current
  evidence still favors the tree ensemble path.
- `src/ensemble.py` supports two-model and three-model convex blends, including
  horizon-specific weights.
- `scripts/run_blind_backtest.py` rebuilds a pseudo Kaggle test window inside
  train data by masking a 91-day score window, rebuilding features with
  time-safe climatology/region statistics, and evaluating five future weeks.
  It warns when `--history-tail-days` is too short for the selected feature
  profile, so smoke tests are not mistaken for model-selection evidence.
- `scripts/leakage_sentinel.py` poisons hidden blind-window scores and asserts
  that final-row pseudo-test features are unchanged.
- `scripts/drift_report.py` compares candidate train tails to the real test
  window with PSI, KS, quantile-Wasserstein distance, weather-only adversarial
  AUC, and weather+region/month adversarial AUC.
- `scripts/fit_blend_weights.py` fits non-negative per-horizon blend weights
  with anchor regularization, optional model caps, and bootstrap stability, so
  public-LB hand tuning is not the only signal.
- `--drop-feature-groups` supports coarse ablations such as
  `score_history`, `climatology`, `calendar`, `short_lag`, `rolling`, `ewm`,
  `long_drought_proxy`, `domain_indices`, `region_stats`, and CatBoost
  `region_id`.

The old separate score-estimation experiment is not the current implementation path.
The active pipeline uses saved feature options from each run and rebuilds the
same direct-horizon feature table during inference.

See `src/features.py` for details.

## 6. Iterating on Algorithms

The repository is organized so each model family can be compared under a stable
artifact contract while preserving previous experiment lineage.

Recommended workflow:

```bash
# Current baseline
uv run python src/train.py \
  --model-family lightgbm_two_stage \
  --experiment-name lgbm_v1

# XGBoost baseline
uv run python src/train_xgb.py \
  --experiment-name xgb_v1

# CatBoost direct-horizon model with native categorical handling
uv run python src/train_catboost.py \
  --experiment-name catboost_v1 \
  --feature-profile lean \
  --validation-mode rolling_origin \
  --regularized

# Submission-like pseudo-private validation for the trained run
uv run python scripts/run_blind_backtest.py \
  --run-dir experiments/<run_id> \
  --origins "5,13,26" \
  --history-tail-days 1100
```

Use rolling-origin metrics as fast diagnostics, then use blind backtest and
drift report evidence before promoting a run to a Kaggle submission candidate.
Do not compare holdout, rolling-origin, blind-backtest, public LB, and private
LB values as if they were the same metric scale.

The `*_two_stage` model-family labels are retained for backward compatibility
with existing run directories. They should be read as historical identifiers,
not as a claim about the current training flow.

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
- Max **6 submissions per day** (updated from 3; resets 8 AM Taiwan time)
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
