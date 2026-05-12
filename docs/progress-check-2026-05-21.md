---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 29px;
  }
  h1 {
    font-size: 42px;
  }
  h2 {
    font-size: 32px;
  }
  table {
    font-size: 23px;
  }
---

# Progress Check

## Natural Disaster Severity Prediction

**Group 5**  
Hsin-Yu Chen, Wei-Hsin Hung, Sky Shih-Kai Hong

May 21, 2026

GitHub: https://github.com/skyhong2002/disaster-severity-prediction

---

# Task and Data

Predict each region's drought severity for the **next five weeks** from the 91-day test window.

| Property | Value |
|---|---:|
| Regions | 2,248 |
| Train rows | 12.3M |
| Train days / region | 5,480 |
| Test days / region | 91 |
| Meteorological features | 14 |

Metric: **MAE**, lower is better.

---

# Main Data Observation

The target `score` is weekly, but the meteorological data is daily.

- About **6/7 daily score values are NaN**, so labels are sparse relative to features.
- The core modeling issue is aligning **daily weather signals** with **weekly drought severity**.
- Test data has no `score`, but score history is still potentially useful for forecasting.

---

# Current Pipeline

We implemented a reproducible **two-stage LightGBM baseline**.

1. Build temporal features from weather, calendar, region statistics, and score history.
2. Reconstruct score-history signals for the test window without using unavailable labels.
3. Train five direct models, one for each future week.
4. Generate a Kaggle-compatible submission and save experiment metadata.

---

# Feature Engineering

Current Stage-2 model input has **327 features**.

- Weather dynamics: lags, rolling mean/std, exponentially weighted means.
- Time and location context: calendar encodings and region-level historical score statistics.
- Drought history: lagged weekly scores and rolling score statistics.

These features are designed for tabular gradient boosting rather than deep sequence modeling.

---

# Two-Stage Modeling

## Stage 1: Score Reconstructor

Predict observed weekly `score` from weather, calendar, and region-level features. This creates estimated score-history signals for the test window.

## Stage 2: Direct Multi-Step Forecasting

Train five LightGBM models: `week1`, `week2`, `week3`, `week4`, and `week5`. Each model predicts one future week directly.

---

# Local Validation Results

| Model / Horizon | MAE |
|---|---:|
| Stage-1 score reconstructor | 0.4869 |
| Stage-2 week 1 | 0.1412 |
| Stage-2 week 2 | 0.1877 |
| Stage-2 week 3 | 0.2263 |
| Stage-2 week 4 | 0.2578 |
| Stage-2 week 5 | 0.2861 |
| **Stage-2 average** | **0.2198** |

Longer forecast horizons are harder: MAE increases from week 1 to week 5.

---

# Kaggle Public Result

Public leaderboard uses about **40%** of the test data.

| Method | Public MAE |
|---|---:|
| Baseline 3 | 0.8056 |
| Team 20 | 0.8062 |
| **Team 5** | **0.8094** |
| Baseline 2 | 0.8623 |
| Baseline 1 | 0.9117 |

First submission is close to Baseline 3 and better than Baseline 1/2.

---

# Engineering Progress

The codebase is prepared for future model iterations.

- Main scripts: `src/train.py`, `src/predict.py`, `src/features.py`.
- Each training run saves `config.json`, `metrics.json`, and submission metadata under `experiments/<run_id>/`.
- Large model files and generated submissions are kept out of git, while small metadata files can support report updates.

This avoids overwriting the current baseline when we test new algorithms.

---

# Current Challenges

- Weekly labels are sparse, while input features are daily and high-volume.
- Public leaderboard is partial feedback and may not match the private leaderboard.
- Large training data requires memory-aware feature engineering and model training.
- Score reconstruction is useful, but reconstruction errors can propagate into final forecasts.

---

# Next Steps

Before the final deadline, we plan to:

- Build stronger local validation based on the 91-day test-window setting.
- Add self-defined baselines such as last score, region mean, and moving average.
- Try XGBoost, LightGBM tuning, weighted ensembling, and feature ablation studies.
- Keep report text, code, experiment metadata, and Kaggle submissions synchronized.

---

# Summary

Completed so far:

- Data pipeline and feature engineering.
- Two-stage LightGBM baseline.
- First Kaggle submission with public MAE **0.8094**.
- Experiment tracking structure for future iterations.

Main focus next: improve validation reliability and iterate models systematically.

## Thank you
