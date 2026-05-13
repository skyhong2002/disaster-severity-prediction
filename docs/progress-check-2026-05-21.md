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

We implemented a reproducible **Direct Forecasting Ensemble**.

1. Build temporal features from weather, calendar, and region statistics.
2. Incorporate newly introduced domain-specific Drought and Dryness Indices.
3. Train direct horizon models for both **LightGBM** and **XGBoost**.
4. Blend the predictions (Ensemble) to generate a robust Kaggle submission.

---

# Feature Engineering

Current model input has a diverse set of features.

- **Weather dynamics**: lags, rolling mean/std, exponentially weighted means.
- **Time and location**: calendar encodings and region-level historical score statistics.
- **Domain Features**: Drought Index (temp/precip) and Dryness Index (temp range * max temp) to capture explicit drought conditions.
- *Note*: Historical score lags were completely removed due to severe train-test noise and discrepancy.

---

## Direct Multi-Step Forecasting

Train five independent models for `week1`, `week2`, `week3`, `week4`, and `week5`. Each model predicts one future week directly.

## Ensemble Strategy

Train both **LightGBM** (leaf-wise growth) and **XGBoost** (depth-wise growth) on the exact same chronological features. The final submission is a 50/50 blend of both, capturing maximum model diversity.

---

# Local Validation Results (Chronological Split)

We fixed a critical **Time-Leakage** bug by switching to a strict chronological holdout (last 20% of time).

| Horizon | LGBM MAE | XGB MAE |
|---|---:|---:|
| Week 1 | 0.6772 | 0.7127 |
| Week 2 | 0.6729 | 0.7365 |
| Week 3 | 0.6795 | 0.7427 |
| Week 4 | 0.6761 | 0.7297 |
| Week 5 | 0.6795 | 0.7384 |
| **Average** | **0.6770** | **0.7320** |

---

# Kaggle Public Result

Public leaderboard uses about **40%** of the test data.

| Method | Public MAE | Insight |
|---|---:|---|
| Baseline 3 | 0.8056 | - |
| Team 20 | 0.8062 | - |
| **Team 5 (v0 Leaky)** | **0.8094** | Data leaked, inflated score |
| **Team 5 (v1 Ensemble)**| **0.8232** | Kept noisy history; best generalization |
| **Team 5 (Strategy B)** | **0.8640** | Perfect Local MAE (0.677), overfit to test |
| Baseline 2 | 0.8623 | - |
| Baseline 1 | 0.9117 | - |

---

# Engineering Progress

The codebase is prepared for future model iterations.

- Main scripts: `src/train.py`, `src/predict.py`, `src/features.py`.
- Each training run saves `config.json`, `metrics.json`, and submission metadata under `experiments/<run_id>/`.
- Large model files and generated submissions are kept out of git, while small metadata files can support report updates.

This avoids overwriting the current baseline when we test new algorithms.

---

# Key Discoveries & Challenges

- **Data Leakage Solved**: Initial region-based splits caused artificial local MAE (0.21) vs LB MAE (0.80). Chronological splitting aligned them.
- **Train-Test Discrepancy**: Attempting to reconstruct missing test labels created $\approx 0.74$ MAE noise, poisoning our lag features.
- Removing label-dependent features entirely yielded a much cleaner, memory-efficient pipeline.

---

# Next Steps

Before the final deadline, we plan to:

- Tune XGBoost hyperparameters formally to close the gap with LightGBM.
- Try more complex rolling windows and feature selection techniques.
- Explore sequence-based models like Temporal Fusion Transformer.
- Finish documenting the methodology in the IEEE LaTeX report.

---

# Summary

Completed so far:

- Resolved critical Train-Test Leakage and Feature Discrepancies.
- Transitioned to Pure Direct Forecasting with Domain Features (Drought/Dryness Index).
- Implemented and evaluated XGBoost + LightGBM Ensemble.
- Kaggle Public MAE: **0.8232** (clean, reproducible, un-leaked).

Main focus next: hyperparameter tuning and model refinement.

## Thank you (∠·ω )⌒★
