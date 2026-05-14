---
marp: true
theme: default
paginate: true
size: 16:9
_class: lead
style: |
  section {
    font-family: "Inter", "Roboto", "Segoe UI", sans-serif;
  }
  h1, h2, h3, h4, h5, h6, strong {
    font-weight: 700;
  }
  code, pre {
    font-family: "Menlo", "Consolas", monospace;
  }
---

# Progress Check Presentation

## Natural Disaster Severity Prediction

**Group 5**  
Hsin-Yu Chen, Wei-Hsin Hung, Sky Shih-Kai Hong

May 21, 2026

---

# Task Description & Dataset Overview

**Objective:** Predict the weekly drought severity level for 2,248 distinct regions over a 5-week forecasting horizon, following a 91-day blind gap.

| Property | Value |
|---|---:|
| Total Regions | 2,248 |
| Training Records | 12,319,040 |
| Weekly Labeled Targets | 1,746,696 |
| Blind Test Period | 91 days |
| Meteorological Features | 14 |

**Evaluation Metric:** Mean Absolute Error (MAE). Lower is better.

---

# Data Observations & Critical Challenges

The primary challenge lies in the discrepancy between the temporal resolution of features and targets.

- **Sparsity:** The target variable (`score`) is recorded only once per week, rendering approximately 85% of target observations as NaN.
- **Temporal Alignment:** Extracting meaningful long-term trends from daily meteorological signals to predict weekly aggregated severity.
- **The Blind Gap:** The 91-day test period completely lacks ground-truth severity scores. Consequently, direct autoregressive feature construction introduces severe Train-Test Discrepancy.

---

# Current Pipeline Architecture

We have established a reproducible, two-stage **Direct Horizon Forecasting Pipeline**.

1. **Feature Engineering:** Extract temporal patterns via lag windows, rolling statistics, and exponentially weighted moving averages (EWMA).
2. **Domain-Specific Indicators:** Synthesize specialized meteorological indices, including drought index approximations and dewpoint spreads.
3. **Multi-Horizon Modeling:** Train five independent models—one for each forecasting horizon (Week 1 through Week 5)—to prevent recursive error accumulation.
4. **Ensembling & Tracking:** Support integration of LightGBM and XGBoost via weighted ensembling, coupled with strict version control for experiment tracking.

---

# Feature Engineering Strategies

The pipeline supports configurable feature profiles to balance predictive capacity and computational constraints:

- **Meteorological Dynamics:** Short-to-long term rolling means and variances.
- **Temporal Encodings:** Cyclical transformations (sine/cosine) of month, week, and day-of-year to capture seasonality.
- **Domain Synthetics:** Extended drought indices and dryness approximations.
- **Gap-Aware Autoregression:** Historical score features strictly computed prior to the 91-day gap to prevent data leakage.
- **Current Optimal Profile:** The `lean` profile yields the best computational efficiency without sacrificing significant representational power.

---

## Direct Multi-Step Forecasting

Independent regressors are allocated for each temporal horizon (`target_week1` to `target_week5`), optimizing specific lag dependencies and mitigating recursive bias.

## Validation Strategy

We transitioned from a random holdout split to a strict **Chronological Holdout Split**. This fundamentally eliminates time-leakage and realistically simulates the 91-day blind forecasting scenario.

---

# Local Validation Results

Key experiments establishing our methodology:

| Model | Validation | Features | Local MAE |
|---|---|---:|---:|
| `lgbm_v2` (Two-Stage) | Chronological | 337 | 0.6942 |
| `xgb_v1` (Two-Stage) | Chronological | 337 | 0.7150 |
| `lgbm_direct` (Pure Weather) | Chronological | 318 | 0.6770 |
| `xgb_direct` (Pure Weather) | Chronological | 318 | 0.7320 |

*Insight: Pure weather models excel locally but require Kaggle validation to prove generalization.*

---

# Kaggle Public Leaderboard (Ablation)

| Method | MAE | Key Takeaway |
|---|---:|---|
| **Naive Persistence (ffill)** | **1.0815** | Static baseline fails entirely; drought is highly dynamic. |
| **Misaligned Ensemble** | **0.8851** | Mixing leaky persistence with clean models degrades generalization. |
| **Pure Weather (Strategy B)** | **0.8640** | Excellent Local MAE, but poor Leaderboard generalization. |
| **Long-Term Weather (365d)** | **0.8604** | Fails to replace unobserved confounders in historical scores. |

---

# Kaggle Leaderboard (Champion)

| Method | MAE | Key Takeaway |
|---|---:|---|
| **Ensemble v1 (Two-Stage)** | **0.8232 🏆** | **Champion Model.** Safely reconstructs blind gap; robustly blends weather and history without leakage. |
| **Initial Submission (Leak)** | **0.8094** | Accidental hybrid due to feature artifact. **Discarded to maintain strict academic rigor.** |

---

# The Kaggle Paradox & Concluding Insights

This extensive ablation study provided a profound insight into the dataset dynamics:

We hypothesized that the inherent noise in historical score reconstruction would degrade performance, prompting us to rely exclusively on pure, long-term meteorological signals (up to 365 days). While this pure approach achieved an outstanding Local MAE of `0.488`, it failed to generalize, scoring only `0.8604` on the Kaggle Leaderboard.

**The Kaggle Paradox:** Historical drought scores, despite their noise and sparsity, encapsulate critical **Unobserved Confounders**—such as local water infrastructure, agricultural habits, and soil retention characteristics—that pure weather data cannot measure. Consequently, the Two-Stage Reconstructor (`0.8232`) remains the scientifically sound and optimal architecture.

---

# System Engineering & Infrastructure

The codebase has been refactored to support robust, production-level iteration:

- **Modular Core:** Feature engineering, training, and inference logic are strictly decoupled (`src/train.py`, `src/predict.py`, `src/features.py`).
- **Ensemble Tooling:** Automated post-processing scripts (`src/ensemble.py`) facilitate dynamic model blending.
- **Experiment Tracking:** Every execution persists configuration metadata, evaluation metrics, and model weights to a versioned `experiments/` directory.

---

# Next Steps & Future Work

Prior to the final submission deadline, our priorities are:

- **Ensemble Optimization:** Fine-tune the blending weights between LightGBM and XGBoost for the Two-Stage Reconstructor pipeline.
- **Temporal Modeling Extensions:** Explore native sequence models such as Temporal Fusion Transformers (TFT) if computational resources permit.
- **Documentation:** Finalize the IEEE standard report (`report/main.tex`), ensuring all findings from the ablation study and the Kaggle Paradox are thoroughly articulated.

---

# Summary

**Achievements:**
- Eradicated critical time-leakage and train-test discrepancies.
- Developed a robust, modular Two-Stage Direct Forecasting pipeline.
- Successfully diagnosed the "Kaggle Paradox," proving the necessity of Unobserved Confounders over pure meteorological data.
- Established a mathematically sound and highly competitive Kaggle Baseline: **0.8232**.

**Current Focus:** Finalizing the IEEE technical report and preparing for the oral presentation.

## Thank you (∠·ω )⌒★