# Experiment and Submission Summary

Last updated: 2026-05-16

This file records the current legal model selection state for the Final Project progress check. It is intended to keep the slides, report, code, and Kaggle submissions consistent.

## Current Leaderboard Interpretation

- Public best legal score: `0.8232` MAE.
- Best legal submission file: `submissions/ensemble_final.csv`.
- 5/15 static private leaderboard: Team 5 is ranked 3, below Baseline 3 and above Baseline 2.
- Baseline 3 pressure: at the time of the TA announcement, at least 13 successful entries were needed to cross Public Baseline 3.
- Strategy implication: optimize for private robustness and reproducibility rather than public-only gains.

## Best Legal Submission

| Field | Value |
|---|---|
| Submission | `submissions/ensemble_final.csv` |
| Public MAE | `0.8232` |
| Rows | `2248` |
| SHA-256 prefix | `28d368bc4be8` |
| Blend | 50% `lgbm_v2` + 50% `xgb_v1` |
| LightGBM source | `submissions/submission_20260512_234155_lgbm_v2.csv` |
| XGBoost source | `submissions/submission_20260513_001713_xgb_v1.csv` |
| Repro check | Max difference from exact 50/50 blend: `4.44e-16` |

## Experiment Table

| Experiment | Model | Feature setup | Validation | Local MAE | Public MAE / Status | Notes |
|---|---|---|---|---:|---:|---|
| `lgbm_v2` | LightGBM two-stage | 337 features, score history | Chronological holdout | `0.6942` | Component of `0.8232` ensemble | Legal baseline model. |
| `xgb_v1` | XGBoost two-stage | 337 features, score history | Chronological holdout | `0.7150` | Component of `0.8232` ensemble | Legal diversity model. |
| `ensemble_final` | 50/50 ensemble | `lgbm_v2` + `xgb_v1` | N/A | N/A | `0.8232` | Current best legal submission. |
| `lgbm_direct` | LightGBM direct | 318 weather-only features | Chronological holdout | `0.6770` | `0.8640` strategy result | Good local score, weak public generalization. |
| `xgb_direct` | XGBoost direct | 318 weather-only features | Chronological holdout | `0.7320` | Used in strategy experiment | Did not beat two-stage ensemble. |
| `ensemble_strategy_b_long_term` | 50/50 ensemble | Long-term weather-only | N/A | N/A | `0.8604` | Shows pure weather cannot replace score history. |
| `lgbm_gap_anomaly_regularized_lean_v2` | LightGBM two-stage | Lean, 91-day-gapped score history, regularized | Rolling origin | `0.1915` | Diagnostic / under review | Local score appears optimistic; do not treat as champion without further validation. |
| `lgbm_leaky_repro` | LightGBM two-stage | Lean, score gap `0` | Holdout | `0.2321` | Discarded | Leaky diagnostic only; excluded from model selection. |

## Reproducibility Requirements

- Keep `Initial Submission (Leak)` and `lgbm_leaky_repro` out of official model claims.
- If the report mentions a Kaggle score, it must point to a submission CSV and a reproducible experiment lineage.
- Future submissions should include the hypothesis being tested, source model files, blend weights, public score, and whether the run is legal.
- Do not overwrite existing run directories; create a new `--experiment-name` for each method change.
