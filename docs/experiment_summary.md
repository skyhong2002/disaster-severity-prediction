# Experiment and Submission Summary

Last updated: 2026-05-20

This file records the current legal model-selection state for the Final Project progress check. It is intended to keep the slides, report, code, and Kaggle submissions consistent.

## Current Leaderboard Interpretation

- Current best legal public score: `0.8124` MAE.
- Current best legal submission file: `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`.
- The previous LGB/XGB anchor remains useful for comparison: `submissions/ensemble_final.csv`, public MAE `0.8232`.
- 5/15 static private leaderboard: Team 5 was ranked 3, below Baseline 3 and above Baseline 2.
- Strategy implication: use public leaderboard gains cautiously and prioritize private robustness, reproducibility, and leakage-free validation evidence.

## Current Best Legal Submission

| Field | Value |
|---|---|
| Submission | `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` |
| Public MAE | `0.8124` |
| Rows | `2248` |
| SHA-256 prefix | `bee6f618828d` |
| Blend | 35% `lgbm_v2` + 35% `xgb_v1` + 30% `catboost_lean_tail2737_regularized_500` |
| LightGBM source | `submissions/submission_20260512_234155_lgbm_v2.csv` |
| XGBoost source | `submissions/submission_20260513_001713_xgb_v1.csv` |
| CatBoost source | `submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` |

## Previous Legal Anchor

| Field | Value |
|---|---|
| Submission | `submissions/ensemble_final.csv` |
| Public MAE | `0.8232` |
| Rows | `2248` |
| SHA-256 prefix | `28d368bc4be8` |
| Blend | 50% `lgbm_v2` + 50% `xgb_v1` |
| Repro check | Max difference from exact 50/50 blend: `4.44e-16` |

## Submitted Candidates

The 2026-05-16 submission batch has two phases. The first phase tested conservative LGB/XGB blends around the previous anchor. The second phase added CatBoost as a third diversity model.

| Submission | Kaggle ref | Blend | Public MAE | SHA-256 prefix | Purpose |
|---|---:|---|---:|---|---|
| `submissions/ensemble_20260516_validation_weighted_v1.csv` | `52689645` | Per-horizon LGBM weights: `0.5191,0.5157,0.5153,0.5132,0.5108` | `0.8232` | `b027d21fe911` | Minimal validation-weighted perturbation around `ensemble_final`. |
| `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `52689654` | 40% `lgbm_v2` / 60% `xgb_v1` | `0.8232` | `e7e6946785f3` | Small XGBoost-tilted hedge. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_45_45_10.csv` | `52698220` | 45% LGB / 45% XGB / 10% CatBoost | `0.8163` | not recorded | CatBoost diversity probe. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_40_40_20.csv` | `52698244` | 40% LGB / 40% XGB / 20% CatBoost | `0.8126` | not recorded | Higher CatBoost weight. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `52698259` | 35% LGB / 35% XGB / 30% CatBoost | `0.8124` | `bee6f618828d` | Current best legal public score. |

Sanity checks for the best current file:

- `2248` rows.
- Same columns as `sample_submission.csv`.
- No missing predictions.
- Predictions clipped to `[0, 5]`.
- The leaky reproduction output was not used.

## Experiment Table

| Experiment | Model | Feature setup | Validation | Local MAE | Public MAE / Status | Notes |
|---|---|---|---|---:|---:|---|
| `lgbm_v2` | LightGBM direct horizon | 337 features, score history | Chronological holdout | `0.6942` | Component of `0.8232` and `0.8124` ensembles | Legal baseline model. |
| `xgb_v1` | XGBoost direct horizon | 337 features, score history | Chronological holdout | `0.7150` | Component of `0.8232` and `0.8124` ensembles | Legal diversity model. |
| `catboost_lean_tail2737_regularized_500` | CatBoost direct horizon | Lean profile, 91-day-gapped score history, native categorical features | Rolling origin | `0.2212` | Component of `0.8124` ensemble | Validation scale differs from holdout; do not compare directly with LGB/XGB holdout MAE. |
| `ensemble_20260516_lgb_xgb_cat2737_35_35_30` | Three-model ensemble | `lgbm_v2` + `xgb_v1` + CatBoost tail2737 | N/A | N/A | `0.8124` | Current best legal public submission. |
| `ensemble_final` | 50/50 ensemble | `lgbm_v2` + `xgb_v1` | N/A | N/A | `0.8232` | Previous legal anchor. |
| `lgbm_direct` | LightGBM direct weather-only | 318 weather-only features | Chronological holdout | `0.6770` | `0.8640` strategy result | Good local score, weak public generalization. |
| `xgb_direct` | XGBoost direct weather-only | 318 weather-only features | Chronological holdout | `0.7320` | Used in strategy experiment | Did not beat score-history ensembles. |
| `ensemble_strategy_b_long_term` | 50/50 ensemble | Long-term weather-only | N/A | N/A | `0.8604` | Shows pure weather cannot replace score-history context. |
| `lgbm_gap_anomaly_regularized_lean_v2` | LightGBM direct horizon | Lean, 91-day-gapped score history, regularized | Rolling origin | `0.1915` | Diagnostic / under review | Local score appears optimistic; do not treat as champion without further validation. |
| `lgbm_leaky_repro` | LightGBM diagnostic | Lean, score gap `0` | Holdout | `0.2321` | Discarded | Leaky diagnostic only; excluded from model selection. |

## Terminology Note

Some run directory names and `model_family` strings still contain `two_stage`. These are retained for compatibility with existing artifacts. In current documentation, the active implementation should be described as direct-horizon boosted-tree models with leakage-aware 91-day-gapped score-history features, not as the older separate score-estimation pipeline.

## Reproducibility Requirements

- Keep `Initial Submission (Leak)` and `lgbm_leaky_repro` out of official model claims.
- If the report mentions a Kaggle score, it must point to a submission CSV and a reproducible experiment lineage.
- When comparing local MAE, state the validation strategy. Holdout and rolling-origin MAE are not directly comparable.
- Future submissions should include the hypothesis being tested, source model files, blend weights, public score, and whether the run is legal.
- Do not overwrite existing run directories; create a new `--experiment-name` for each method change.
