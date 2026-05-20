# 5/22 Static Private Leaderboard Submission Plan

Date prepared: 2026-05-16
Status updated: 2026-05-20

This plan was originally written before the CatBoost submissions. It is now kept as a private-leaderboard readout plan, with the current CatBoost blend added so it does not conflict with `docs/experiment_summary.md`.

## Objective

Compare conservative legal submissions on the 2026-05-22 static private leaderboard. The goal is not to chase a single public leaderboard snapshot, but to decide whether the CatBoost diversity blend is privately robust enough to replace the older LGB/XGB anchor.

## Source Models

| Source | Public MAE | Status |
|---|---:|---|
| `submissions/submission_20260512_234155_lgbm_v2.csv` | `0.8299` | Legal LightGBM baseline |
| `submissions/submission_20260513_001713_xgb_v1.csv` | Not submitted alone | Legal XGBoost diversity model |
| `submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` | Not submitted alone | Legal CatBoost diversity model |
| `submissions/ensemble_final.csv` | `0.8232` | Previous legal LGB/XGB anchor |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `0.8124` | Current best legal public submission |

Excluded from official model claims:

- `submission_20260512_195951.csv`: public `0.8094`, treated as accidental/leaky and not used.
- `submission_20260514_122628_20260514_121452_lightgbm_two_stage_lgbm_leaky_repro.csv`: leaky diagnostic, not used.
- `submission_20260514_134141_20260513_165547_lightgbm_two_stage_lgbm_gap_anomaly_regularized_lean_v2.csv`: public `0.9723`, not competitive.

## Submitted Candidates

| Candidate | File | Kaggle ref | Public MAE | Rationale |
|---|---|---:|---:|---|
| Validation-weighted LGB/XGB | `submissions/ensemble_20260516_validation_weighted_v1.csv` | `52689645` | `0.8232` | Small per-horizon perturbation around the old LGB/XGB anchor. |
| XGB-tilted LGB/XGB | `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `52689654` | `0.8232` | Tests whether a stronger XGBoost contribution helps private robustness. |
| CatBoost 10% blend | `submissions/ensemble_20260516_lgb_xgb_cat2737_45_45_10.csv` | `52698220` | `0.8163` | First legal CatBoost diversity probe. |
| CatBoost 20% blend | `submissions/ensemble_20260516_lgb_xgb_cat2737_40_40_20.csv` | `52698244` | `0.8126` | Higher CatBoost contribution. |
| CatBoost 30% blend | `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `52698259` | `0.8124` | Current best legal public submission. |

## Exact Commands for Current Best Blend

```bash
uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --cat submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv \
  --out submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  --weights 'lgb=0.35;xgb=0.35;cat=0.30'

kaggle competitions submit \
  -c data-mining-2026-final-project \
  -f submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  -m "LGB/XGB/CatBoost 35/35/30 blend with CatBoost tail2737"
```

## File Audit

| File | SHA-256 prefix | Notes |
|---|---|---|
| `submissions/ensemble_20260516_validation_weighted_v1.csv` | `b027d21fe911` | Legal LGB/XGB conservative candidate. |
| `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `e7e6946785f3` | Legal LGB/XGB conservative candidate. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `bee6f618828d` | Legal three-model current public best. |

All tracked candidate files have `2248` rows, matching submission columns, no NaN values, and predictions clipped to `[0, 5]`.

## Readout Plan After 5/22

1. Compare private ranks of the old LGB/XGB anchor, the conservative LGB/XGB perturbations, and the CatBoost blends.
2. If the CatBoost 30% blend wins or ties privately, keep it as the legal anchor and only test nearby weights with a clear hypothesis.
3. If the old LGB/XGB anchor is stronger privately, treat CatBoost as public-overfit risk and reduce or remove its weight.
4. If all blend variants remain unstable privately, stop blend-only exploration and invest in validation redesign.
