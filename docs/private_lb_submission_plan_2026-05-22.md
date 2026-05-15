# 5/22 Static Private Leaderboard Submission Plan

Date prepared: 2026-05-16

## Objective

Prepare 1-2 conservative submissions before the 2026-05-22 static private leaderboard update without sacrificing the current best legal public score.

## Source Models

| Source | Public MAE | Status |
|---|---:|---|
| `submissions/submission_20260512_234155_lgbm_v2.csv` | `0.8299` | Legal |
| `submissions/submission_20260513_001713_xgb_v1.csv` | Not submitted alone | Legal |
| `submissions/ensemble_final.csv` | `0.8232` | Current best legal anchor |

Excluded:

- `submission_20260512_195951.csv`: public `0.8094`, treated as accidental/leaky and not used.
- `submission_20260514_122628_20260514_121452_lightgbm_two_stage_lgbm_leaky_repro.csv`: leaky diagnostic, not used.
- `submission_20260514_134141_20260513_165547_lightgbm_two_stage_lgbm_gap_anomaly_regularized_lean_v2.csv`: public `0.9723`, not competitive.

## Submitted Candidates

| Candidate | File | Kaggle ref | Public MAE | Rationale |
|---|---|---:|---:|---|
| Validation-weighted blend | `submissions/ensemble_20260516_validation_weighted_v1.csv` | `52689645` | `0.8232` | Uses per-horizon local validation MAE to slightly favor LGBM while staying near the current 50/50 anchor. |
| XGB-tilted hedge | `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `52689654` | `0.8232` | Tests whether a modestly stronger XGBoost contribution helps private robustness; public score stayed tied with the best legal anchor. |

## Exact Commands

```bash
uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --out submissions/ensemble_20260516_validation_weighted_v1.csv \
  --lgb-weights 0.5191,0.5157,0.5153,0.5132,0.5108

kaggle competitions submit \
  -c data-mining-2026-final-project \
  -f submissions/ensemble_20260516_validation_weighted_v1.csv \
  -m "Conservative validation-weighted LGB/XGB blend from legal v2/v1"

uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --out submissions/ensemble_20260516_xgb_tilt_40_60.csv \
  --lgb-weight 0.4

kaggle competitions submit \
  -c data-mining-2026-final-project \
  -f submissions/ensemble_20260516_xgb_tilt_40_60.csv \
  -m "Conservative XGB-tilted 40/60 blend from legal v2/v1"
```

## File Audit

| File | SHA-256 prefix | Mean absolute delta vs `ensemble_final` | Max delta vs `ensemble_final` |
|---|---|---:|---:|
| `submissions/ensemble_20260516_validation_weighted_v1.csv` | `b027d21fe911` | `0.003238` | `0.028077` |
| `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `e7e6946785f3` | `0.021799` | `0.176259` |

Both files have `2248` rows, matching submission columns, no NaN values, and predictions clipped to `[0, 5]`.

## Readout Plan After 5/22

When the 2026-05-22 static private leaderboard is released:

1. Compare private ranks of `ensemble_final`, `ensemble_20260516_validation_weighted_v1`, and `ensemble_20260516_xgb_tilt_40_60`.
2. If validation-weighted wins or ties privately, keep future blends very close to 50/50.
3. If XGB-tilted wins privately, explore one more nearby weight, such as 30/70 or per-horizon XGB tilt.
4. If neither improves private rank, stop blend-only exploration and invest in validation redesign.
