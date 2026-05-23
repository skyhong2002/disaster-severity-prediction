# Feature Missingness Audit

Generated: 2026-05-23T13:23:31
Run directory: `/Users/skyhong/Documents/GitHub/disaster-severity-prediction/experiments/20260521_153911_lightgbm_two_stage_lgbm_refit_full_lean_tail1095_20260521`

## Feature Options

```json
{
  "drop_feature_groups": [],
  "drop_feature_nan_rows": false,
  "feature_profile": "lean",
  "max_score_lag_weeks": null,
  "score_gap_days": 91,
  "train_tail_days": 1095,
  "use_climatology": true,
  "use_region_stats": false,
  "use_score_history": true
}
```

## Summary

| matrix | rows | columns | missing columns | missing cells | max missing rate |
|---|---:|---:|---:|---:|---:|
| raw train weather | 12,319,040 | 16 | 0 | 0 | 0.0000 |
| raw test weather | 204,568 | 16 | 0 | 0 | 0.0000 |
| inference test_last features | 2,248 | 401 | 0 | 0 | 0.0000 |
| training weekly features | 339,448 | 401 | 120 | 1,782,664 | 0.4305 |

## Top Missing Engineered Train Features

Top 40 columns by missing rate on supervised weekly training rows.

| feature | missing | missing rate |
|---|---:|---:|
| `score_gap_lag52w` | 146,120 | 0.4305 |
| `score_gap_lag26w` | 87,672 | 0.2583 |
| `score_gap_lag12w` | 56,200 | 0.1656 |
| `score_gap_lag8w` | 47,208 | 0.1391 |
| `score_momentum_28d` | 38,216 | 0.1126 |
| `score_gap_lag4w` | 38,216 | 0.1126 |
| `score_momentum_14d` | 33,720 | 0.0993 |
| `score_gap_lag2w` | 33,720 | 0.0993 |
| `score_velocity_1w` | 31,472 | 0.0927 |
| `score_gap_std365d` | 31,472 | 0.0927 |
| `score_gap_lag1w` | 31,472 | 0.0927 |
| `score_gap_std182d` | 31,472 | 0.0927 |
| `score_gap_std91d` | 31,472 | 0.0927 |
| `score_gap_lag0w` | 29,224 | 0.0861 |
| `score_gap_trend_13w_52w` | 29,224 | 0.0861 |
| `score_gap_mean91d` | 29,224 | 0.0861 |
| `score_gap_max91d` | 29,224 | 0.0861 |
| `score_gap_mean182d` | 29,224 | 0.0861 |
| `score_gap_max182d` | 29,224 | 0.0861 |
| `score_gap_mean365d` | 29,224 | 0.0861 |
| `score_gap_max365d` | 29,224 | 0.0861 |
| `last_known_score` | 29,224 | 0.0861 |
| `wind_range_lag49` | 15,736 | 0.0464 |
| `wind_lag49` | 15,736 | 0.0464 |
| `tmp_max_lag49` | 15,736 | 0.0464 |
| `wb_tmp_lag49` | 15,736 | 0.0464 |
| `wind_max_lag49` | 15,736 | 0.0464 |
| `surf_tmp_lag49` | 15,736 | 0.0464 |
| `tmp_range_lag49` | 15,736 | 0.0464 |
| `wind_min_lag49` | 15,736 | 0.0464 |
| `dp_tmp_lag49` | 15,736 | 0.0464 |
| `tmp_lag49` | 15,736 | 0.0464 |
| `surf_pre_lag49` | 15,736 | 0.0464 |
| `tmp_min_lag49` | 15,736 | 0.0464 |
| `prec_lag49` | 15,736 | 0.0464 |
| `humidity_lag49` | 15,736 | 0.0464 |
| `wind_range_lag42` | 13,488 | 0.0397 |
| `tmp_range_lag42` | 13,488 | 0.0397 |
| `prec_lag42` | 13,488 | 0.0397 |
| `surf_tmp_lag42` | 13,488 | 0.0397 |

## Top Missing Inference Features

| feature | missing | missing rate |
|---|---:|---:|
| none | 0 | 0.0000 |

## Readout

- Raw weather columns have no missing values in the current train/test files.
- Most engineered missingness comes from warmup windows: long score-history lags and long meteorological lags at the beginning of each region history.
- The final Kaggle-style inference row per region should have no missing values; `src/predict.py` now audits and fails fast if that changes.
