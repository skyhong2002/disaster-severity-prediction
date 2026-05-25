# Missingness / Shift Experiments: 20260523

Started: 2026-05-23T05:27:59Z


## missingness_lgbm_lean_tail1095_drop_feature_nan_rows_20260523

```bash
uv run python src/train.py --experiment-name missingness_lgbm_lean_tail1095_drop_feature_nan_rows_20260523 --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --train-tail-days 1095 --recency-half-life-days 1095 --drop-feature-nan-rows
```

## missingness_lgbm_lean_tail1095_score_lag26_20260523

```bash
uv run python src/train.py --experiment-name missingness_lgbm_lean_tail1095_score_lag26_20260523 --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --train-tail-days 1095 --recency-half-life-days 1095 --max-score-lag-weeks 26
```

## shift_lgbm_lean_tail1095_recency365_20260523

```bash
uv run python src/train.py --experiment-name shift_lgbm_lean_tail1095_recency365_20260523 --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --train-tail-days 1095 --recency-half-life-days 1095 --recency-half-life-days 365
```

## shift_lgbm_lean_tail1095_seasonmatch2_20260523

```bash
uv run python src/train.py --experiment-name shift_lgbm_lean_tail1095_seasonmatch2_20260523 --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --train-tail-days 1095 --recency-half-life-days 1095 --season-match-weight 2.0
```

## shift_lgbm_lean_tail365_lag26_recency365_20260523

```bash
uv run python src/train.py --experiment-name shift_lgbm_lean_tail365_lag26_recency365_20260523 --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --train-tail-days 1095 --recency-half-life-days 1095 --train-tail-days 365 --recency-half-life-days 365 --max-score-lag-weeks 26
```
