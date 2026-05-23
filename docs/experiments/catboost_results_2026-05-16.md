# CatBoost Experiment and Submission Results - 2026-05-16

## Summary

Following the deep-research recommendation to add CatBoost as a third boosted-tree model family, we implemented a time-aware CatBoost direct-horizon pipeline and used it as a diversity model in the Kaggle submission ensemble.

The best public leaderboard result from today's submissions was:

| Submission | Kaggle ref | Public MAE |
|---|---:|---:|
| `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `52698259` | `0.8124` |

This improves over the previous legal anchor `ensemble_final.csv` / conservative LGB-XGB blends at public MAE `0.8232`.

## Implementation Changes

- Added `src/train_catboost.py` for direct-horizon CatBoost training.
- Added CatBoost as a dependency in `pyproject.toml`.
- Updated `src/predict.py` so inference can load `catboost_models.pkl`, `xgb_models.pkl`, or `lgbm_models.pkl` from a run directory.
- Updated `src/ensemble.py` to support optional three-model convex blends: LightGBM, XGBoost, and CatBoost.
- Fixed `src/train_xgb.py` so XGBoost runs save to `xgb_models.pkl` instead of `lgbm_models.pkl`.
- Updated `README.md` with `uv` commands for CatBoost training.

CatBoost settings used:

- `loss_function=MAE`
- `eval_metric=MAE`
- `has_time=True`
- direct one-model-per-horizon training for weeks 1-5
- native categorical features: `region_id`, `month`, `quarter`, `weekofyear`
- `feature_profile=lean`
- `score_gap_days=91`
- rolling-origin validation with 3 folds
- regularized CatBoost preset
- recency half-life of 1095 days

## Memory Management

The first full-history CatBoost attempt was killed with exit code `137`, consistent with memory pressure on a 32 GB RAM machine. To keep the run reproducible and memory-safe, `src/train_catboost.py` now supports:

```bash
--train-tail-days N
```

This trims each region to the latest `N` daily rows before feature construction. The successful runs used 1095, 1825, and 2737 days. The 2737-day run stayed within the available RAM during monitoring and gave the best validation score.

## CatBoost Validation Results

| Run | Tail days | Iterations | Weekly rows | Average MAE | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `catboost_lean_tail1095_regularized_300` | 1095 | 300 | 339,448 | 0.3091 | 0.2931 | 0.2990 | 0.3075 | 0.3305 | 0.3156 |
| `catboost_lean_tail1825_regularized_500` | 1825 | 500 | 573,240 | 0.2343 | 0.2203 | 0.2228 | 0.2311 | 0.2420 | 0.2555 |
| `catboost_lean_tail2737_regularized_500` | 2737 | 500 | 867,728 | 0.2212 | 0.2152 | 0.2198 | 0.2221 | 0.2228 | 0.2263 |

Best run directory:

```text
experiments/20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500
```

Best CatBoost-only submission file generated locally:

```text
submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv
```

The CatBoost-only file was not submitted on 2026-05-16. It was used as an ensemble member only.

## Kaggle Submissions

After the daily submission limit reset, three conservative CatBoost blends were submitted:

| Kaggle ref | File | Weights `(LGB, XGB, CatBoost)` | Public MAE |
|---:|---|---:|---:|
| `52698220` | `ensemble_20260516_lgb_xgb_cat2737_45_45_10.csv` | `(0.45, 0.45, 0.10)` | `0.8163` |
| `52698244` | `ensemble_20260516_lgb_xgb_cat2737_40_40_20.csv` | `(0.40, 0.40, 0.20)` | `0.8126` |
| `52698259` | `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `(0.35, 0.35, 0.30)` | `0.8124` |

Increasing the CatBoost weight from 10% to 30% improved the public score, but the gain flattened between 20% and 30%. The best current legal public score is therefore `0.8124`.

## Reproduction Commands

Install dependencies:

```bash
uv sync
```

Train the best CatBoost model:

```bash
PYTHONUNBUFFERED=1 uv run python src/train_catboost.py \
  --experiment-name catboost_lean_tail2737_regularized_500 \
  --feature-profile lean \
  --train-tail-days 2737 \
  --validation-mode rolling_origin \
  --rolling-folds 3 \
  --regularized \
  --recency-half-life-days 1095 \
  --iterations 500
```

Generate CatBoost predictions:

```bash
PYTHONUNBUFFERED=1 uv run python src/predict.py \
  --run-dir experiments/20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500
```

Create the best submitted blend:

```bash
uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --cat submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv \
  --out submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  --weights 'lgb=0.35;xgb=0.35;cat=0.30'
```

Submit:

```bash
uv run kaggle competitions submit \
  -c data-mining-2026-final-project \
  -f submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  -m "LGB/XGB/CatBoost 35/35/30 blend with CatBoost tail2737"
```

## Next Steps

- Test nearby CatBoost weights around 30%-45% after the next daily reset.
- Consider a memory-safe 3650-day CatBoost run if the machine is idle and RAM remains below 28 GB.
- Avoid overreacting to small public-LB deltas; keep the 35/35/30 blend as the current legal anchor until private-LB evidence suggests otherwise.
