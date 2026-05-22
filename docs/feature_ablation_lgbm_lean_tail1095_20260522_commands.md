# Feature Ablation Commands: feature_ablation_lgbm_lean_tail1095_20260522

Generated: 2026-05-22T23:11:16

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_baseline --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --train-tail-days 1095
```

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_minus_score_history --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --drop-feature-groups score_history --train-tail-days 1095
```

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_minus_climatology --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --drop-feature-groups climatology --train-tail-days 1095
```

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_minus_region_stats --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --drop-feature-groups region_stats --train-tail-days 1095
```

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_minus_long_drought_proxy --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --drop-feature-groups long_drought_proxy --train-tail-days 1095
```

```bash
uv run python src/train.py --experiment-name feature_ablation_lgbm_lean_tail1095_20260522_minus_domain_indices --feature-profile lean --validation-mode rolling_origin --rolling-folds 3 --final-train-mode refit_full --regularized --recency-half-life-days 1095.0 --drop-feature-groups domain_indices --train-tail-days 1095
```
