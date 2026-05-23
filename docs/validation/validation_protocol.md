# Validation Protocol

Last updated: 2026-05-21

The project now treats model selection as a three-layer validation problem:
fast local diagnostics, Kaggle-like pseudo-private backtesting, and leaderboard
readout. Public leaderboard movement alone is not enough to promote a model.

## 1. Train With A Real Final-Model Contract

All boosted-tree trainers now support:

```bash
--final-train-mode last_fold
--final-train-mode fold_ensemble
--final-train-mode refit_full
```

`refit_full` is the default. Each horizon first runs the selected validation
folds, records fold MAE and best iteration counts, then retrains one final model
on all legal rows for that horizon using the median best iteration. `fold_ensemble`
saves a fold ensemble wrapper and averages fold predictions at inference time.
`last_fold` remains only for debugging and backward comparison.

This fixes the previous mismatch where the reported CV score was an average
over folds but the saved submission model was the final validation fold model.

## 2. Use Blind Backtest Before Promoting A Candidate

Run:

```bash
uv run python scripts/run_blind_backtest.py \
  --run-dir experiments/<run_id> \
  --origins "5,13,26" \
  --history-tail-days 1100
```

For each pseudo origin, the script:

1. Selects a historical weekly forecast origin.
2. Masks scores in the preceding 91-day blind window.
3. Rebuilds features from prior history plus blind-window weather.
4. Predicts from the final blind-window row for each region.
5. Scores the next five real weekly labels.

Outputs are written under `experiments/<run_id>/validation/`:

- `blind_predictions_<origin>.csv`
- `blind_backtest_rows.csv`
- `blind_backtest_metrics.json`

The metrics include overall MAE, MAE by horizon, MAE by region-climate cluster,
MAE by calendar month, MAE by origin, MAE by region, zero/nonzero/high-severity
MAE, prediction stats, and target stats.

Before trusting a blind backtest as model-selection evidence, run:

```bash
uv run python scripts/leakage_sentinel.py \
  --origin 5 \
  --history-tail-days 1100 \
  --feature-profile micro
```

The sentinel poisons weekly scores inside the hidden 91-day window and verifies
that final-row features do not change. A failure means score-history,
climatology, or region priors are leaking blind-window labels.

## 3. Check Distribution Shift Before Choosing Tail Data

Run:

```bash
uv run python scripts/drift_report.py \
  --tail-days "1095,1825,2737,3650,0" \
  --out docs/validation/drift_report_20260521.md
```

The report compares train-tail candidates against `data/test.csv` using PSI,
KS distance, quantile-Wasserstein distance, weather-only adversarial validation
AUC, and weather+region/month adversarial AUC. Use this to decide whether
`1095`, `1825`, `2737`, `3650`, or full-history training should be tested in
blind backtest.

## 4. Fit Blends From Validation Predictions

When OOF or blind-backtest predictions exist for multiple model families, fit
weights with an anchor instead of hand-tuning public LB:

```bash
uv run python scripts/fit_blend_weights.py \
  --preds lgb=validation/lgb.csv,xgb=validation/xgb.csv,cat=validation/cat.csv \
  --target validation/target.csv \
  --anchor 'lgb=0.35;xgb=0.35;cat=0.30' \
  --lambda-reg 0.05 \
  --bootstrap 100 \
  --out-json experiments/blend_weights.json
```

Weights are non-negative and sum to one per horizon. The regularization keeps
the solution near the current legal 35/35/30 anchor unless validation evidence
justifies moving away. Use `--caps lgbm_micro=0.15` when evaluating diagnostic
models that should not be allowed to dominate the blend.

## 5. Run Feature-Group Ablations

Use `--drop-feature-groups` for targeted ablations:

```bash
uv run python src/train.py \
  --experiment-name lgbm_minus_score_history_20260521 \
  --feature-profile micro \
  --validation-mode rolling_origin \
  --regularized \
  --drop-feature-groups score_history
```

To generate a command matrix first:

```bash
uv run python scripts/feature_ablation.py \
  --model-family lightgbm \
  --groups score_history,climatology,calendar,rolling,ewm,long_drought_proxy,domain_indices,region_stats \
  --feature-profile micro \
  --out docs/experiments/feature_ablation_20260521_commands.md
```

Promote an ablation only if blind backtest improves, or if MAE is flat but
residual diversity improves in a constrained blend.

## 6. Decision Rule

Promote a candidate only when it has a distinct hypothesis and at least one of:

- blind-backtest MAE improves without prediction distribution collapse;
- blind-backtest MAE is similar but residuals add useful ensemble diversity;
- private leaderboard readout confirms the hypothesis.

Do not promote a run solely because rolling-origin MAE improved or public LB
nudged upward.
