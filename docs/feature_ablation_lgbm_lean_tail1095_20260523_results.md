# Feature Ablation Results - LGBM Lean Tail1095 - 2026-05-23

This batch evaluates high-risk feature groups with LightGBM, `lean` profile,
`train-tail-days=1095`, `recency-half-life-days=1095`, rolling-origin
validation with 3 folds, and `refit_full`.

## Rolling-Origin Results

| Run | Feature columns | Avg MAE | Delta vs baseline | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minus_climatology` | `386` | `0.3046` | `-0.0078` | `0.2711` | `0.2868` | `0.3094` | `0.3189` | `0.3368` | Local-only improvement; rejected after blind backtest. |
| `baseline` | `401` | `0.3124` | `+0.0000` | `0.2865` | `0.2960` | `0.3215` | `0.3275` | `0.3305` | Reference. |
| `minus_region_stats` | `401` | `0.3124` | `+0.0000` | `0.2865` | `0.2960` | `0.3215` | `0.3275` | `0.3305` | No-op because region stats were disabled. |
| `minus_domain_indices` | `367` | `0.3144` | `+0.0020` | `0.2794` | `0.2958` | `0.3211` | `0.3217` | `0.3541` | Reject for now; hurts week 5. |
| `minus_long_drought_proxy` | `380` | `0.3645` | `+0.0521` | `0.3625` | `0.3683` | `0.3568` | `0.3601` | `0.3748` | Reject; long drought proxies carry strong signal. |
| `minus_score_history` | `379` | `0.3645` | `+0.0521` | `0.3446` | `0.3479` | `0.3630` | `0.3758` | `0.3914` | Reject; score history remains essential. |

## Readout

The only positive local ablation was `minus_climatology`, improving average
rolling MAE by `0.0078`. The largest negative controls were `score_history` and
`long_drought_proxy`; removing either degraded average MAE by about `0.052`.

However, `minus_climatology` failed Kaggle-like blind backtesting. This is a
clear validation-discipline example: rolling-origin improvement alone was not
enough evidence for promotion.

## Blind Backtest

Protocol:

- Origins: `5,13,26,39,52,78,104`
- History context: `history-tail-days=1100`
- Output: `experiments/blind_20260523_lgbm_minus_climatology_tail1095_h1100`

| Metric | Value |
|---|---:|
| Overall blind MAE | `0.4463` |
| Week 1 | `0.4074` |
| Week 2 | `0.4344` |
| Week 3 | `0.4474` |
| Week 4 | `0.4646` |
| Week 5 | `0.4775` |
| Zero-score MAE | `0.1824` |
| Nonzero-score MAE | `0.9408` |
| High-severity MAE (`target >= 3`) | `1.7394` |

The baseline LGBM blind anchor recorded earlier was `0.3549`, so
`minus_climatology` is substantially worse despite better rolling-origin MAE.

## Decision

- Do not submit `minus_climatology`.
- Keep `score_history` and `long_drought_proxy`; both are essential.
- Treat `region_stats` ablation as a no-op in this configuration because
  `use_region_stats=false`.
- `domain_indices` is not promoted; it slightly hurts average MAE and worsens
  week 5.
- Future feature pruning should be promoted only if it improves blind backtest,
  not just rolling-origin MAE.
