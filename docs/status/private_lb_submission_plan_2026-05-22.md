# 5/22 Static Private Leaderboard Submission Plan

Date prepared: 2026-05-16
Last updated: 2026-05-20

This plan was originally written before the CatBoost submissions. It is now kept
as a private-leaderboard readout plan, with the current CatBoost blend family and
2026-05-20 local reruns recorded so it stays consistent with
`docs/status/experiment_summary.md`.

## Objective

Compare conservative legal submissions on the 2026-05-22 static private
leaderboard. The goal is not to chase a single public leaderboard snapshot, but
to decide whether the CatBoost diversity blend is privately robust enough to
replace the older LGB/XGB anchor.

## Source Models

| Source | Public MAE | Status |
|---|---:|---|
| `submissions/submission_20260512_234155_lgbm_v2.csv` | `0.8299` | Legal LightGBM baseline |
| `submissions/submission_20260513_001713_xgb_v1.csv` | Not submitted alone | Legal XGBoost diversity model |
| `submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` | Not submitted alone | Legal CatBoost diversity source |
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
| CatBoost 20% blend | `submissions/ensemble_20260516_lgb_xgb_cat2737_40_40_20.csv` | `52698244` | `0.8126` | Higher CatBoost contribution while keeping LGB/XGB as the majority vote. |
| CatBoost 30% blend | `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `52698259` | `0.8124` | Best current legal public score; main 5/22 private-LB anchor. |
| CatBoost 35% blend | `submissions/ensemble_20260519_lgb_xgb_cat2737_325_325_35.csv` | `52796551` | `0.8141` | Nearby weight probe after the 30% blend; public score regressed, so treat as secondary private hedge only. |
| CatBoost 40% blend | `submissions/ensemble_20260519_lgb_xgb_cat2737_30_30_40.csv` | `52796554` | `0.8168` | Higher CatBoost-weight probe; public score regressed further. |
| Horizon CatBoost ramp | `submissions/ensemble_20260519_lgb_xgb_cat2737_horizon_cat_ramp.csv` | `52796562` | `0.8138` | Horizon-specific CatBoost-ramp hypothesis; worse than 30% publicly but close enough to inspect on 5/22. |

Primary 5/22 readout focus should stay narrow: treat the 30% CatBoost blend
(`52698259`) as the main conservative candidate and the horizon CatBoost ramp
(`52796562`) as the most meaningful secondary hedge. The 35% and 40%
CatBoost-weight probes are recorded for audit completeness, but their weaker
public scores make them lower-priority private-LB evidence.

## 2026-05-20 Local Experiments

As of this update, no additional Kaggle submission was made on 2026-05-20.

The reproducible local rerun
`experiments/20260520_111133_catboost_two_stage_catboost_lean_tail2737_regularized_500`
is legal and non-leaky (`score_gap_days=91`, `use_region_stats=false`,
rolling-origin validation). Its average rolling-origin MAE improved only
slightly versus the 2026-05-16 CatBoost source (`0.2192` vs `0.2212`). Because
the 2026-05-19 CatBoost weight probes already tested the same model-family
hypothesis and all regressed from the 30% CatBoost anchor on public LB,
submitting the 2026-05-20 CatBoost-only file is low signal before the 2026-05-22
static private update.

The local LGBM micro run
`experiments/20260520_142456_lightgbm_two_stage_lgbm_micro_rolling_regularized_20260520`
completed with rolling-origin MAE `0.2002` and produced
`submissions/submission_20260520_163323_20260520_142456_lightgbm_two_stage_lgbm_micro_rolling_regularized_20260520.csv`.
It is a promising diagnostic or blend input, but should not replace the current
legal anchor until it has either public-LB evidence or pseudo-private validation
evidence that explains why this low local MAE will generalize.

## 2026-05-21 Quota Probe Readout

Three additional legal submissions were made on 2026-05-21 after the blind
backtest framework was merged. They should be interpreted as diagnostic quota
probes, not as updates to the main 5/22 private-leaderboard anchor set.

| Candidate | File | Kaggle ref | Public MAE | Rationale |
|---|---|---:|---:|---|
| LGBM blind anchor | `submissions/submission_20260521_154656_20260521_153911_lightgbm_two_stage_lgbm_refit_full_lean_tail1095_20260521.csv` | `52882449` | `0.9380` | Tests whether the best blind single model transfers to public LB. It did not. |
| Blind-fit unregularized horizon blend | `submissions/ensemble_20260521_blindfit_unregularized_lgb_w1w4_cat_w5.csv` | `52882455` | `0.9302` | Uses LGBM for weeks 1-4 and an LGBM/CatBoost week-5 blend; best of the quota probes but still weak publicly. |
| Blind-fit regularized horizon blend | `submissions/ensemble_20260521_blindfit_regularized_horizon_lgb_xgb_cat.csv` | `52882462` | `0.9303` | Tests anchor-regularized blind weights; similar public result to the unregularized blend. |

Readout: these results do not challenge the existing `0.8124` public anchor.
They are useful evidence that the newly constructed blind benchmark is not yet
aligned enough with public LB to drive submissions by itself.

## Exact Commands for Current Best Blend

```bash
uv run python src/ensemble.py \
  --lgb submissions/submission_20260512_234155_lgbm_v2.csv \
  --xgb submissions/submission_20260513_001713_xgb_v1.csv \
  --cat submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv \
  --out submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  --weights 'lgb=0.35;xgb=0.35;cat=0.30'

uv run kaggle competitions submit \
  -c data-mining-2026-final-project \
  -f submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv \
  -m "LGB/XGB/CatBoost 35/35/30 blend with CatBoost tail2737"
```

The neighboring CatBoost 10%, 20%, 35%, 40%, and horizon-ramp blends use the
same legal LGB/XGB/CatBoost source files and `src/ensemble.py`; exact 2026-05-16
CatBoost training and prediction commands are recorded in
`docs/experiments/catboost_results_2026-05-16.md`.

## File Audit

| File | SHA-256 prefix | Notes |
|---|---|---|
| `submissions/ensemble_20260516_validation_weighted_v1.csv` | `b027d21fe911` | Legal LGB/XGB conservative candidate. |
| `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `e7e6946785f3` | Legal LGB/XGB conservative candidate. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `bee6f618828d` if restored locally | Legal three-model current public best. |
| `submissions/submission_20260520_115925_20260520_111133_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` | `0fba74bfc925` | Local CatBoost-only rerun; valid but not submitted. |
| `submissions/submission_20260520_163323_20260520_142456_lightgbm_two_stage_lgbm_micro_rolling_regularized_20260520.csv` | `25bf0deb1080` | Local LGBM micro rerun; valid but not submitted. |
| `submissions/submission_20260521_154656_20260521_153911_lightgbm_two_stage_lgbm_refit_full_lean_tail1095_20260521.csv` | `6be765ac8579` | Submitted 2026-05-21 quota probe; public `0.9380`. |
| `submissions/ensemble_20260521_blindfit_unregularized_lgb_w1w4_cat_w5.csv` | `ebb7b3af4087` | Submitted 2026-05-21 quota probe; public `0.9302`. |
| `submissions/ensemble_20260521_blindfit_regularized_horizon_lgb_xgb_cat.csv` | `443c8add294c` | Submitted 2026-05-21 quota probe; public `0.9303`. |

All local audit files have `2248` rows, matching submission columns, no NaN
values, and predictions clipped to `[0, 5]`. The submitted 2026-05-16 and
2026-05-19 CatBoost blend files are visible in Kaggle history with refs above;
backfill SHA-256 hashes if those exact files are restored or downloaded.

## Readout Plan After 5/22

When the 2026-05-22 static private leaderboard is released:

1. Compare private ranks of `ensemble_final`, the two LGB/XGB conservative hedges, and the CatBoost blend family.
2. If `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` wins privately, keep CatBoost near 30% and avoid heavier CatBoost weights.
3. If the 35%, 40%, or horizon-ramp CatBoost blends win privately despite weaker public scores, investigate private robustness by horizon before making one nearby follow-up submission.
4. If the LGB/XGB hedges beat the CatBoost family privately, roll back to the simpler two-model ensemble and invest in validation redesign.
5. Treat the 2026-05-20 local CatBoost and LGBM micro reruns as post-readout inputs for new blends, not as evidence that the current private-LB plan should change before 5/22.
