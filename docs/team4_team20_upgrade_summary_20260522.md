# Team 4 / Team 20 Upgrade Summary - 2026-05-22

This note records the serious reproduction pass inspired by the stronger
teams' progress-check talks. The goal was not to chase a new architecture for
its own sake, but to test two missing capabilities in our validation-first
workflow:

- Team 4 style region historical priors plus horizon-wise postprocessing.
- Team 20 style lightweight TCN and feature-fused TCN sequence signals.

## Protocol

- Blind backtest origins: `5,13,26,39,52,78,104`.
- Blind window: `91` days with hidden scores masked.
- History context: `history-tail-days=1100`.
- Submission sanity checks: `2248` rows, sample-submission region order, no
  missing predictions, predictions clipped to `[0, 5]`.
- Kaggle quota status: after the TCN single-model submission, the 2026-05-21
  Kaggle-day quota reached `6/6`; the horizon blend could not be submitted.

## Team 4 Features and Postprocessing

Implemented in `src/features.py`:

- Region severity ratios:
  - `region_score_eq0_ratio`
  - `region_score_ge1_ratio`
  - `region_score_ge3_ratio`
  - `region_score_eq5_ratio`
- Recent region priors for `365` and `730` days:
  - recent mean, max, zero ratio, high-severity ratio.
- Smoothed region/month prior:
  - `region_month_score_mean_smooth`.

Implemented in `scripts/tune_postprocess.py`:

- Horizon-wise clipped affine calibration.
- Grid search over scale and bias with regularization toward identity.
- Optional application to a submission CSV.

### Result

| Candidate | Local / blind evidence | Decision |
|---|---:|---|
| `lgbm_team4_region_priors_lean_tail1095_20260522` | Rolling MAE `0.2680`; blind MAE `0.5068` | Reject. Direct full-train region priors are too optimistic / too conservative. |
| Same run after postprocess | Blind MAE `0.4973` | Reject. Calibration cannot rescue the bad base. |
| Existing LGBM blind anchor after postprocess | Blind MAE `0.3549 -> 0.3418` | Keep as diagnostic only because its public MAE is weak (`0.9380`). |

Team 4's idea is valid, but our first direct region-prior implementation did
not transfer. The useful part is the postprocess tool; the feature-prior part
needs a time-safe ablation or lower-variance smoothing before promotion.

## Team 20 TCN Reproduction

Implemented in:

- `src/train_tcn.py`
- `src/predict_tcn.py`
- `scripts/run_tcn_blind_backtest.py`

The TCN uses a 91-day weather sequence, region embedding, cyclic date context,
and a direct 5-horizon head. The `feature_fused` variant adds time-safe visible
score and region/month priors computed only from labels at least `91` days
before the forecast origin.

### Single-Model Results

| Model | Run | Local val MAE | Blind MAE | w1 | w2 | w3 | w4 | w5 | Public MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weather-only TCN | `20260522_021808_tcn_tcn_weather_tail1825_8ep_20260522` | `0.3151` | `0.4040` | `0.4053` | `0.3937` | `0.3922` | `0.4044` | `0.4245` | Not submitted |
| Feature-fused TCN | `20260522_022815_tcn_tcn_feature_fused_tail1825_8ep_20260522` | `0.2654` | `0.3508` | `0.3495` | `0.3362` | `0.3357` | `0.3546` | `0.3780` | `0.9450` |

Feature fusion is the important result: it cuts blind MAE by about `0.053`
versus weather-only TCN and slightly beats the LGBM blind anchor (`0.3549`).
However, the public score is poor, so it should not replace the public anchor.
Its value is diversity, especially for later horizons.

Submitted TCN artifact:

| Field | Value |
|---|---|
| Submission | `submissions/submission_20260522_023818_20260522_022815_tcn_tcn_feature_fused_tail1825_8ep_20260522_feature_fused.csv` |
| Kaggle ref | `52900168` |
| Public MAE | `0.9450` |
| SHA-256 prefix | `d62f9e32e312` |

## LGBM + Feature-Fused TCN Blend

The constrained blend used blind predictions from:

- LGBM blind anchor:
  `experiments/blind_20260521_lgbm_refit_full_lean_tail1095/blind_backtest_rows.csv`
- Feature-fused TCN:
  `experiments/blind_20260522_tcn_feature_fused_tail1825_h1100/blind_backtest_rows.csv`

### Conservative Regularized Blend

Anchor: `lgb=0.60;tcnf=0.40`, `lambda_reg=0.05`.

| Horizon | LGBM | TCNF | Blind MAE | Bootstrap TCNF mean/std |
|---|---:|---:|---:|---:|
| week 1 | `0.70` | `0.30` | `0.2961` | `0.29 +/- 0.07` |
| week 2 | `0.66` | `0.34` | `0.2983` | `0.33 +/- 0.06` |
| week 3 | `0.64` | `0.36` | `0.3041` | `0.35 +/- 0.06` |
| week 4 | `0.48` | `0.52` | `0.3504` | `0.52 +/- 0.07` |
| week 5 | `0.30` | `0.70` | `0.3871` | `0.69 +/- 0.09` |

Mean horizon MAE: `0.3272`.

Artifact:

- `submissions/submission_20260522_lgb_tcnf_horizon_blend_conservative.csv`
- SHA-256 prefix: `a4a1744ec62c`

### Loose Blend

Anchor: `lgb=0.50;tcnf=0.50`, `lambda_reg=0.01`.

| Horizon | LGBM | TCNF | Blind MAE | Bootstrap TCNF mean/std |
|---|---:|---:|---:|---:|
| week 1 | `0.76` | `0.24` | `0.2951` | `0.24 +/- 0.11` |
| week 2 | `0.68` | `0.32` | `0.2980` | `0.31 +/- 0.10` |
| week 3 | `0.66` | `0.34` | `0.3039` | `0.35 +/- 0.10` |
| week 4 | `0.36` | `0.64` | `0.3486` | `0.65 +/- 0.13` |
| week 5 | `0.08` | `0.92` | `0.3788` | `0.89 +/- 0.12` |

Mean horizon MAE: `0.3249`.

Artifact:

- `submissions/submission_20260522_lgb_tcnf_horizon_blend_loose.csv`
- SHA-256 prefix: `005c49e67b41`

The loose blend is the next Kaggle candidate after quota reset. It is
validation-strong, but it carries public-LB risk because both source models are
weak on public (`0.9380` and `0.9450`). Submit it as a diagnostic late-horizon
blend, not as a replacement for the `0.8124` anchor.

## Decisions

- Keep current public anchor: `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`
  with public MAE `0.8124`.
- Do not submit more pure TCN variants until public/private evidence supports
  sequence-only signals.
- Submit the loose LGBM+TCNF horizon blend after quota reset if we want one
  controlled probe of Team 20 style diversity.
- Do not submit the Team 4 region-prior LGBM run.
- Next technical step: blend TCNF with the existing public-strong
  LGB/XGB/CatBoost anchor family, not only with the blind-strong LGBM anchor.
