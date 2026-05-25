# Experiment and Submission Summary

Last updated: 2026-05-25

This file records the current legal model-selection state for the Final Project
progress check. It is intended to keep the slides, report, code, and Kaggle
submissions consistent.

## Current Leaderboard Interpretation

- Current public leaderboard best: `0.7922` MAE from `submissions/baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1.csv` (Kaggle ref `53003220`), which crosses Baseline 3 (`0.8056`) and places Team 5 rank 3 on the live public leaderboard.
- The `0.7922` artifact is a public-chase/final-selection horizon hedge and should not be described as a clean reportable modeling claim.
- Current best legal public score: `0.8124` MAE.
- Current best legal submission file: `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`.
- Previous legal LGB/XGB anchor: `submissions/ensemble_final.csv`, public MAE `0.8232`.
- 5/15 static private leaderboard: Team 5 was ranked 3, below Baseline 3 and above Baseline 2.
- Baseline 3 pressure: at the time of the TA announcement, at least 13 successful entries were needed to cross Public Baseline 3.
- Strategy implication: use public leaderboard gains cautiously and prioritize private robustness, reproducibility, and leakage-free validation evidence.

## 2026-05-21 Validation Architecture Update

The codebase now includes a stronger model-selection contract before future
submission decisions:

- `src/train.py`, `src/train_xgb.py`, and `src/train_catboost.py` support
  `--final-train-mode {last_fold,fold_ensemble,refit_full}`. The default is
  `refit_full`, so CV is used to estimate best iterations and the saved
  submission model is retrained on all legal rows instead of silently keeping
  only the last validation fold.
- `scripts/run_blind_backtest.py` creates a Kaggle-like pseudo-private test by
  masking a 91-day historical score window, rebuilding time-safe features, and
  scoring the next five weekly labels.
- `scripts/drift_report.py` compares train-tail candidates to the real test
  meteorological window using PSI, KS, quantile-Wasserstein distance,
  weather-only adversarial AUC, and weather+region/month adversarial AUC.
- `scripts/fit_blend_weights.py` fits non-negative per-horizon blend weights
  from OOF or blind-backtest predictions with anchor regularization, optional
  caps, and bootstrap stability around the current legal 35/35/30 blend.
- `scripts/leakage_sentinel.py` poisons hidden blind-window labels to assert
  that pseudo-test features do not change.
- Trainers and prediction now support `drop_feature_groups` for coarse feature
  ablations, including score history, climatology, rolling/EWM weather,
  long-window drought proxies, region stats, and CatBoost `region_id`.

These tools do not change the current best legal public submission by
themselves. They define the required evidence for promoting future reruns from
diagnostics to submission candidates.

## Current Best Legal Submission

| Field | Value |
|---|---|
| Submission | `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` |
| Kaggle ref | `52698259` |
| Public MAE | `0.8124` |
| Rows | `2248` |
| SHA-256 prefix | `bee6f618828d` if restored locally; exact raw CSV is not present in the current checkout |
| Blend | 35% `lgbm_v2` + 35% `xgb_v1` + 30% `catboost_lean_tail2737_regularized_500` |
| LightGBM source | `submissions/submission_20260512_234155_lgbm_v2.csv` |
| XGBoost source | `submissions/submission_20260513_001713_xgb_v1.csv` |
| CatBoost source | `submissions/submission_20260516_063135_20260516_060249_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` |
| Repro check | Command lineage is recorded in `docs/experiments/catboost_results_2026-05-16.md`. |

## Submitted Candidates

The submission set for the 2026-05-22 static private leaderboard contains the
original conservative LGB/XGB hedges plus a CatBoost blend family. All official
candidates below use reproducible legal sources; leaky diagnostics are excluded
from model selection.

| Submission | Kaggle ref | Blend | Public MAE | SHA-256 prefix | Purpose |
|---|---:|---|---:|---|---|
| `submissions/ensemble_20260516_validation_weighted_v1.csv` | `52689645` | Per-horizon LGBM weights: `0.5191,0.5157,0.5153,0.5132,0.5108` | `0.8232` | `b027d21fe911` | Minimal validation-weighted perturbation around `ensemble_final`. |
| `submissions/ensemble_20260516_xgb_tilt_40_60.csv` | `52689654` | 40% `lgbm_v2` / 60% `xgb_v1` | `0.8232` | `e7e6946785f3` | Small XGBoost-tilted hedge. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_45_45_10.csv` | `52698220` | 45% `lgbm_v2` / 45% `xgb_v1` / 10% CatBoost | `0.8163` | Not available locally | Low CatBoost-weight probe. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_40_40_20.csv` | `52698244` | 40% `lgbm_v2` / 40% `xgb_v1` / 20% CatBoost | `0.8126` | Not available locally | Conservative CatBoost blend near the best public score. |
| `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `52698259` | 35% `lgbm_v2` / 35% `xgb_v1` / 30% CatBoost | `0.8124` | `bee6f618828d` if restored locally | Current best legal public score and main 5/22 anchor. |
| `submissions/ensemble_20260519_lgb_xgb_cat2737_325_325_35.csv` | `52796551` | 32.5% `lgbm_v2` / 32.5% `xgb_v1` / 35% CatBoost | `0.8141` | Not available locally | Nearby CatBoost-weight robustness probe. |
| `submissions/ensemble_20260519_lgb_xgb_cat2737_30_30_40.csv` | `52796554` | 30% `lgbm_v2` / 30% `xgb_v1` / 40% CatBoost | `0.8168` | Not available locally | Higher CatBoost-weight probe; public regression suggests no further public-only increase. |
| `submissions/ensemble_20260519_lgb_xgb_cat2737_horizon_cat_ramp.csv` | `52796562` | Horizon-specific LGB/XGB/CatBoost ramp | `0.8138` | Not available locally | Tests whether CatBoost helps more on later horizons. |

Sanity checks for local candidate files:

- `2248` rows.
- Same columns and `region_id` order as `data/sample_submission.csv`.
- No missing predictions.
- Predictions clipped to `[0, 5]`.
- The leaky reproduction output was not used.

## 2026-05-20 Local Reruns

These runs are legal and useful for future blend experiments, but they were not
submitted as of this update.

| Run | Model | Validation | Local MAE | Submission | Notes |
|---|---|---|---:|---|---|
| `20260520_111133_catboost_two_stage_catboost_lean_tail2737_regularized_500` | CatBoost | Rolling origin | `0.2192` | `submissions/submission_20260520_115925_20260520_111133_catboost_two_stage_catboost_lean_tail2737_regularized_500.csv` | Valid rerun of the CatBoost tail2737 hypothesis; SHA-256 prefix `0fba74bfc925`. |
| `20260520_142456_lightgbm_two_stage_lgbm_micro_rolling_regularized_20260520` | LightGBM | Rolling origin | `0.2002` | `submissions/submission_20260520_163323_20260520_142456_lightgbm_two_stage_lgbm_micro_rolling_regularized_20260520.csv` | Memory-safe micro-profile LGBM; strong local score but should be treated as diagnostic/blend input until Kaggle evidence confirms it. |

## 2026-05-21 Quota Probe Submissions

These submissions intentionally spent daily Kaggle quota after the blind
backtest refactor. They are negative or diagnostic evidence, not replacements
for the `0.8124` public anchor.

| Submission | Kaggle ref | Public MAE | SHA-256 prefix | Blind evidence | Decision |
|---|---:|---:|---|---|---|
| `submissions/submission_20260521_154656_20260521_153911_lightgbm_two_stage_lgbm_refit_full_lean_tail1095_20260521.csv` | `52882449` | `0.9380` | `6be765ac8579` | Best single-model blind anchor, blind MAE `0.3549`. | Do not promote on public; keep as pseudo-private anchor evidence only. |
| `submissions/ensemble_20260521_blindfit_unregularized_lgb_w1w4_cat_w5.csv` | `52882455` | `0.9302` | `ebb7b3af4087` | Slightly beat LGBM in blind MAE (`0.3536`) by using LGBM for weeks 1-4 and LGB/Cat for week 5. | Best of this quota probe, but far from current public anchor. |
| `submissions/ensemble_20260521_blindfit_regularized_horizon_lgb_xgb_cat.csv` | `52882462` | `0.9303` | `443c8add294c` | Regularized blind-fit blend MAE `0.3614`; worse than LGBM alone in blind. | Useful contrast against unregularized blend, not a candidate anchor. |

Readout: the blind-validation candidates did not transfer to public LB. This
reinforces the need to compare 5/22 private leaderboard behavior before using
blind-backtest MAE as the sole promotion metric.

## 2026-05-22 Team 4 / Team 20 Reproduction Pass

The serious reproduction pass is recorded in
`docs/experiments/team4_team20_upgrade_summary_20260522.md`.

| Candidate | Evidence | Kaggle status | Decision |
|---|---|---|---|
| Team 4 region-prior LGBM | Rolling MAE `0.2680`; blind MAE `0.5068`; postprocessed blind MAE `0.4973` | Not submitted | Reject as an overfit/overconservative feature-prior variant. |
| Existing LGBM blind anchor postprocessed | Blind MAE `0.3549 -> 0.3418` | Not submitted | Keep as diagnostic; source public MAE is `0.9380`. |
| Weather-only TCN | Local val MAE `0.3151`; blind MAE `0.4040` | Not submitted | Reject as standalone; useful only as an architecture smoke. |
| Feature-fused TCN | Local val MAE `0.2654`; blind MAE `0.3508` | Submitted as ref `52900168`, public MAE `0.9450`, SHA-256 prefix `d62f9e32e312` | Negative public readout as a single model; keep as diversity input. |
| LGBM + feature-fused TCN loose horizon blend | Blind mean horizon MAE `0.3249`; weights `lgb=0.76,0.68,0.66,0.36,0.08` | Submitted as ref `52912241`, public MAE `0.9197`, SHA-256 prefix `005c49e67b41` | Negative public readout; keep as diagnostic only. |
| 4-seed feature-fused TCN equal ensemble | Blind MAE `0.3772`; seed blind range `0.3508` to `0.4164` | Not submitted; SHA-256 prefix `6181c02e69eb` | Reject; seed averaging worsened blind MAE. |
| LGBM + 4-seed TCNF fitted horizon blend | Blind MAE `0.3377`; weights `lgb=0.88,0.84,0.82,0.48,0.02` | Submitted as ref `52912252`, public MAE `0.9050`, SHA-256 prefix `8085cb7953cf` | Best TCNF-family public readout, still far behind the `0.8124` anchor. |

Readout: Team 20 style feature-fused sequence modeling produced a real
pseudo-private signal, especially for later horizons, but every submitted TCNF
family candidate transferred poorly to public LB. The next blend should combine
TCNF with the public-strong LGB/XGB/CatBoost anchor family, not only with the
blind-strong LGBM tail1095 anchor.

## 2026-05-22 Six-Submission Quota Probe

The course daily Kaggle submission limit was increased from 3 to 6. The first
extra-quota probe batch is recorded in
`docs/experiments/anchor_family_probe_summary_20260522.md`.

| Candidate | Kaggle ref | Public MAE | Decision |
|---|---:|---:|---|
| `submissions/ensemble_20260522_lgb_xgb_cat2737_375_375_25.csv` | `52928386` | `0.9546` | Reject; restored local component files are not safe anchor sources. |
| `submissions/ensemble_20260522_lgb_xgb_cat2737_soft_cat_ramp.csv` | `52928403` | `0.9561` | Reject; same artifact-lineage issue. |
| `submissions/ensemble_20260522_anchor_tcnf_cap5_global.csv` | `52928409` | `0.9509` | Reject; TCNF cap cannot rescue bad restored anchor sources. |
| `submissions/ensemble_20260522_anchor_tcnf_late_ramp_cap10.csv` | `52928422` | `0.9531` | Reject; diagnostic only. |

Readout: do not use the locally restored `restored_20260522_*_component` files
for future submissions. The `0.8124` public anchor remains valid as a historical
Kaggle submission, but new anchor-family probes must be built from reproducible
current runs or exact recovered historical CSVs.

## 2026-05-23 LGBM Feature Ablation Readout

The first high-risk feature-group ablation batch is recorded in
`docs/experiments/feature_ablation_lgbm_lean_tail1095_20260523_results.md`.

| Variant | Rolling MAE | Delta vs baseline | Blind MAE | Decision |
|---|---:|---:|---:|---|
| Baseline LGBM lean tail1095 | `0.3124` | `+0.0000` | Prior anchor `0.3549` | Reference. |
| `minus_climatology` | `0.3046` | `-0.0078` | `0.4463` | Reject; local-only improvement failed blind validation. |
| `minus_score_history` | `0.3645` | `+0.0521` | Not run | Reject; score history is essential. |
| `minus_long_drought_proxy` | `0.3645` | `+0.0521` | Not run | Reject; long drought proxies carry strong signal. |
| `minus_domain_indices` | `0.3144` | `+0.0020` | Not run | Reject for now; worsens week 5. |
| `minus_region_stats` | `0.3124` | `+0.0000` | Not run | No-op because region stats were disabled. |

Readout: rolling-origin ablation alone is not a promotion signal. The
`minus_climatology` run is a concrete example: it looked better locally but
substantially worse in Kaggle-like blind validation.

## 2026-05-23 GRU Quota Probe

The 2026-05-23 daily quota automation submitted the top two GRU-family
candidates after a live-history guard and full submission sanity check. Both
were legal/non-leaky, had `2248` rows, matched `data/sample_submission.csv`
column and region order, contained no NaN values, and stayed within `[0, 5]`.

| Candidate | Kaggle time | Public MAE | Decision |
|---|---:|---:|---|
| `submissions/submission_20260522_gru_family_constrained_stack.csv` | `2026-05-23 05:16:04.340000` | `0.9850` | Reject; GRU stack did not transfer to public LB. |
| `submissions/submission_20260521_193438_20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521_group3_ar_gru.csv` | `2026-05-23 05:16:16.130000` | `0.9916` | Reject; raw GRU was worse than the stack. |

Readout: pause GRU-family single models, stacks, affine calibration,
calendar-gated variants, and GRU consensus submissions. The next candidate
should return to reproducible boosted-tree anchor lineage or first explain why
blind/LOO evidence is misaligned with public LB.

## 2026-05-23 Baseline 3 Public-Chase Readout

After the missingness/shift queue finished, the strongest validation/blind
candidate was `shift_lgbm_lean_tail1095_seasonmatch2_20260523` with rolling
MAE `0.24574` and blind MAE `0.38370`. Because recent validation-strong
families transferred poorly to public LB, the quota was spent on exact-history
public-chase blends instead of on the new LGBM runs.

| Submission | Kaggle ref | Public MAE | Decision |
|---|---:|---:|---|
| `submissions/baseline3_public_chase_v0_cat35_08124_alpham0p50.csv` | `52946549` | `0.8786` | Wrong direction; high-amplitude extrapolation failed. |
| `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p10.csv` | `52946569` | `0.8049` | First Baseline 3 crossing. |
| `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p20.csv` | `52946598` | `0.8019` | Improved public score. |
| `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` | `52946620` | `0.7991` | Best public readout; Team 5 now beats Baseline 3. |

Readout: for leaderboard/selection status, Team 5 has crossed Baseline 3 with
public MAE `0.7991`. For report methodology, keep the legal `0.8124` CatBoost
blend as the clean anchor and label the `0.7991` file as `public-chase`.

## 2026-05-24 Manual Private-Hedge Readout

After the user raised the private-LB risk of the public-chase artifact, a
manual run spent all six useful 2026-05-24 quota slots on anchor-tilted hedge
submissions. Each file passed the standard submission sanity checks and remains
labelled `public-chase`, not `reportable`.

| Submission | Kaggle ref | Public MAE | Decision |
|---|---:|---:|---|
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p50.csv` | `52972055` | `0.7976` | Improved prior public best and moved closer to clean anchor. |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8.csv` | `52972132` | `0.7930` | Best public artifact; primary selectable public-chase candidate. |
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p65.csv` | `52972150` | `0.7982` | Still passes Baseline 3; uniform shift begins to regress public. |
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p80.csv` | `52972169` | `0.8022` | High-anchor-weight upper bound; still below Baseline 3. |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p4_0p45_0p55_0p7_0p85.csv` | `52972194` | `0.7933` | Near-best public and stronger private hedge than the `0.7930` file. |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p45_0p55_0p65_0p85_1p0.csv` | `52972219` | `0.7945` | Most anchor-tilted submitted horizon hedge; still below Baseline 3. |

Readout: Team 5 public MAE is now `0.7930`, ahead of Baseline 3 `0.8056`.
For final-selection discussion, keep the `0.7930` artifact as the public-best
choice and the `0.7933`/`0.7945` horizon hedges as private-risk alternatives.
For report methodology, the clean `0.8124` LGB/XGB/CatBoost lineage remains the
reportable anchor.

## 2026-05-25 Manual Private-Hedge Frontier Readout

After the 08:12 DNS/API blocker recovered, a manual local follow-up spent all
six 2026-05-25 quota slots on adjacent v1 horizon hedges. The slate was built
from exact recovered Team 5 submissions only: the `0.8094` source reference and
the clean `0.8124` reportable anchor. No private labels, external answers, or
`restored_20260522_*` / `restored_unverified_*` sources were used. Every file
passed the standard sanity checks before submission.

| Submission | Kaggle ref | Public MAE | SHA-12 | Decision |
|---|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v1_cat35_left_of_best_public_horizon_0p325_0p375_0p475_0p625_0p775.csv` | `53003193` | `0.7929` | `293531f60aa1` | Improved over the prior `0.7930` best but farther from the clean anchor. |
| `submissions/baseline3_private_hedge_v1_cat35_midpoint_best_to_nearbest_horizon_0p375_0p425_0p525_0p675_0p825.csv` | `53003210` | `0.7931` | `eef241fc2218` | Near-best midpoint hedge. |
| `submissions/baseline3_private_hedge_v1_cat35_preserve_early_anchor_late_horizon_0p35_0p425_0p55_0p725_0p9.csv` | `53003215` | `0.7927` | `2b58d70fc4a7` | Strong public score while moving late horizons closer to the clean anchor. |
| `submissions/baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1.csv` | `53003220` | `0.7922` | `695c62a4eb28` | New best public artifact and primary selectable public-chase candidate. |
| `submissions/baseline3_private_hedge_v1_cat35_robust_mid_frontier_horizon_0p4_0p475_0p575_0p75_0p9.csv` | `53003222` | `0.7933` | `7adc27154921` | Stronger private hedge alternative with lower delta to the reportable anchor. |
| `submissions/baseline3_private_hedge_v1_cat35_smooth_high_anchor_horizon_0p425_0p5_0p6_0p8_0p95.csv` | `53003227` | `0.7937` | `92f02f3f2a20` | Most anchor-tilted v1 hedge submitted today while still below Baseline 3. |

Readout: Team 5 public MAE is now `0.7922`, ahead of Baseline 3 `0.8056` and
Team 20 `0.7942`. For final-selection discussion, use ref `53003220` as the
public-best artifact and refs `53003222` / `53003227` as stronger private-risk
hedge alternatives. For report methodology, the clean `0.8124`
LGB/XGB/CatBoost lineage remains the reportable anchor.

## Experiment Table

| Experiment | Model | Feature setup | Validation | Local MAE | Public MAE / Status | Notes |
|---|---|---|---|---:|---:|---|
| `baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.30,0.40,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7922` | Current best public artifact; `public-chase`, not reportable method claim. |
| `baseline3_private_hedge_v1_cat35_smooth_high_anchor_horizon_0p425_0p5_0p6_0p8_0p95` | Exact-history horizon hedge | More anchor-tilted v1 hedge with alphas `[0.425,0.50,0.60,0.80,0.95]` | Public LB feedback | N/A | `0.7937` | Stronger private-risk alternative while still below Baseline 3. |
| `baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.35,0.40,0.50,0.65,0.80]` | Public LB feedback | N/A | `0.7930` | Previous public-best artifact; `public-chase`, not reportable method claim. |
| `baseline3_public_chase_v0_cat35_08124_alphap0p35` | Exact-history public-chase blend | 65% recovered `0.8094` public reference + 35% recovered legal `0.8124` anchor | Public LB feedback | N/A | `0.7991` | Crossed Baseline 3; not a clean reportable method claim. |
| `lgbm_v2` | LightGBM direct horizon | 337 features, score history | Chronological holdout | `0.6942` | Component of `0.8232` and `0.8124` ensembles | Legal baseline model. |
| `xgb_v1` | XGBoost direct horizon | 337 features, score history | Chronological holdout | `0.7150` | Component of `0.8232` and `0.8124` ensembles | Legal diversity model. |
| `catboost_lean_tail2737_regularized_500` | CatBoost direct horizon | Lean profile, 91-day-gapped score history, native categorical features | Rolling origin | `0.2212`; `0.2192` rerun | Component of `0.8124` ensemble | Validation scale differs from holdout; do not compare directly with LGB/XGB holdout MAE. |
| `lgbm_micro_rolling_regularized_20260520` | LightGBM direct horizon | Micro profile, 91-day-gapped score history, regularized | Rolling origin | `0.2002` | Local diagnostic / candidate blend input | Strong local run; needs Kaggle or pseudo-private evidence before becoming an anchor. |
| `ensemble_20260516_lgb_xgb_cat2737_35_35_30` | Three-model ensemble | `lgbm_v2` + `xgb_v1` + CatBoost tail2737 | N/A | N/A | `0.8124` | Current best legal public submission. |
| `ensemble_20260519_lgb_xgb_cat2737_horizon_cat_ramp` | Three-model ensemble | Horizon-specific CatBoost ramp | Private-robustness probe | N/A | `0.8138` | Keep for 5/22 readout; do not continue unless private rank supports it. |
| `ensemble_final` | 50/50 ensemble | `lgbm_v2` + `xgb_v1` | N/A | N/A | `0.8232` | Previous legal anchor. |
| `lgbm_direct` | LightGBM direct weather-only | 318 weather-only features | Chronological holdout | `0.6770` | `0.8640` strategy result | Good local score, weak public generalization. |
| `xgb_direct` | XGBoost direct weather-only | 318 weather-only features | Chronological holdout | `0.7320` | Used in strategy experiment | Did not beat score-history ensembles. |
| `ensemble_strategy_b_long_term` | 50/50 ensemble | Long-term weather-only | N/A | N/A | `0.8604` | Shows pure weather cannot replace score-history context. |
| `lgbm_gap_anomaly_regularized_lean_v2` | LightGBM direct horizon | Lean, 91-day-gapped score history, regularized | Rolling origin | `0.1915` | Diagnostic / under review | Local score appears optimistic; do not treat as champion without further validation. |
| `lgbm_leaky_repro` | LightGBM diagnostic | Lean, score gap `0` | Holdout | `0.2321` | Discarded | Leaky diagnostic only; excluded from model selection. |

## Terminology Note

Some run directory names and `model_family` strings still contain `two_stage`.
These are retained for compatibility with existing artifacts. In current
documentation, the active implementation should be described as direct-horizon
boosted-tree models with leakage-aware 91-day-gapped score-history features, not
as the older separate score-estimation pipeline.

## Reproducibility Requirements

- Keep `Initial Submission (Leak)` and `lgbm_leaky_repro` out of official model claims.
- If the report mentions a Kaggle score, it must point to a submission CSV and a reproducible experiment lineage.
- When comparing local MAE, state the validation strategy. Holdout and rolling-origin MAE are not directly comparable.
- Future submissions should include the hypothesis being tested, source model files, blend weights, public score, and whether the run is legal.
- Do not overwrite existing run directories; create a new `--experiment-name` for each method change.
