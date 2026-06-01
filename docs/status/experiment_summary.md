# Experiment and Submission Summary

Last updated: 2026-06-01

This file records the current legal model-selection state for the Final Project
progress check. It is intended to keep the slides, report, code, and Kaggle
submissions consistent.

## Current Leaderboard Interpretation

- Current public leaderboard best: `0.7905` MAE from `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` (Kaggle ref `53204258`), which crosses Baseline 3 (`0.8056`) and places Team 5 rank 6 on the live public leaderboard after the 2026-06-01 v9 batch.
- The `0.7905` artifact is a public-chase/final-selection horizon hedge and should not be described as a clean reportable modeling claim.
- Static Private / final-selection pair: select ref `53204258` as the public-best tie / private-safer slot and ref `53259683` (`0.7909`, SHA-12 `262f683b87b6`) as the near-public private-robust hedge slot. Ref `53204319` remains the pure-delta fallback; the previous v7 pair `53186508` / `53186571` remains the manual fallback if the team prefers the earlier selection.
- Current best legal public score: `0.8124` MAE.
- Current best legal submission file: `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`.
- Previous legal LGB/XGB anchor: `submissions/ensemble_final.csv`, public MAE `0.8232`.
- 5/15 static private leaderboard: Team 5 was ranked 3, below Baseline 3 and above Baseline 2.
- 5/29 static private leaderboard snapshot: Team 5 was ranked 5 at release time `5/29 23:46:29`; the sheet notes this is not the final ranking, so the official Kaggle private selection remains risk-sensitive.
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
| `submissions/baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1.csv` | `53003220` | `0.7922` | `695c62a4eb28` | Best v1 public artifact; later superseded by v2 ref `53038031`. |
| `submissions/baseline3_private_hedge_v1_cat35_robust_mid_frontier_horizon_0p4_0p475_0p575_0p75_0p9.csv` | `53003222` | `0.7933` | `7adc27154921` | Stronger private hedge alternative with lower delta to the reportable anchor. |
| `submissions/baseline3_private_hedge_v1_cat35_smooth_high_anchor_horizon_0p425_0p5_0p6_0p8_0p95.csv` | `53003227` | `0.7937` | `92f02f3f2a20` | Most anchor-tilted v1 hedge submitted today while still below Baseline 3. |

Readout: Team 5 public MAE is now `0.7922`, ahead of Baseline 3 `0.8056` and
Team 20 `0.7942`. At this v1 checkpoint, ref `53003220` was the public-best
artifact and refs `53003222` / `53003227` were stronger private-risk
hedge alternatives. For report methodology, the clean `0.8124`
LGB/XGB/CatBoost lineage remains the reportable anchor.

## 2026-05-26 Manual Private-Hedge Frontier v2 Readout

After the 2026-05-26 08:00 Taipei quota reset, a manual local follow-up spent
all six quota slots on v2 local horizon probes around the previous best
`[0.30,0.40,0.55,0.75,1.00]`. The slate again used exact recovered Team 5
submissions only: the `0.8094` source reference and the clean `0.8124`
reportable anchor. No private labels, external answers, or
`restored_20260522_*` / `restored_unverified_*` sources were used. Every file
passed the standard sanity checks before submission.

| Submission | Kaggle ref | Public MAE | SHA-12 | Decision |
|---|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v2_cat35_lower_w1_only_horizon_0p275_0p4_0p55_0p75_1.csv` | `53038024` | `0.7920` | `d7a9675828cf` | Improved over v1 by lowering week-1 anchor weight only. |
| `submissions/baseline3_private_hedge_v2_cat35_lower_w2_only_horizon_0p3_0p375_0p55_0p75_1.csv` | `53038028` | `0.7922` | `c14f6b929192` | Matched the prior best. |
| `submissions/baseline3_private_hedge_v2_cat35_lower_early_pair_horizon_0p25_0p375_0p55_0p75_1.csv` | `53038031` | `0.7917` | `eae8c5b0dc6f` | New best public artifact and primary selectable public-chase candidate. |
| `submissions/baseline3_private_hedge_v2_cat35_late_more_anchor_horizon_0p3_0p4_0p575_0p8_1.csv` | `53038033` | `0.7923` | `a495c0ef028e` | More anchor-tilted late-horizon hedge; still near public best. |
| `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `ec7984e858b2` | Stronger private hedge alternative with lower delta to the reportable anchor. |
| `submissions/baseline3_private_hedge_v2_cat35_smooth_high_anchor_v2_horizon_0p35_0p45_0p6_0p8_1.csv` | `53038040` | `0.7929` | `3b16648a2c5c` | Most anchor-tilted v2 hedge submitted today while still below Baseline 3. |

Readout: Team 5 public MAE is now `0.7917`, ahead of Baseline 3 `0.8056`,
Team 1 `0.8043`, and Team 20 `0.7942`. For final-selection discussion, use
ref `53038031` as the public-best artifact and refs `53038040` / `53038036` /
`53038033` as stronger private-risk hedge alternatives. For report methodology,
the clean `0.8124` LGB/XGB/CatBoost lineage remains the reportable anchor.

## 2026-05-26 Teammate Candidate Screen

A teammate-provided attachment,
`/Users/skyhong/.codex/attachments/3a7c1688-1166-49c5-97c3-5a6b48128735/submission_20260526_145450_20260521_164434_lightgbm_two_stage_lgbm_v3_enhanced.csv`,
was screened as a possible 2026-05-27 post-reset probe. It passed the standard
submission-format sanity gate with SHA-256
`66f7815e7ff30430fe543d1d1df0d3b1c0167594edbe2a34453418c25c1240cf`, but no
matching run directory, exact filename, or exact local artifact was found in
the checkout. Treat it as `external-teammate-candidate` /
`provenance_pending`, not as a reportable method claim.

The prediction surface is substantially different from the successful v2
frontier: mean prediction `0.379247` with `2458` exact zeros, versus roughly
`0.94` to `0.95` mean and `120` exact zeros for the then-current v2 public-best and
private-hedge files. Mean absolute difference to then-current public-best ref
`53038031` is `0.803732` with flat correlation `0.394512`; mean absolute
difference to the clean reportable anchor ref `52698259` is `0.695846` with flat
correlation `0.526317`.

Decision update: per user directive on 2026-05-26, queue this candidate as rank
`1` for the first 2026-05-27 post-reset submission slot, despite the risk
screen. The staged path is
`submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`.
Submit only after live Kaggle re-gating, exact-file sanity revalidation, SHA
confirmation, and no evidence of private labels, external answers, unauthorized
data, `restored_20260522_*`, or `restored_unverified_*` sources. Readout:
`experiments/baseline3_push_20260523/teammate_candidate_20260526_1503/candidate_readout.json`;
queue:
`experiments/baseline3_push_20260523/teammate_candidate_20260526_1503/next_submission_queue_20260527.json`.

2026-05-27 live readout: the teammate candidate was already submitted as ref
`53074655` by `veldahung` and scored public MAE `1.0685`. It is a negative
public readout and remains `provenance_pending`; do not promote it.

## 2026-05-27 Manual Private-Hedge Frontier v3 Readout

After the teammate first-queue readout, the remaining five 2026-05-27 quota
slots were spent on v3 local horizon probes around the previous best
`[0.25,0.375,0.55,0.75,1.00]`. The slate again used exact recovered Team 5
submissions only: the `0.8094` source reference and the clean `0.8124`
reportable anchor. No private labels, external answers, or
`restored_20260522_*` / `restored_unverified_*` sources were used. Every file
passed the standard sanity checks before submission.

| Submission | Kaggle ref | Public MAE | SHA-12 | Decision |
|---|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v3_cat35_lower_w1_more_horizon_0p225_0p375_0p55_0p75_1.csv` | `53074980` | `0.7915` | `003351676087` | New public-best tie and selected public-chase candidate because it is closer to the clean anchor than ref `53075001`. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_w2_more_horizon_0p25_0p35_0p55_0p75_1.csv` | `53074990` | `0.7917` | `ab5bad17e277` | Matched the prior best. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_early_more_horizon_0p225_0p35_0p55_0p75_1.csv` | `53075001` | `0.7915` | `a76ae651b3f6` | Public-best tie, but farther from the clean anchor. |
| `submissions/baseline3_private_hedge_v3_cat35_week3_slight_down_horizon_0p25_0p375_0p525_0p75_1.csv` | `53075011` | `0.7918` | `0bd41f3de686` | Slight public regression; week-3 lower anchor is not the main driver. |
| `submissions/baseline3_private_hedge_v3_cat35_public_best_late_more_anchor_horizon_0p25_0p375_0p575_0p8_1.csv` | `53075022` | `0.7918` | `20aa554510a5` | Strongest same-day private hedge alternative. |

Readout: Team 5 public MAE is now `0.7915`, ahead of Baseline 3 `0.8056`,
Team 1 `0.8043`, and Team 20 `0.7942`. For final-selection discussion, use
ref `53074980` as the public-best artifact; ref `53075001` is a same-score but
less private-robust tie. Same-day private hedge ref `53075022` and cross-day
refs `53038040` / `53038036` / `53038033` remain stronger private-risk
alternatives. For report methodology, the clean `0.8124` LGB/XGB/CatBoost
lineage remains the reportable anchor.

## 2026-05-28 Manual Private-Hedge Frontier v4 Readout

After a fresh live Kaggle gate on 2026-05-28, the first six quota slots were
spent on v4 local horizon probes around the v3 week-1 signal before the daily
limit was confirmed as `10/day`. The slate again used exact
recovered Team 5 submissions only: the `0.8094` source reference and the clean
`0.8124` reportable anchor. No private labels, external answers, or
`restored_20260522_*` / `restored_unverified_*` sources were used. Every file
passed the standard sanity checks before submission.

| Submission | Kaggle ref | Public MAE | SHA-12 | Decision |
|---|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p2125_keep_shape_horizon_0p2125_0p375_0p55_0p75_1.csv` | `53109107` | `0.7914` | `ec1cea965464` | Improved over v3 by moving slightly left on week 1. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_keep_shape_horizon_0p2_0p375_0p55_0p75_1.csv` | `53109122` | `0.7913` | `4138156b1e10` | Continued the week-1 public-side trend. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` | `53109133` | `0.7912` | `c02dc54a2a79` | v4 public-best at the time; later superseded by v5, still a near-public private hedge. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_w2_mid_horizon_0p225_0p3625_0p55_0p75_1.csv` | `53109150` | `0.7915` | `56284d0e5a27` | Near-best hedge with less week-1 drift. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_late_mid_anchor_horizon_0p225_0p375_0p565_0p775_1.csv` | `53109157` | `0.7916` | `4569505de0e4` | Same-day late-anchor private-risk hedge. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_late_anchor_horizon_0p2_0p375_0p575_0p8_1.csv` | `53109166` | `0.7914` | `5d6406745e87` | Strongest same-day private hedge among v4 candidates. |

Readout at the time: Team 5 public MAE became `0.7912`, ahead of Baseline 3 `0.8056`,
Team 1 `0.8043`, and Team 20 `0.7942`. For final-selection discussion, use
ref `53109133` as the v4 public-best artifact. Same-day private hedge refs
`53109166`, `53109157`, and `53109150`, plus cross-day refs `53075022`,
`53038040`, `53038036`, and `53038033`, remain private-risk alternatives. For
report methodology, the clean `0.8124` LGB/XGB/CatBoost lineage remains the
reportable anchor.

## 2026-05-28 Quota-10 v5 Public-Chase Readout

After the user confirmed the daily Kaggle limit is now `10/day` (updated from
6/day), the remaining four 2026-05-28 UTC slots were used on v5 lower-week1
keep-shape probes. The teammate CSV stayed first in the manual queue, but live
history confirmed it was already submitted as ref `53074655` with public MAE
`1.0685`, so the duplicate guard skipped it. All v5 files passed the standard
sanity checks before submission.

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to clean anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p1625_keep_shape_horizon_0p1625_0p375_0p55_0p75_1.csv` | `53110653` | `0.7911` | `530c7e912705` | `0.184780` | Improved public best while continuing the lower week-1 curve. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p15_keep_shape_horizon_0p15_0p375_0p55_0p75_1.csv` | `53110796` | `0.7910` | `524a8ddebdd6` | `0.185865` | Improved again; public-side trend remained monotone. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p125_keep_shape_horizon_0p125_0p375_0p55_0p75_1.csv` | `53110803` | `0.7909` | `7fc12846493c` | `0.188035` | Improved public score, with higher distance from the clean anchor. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p10_keep_shape_horizon_0p1_0p375_0p55_0p75_1.csv` | `53110808` | `0.7907` | `12016072d85f` | `0.190205` | v5 public-best at the time; later superseded by v7 ref `53186508`. |

Readout: Team 5 public MAE is now `0.7907`, ahead of Baseline 3 `0.8056`,
Team 1 `0.8043`, and Team 20 `0.7942`. For final-selection discussion, use
ref `53110808` as the public-best artifact, but keep ref `53109133` (`0.7912`)
as the same-day near-public private hedge because it is closer to the clean
`0.8124` anchor. Ref `53038036` (`0.7925`) is the stronger private fallback
inside the current `0.0020` public window. Quota is now `10/10` used for
2026-05-28 UTC.

Queue and submission log:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260529_queue_20260528_1555/next_submission_queue_20260529.json`;
`experiments/baseline3_push_20260523/private_hedge_frontier_20260529_queue_20260528_1555/quota10_v5_submit_20260528_1648.json`.

## 2026-05-28 Prepared v6 Private-Robust Backup

A v6 backup queue was also generated and revalidated after the v5 quota-10
readout. These candidates are not meant to supersede the public-best v5 file
automatically. They are a private-risk backup around the v4 late-anchor signal:
the alphas add week-2 or week-3/4 anchor weight, so they are closer to the clean
`0.8124` anchor than the pure public-best v5 file.

| Candidate | SHA-12 | Delta to clean anchor | Role |
|---|---:|---:|---|
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv` | `2a9f1230b65f` | `0.178586` | Public-near backup with v5 week-1 and late-anchor protection. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p20_late_anchor_w2_0p40_horizon_0p2_0p4_0p575_0p8_1.csv` | `d355e20f2ce1` | `0.173202` | Adds week-2 anchor to the v4 `53109166` hedge. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p175_late_anchor_w2_0p40_horizon_0p175_0p4_0p575_0p8_1.csv` | `efcbbfe922f6` | `0.175372` | Public-best week-1 setting with more week-2 and late anchor. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p20_stronger_late_anchor_horizon_0p2_0p375_0p6_0p825_1.csv` | `5838dee864bb` | `0.171162` | Stronger late-anchor version of the v4 private hedge. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p225_stronger_late_anchor_horizon_0p225_0p375_0p6_0p825_1.csv` | `e337885b660d` | `0.168992` | V3 early shape with stronger late-anchor protection. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p25_stronger_late_anchor_horizon_0p25_0p375_0p6_0p825_1.csv` | `86d7dd4da810` | `0.166822` | Most conservative v6 backup, closest to the clean anchor. |

Readout:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260530_backup_20260528_1610/frontier_readout.json`.

## 2026-05-30 Quota-10 v7 Static Private Readout

After the 5/29 static private snapshot showed Team 5 rank `5`, the 2026-05-30
live gate confirmed that no Team 5 submissions had been spent that day and that
the current policy is `10/day`. All ten quota slots were used on a v7
public/private frontier around the week-1 public optimum and late-horizon
anchor protection. Every file passed the standard sanity checks before
submission, including `2248` rows, exact sample columns and region order, no
NaN, prediction range in `[0,5]`, SHA-256 recording, and no
`restored_20260522_*` / `restored_unverified_*` source pattern.

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to clean anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_late_anchor_horizon_0p1_0p375_0p575_0p8_1.csv` | `53186451` | `0.7908` | `2669fcc54a42` | `0.184011` | Late-anchor hedge at the previous v5 week-1 setting. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_keep_shape_horizon_0p075_0p375_0p55_0p75_1.csv` | `53186458` | `0.7906` | `79792c199111` | `0.192375` | Near-public keep-shape probe. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_late_anchor_horizon_0p075_0p375_0p575_0p8_1.csv` | `53186470` | `0.7907` | `81fe2fae4b96` | `0.186181` | Near-public private hedge. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_keep_shape_horizon_0p05_0p375_0p55_0p75_1.csv` | `53186480` | `0.7906` | `63b3987da5fa` | `0.194544` | Aggressive week-1 public-side probe. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_late_anchor_horizon_0p05_0p375_0p575_0p8_1.csv` | `53186493` | `0.7906` | `f1c733d6d2e2` | `0.188351` | Public-biased alternate hedge. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1.csv` | `53186508` | `0.7905` | `9b452689c221` | `0.196714` | v7 public-best at the time; later superseded for selection by v8 ref `53204258`. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_late_anchor_horizon_0p025_0p375_0p575_0p8_1.csv` | `53186528` | `0.7906` | `4dfc45e15a42` | `0.190521` | Near-public late-anchor hedge. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p00_keep_shape_horizon_0_0p375_0p55_0p75_1.csv` | `53186548` | `0.7906` | `f105abad604f` | `0.198884` | Zero week-1 anchor boundary. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_stronger_late_anchor_horizon_0p075_0p375_0p6_0p825_1.csv` | `53186562` | `0.7908` | `6cb910b58f59` | `0.182012` | Stronger late-anchor hedge. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1.csv` | `53186571` | `0.7909` | `6c2e32e92647` | `0.179842` | v7 Static Private slot 2; later kept as fallback after v8 ref `53204319`. |

Readout: Team 5 public MAE is now `0.7905`, ahead of Baseline 3 `0.8056`.
For Static Private / final selection, manually select refs `53186508` and
`53186571`. The first maximizes public score; the second gives a near-public
late-anchor hedge only `0.0004` behind the public-best while reducing distance
to the clean reportable anchor. Public-biased alternate ref `53186493` is
available if the second slot must stay closer to public-best. Quota is `10/10`
used for 2026-05-30; next reset is `2026-05-31T08:00:00+08:00`.

Readouts:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260530_quota_20260530_2155/frontier_readout.json`;
`experiments/baseline3_push_20260523/final_selection_matrix_20260530_2205/final_selection_matrix.json`.

## 2026-05-31 Quota-10 v8 Static Private Readout

After the 2026-05-31 08:00 Taipei reset, live Kaggle history showed `0/10`
Team 5 submissions for the new UTC quota day. The v8 slate used all ten slots
on a refinement of the v7 public plateau plus stronger private hedges. The live
leaderboard after the batch kept Team 5 at displayed public MAE `0.7905` and
rank `5`; Team 15 had moved ahead at `0.7799`. Every v8 file passed the
standard submission sanity checks and remains a `public-chase` /
final-selection artifact, not a reportable method claim.

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to clean anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p0125_keep_shape_horizon_0p0125_0p375_0p55_0p75_1.csv` | `53204251` | `0.7906` | `d12fb86c376f` | `0.197799` | Fine public-plateau probe; did not improve. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` | `53204258` | `0.7905` | `390565517cb3` | `0.195629` | Updated public-best tie; lower clean-anchor delta than v7 ref `53186508`. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_m0p025_keep_shape_horizon_m0p025_0p375_0p55_0p75_1.csv` | `53204263` | `0.7907` | `3186f3882dbb` | `0.201035` | Left-of-zero week-1 boundary regressed. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w2_0p35_keep_shape_horizon_0p025_0p35_0p55_0p75_1.csv` | `53204270` | `0.7905` | `ecb762e5373a` | `0.198844` | Public tie but farther from clean anchor than ref `53204258`. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w2_0p40_keep_shape_horizon_0p025_0p4_0p55_0p75_1.csv` | `53204281` | `0.7906` | `19b70bdfe92c` | `0.194585` | Slight public regression; useful week-2 anchor evidence. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w3_0p525_keep_shape_horizon_0p025_0p375_0p525_0p75_1.csv` | `53204287` | `0.7906` | `a494bf9fd884` | `0.198860` | Week-3 lower-anchor probe regressed slightly. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w4_0p725_keep_shape_horizon_0p025_0p375_0p55_0p725_1.csv` | `53204292` | `0.7905` | `d94440195bf0` | `0.198738` | Public-biased alternate if slot 2 must keep displayed public `0.7905`. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_late_soft_anchor_horizon_0p025_0p375_0p5625_0p775_1.csv` | `53204297` | `0.7906` | `874de0bed5a2` | `0.193618` | Soft late-anchor hedge; near public-best. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_stronger_late_anchor_horizon_0p025_0p375_0p6_0p825_1.csv` | `53204307` | `0.7907` | `a2219cb4bdb5` | `0.186352` | Stronger late-anchor hedge but less attractive than ref `53204319`. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1.csv` | `53204319` | `0.7910` | `a5a5e9188d9e` | `0.177713` | v8 Static Private slot 2; now kept as pure-delta fallback after v9 ref `53259683`. |

Readout: v8 did not beat the displayed public MAE `0.7905`, but it improved
the final-selection tradeoff. Ref `53204258` ties the public best and is closer
to the clean anchor than v7 ref `53186508`; ref `53204319` became the v8 private-robust hedge slot and is now kept as the pure-delta fallback after v9 ref `53259683`. Quota is `10/10` used for 2026-05-31; next reset is
`2026-06-01T08:00:00+08:00`.

Readouts:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260531_quota_20260531_1245/frontier_readout.json`;
`experiments/baseline3_push_20260523/final_selection_matrix_20260531_1255/final_selection_matrix.json`.


## 2026-06-01 Quota-10 v9 Static Private Readout

After the 2026-06-01 08:00 Taipei reset, live Kaggle history showed `0/10`
Team 5 submissions for the new UTC quota day. The v9 slate used all ten slots
on crossed v8 public-tie signals plus stronger week-2 and late-anchor hedges.
The live leaderboard after the batch kept Team 5 at displayed public MAE
`0.7905` and rank `6`; Team 2 and Team 15 had moved ahead. Every v9 file
passed the standard submission sanity checks and remains a `public-chase` /
final-selection artifact, not a reportable method claim.

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to clean anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p35_w4_0p725_horizon_0p0375_0p35_0p55_0p725_1.csv` | `53259586` | `0.7905` | `5b157f279d2b` | `0.199783` | Displayed public tie, but farther from clean anchor than the selected v8 public-best. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_keep_shape_horizon_0p0375_0p4_0p55_0p75_1.csv` | `53259597` | `0.7906` | `43091b79466b` | `0.193500` | Week-2 anchor increase regressed slightly; useful private-risk evidence. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1.csv` | `53259609` | `0.7905` | `38285232a483` | `0.197653` | Best v9 displayed-public tie; public-biased alternate for slot 2. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p025_w2_0p35_w4_0p725_horizon_0p025_0p35_0p55_0p725_1.csv` | `53259623` | `0.7905` | `b2eba6486837` | `0.200868` | Displayed public tie, but most distant from the clean anchor among v9 ties. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p35_w4_0p725_horizon_0p05_0p35_0p55_0p725_1.csv` | `53259642` | `0.7905` | `2993e16cf7b1` | `0.198698` | Displayed public tie with slightly more week-1 anchor than the v7/v8 boundary. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_keep_shape_horizon_0p05_0p4_0p55_0p75_1.csv` | `53259652` | `0.7906` | `ddcbd1e7853e` | `0.192415` | Private-safer keep-shape point; public regressed to 0.7906. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_late_soft_anchor_horizon_0p0375_0p4_0p5625_0p775_1.csv` | `53259657` | `0.7906` | `a304c903000d` | `0.190404` | Soft late-anchor hedge; public 0.7906 with lower delta than public-tie v9 points. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_late_soft_anchor_horizon_0p05_0p4_0p5625_0p775_1.csv` | `53259668` | `0.7906` | `87c19ce24e72` | `0.189319` | Soft late-anchor hedge with more week-1 anchor; near-public private hedge. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv` | `53259683` | `0.7909` | `262f683b87b6` | `0.177753` | Updated Static/private hedge slot 2: public 0.7909 with strong week-2 and late-anchor protection. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p125_w2_0p45_stronger_late_anchor_horizon_0p125_0p45_0p625_0p85_1.csv` | `53259697` | `0.7914` | `458c8d1b7f68` | `0.167115` | Most conservative v9 point; useful private fallback, but public gap is larger. |

Readout: v9 did not beat displayed public MAE `0.7905`, so ref `53204258`
remains the public-best selection. It did add a better balanced second slot:
ref `53259683` has public MAE `0.7909` with strong week-2 and late-anchor
private protection. Ref `53204319` remains the pure-delta fallback, while ref
`53038036` is the stronger historical private fallback inside the matrix window.
Quota is `10/10` used for 2026-06-01; next reset is
`2026-06-02T08:00:00+08:00`.

Readouts:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260601_quota_20260601_2325/frontier_readout.json`;
`experiments/baseline3_push_20260523/final_selection_matrix_20260601_2335/final_selection_matrix.json`.

## Experiment Table

| Experiment | Model | Feature setup | Validation | Local MAE | Public MAE / Status | Notes |
|---|---|---|---|---:|---:|---|
| `baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1` | Exact-history horizon hedge | Near-public private hedge with horizon alphas `[0.075,0.425,0.60,0.825,1.00]` | Public LB feedback | N/A | `0.7909` | Updated Static Private slot 2, ref `53259683`; better public than v8 ref `53204319` with nearly identical clean-anchor distance; `public-chase`, not reportable. |
| `baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1` | Exact-history horizon hedge | Crossed v8 public-tie signal with lower week-4 anchor | Public LB feedback | N/A | `0.7905` | Public-biased v9 alternate, ref `53259609`; displayed public tie but farther from clean anchor than ref `53204258`. |
| `baseline3_private_hedge_v9_cat35_w1_0p125_w2_0p45_stronger_late_anchor_horizon_0p125_0p45_0p625_0p85_1` | Exact-history horizon hedge | Most conservative v9 hedge with alphas `[0.125,0.45,0.625,0.85,1.00]` | Public LB feedback | N/A | `0.7914` | Strong v9 private fallback, ref `53259697`; matrix still keeps historical ref `53038036` as the lower-delta stronger private fallback. |
| `baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.0375,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7905` | Updated public-best tie, ref `53204258`; same displayed public as v7 but lower clean-anchor delta; `public-chase`, not reportable. |
| `baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1` | Exact-history horizon hedge | More conservative late-anchor hedge with horizon alphas `[0.10,0.40,0.60,0.825,1.00]` | Public LB feedback | N/A | `0.7910` | Previous Static Private fallback, ref `53204319`; pure-delta fallback after v9 ref `53259683`. |
| `baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.025,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7905` | Previous public-best tie, ref `53186508`; superseded for selection by v8 ref `53204258`; `public-chase`, not reportable method claim. |
| `baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1` | Exact-history horizon hedge | Stronger late-anchor hedge by horizon alphas `[0.10,0.375,0.60,0.825,1.00]` | Public LB feedback | N/A | `0.7909` | Static Private slot 2, ref `53186571`; more private-robust than public-best while still near public. |
| `baseline3_private_hedge_v7_cat35_w1_0p05_late_anchor_horizon_0p05_0p375_0p575_0p8_1` | Exact-history horizon hedge | Lower week-1 alpha with late-anchor protection | Public LB feedback | N/A | `0.7906` | Public-biased alternate hedge, ref `53186493`; not reportable. |
| `baseline3_private_hedge_v5_cat35_w1_0p10_keep_shape_horizon_0p1_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.10,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7907` | Previous public-best artifact, ref `53110808`; superseded by v7 ref `53186508`; `public-chase`, not reportable method claim. |
| `baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.175,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7912` | Same-day near-public private hedge, ref `53109133`; closer to clean anchor than v5 public-best. |
| `baseline3_private_hedge_v6_cat35_w1_0p20_stronger_late_anchor_horizon_0p2_0p375_0p6_0p825_1` | Exact-history horizon hedge | Prepared backup with lower week-1 alpha and stronger late-anchor protection | Sanity + backup queue | N/A | Not submitted | Private-robust backup after v5 readout; SHA `5838dee864bb`; no score yet. |
| `baseline3_private_hedge_v4_cat35_w1_0p20_late_anchor_horizon_0p2_0p375_0p575_0p8_1` | Exact-history horizon hedge | Lower week-1 alpha with more anchor on weeks 3-4 | Public LB feedback | N/A | `0.7914` | Strongest same-day v4 private-risk alternative while still near the public best. |
| `baseline3_private_hedge_v3_cat35_lower_w1_more_horizon_0p225_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.225,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7915` | Previous public-best artifact; selected over same-score ref `53075001` because it is closer to the clean anchor; `public-chase`, not reportable method claim. |
| `submission_20260526_145450_20260521_164434_lightgbm_two_stage_lgbm_v3_enhanced` | LightGBM candidate attachment | Source lineage not found in checkout; teammate candidate only | Submission sanity + prediction-surface screen | N/A | `1.0685` | Negative public readout, ref `53074655`; keep `provenance_pending`, not reportable. |
| `baseline3_private_hedge_v2_cat35_lower_early_pair_horizon_0p25_0p375_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.25,0.375,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7917` | Previous public-best artifact; `public-chase`, not reportable method claim. |
| `baseline3_private_hedge_v2_cat35_smooth_high_anchor_v2_horizon_0p35_0p45_0p6_0p8_1` | Exact-history horizon hedge | More anchor-tilted v2 hedge with alphas `[0.35,0.45,0.60,0.80,1.00]` | Public LB feedback | N/A | `0.7929` | Stronger private-risk alternative while still below Baseline 3. |
| `baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1` | Exact-history horizon hedge | Recovered `0.8094` public reference moved toward recovered `0.8124` clean anchor by horizon alphas `[0.30,0.40,0.55,0.75,1.00]` | Public LB feedback | N/A | `0.7922` | Previous public-best artifact; `public-chase`, not reportable method claim. |
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
