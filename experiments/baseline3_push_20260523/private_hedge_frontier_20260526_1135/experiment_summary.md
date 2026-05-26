# Private-Hedge Frontier 2026-05-26

- Created UTC: `2026-05-26T03:35:31+00:00`
- Updated UTC: `2026-05-26T03:39:01+00:00`
- Experiment label: `private_hedge_frontier_20260526_1135`
- Spec set: `v2`
- Role: public-chase / final-selection hedge only; not a reportable method claim.
- Source policy: exact recovered Team 5 submissions only; no private labels, external answers, or restored/unverified source files.
- Live leaderboard after batch: Team 5 public MAE `0.7917`, rank `3`, Baseline 3 `0.8056`.
- Quota: `6/6` used for 2026-05-26 UTC; next reset `2026-05-27T08:00:00+08:00`.

## Sources

| Key | Role | Public MAE | SHA-12 | Path |
|---|---|---:|---:|---|
| `cat35_08124` | `clean_reportable_anchor` | `0.8124` | `bee6f618828d` | `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` |
| `v0_08094` | `source_reference_only_public_chase` | `0.8094` | `2f1eb3575419` | `experiments/recovered_submissions_20260523/submission_20260512_195951.csv` |

## Kaggle Submission Results

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to reportable anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v2_cat35_lower_w1_only_horizon_0p275_0p4_0p55_0p75_1.csv` | `53038024` | `0.7920` | `d7a9675828cf` | `0.172885` | passes_baseline3_and_adds_frontier_evidence |
| `submissions/baseline3_private_hedge_v2_cat35_lower_w2_only_horizon_0p3_0p375_0p55_0p75_1.csv` | `53038028` | `0.7922` | `c14f6b929192` | `0.172845` | passes_baseline3_and_adds_frontier_evidence |
| `submissions/baseline3_private_hedge_v2_cat35_lower_early_pair_horizon_0p25_0p375_0p55_0p75_1.csv` | `53038031` | `0.7917` | `eae8c5b0dc6f` | `0.177185` | best_public_and_primary_selected_public_chase_candidate |
| `submissions/baseline3_private_hedge_v2_cat35_late_more_anchor_horizon_0p3_0p4_0p575_0p8_1.csv` | `53038033` | `0.7923` | `a495c0ef028e` | `0.164522` | stronger_private_hedge_alternative_below_baseline3 |
| `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `ec7984e858b2` | `0.162247` | stronger_private_hedge_alternative_below_baseline3 |
| `submissions/baseline3_private_hedge_v2_cat35_smooth_high_anchor_v2_horizon_0p35_0p45_0p6_0p8_1.csv` | `53038040` | `0.7929` | `3b16648a2c5c` | `0.153778` | stronger_private_hedge_alternative_below_baseline3 |

## Final-Selection Readout

- Best public/frontier candidate: `submissions/baseline3_private_hedge_v2_cat35_lower_early_pair_horizon_0p25_0p375_0p55_0p75_1.csv` / ref `53038031` / public MAE `0.7917` / SHA-12 `eae8c5b0dc6f`.
- Stronger private-risk alternatives: ref `53038040` (`0.7929`, delta `0.153778`), ref `53038036` (`0.7925`, delta `0.162247`), ref `53038033` (`0.7923`, delta `0.164522`).
- Reportable method lineage remains the clean 35/35/30 LGB/XGB/CatBoost anchor at public MAE `0.8124` / ref `52698259` / SHA-12 `bee6f618828d`.
- Do not describe any v2 hedge as a new reportable training method; it is a final-selection/public-chase artifact built from exact recovered team submissions.

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.
