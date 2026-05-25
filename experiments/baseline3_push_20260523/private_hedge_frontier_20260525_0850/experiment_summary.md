# Private-Hedge Frontier 2026-05-25

- Created UTC: `2026-05-25T00:52:25+00:00`
- Updated UTC: `2026-05-25T00:59:27+00:00`
- Experiment label: `private_hedge_frontier_20260525_0850`
- Role: public-chase / final-selection hedge only; not a reportable method claim.
- Source policy: exact recovered Team 5 submissions only; no private labels, external answers, or restored/unverified source files.
- Live leaderboard after batch: Team 5 public MAE `0.7922`, rank `3`, Baseline 3 `0.8056`.
- Quota: `6/6` used for 2026-05-25 UTC; next reset `2026-05-26T08:00:00+08:00`.

## Sources

| Key | Role | Public MAE | SHA-12 | Path |
|---|---|---:|---:|---|
| `cat35_08124` | `clean_reportable_anchor` | `0.8124` | `bee6f618828d` | `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` |
| `v0_08094` | `source_reference_only_public_chase` | `0.8094` | `2f1eb3575419` | `experiments/recovered_submissions_20260523/submission_20260512_195951.csv` |

## Kaggle Submission Results

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to reportable anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v1_cat35_left_of_best_public_horizon_0p325_0p375_0p475_0p625_0p775.csv` | `53003193` | `0.7929` | `293531f60aa1` | `0.205233` | passes_baseline3_and_adds_frontier_evidence |
| `submissions/baseline3_private_hedge_v1_cat35_midpoint_best_to_nearbest_horizon_0p375_0p425_0p525_0p675_0p825.csv` | `53003210` | `0.7931` | `eef241fc2218` | `0.184295` | passes_baseline3_and_adds_frontier_evidence |
| `submissions/baseline3_private_hedge_v1_cat35_preserve_early_anchor_late_horizon_0p35_0p425_0p55_0p725_0p9.csv` | `53003215` | `0.7927` | `2b58d70fc4a7` | `0.174271` | passes_baseline3_and_adds_frontier_evidence |
| `submissions/baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1.csv` | `53003220` | `0.7922` | `695c62a4eb28` | `0.170715` | best_public_and_primary_selected_public_chase_candidate |
| `submissions/baseline3_private_hedge_v1_cat35_robust_mid_frontier_horizon_0p4_0p475_0p575_0p75_0p9.csv` | `53003222` | `0.7933` | `7adc27154921` | `0.161504` | stronger_private_hedge_alternative_below_baseline3 |
| `submissions/baseline3_private_hedge_v1_cat35_smooth_high_anchor_horizon_0p425_0p5_0p6_0p8_0p95.csv` | `53003227` | `0.7937` | `92f02f3f2a20` | `0.147011` | stronger_private_hedge_alternative_below_baseline3 |

## Final-Selection Readout

- Best public/frontier candidate: `submissions/baseline3_private_hedge_v1_cat35_public_early_full_late_anchor_horizon_0p3_0p4_0p55_0p75_1.csv` / ref `53003220` / public MAE `0.7922` / SHA-12 `695c62a4eb28`.
- Stronger private-risk alternatives: ref `53003227` (`0.7937`, delta `0.147011`), ref `53003222` (`0.7933`, delta `0.161504`).
- Reportable method lineage remains the clean 35/35/30 LGB/XGB/CatBoost anchor at public MAE `0.8124` / ref `52698259` / SHA-12 `bee6f618828d`.
- Do not describe any v1 hedge as a new reportable training method; it is a final-selection/public-chase artifact built from exact recovered team submissions.

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.
