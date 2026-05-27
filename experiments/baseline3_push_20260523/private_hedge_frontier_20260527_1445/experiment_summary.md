# Private-Hedge Frontier private_hedge_frontier_20260527_1445

- Created UTC: `2026-05-27T06:46:00+00:00`
- Experiment label: `private_hedge_frontier_20260527_1445`
- Spec set: `v3`
- Role: public-chase / final-selection hedge only; not a reportable method claim.
- Source policy: exact recovered Team 5 submissions only; no private labels, external answers, or restored/unverified source files.

## Sources

| Key | Role | Public MAE | SHA-12 | Path |
|---|---|---:|---:|---|
| `v0_08094` | `source_reference_only_public_chase` | `0.8094` | `2f1eb3575419` | `experiments/recovered_submissions_20260523/submission_20260512_195951.csv` |
| `cat35_08124` | `clean_reportable_anchor` | `0.8124` | `bee6f618828d` | `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` |

## Candidate Slate

| Candidate | Horizon alphas | SHA-12 | Delta to reportable anchor | Rationale |
|---|---|---:|---:|---|
| `submissions/baseline3_private_hedge_v3_cat35_lower_w1_more_horizon_0p225_0p375_0p55_0p75_1.csv` | `0.225, 0.375, 0.55, 0.75, 1` | `003351676087` | `0.179355` | After v2 lower_early_pair became public-best, reduce week-1 anchor weight slightly more while preserving weeks 2-5. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_w2_more_horizon_0p25_0p35_0p55_0p75_1.csv` | `0.25, 0.35, 0.55, 0.75, 1` | `ab5bad17e277` | `0.179314` | Local v3 probe: keep the best week-1 setting and reduce week-2 anchor weight to test the early-horizon public optimum. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_early_more_horizon_0p225_0p35_0p55_0p75_1.csv` | `0.225, 0.35, 0.55, 0.75, 1` | `a76ae651b3f6` | `0.181484` | More aggressive early-horizon public-side probe; higher public-chase risk but useful as a boundary point. |
| `submissions/baseline3_private_hedge_v3_cat35_week3_slight_down_horizon_0p25_0p375_0p525_0p75_1.csv` | `0.25, 0.375, 0.525, 0.75, 1` | `0bd41f3de686` | `0.179330` | Tests whether the public optimum also prefers slightly less anchor weight on week 3. |
| `submissions/baseline3_private_hedge_v3_cat35_public_best_late_more_anchor_horizon_0p25_0p375_0p575_0p8_1.csv` | `0.25, 0.375, 0.575, 0.8, 1` | `20aa554510a5` | `0.170991` | Private-robust hedge around the v2 public best: preserve early weights and move weeks 3-4 toward the clean anchor. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

## Kaggle Submission Results

The teammate-provided first queue item was already present in live history at the 2026-05-27 gate: ref `53074655`, public MAE `1.0685`, submitted by `veldahung`. It is a negative public readout and remains `external-teammate-candidate` / `provenance_pending`, not reportable.

The remaining five 2026-05-27 quota slots were spent on the v3 frontier:

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to reportable anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v3_cat35_lower_w1_more_horizon_0p225_0p375_0p55_0p75_1.csv` | `53074980` | `0.7915` | `003351676087` | `0.179355` | Public-best tie and primary selected public-chase candidate because it is closer to the clean anchor than the other `0.7915` tie. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_w2_more_horizon_0p25_0p35_0p55_0p75_1.csv` | `53074990` | `0.7917` | `ab5bad17e277` | `0.179314` | Matches prior v2 best while testing lower week-2 anchor weight. |
| `submissions/baseline3_private_hedge_v3_cat35_lower_early_more_horizon_0p225_0p35_0p55_0p75_1.csv` | `53075001` | `0.7915` | `a76ae651b3f6` | `0.181484` | Public-best tie, but farther from the clean anchor than ref `53074980`. |
| `submissions/baseline3_private_hedge_v3_cat35_week3_slight_down_horizon_0p25_0p375_0p525_0p75_1.csv` | `53075011` | `0.7918` | `0bd41f3de686` | `0.179330` | Slightly worse public score; suggests week-3 lower anchor is not the main driver. |
| `submissions/baseline3_private_hedge_v3_cat35_public_best_late_more_anchor_horizon_0p25_0p375_0p575_0p8_1.csv` | `53075022` | `0.7918` | `20aa554510a5` | `0.170991` | Strongest same-day private hedge: closer to the clean anchor while still near the public best. |

## Final-Selection Readout

- Live leaderboard after the batch: Team 5 public MAE `0.7915`, rank `3`; Baseline 3 `0.8056`.
- Best public/frontier candidate: `submissions/baseline3_private_hedge_v3_cat35_lower_w1_more_horizon_0p225_0p375_0p55_0p75_1.csv` / ref `53074980` / public MAE `0.7915` / SHA-12 `003351676087`.
- Same-score public tie: ref `53075001` (`0.7915`, SHA-12 `a76ae651b3f6`), but it is farther from the clean anchor.
- Stronger same-day private-risk alternative: ref `53075022` (`0.7918`, SHA-12 `20aa554510a5`, delta `0.170991`).
- Cross-day stronger private-risk alternatives remain refs `53038040` (`0.7929`, delta `0.153778`), `53038036` (`0.7925`, delta `0.162247`), and `53038033` (`0.7923`, delta `0.164522`).
- Reportable method lineage remains the clean 35/35/30 LGB/XGB/CatBoost anchor at public MAE `0.8124` / ref `52698259` / SHA-12 `bee6f618828d`.
- Quota: `6/6` used for 2026-05-27 UTC; next reset `2026-05-28T08:00:00+08:00`.
