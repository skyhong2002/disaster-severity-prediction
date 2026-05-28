# Private-Hedge Frontier private_hedge_frontier_20260528_queue_20260527_1500

- Created UTC: `2026-05-27T06:58:52+00:00`
- Experiment label: `private_hedge_frontier_20260528_queue_20260527_1500`
- Spec set: `v4`
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
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_keep_shape_horizon_0p2_0p375_0p55_0p75_1.csv` | `0.2, 0.375, 0.55, 0.75, 1` | `4138156b1e10` | `0.181525` | Follows the v3 week-1 signal: reduce only week-1 anchor weight from 0.225 to 0.200 while preserving the selected v3 shape. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` | `0.175, 0.375, 0.55, 0.75, 1` | `c02dc54a2a79` | `0.183695` | Boundary probe for the week-1 public optimum after v3 showed week-1 reduction helped. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p2125_keep_shape_horizon_0p2125_0p375_0p55_0p75_1.csv` | `0.212, 0.375, 0.55, 0.75, 1` | `ec1cea965464` | `0.180440` | Fine-grained interpolation between the v3 selected 0.225 week-1 weight and the next lower 0.200 probe. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_w2_mid_horizon_0p225_0p3625_0p55_0p75_1.csv` | `0.225, 0.362, 0.55, 0.75, 1` | `56284d0e5a27` | `0.180419` | Tests whether a milder week-2 reduction than v3 lower_early_more can keep the 0.7915 public score with less anchor drift. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_late_anchor_horizon_0p2_0p375_0p575_0p8_1.csv` | `0.2, 0.375, 0.575, 0.8, 1` | `5d6406745e87` | `0.175331` | Private-robust hedge combining the new lower week-1 probe with more anchor weight on weeks 3-4. |
| `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_late_mid_anchor_horizon_0p225_0p375_0p565_0p775_1.csv` | `0.225, 0.375, 0.565, 0.775, 1` | `4569505de0e4` | `0.176043` | Softer late-anchor hedge between the v3 selected public-best shape and the same-day late-anchor private hedge. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

## Live Submission Results

- Checked Taipei: `2026-05-28T15:53:27+08:00`.
- Live gate passed before submission: Kaggle submission history and leaderboard were reachable, quota was available, and each file was revalidated by SHA and format sanity.
- Quota status after batch: `6/10` used for 2026-05-28 UTC after the daily limit was confirmed as 10/day; the remaining four slots were later used by the v5 readout.
- Team 5 public MAE after batch: `0.7912`, rank `3`; Baseline 3 remains `0.8056`.
- Live snapshots: `kaggle_submissions_live.txt` and `kaggle_leaderboard_live.txt` in this directory.

| Rank | Candidate | Kaggle ref | Public MAE | SHA-12 | Purpose |
|---:|---|---:|---:|---:|---|
| 1 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p2125_keep_shape_horizon_0p2125_0p375_0p55_0p75_1.csv` | `53109107` | `0.7914` | `ec1cea965464` | Fine interpolation just left of v3 best week-1 weight. |
| 2 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_keep_shape_horizon_0p2_0p375_0p55_0p75_1.csv` | `53109122` | `0.7913` | `4138156b1e10` | Main public-side week-1 probe. |
| 3 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` | `53109133` | `0.7912` | `c02dc54a2a79` | New public best and selected public-chase candidate. |
| 4 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_w2_mid_horizon_0p225_0p3625_0p55_0p75_1.csv` | `53109150` | `0.7915` | `56284d0e5a27` | Milder week-2 adjustment with less anchor drift than ref `53075001`. |
| 5 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p225_late_mid_anchor_horizon_0p225_0p375_0p565_0p775_1.csv` | `53109157` | `0.7916` | `4569505de0e4` | Softer private hedge around the v3 public best. |
| 6 | `submissions/baseline3_private_hedge_v4_cat35_w1_0p20_late_anchor_horizon_0p2_0p375_0p575_0p8_1.csv` | `53109166` | `0.7914` | `5d6406745e87` | Stronger private hedge with lower delta to the clean anchor. |

Best v4 public artifact at the time: `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` / ref `53109133` / public MAE `0.7912`. It was later superseded on public score by the v5 quota-10 readout. Stronger private-risk alternatives remain same-day refs `53109166`, `53109157`, and `53109150`, plus cross-day refs `53075022`, `53038040`, `53038036`, and `53038033`.

All v4 files are public-chase/final-selection candidates only. They must not be described as new reportable training methods.
