# Private-Hedge Frontier private_hedge_frontier_20260601_quota_20260601_2325

- Created UTC: `2026-06-01T15:22:04+00:00`
- Experiment label: `private_hedge_frontier_20260601_quota_20260601_2325`
- Spec set: `v9`
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
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p35_w4_0p725_horizon_0p0375_0p35_0p55_0p725_1.csv` | `0.0375, 0.35, 0.55, 0.725, 1` | `5b157f279d2b` | `0.199783` | Crosses the three v8 displayed-public tie signals: selected 0.0375 week-1, lower week-2 anchor, and lower week-4 anchor. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_keep_shape_horizon_0p0375_0p4_0p55_0p75_1.csv` | `0.0375, 0.4, 0.55, 0.75, 1` | `43091b79466b` | `0.193500` | Keeps the v8 selected week-1 setting but moves week 2 toward the clean anchor to reduce private-risk distance. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1.csv` | `0.0375, 0.375, 0.55, 0.725, 1` | `38285232a483` | `0.197653` | Keeps the v8 selected week-1 setting and tests whether the v8 lower week-4 public tie transfers when paired with 0.0375. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p025_w2_0p35_w4_0p725_horizon_0p025_0p35_0p55_0p725_1.csv` | `0.025, 0.35, 0.55, 0.725, 1` | `b2eba6486837` | `0.200868` | Combines the v8 same-score lower week-2 and lower week-4 public probes around the original v7 public-best week-1. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p35_w4_0p725_horizon_0p05_0p35_0p55_0p725_1.csv` | `0.05, 0.35, 0.55, 0.725, 1` | `2993e16cf7b1` | `0.198698` | Boundary point just to the right of the v8 public-tie cluster, useful if week-1 0.0375 is under-anchored on private. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_keep_shape_horizon_0p05_0p4_0p55_0p75_1.csv` | `0.05, 0.4, 0.55, 0.75, 1` | `ddcbd1e7853e` | `0.192415` | Private-safer local point near the v8 selected public tie, increasing early-horizon anchor without changing late shape. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_late_soft_anchor_horizon_0p0375_0p4_0p5625_0p775_1.csv` | `0.0375, 0.4, 0.562, 0.775, 1` | `a304c903000d` | `0.190404` | Private-robust soft late-anchor hedge at the v8 selected week-1 setting with extra week-2 protection. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_late_soft_anchor_horizon_0p05_0p4_0p5625_0p775_1.csv` | `0.05, 0.4, 0.562, 0.775, 1` | `87c19ce24e72` | `0.189319` | Slightly more anchored version of the soft late-anchor hedge for the second Static Private slot. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv` | `0.075, 0.425, 0.6, 0.825, 1` | `262f683b87b6` | `0.177753` | Near-public private hedge that extends the v8 selected second slot with more week-2 anchor weight. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p125_w2_0p45_stronger_late_anchor_horizon_0p125_0p45_0p625_0p85_1.csv` | `0.125, 0.45, 0.625, 0.85, 1` | `458c8d1b7f68` | `0.167115` | Most conservative v9 hedge; intended to lower private flip risk if the public plateau is overfit. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

After each Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.

## Submission Readout

| Submission | Kaggle ref | Public MAE | SHA-12 | Delta to clean anchor | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p35_w4_0p725_horizon_0p0375_0p35_0p55_0p725_1.csv` | `53259586` | `0.7905` | `5b157f279d2b` | `0.199783` | Displayed public tie; not better than v8 public-best on clean-anchor delta. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_keep_shape_horizon_0p0375_0p4_0p55_0p75_1.csv` | `53259597` | `0.7906` | `43091b79466b` | `0.193500` | Near-public hedge evidence; public regressed slightly. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1.csv` | `53259609` | `0.7905` | `38285232a483` | `0.197653` | Best v9 public-tie alternate. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p025_w2_0p35_w4_0p725_horizon_0p025_0p35_0p55_0p725_1.csv` | `53259623` | `0.7905` | `b2eba6486837` | `0.200868` | Displayed public tie; not better than v8 public-best on clean-anchor delta. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p35_w4_0p725_horizon_0p05_0p35_0p55_0p725_1.csv` | `53259642` | `0.7905` | `2993e16cf7b1` | `0.198698` | Displayed public tie; not better than v8 public-best on clean-anchor delta. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_keep_shape_horizon_0p05_0p4_0p55_0p75_1.csv` | `53259652` | `0.7906` | `ddcbd1e7853e` | `0.192415` | Near-public hedge evidence; public regressed slightly. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w2_0p40_late_soft_anchor_horizon_0p0375_0p4_0p5625_0p775_1.csv` | `53259657` | `0.7906` | `a304c903000d` | `0.190404` | Near-public hedge evidence; public regressed slightly. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p05_w2_0p40_late_soft_anchor_horizon_0p05_0p4_0p5625_0p775_1.csv` | `53259668` | `0.7906` | `87c19ce24e72` | `0.189319` | Near-public hedge evidence; public regressed slightly. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv` | `53259683` | `0.7909` | `262f683b87b6` | `0.177753` | Selected v9 near-public private hedge slot. |
| `submissions/baseline3_private_hedge_v9_cat35_w1_0p125_w2_0p45_stronger_late_anchor_horizon_0p125_0p45_0p625_0p85_1.csv` | `53259697` | `0.7914` | `458c8d1b7f68` | `0.167115` | Strongest v9 private fallback; public gap larger. |

- Final two-slot recommendation after v9: keep public-best ref `53204258`; select v9 private hedge ref `53259683` as slot 2.
- Fallbacks: v8 pure-delta hedge ref `53204319`; stronger historical private fallback ref `53038036`.
- Quota status: `10/10` used for 2026-06-01 UTC; next reset `2026-06-02T08:00:00+08:00`.
