# Private-Hedge Frontier private_hedge_frontier_20260531_quota_20260531_1245

- Created UTC: `2026-05-31T04:41:53+00:00`
- Experiment label: `private_hedge_frontier_20260531_quota_20260531_1245`
- Spec set: `v8`
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
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p0125_keep_shape_horizon_0p0125_0p375_0p55_0p75_1.csv` | `0.0125, 0.375, 0.55, 0.75, 1` | `d12fb86c376f` | `0.197799` | Fine-grained public plateau probe between the v7 zero and 0.025 week-1 anchor settings. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` | `0.0375, 0.375, 0.55, 0.75, 1` | `390565517cb3` | `0.195629` | Fine-grained public plateau probe between the v7 0.025 and 0.050 week-1 anchor settings. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_m0p025_keep_shape_horizon_m0p025_0p375_0p55_0p75_1.csv` | `-0.025, 0.375, 0.55, 0.75, 1` | `3186f3882dbb` | `0.201035` | Small left-of-zero week-1 boundary probe; high public-chase risk but useful to confirm the v7 week-1 optimum did not sit outside the tested range. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w2_0p35_keep_shape_horizon_0p025_0p35_0p55_0p75_1.csv` | `0.025, 0.35, 0.55, 0.75, 1` | `ecb762e5373a` | `0.198844` | Tests whether the v7 public-best shape also benefits from slightly less week-2 anchor weight. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w2_0p40_keep_shape_horizon_0p025_0p4_0p55_0p75_1.csv` | `0.025, 0.4, 0.55, 0.75, 1` | `19b70bdfe92c` | `0.194585` | Symmetric week-2 anchor probe around the v7 public-best, also mildly closer to the clean anchor. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w3_0p525_keep_shape_horizon_0p025_0p375_0p525_0p75_1.csv` | `0.025, 0.375, 0.525, 0.75, 1` | `a494bf9fd884` | `0.198860` | Tests whether week 3 prefers a little less anchor weight while preserving the current public-best early shape. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w4_0p725_keep_shape_horizon_0p025_0p375_0p55_0p725_1.csv` | `0.025, 0.375, 0.55, 0.725, 1` | `d94440195bf0` | `0.198738` | Tests whether week 4 prefers a little less anchor weight while preserving the current public-best early shape. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_late_soft_anchor_horizon_0p025_0p375_0p5625_0p775_1.csv` | `0.025, 0.375, 0.562, 0.775, 1` | `874de0bed5a2` | `0.193618` | Soft late-anchor hedge halfway between the v7 public-best and v7 late-anchor shape. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_stronger_late_anchor_horizon_0p025_0p375_0p6_0p825_1.csv` | `0.025, 0.375, 0.6, 0.825, 1` | `a2219cb4bdb5` | `0.186352` | Private-robust hedge at the v7 public-best week-1 setting with stronger weeks 3-4 anchor protection. |
| `submissions/baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1.csv` | `0.1, 0.4, 0.6, 0.825, 1` | `a5a5e9188d9e` | `0.177713` | More conservative private-robust hedge extending the selected v7 Static Private slot with extra week-2 anchor weight. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

After each Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.

## Submission Readout

- Live gate before submit: 2026-05-31 had `0/10` Team 5 submissions, so all ten quota slots were available.
- Live leaderboard after submit: Team 5 public MAE `0.7905`, Baseline 3 `0.8056`, Team 5 public rank `5`.
- Quota status after submit: `10/10` used for 2026-05-31 UTC; next reset `2026-06-01T08:00:00+08:00`.
- Selection context: v8 did not improve the displayed public score beyond `0.7905`, but it found a public-best tie closer to the clean anchor and a stronger private hedge than the v7 Static Private pair.

| Candidate | Kaggle ref | Public MAE | SHA-12 | Delta to reportable anchor | Selection note |
|---|---:|---:|---:|---:|---|
| `w1_0p0125_keep_shape` | `53204251` | `0.7906` | `d12fb86c376f` | `0.197799` | Fine public-plateau probe; did not improve. |
| `w1_0p0375_keep_shape` | `53204258` | `0.7905` | `390565517cb3` | `0.195629` | Select as Static Private slot 1 / public-best tie. |
| `w1_m0p025_keep_shape` | `53204263` | `0.7907` | `3186f3882dbb` | `0.201035` | Left-of-zero boundary regressed; do not select. |
| `w1_0p025_w2_0p35_keep_shape` | `53204270` | `0.7905` | `ecb762e5373a` | `0.198844` | Public tie but farther from clean anchor than ref `53204258`. |
| `w1_0p025_w2_0p40_keep_shape` | `53204281` | `0.7906` | `19b70bdfe92c` | `0.194585` | Slight public regression; useful week-2 anchor evidence. |
| `w1_0p025_w3_0p525_keep_shape` | `53204287` | `0.7906` | `a494bf9fd884` | `0.198860` | Week-3 lower-anchor probe regressed slightly. |
| `w1_0p025_w4_0p725_keep_shape` | `53204292` | `0.7905` | `d94440195bf0` | `0.198738` | Public-biased alternate if slot 2 must keep displayed `0.7905`. |
| `w1_0p025_late_soft_anchor` | `53204297` | `0.7906` | `874de0bed5a2` | `0.193618` | Soft late-anchor hedge; near public-best. |
| `w1_0p025_stronger_late_anchor` | `53204307` | `0.7907` | `a2219cb4bdb5` | `0.186352` | Stronger late-anchor hedge but public regressed more than the selected hedge. |
| `w1_0p10_w2_0p40_stronger_late_anchor` | `53204319` | `0.7910` | `a5a5e9188d9e` | `0.177713` | Select as Static Private slot 2 / private-robust hedge. |

Static Private recommendation: select refs `53204258` and `53204319`. The
first ties the displayed public-best while reducing clean-anchor delta versus
v7 ref `53186508`; the second is the most private-robust near-public hedge
inside the displayed `0.0005` public window. The previous v7 pair
`53186508`/`53186571` remains a fallback if the team wants the earlier manual
selection.
