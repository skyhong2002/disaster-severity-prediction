# Private-Hedge Frontier private_hedge_frontier_20260530_quota_20260530_2155

- Created UTC: `2026-05-30T13:53:29+00:00`
- Experiment label: `private_hedge_frontier_20260530_quota_20260530_2155`
- Spec set: `v7`
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
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_late_anchor_horizon_0p1_0p375_0p575_0p8_1.csv` | `0.1, 0.375, 0.575, 0.8, 1` | `2669fcc54a42` | `0.184011` | Private-robust check at the current v5 public-best week-1 weight, moving weeks 3-4 back toward the clean anchor. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_keep_shape_horizon_0p075_0p375_0p55_0p75_1.csv` | `0.075, 0.375, 0.55, 0.75, 1` | `79792c199111` | `0.192375` | Next public-side week-1 boundary point after v5 improved monotonically through 0.100. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_late_anchor_horizon_0p075_0p375_0p575_0p8_1.csv` | `0.075, 0.375, 0.575, 0.8, 1` | `81fe2fae4b96` | `0.186181` | Pairs the next lower week-1 public probe with late-anchor private-risk protection. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_keep_shape_horizon_0p05_0p375_0p55_0p75_1.csv` | `0.05, 0.375, 0.55, 0.75, 1` | `63b3987da5fa` | `0.194544` | More aggressive week-1 public-side boundary probe after 0.075. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_late_anchor_horizon_0p05_0p375_0p575_0p8_1.csv` | `0.05, 0.375, 0.575, 0.8, 1` | `f1c733d6d2e2` | `0.188351` | More aggressive week-1 public probe with late-anchor protection. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1.csv` | `0.025, 0.375, 0.55, 0.75, 1` | `9b452689c221` | `0.196714` | Near-zero week-1 anchor boundary to test whether the public optimum has already reversed. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_late_anchor_horizon_0p025_0p375_0p575_0p8_1.csv` | `0.025, 0.375, 0.575, 0.8, 1` | `4dfc45e15a42` | `0.190521` | Near-zero week-1 anchor boundary with private-robust late-horizon anchor weight. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p00_keep_shape_horizon_0_0p375_0p55_0p75_1.csv` | `0, 0.375, 0.55, 0.75, 1` | `f105abad604f` | `0.198884` | Zero week-1 anchor upper-risk boundary; submit only after the safer 0.075/0.050 readouts remain public-competitive. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p075_stronger_late_anchor_horizon_0p075_0p375_0p6_0p825_1.csv` | `0.075, 0.375, 0.6, 0.825, 1` | `6cb910b58f59` | `0.182012` | Stronger late-anchor hedge around the likely public-side 0.075 week-1 region. |
| `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1.csv` | `0.1, 0.375, 0.6, 0.825, 1` | `6c2e32e92647` | `0.179842` | Stronger late-anchor hedge at the current v5 public-best week-1 setting, designed as a near-public private fallback. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

After each Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.

## Submission Readout

- Live gate before submit: 2026-05-30 had `0/10` Team 5 submissions, so all ten quota slots were available.
- Static private context: user-provided 5/29 Google Sheet snapshot showed Team 5 rank `5` at release time `5/29 23:46:29`; the sheet says this is not the final Kaggle private ranking.
- Live leaderboard after submit: Team 5 public MAE `0.7905`, Baseline 3 `0.8056`, Team 5 public rank `4`.
- Quota status after submit: `10/10` used for 2026-05-30 UTC; next reset `2026-05-31T08:00:00+08:00`.

| Candidate | Kaggle ref | Public MAE | SHA-12 | Delta to reportable anchor | Selection note |
|---|---:|---:|---:|---:|---|
| `w1_0p10_late_anchor` | `53186451` | `0.7908` | `2669fcc54a42` | `0.184011` | Passes Baseline 3; late-anchor hedge at previous v5 week-1 setting. |
| `w1_0p075_keep_shape` | `53186458` | `0.7906` | `79792c199111` | `0.192375` | Near-public keep-shape probe. |
| `w1_0p075_late_anchor` | `53186470` | `0.7907` | `81fe2fae4b96` | `0.186181` | Near-public private hedge. |
| `w1_0p05_keep_shape` | `53186480` | `0.7906` | `63b3987da5fa` | `0.194544` | Aggressive week-1 public-side probe. |
| `w1_0p05_late_anchor` | `53186493` | `0.7906` | `f1c733d6d2e2` | `0.188351` | Public-biased alternate for slot 2 if needed. |
| `w1_0p025_keep_shape` | `53186508` | `0.7905` | `9b452689c221` | `0.196714` | Select as Static Private slot 1 / public-best. |
| `w1_0p025_late_anchor` | `53186528` | `0.7906` | `4dfc45e15a42` | `0.190521` | Near-public late-anchor hedge. |
| `w1_0p00_keep_shape` | `53186548` | `0.7906` | `f105abad604f` | `0.198884` | Public stayed near-best but private risk is higher. |
| `w1_0p075_stronger_late_anchor` | `53186562` | `0.7908` | `6cb910b58f59` | `0.182012` | Stronger late-anchor hedge. |
| `w1_0p10_stronger_late_anchor` | `53186571` | `0.7909` | `6c2e32e92647` | `0.179842` | Select as Static Private slot 2 / private-robust hedge. |

Static Private recommendation: select refs `53186508` and `53186571`. The first
maximizes public score; the second is only `0.0004` public MAE behind while
staying closer to the clean `0.8124` reportable anchor. Both remain
`public-chase` final-selection artifacts, not reportable method claims.
