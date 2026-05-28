# Private-Hedge Frontier private_hedge_frontier_20260529_queue_20260528_1555

- Created UTC: `2026-05-28T07:53:53+00:00`
- Experiment label: `private_hedge_frontier_20260529_queue_20260528_1555`
- Spec set: `v5`
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
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p1625_keep_shape_horizon_0p1625_0p375_0p55_0p75_1.csv` | `0.163, 0.375, 0.55, 0.75, 1` | `530c7e912705` | `0.184780` | Fine interpolation just below the v4 public-best week-1 weight 0.175. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p15_keep_shape_horizon_0p15_0p375_0p55_0p75_1.csv` | `0.15, 0.375, 0.55, 0.75, 1` | `524a8ddebdd6` | `0.185865` | Main continuation of the week-1 public-side curve after v4 improved monotonically down to 0.175. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p125_keep_shape_horizon_0p125_0p375_0p55_0p75_1.csv` | `0.125, 0.375, 0.55, 0.75, 1` | `7fc12846493c` | `0.188035` | Boundary probe to locate the left side of the week-1 optimum without jumping all the way to zero. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p10_keep_shape_horizon_0p1_0p375_0p55_0p75_1.csv` | `0.1, 0.375, 0.55, 0.75, 1` | `12016072d85f` | `0.190205` | More aggressive week-1 public-side boundary probe; use after the safer 0.1625/0.15/0.125 points. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p175_late_anchor_horizon_0p175_0p375_0p575_0p8_1.csv` | `0.175, 0.375, 0.575, 0.8, 1` | `44798cc5134a` | `0.177501` | Private-robust hedge at the v4 public-best week-1 weight with weeks 3-4 moved toward the clean anchor. |
| `submissions/baseline3_private_hedge_v5_cat35_w1_0p15_late_anchor_horizon_0p15_0p375_0p575_0p8_1.csv` | `0.15, 0.375, 0.575, 0.8, 1` | `34127e357aa5` | `0.179671` | Private-robust hedge combining the next lower week-1 public-side probe with late-anchor protection. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

## Live Gate Context

- Checked Taipei: `2026-05-28T15:53:27+08:00`.
- Team 5 public MAE is `0.7912`, rank `3`; Baseline 3 is `0.8056`.
- Quota status: `6/6` used for 2026-05-28 UTC, so no v5 candidate was submitted in this preparation pass.
- Next reset: `2026-05-29T08:00:00+08:00`.
- Live snapshots: `kaggle_submissions_live.txt` and `kaggle_leaderboard_live.txt` in this directory.

## Manual 2026-05-29 Queue

Per user directive, the teammate CSV is recorded as rank `1` for the next-reset queue:
`submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`
(SHA-12 `66f7815e7ff3`). It passed the standard sanity gate, but the same file
is already recorded in live history as Kaggle ref `53074655` with public MAE
`1.0685`. Therefore the next live gate must check duplicate history first; if
the duplicate is confirmed, skip the teammate item and promote the v5 candidates
in order rather than silently spending quota on a known negative duplicate.

Manual queue JSON:
`experiments/baseline3_push_20260523/private_hedge_frontier_20260529_queue_20260528_1555/next_submission_queue_20260529.json`.

After each Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.
