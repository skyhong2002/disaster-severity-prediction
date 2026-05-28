# Private-Hedge Frontier private_hedge_frontier_20260530_backup_20260528_1610

- Created UTC: `2026-05-28T08:09:25+00:00`
- Experiment label: `private_hedge_frontier_20260530_backup_20260528_1610`
- Spec set: `v6`
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
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv` | `0.163, 0.375, 0.575, 0.8, 1` | `2a9f1230b65f` | `0.178586` | Backup private hedge pairing the v5 lower week-1 probe with the v4 late-anchor protection that stayed near public best. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p20_late_anchor_w2_0p40_horizon_0p2_0p4_0p575_0p8_1.csv` | `0.2, 0.4, 0.575, 0.8, 1` | `d355e20f2ce1` | `0.173202` | Adds week-2 anchor weight to the v4 ref 53109166 late-anchor hedge, seeking lower private risk with limited public drift. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p175_late_anchor_w2_0p40_horizon_0p175_0p4_0p575_0p8_1.csv` | `0.175, 0.4, 0.575, 0.8, 1` | `efcbbfe922f6` | `0.175372` | Combines the v4 public-best week-1 setting with more week-2 and late-horizon anchor weight. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p20_stronger_late_anchor_horizon_0p2_0p375_0p6_0p825_1.csv` | `0.2, 0.375, 0.6, 0.825, 1` | `5838dee864bb` | `0.171162` | Stronger late-anchor version of ref 53109166 for private-risk coverage if v5 public-side probes overfit. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p225_stronger_late_anchor_horizon_0p225_0p375_0p6_0p825_1.csv` | `0.225, 0.375, 0.6, 0.825, 1` | `e337885b660d` | `0.168992` | Uses the v3 selected early shape with stronger week-3 and week-4 anchor protection. |
| `submissions/baseline3_private_hedge_v6_cat35_w1_0p25_stronger_late_anchor_horizon_0p25_0p375_0p6_0p825_1.csv` | `0.25, 0.375, 0.6, 0.825, 1` | `86d7dd4da810` | `0.166822` | Most conservative v6 late-anchor backup, staying closer to prior private-robust refs while still below Baseline 3 in nearby submitted points. |

## Required Submission Sanity

All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

## Live Gate Context

- Checked Taipei: `2026-05-28T17:00:54+08:00` after the v5 quota-10 readout.
- Team 5 public MAE is `0.7907`, rank `3`; Baseline 3 is `0.8056`.
- Quota status: `10/10` used for 2026-05-28 UTC, so no v6 candidate was submitted in this preparation pass.
- Next reset: `2026-05-29T08:00:00+08:00`.
- Live snapshots: `kaggle_submissions_live.txt` and `kaggle_leaderboard_live.txt` in this directory.

## Backup Use

This is not an automatic 2026-05-29 submission queue. The teammate duplicate
guard and v5 keep-shape readout have already been consumed on 2026-05-28 after
the limit changed to 10/day. Use v6 only after a fresh live gate, or on a later
quota reset, if the team needs more private-robust hedges around the v4
late-anchor signal.

The v6 candidates are intentionally closer to the clean `0.8124` anchor than
the pure public-best v5 file. The closest v6 backup is
`submissions/baseline3_private_hedge_v6_cat35_w1_0p25_stronger_late_anchor_horizon_0p25_0p375_0p6_0p825_1.csv`
with SHA-12 `86d7dd4da810` and mean absolute delta `0.166822`; the more
public-near backup is
`submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv`
with SHA-12 `2a9f1230b65f` and mean absolute delta `0.178586`.

After each future Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.
