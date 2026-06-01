# Final Selection Matrix 2026-06-01

- Created UTC: `2026-06-01T15:29:48+00:00`
- Live Team 5 public MAE: `0.7905`
- Baseline 3 public MAE: `0.8056`
- Team 5 public rank: `6`
- Static private snapshot: Team 5 rank `5` at `5/29 23:46:29`.
- Role: decision support for public/private final selection; not a reportable method claim.

## Recommendations

- Select 1, public-best artifact: `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` / ref `53204258` / public `0.7905`.
- Select 2, static/private hedge: `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv` / ref `53259683` / public `0.7909` / delta `0.177753`.
- Public-biased alternate if the second slot must stay closer to public-best: `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1.csv` / ref `53259609` / public `0.7905` / delta `0.197653`.
- Stronger private fallback by the matrix score: `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` / ref `53038036` / public `0.7925` / delta `0.162247`.

## Selected Submitted Frontier

| Role | File | Ref | Public MAE | Delta | SHA-12 |
|---|---|---:|---:|---:|---:|
| `select_1_public_best` | `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` | `53204258` | `0.7905` | `0.195629` | `390565517cb3` |
| `select_2_private_robust_hedge` | `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv` | `53259683` | `0.7909` | `0.177753` | `262f683b87b6` |
| `public_biased_alternate` | `submissions/baseline3_private_hedge_v9_cat35_w1_0p0375_w4_0p725_keep_shape_horizon_0p0375_0p375_0p55_0p725_1.csv` | `53259609` | `0.7905` | `0.197653` | `38285232a483` |
| `stronger_private_fallback` | `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `0.162247` | `ec7984e858b2` |

## Next Quota Rules

- Current 2026-06-01 UTC quota is 10/10 used after the v9 quota-10 frontier.
- For Static Private / final-selection UI, manually select refs 53204258 and 53259683; do not rely only on auto-selection.
- Live-gate Kaggle submission history and leaderboard after 2026-06-02T08:00:00+08:00 Taipei before spending any new quota.
- Do not resubmit the teammate file: live history already confirms ref 53074655 / public 1.0685 for the same artifact.
- Use the v8 pair refs 53204258/53204319 or the v7 pair refs 53186508/53186571 as fallback if the UI or team policy prefers a previous manually selected Static Private pair.
- Do not describe any v5/v6/v7/v8/v9 public-chase artifact as a reportable method claim.

## Queue Pointers

- Teammate gate: `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`
- Latest public-best selected: `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv`
- Latest static/private hedge selected: `submissions/baseline3_private_hedge_v9_cat35_w1_0p075_w2_0p425_stronger_late_anchor_horizon_0p075_0p425_0p6_0p825_1.csv`
- First v6/private-robust backup if future quota is reopened: `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv`
