# Final Selection Matrix 2026-05-31

- Created UTC: `2026-05-31T04:49:57+00:00`
- Live Team 5 public MAE: `0.7905`
- Baseline 3 public MAE: `0.8056`
- Team 5 public rank: `5`
- Static private snapshot: Team 5 rank `5` at `5/29 23:46:29`.
- Role: decision support for public/private final selection; not a reportable method claim.

## Recommendations

- Select 1, public-best artifact: `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` / ref `53204258` / public `0.7905`.
- Select 2, static/private hedge: `submissions/baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1.csv` / ref `53204319` / public `0.791` / delta `0.177713`.
- Public-biased alternate if the second slot must stay closer to public-best: `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w4_0p725_keep_shape_horizon_0p025_0p375_0p55_0p725_1.csv` / ref `53204292` / public `0.7905` / delta `0.198738`.
- Stronger private fallback by the matrix score: `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` / ref `53038036` / public `0.7925` / delta `0.162247`.

## Selected Submitted Frontier

| Role | File | Ref | Public MAE | Delta | SHA-12 |
|---|---|---:|---:|---:|---:|
| `select_1_public_best` | `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv` | `53204258` | `0.7905` | `0.195629` | `390565517cb3` |
| `select_2_private_robust_hedge` | `submissions/baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1.csv` | `53204319` | `0.791` | `0.177713` | `a5a5e9188d9e` |
| `public_biased_alternate` | `submissions/baseline3_private_hedge_v8_cat35_w1_0p025_w4_0p725_keep_shape_horizon_0p025_0p375_0p55_0p725_1.csv` | `53204292` | `0.7905` | `0.198738` | `d94440195bf0` |
| `stronger_private_fallback` | `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `0.162247` | `ec7984e858b2` |

## Next Quota Rules

- Current 2026-05-31 UTC quota is 10/10 used after the v8 quota-10 frontier.
- For Static Private / final-selection UI, manually select refs 53204258 and 53204319; do not rely only on auto-selection.
- Live-gate Kaggle submission history and leaderboard after 2026-06-01T08:00:00+08:00 Taipei before spending any new quota.
- Do not resubmit the teammate file: live history already confirms ref 53074655 / public 1.0685 for the same artifact.
- Use the v7 pair refs 53186508/53186571 as fallback if the UI or team policy prefers the previous manually selected Static Private pair.
- Do not describe any v5/v6/v7/v8 public-chase artifact as a reportable method claim.

## Queue Pointers

- Teammate gate: `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`
- Latest public-best selected: `submissions/baseline3_private_hedge_v8_cat35_w1_0p0375_keep_shape_horizon_0p0375_0p375_0p55_0p75_1.csv`
- Latest static/private hedge selected: `submissions/baseline3_private_hedge_v8_cat35_w1_0p10_w2_0p40_stronger_late_anchor_horizon_0p1_0p4_0p6_0p825_1.csv`
- First v6/private-robust backup if future quota is reopened: `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv`
