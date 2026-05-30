# Final Selection Matrix 2026-05-30

- Created UTC: `2026-05-30T14:06:37+00:00`
- Live Team 5 public MAE: `0.7905`
- Baseline 3 public MAE: `0.8056`
- Team 5 public rank: `4`
- Static private snapshot: Team 5 rank `5` at `5/29 23:46:29`.
- Role: decision support for public/private final selection; not a reportable method claim.

## Recommendations

- Select 1, public-best artifact: `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1.csv` / ref `53186508` / public `0.7905`.
- Select 2, static/private hedge: `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1.csv` / ref `53186571` / public `0.7909` / delta `0.179842`.
- Public-biased alternate if the second slot must stay closer to public-best: `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_late_anchor_horizon_0p05_0p375_0p575_0p8_1.csv` / ref `53186493` / public `0.7906` / delta `0.188351`.
- Stronger private fallback by the matrix score: `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` / ref `53038036` / public `0.7925` / delta `0.162247`.

## Selected Submitted Frontier

| Role | File | Ref | Public MAE | Delta | SHA-12 |
|---|---|---:|---:|---:|---:|
| `select_1_public_best` | `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1.csv` | `53186508` | `0.7905` | `0.196714` | `9b452689c221` |
| `select_2_private_robust_hedge` | `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1.csv` | `53186571` | `0.7909` | `0.179842` | `6c2e32e92647` |
| `public_biased_alternate` | `submissions/baseline3_private_hedge_v7_cat35_w1_0p05_late_anchor_horizon_0p05_0p375_0p575_0p8_1.csv` | `53186493` | `0.7906` | `0.188351` | `f1c733d6d2e2` |
| `stronger_private_fallback` | `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `0.162247` | `ec7984e858b2` |

## Next Quota Rules

- Current 2026-05-30 UTC quota is 10/10 used after the v7 quota-10 frontier.
- For Static Private / final-selection UI, manually select refs 53186508 and 53186571; do not rely only on auto-selection.
- Live-gate Kaggle submission history and leaderboard after 2026-05-31T08:00:00+08:00 Taipei before spending any new quota.
- Do not resubmit the teammate file: live history already confirms ref 53074655 / public 1.0685 for the same artifact.
- Use ref 53186493 as the public-biased alternate only if the second final-selection slot must stay closer to public-best.
- Do not describe any v5/v6/v7 public-chase artifact as a reportable method claim.

## Queue Pointers

- Teammate gate: `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`
- Latest v7 public-best selected: `submissions/baseline3_private_hedge_v7_cat35_w1_0p025_keep_shape_horizon_0p025_0p375_0p55_0p75_1.csv`
- Latest v7 static/private hedge selected: `submissions/baseline3_private_hedge_v7_cat35_w1_0p10_stronger_late_anchor_horizon_0p1_0p375_0p6_0p825_1.csv`
- First v6/private-robust backup if future quota is reopened: `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv`
