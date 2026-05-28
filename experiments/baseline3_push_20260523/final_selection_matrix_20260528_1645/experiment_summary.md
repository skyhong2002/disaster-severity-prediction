# Final Selection Matrix 2026-05-28

- Created UTC: `2026-05-28T09:05:01+00:00`
- Live Team 5 public MAE: `0.7907`
- Baseline 3 public MAE: `0.8056`
- Role: decision support for public/private final selection; not a reportable method claim.

## Recommendations

- Public-best selected artifact: `submissions/baseline3_private_hedge_v5_cat35_w1_0p10_keep_shape_horizon_0p1_0p375_0p55_0p75_1.csv` / ref `53110808` / public `0.7907`.
- Same-day private hedge: `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` / ref `53109133` / public `0.7912` / delta `0.183695`.
- Stronger private fallback: `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` / ref `53038036` / public `0.7925` / delta `0.162247`.

## Submitted Frontier

| Role | File | Ref | Public MAE | Delta | SHA-12 |
|---|---|---:|---:|---:|---:|
| `public_best` | `submissions/baseline3_private_hedge_v5_cat35_w1_0p10_keep_shape_horizon_0p1_0p375_0p55_0p75_1.csv` | `53110808` | `0.7907` | `0.190205` | `12016072d85f` |
| `same_day_private_hedge` | `submissions/baseline3_private_hedge_v4_cat35_w1_0p175_keep_shape_horizon_0p175_0p375_0p55_0p75_1.csv` | `53109133` | `0.7912` | `0.183695` | `c02dc54a2a79` |
| `stronger_private_fallback` | `submissions/baseline3_private_hedge_v2_cat35_balanced_private_frontier_horizon_0p325_0p425_0p575_0p775_1.csv` | `53038036` | `0.7925` | `0.162247` | `ec7984e858b2` |

## Next Quota Rules

- Current 2026-05-28 UTC quota is 10/10 used after the quota-limit update and v5 keep-shape readout.
- Live-gate Kaggle submission history and leaderboard after 2026-05-29T08:00:00+08:00 Taipei before spending any new quota.
- Do not resubmit the teammate file: live history already confirms ref 53074655 / public 1.0685 for the same artifact.
- Do not resubmit v5 keep-shape files already used on 2026-05-28; use them only as readout evidence for final selection.
- If more quota is spent, prioritize private-robust v5 late-anchor or v6 backup candidates when the team wants lower private-risk alternatives to the public-best v5 artifact.
- Do not describe any v5/v6 public-chase artifact as a reportable method claim.

## Queue Pointers

- Teammate gate: `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`
- First v5 keep-shape already consumed after duplicate skip: `submissions/baseline3_private_hedge_v5_cat35_w1_0p1625_keep_shape_horizon_0p1625_0p375_0p55_0p75_1.csv`
- First v6/private-robust backup after fresh live gate: `submissions/baseline3_private_hedge_v6_cat35_w1_0p1625_late_anchor_horizon_0p1625_0p375_0p575_0p8_1.csv`
