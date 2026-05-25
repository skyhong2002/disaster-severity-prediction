# 20260523 1549 Post-Cross Cleanup

## Decision

Team 5 is now below Baseline 3 on the live public leaderboard. Stop public-chase submissions and move to private robustness, reportable lineage cleanup, and final selection rationale.

## Live Readout

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Team 5 live rank: `5`
- Stop public-chase: `True`

## Submitted Public-Chase Curve

| File | Public MAE | SHA-12 | Label | Reportable |
|---|---:|---|---|---|
| `baseline3_public_chase_v0_cat35_08124_alpham0p50.csv` | `0.8786` | `3528a348ddb1` | `public-chase` | `False` |
| `baseline3_public_chase_v0_cat35_08124_alphap0p10.csv` | `0.8049` | `4fd16f09a095` | `public-chase` | `False` |
| `baseline3_public_chase_v0_cat35_08124_alphap0p20.csv` | `0.8019` | `3ddeab41aa75` | `public-chase` | `False` |
| `baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` | `0.7991` | `d550c9cbc465` | `public-chase` | `False` |

## Reportable Lineage

- Primary reportable model: `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`
- Public MAE: `0.8124`
- SHA-12: `bee6f618828d`
- Method claim: 35% LightGBM / 35% XGBoost / 30% CatBoost with CatBoost tail2737

The public-chase best `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` has public MAE `0.7991` and SHA-12 `d550c9cbc465`, but it is labeled `public-chase`, not `reportable`.

## Recommendation

Do not submit more public-chase variants today. Preserve the `0.7991` file for public leaderboard/final selection consideration, but keep the `0.8124` legal anchor as the clean reportable model lineage. Next work should focus on private robustness checks, final submission selection notes, and making the artifact ledger unambiguous.

## Artifacts

- `post_cross_cleanup_report.json`
- `experiments/baseline3_push_20260523/heartbeat_1549_live_check/kaggle_submissions_live.txt`
- `experiments/baseline3_push_20260523/heartbeat_1549_live_check/kaggle_leaderboard_live.txt`
