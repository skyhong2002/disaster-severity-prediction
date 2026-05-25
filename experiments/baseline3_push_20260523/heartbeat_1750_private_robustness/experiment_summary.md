# 20260523 1750 Private Robustness Audit

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Stop public-chase: `True`
- Quota recommendation: `do_not_submit_more_public_chase_or_gru_today`

## Role Separation

| Role | File | Public MAE | SHA-12 | Use |
|---|---|---:|---|---|
| leaderboard/final-selection | `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` | `0.7991` | `d550c9cbc465` | Already-submitted best public artifact; usable for final selection if rules allow. |
| reportable lineage | `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` | `0.8124` | `bee6f618828d` | Clean model lineage for report/method claim. |
| source reference only | `experiments/recovered_submissions_20260523/submission_20260512_195951.csv` | `0.8094` | `2f1eb3575419` | Public-chase source reference; not a current reportable method. |

## Robustness Signals

- Public best vs reportable anchor mean absolute delta: `0.272186`
- Public best vs historical v0 mean absolute delta: `0.146562`
- Public best prediction mean: `0.991763`
- Reportable anchor prediction mean: `0.881952`
- Public best clip-at-5 fraction: `0.000000`

## Decision

Do not spend more quota on public-chase or GRU variants today. The `0.7991` file is the leaderboard-optimal already-submitted artifact, but it is public-feedback optimized. The `0.8124` anchor remains the reportable method lineage.

## Next Action

Prepare final report wording and final/private selection checklist that explicitly separates selected submission from reportable model lineage.

## Artifacts

- `private_robustness_audit.json`
- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
