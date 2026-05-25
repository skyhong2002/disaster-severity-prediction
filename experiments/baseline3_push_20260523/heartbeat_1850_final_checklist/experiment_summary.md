# 20260523 1850 Final/Private Selection Checklist

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Crossed Baseline 3: `True`
- Quota policy: no more public-chase or GRU submissions today.

## Selectable Submitted Artifact

Use this only as a Kaggle leaderboard/final-selection artifact if rules allow selecting submitted files:

- File: `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv`
- Public MAE: `0.7991`
- SHA-12: `d550c9cbc465`
- Label: `public-chase`
- Reportable method claim: `false`

## Reportable Method Lineage

Use this for report/method discussion:

- File: `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`
- Public MAE: `0.8124`
- SHA-12: `bee6f618828d`
- Method: 35% LightGBM / 35% XGBoost / 30% CatBoost with CatBoost tail2737
- Label: `reportable`

## Do Not Select Or Submit

- `restored_20260522_*`: blocked submit sources.
- `restored_unverified_*`: unverified restore; not legal/reportable.
- GRU stack/raw: live public `0.9850`/`0.9916`; validation did not transfer.
- Additional `baseline3_public_chase_v0_cat35_08124_alpha*` variants: stop condition already crossed; no more quota.

## Report Wording

Safe wording:

> Our clean reportable model lineage is the recovered exact 35/35/30 LGB/XGB/CatBoost anchor with public MAE 0.8124. For leaderboard/final selection, the best already-submitted artifact is a public-chase blend with public MAE 0.7991; this artifact is labeled public-chase and not used as a method claim.

Avoid claiming that the `0.7991` artifact is the primary model improvement, a new training method, or private-robust unless final private evidence later supports that claim.

## Inputs

- `experiments/baseline3_push_20260523/heartbeat_1650_final_selection/final_selection_matrix.json`
- `experiments/baseline3_push_20260523/heartbeat_1750_private_robustness/private_robustness_audit.json`
- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
