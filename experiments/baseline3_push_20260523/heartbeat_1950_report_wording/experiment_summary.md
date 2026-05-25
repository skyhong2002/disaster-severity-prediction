# 20260523 1950 Report Wording Package

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Crossed Baseline 3: `True`
- Quota policy: no more public-chase or GRU submissions today.

## One-Sentence Result

Team 5 crossed Baseline 3 on the live public leaderboard with public MAE 0.7991; for method reporting, we keep the clean 35/35/30 LGB/XGB/CatBoost lineage at public MAE 0.8124 separate from the public-chase selected artifact.

## Methods Paragraph

Our reportable model lineage is an exact recovered ensemble of 35% LightGBM, 35% XGBoost, and 30% CatBoost with CatBoost tail2737. The recovered artifact has SHA-12 bee6f618828d, passes the official submission sanity checks, and has public MAE 0.8124. This is the model lineage we use for methodological discussion because it is reproducible, non-leaky, and rule-safe.

## Leaderboard/Final-Selection Paragraph

For Kaggle leaderboard/final-selection purposes, the best already-submitted artifact is submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv, with public MAE 0.7991 and SHA-12 d550c9cbc465. This artifact is labeled public-chase: it is valid as a submitted leaderboard artifact, but it should not be described as the primary reportable modeling improvement.

## Limitations Paragraph

The 0.7991 artifact was chosen after public-feedback-guided alpha probing between recovered historical submissions. Therefore, it carries public-overfit risk and should be separated from the reportable model claim. GRU stack/raw candidates were rejected for final selection after live public MAE 0.9850/0.9916, despite promising local validation.

## Final Selection Paragraph

If final selection is based on submitted Kaggle artifacts, select the 0.7991 public-chase submission. In the written report, describe the clean 0.8124 LGB/XGB/CatBoost anchor as the reportable method lineage and explicitly label the 0.7991 artifact as a public-chase final-selection artifact.

## Must Not Claim

- Do not claim the 0.7991 public-chase artifact is a new training method.
- Do not claim the 0.7991 artifact is private-robust without private evidence.
- Do not use restored_20260522_* or restored_unverified_* as legal/reportable sources.
- Do not select GRU stack/raw as final anchors after their public MAE 0.9850/0.9916 readout.

## Artifacts

- `report_wording_package.json`
- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
