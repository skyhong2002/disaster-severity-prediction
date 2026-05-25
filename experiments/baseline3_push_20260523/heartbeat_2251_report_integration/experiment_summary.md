# 22:51 Report Integration

- Live Kaggle check: Team 5 remains at public MAE `0.7991`, rank `5`, below Baseline 3 `0.8056`.
- Public-chase remains closed. Do not submit more public-chase or GRU variants today.
- Updated `report/main.tex` to separate:
  - selected Kaggle/public-chase artifact: `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv`, public MAE `0.7991`, SHA-12 `d550c9cbc465`;
  - reportable method lineage: `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`, public MAE `0.8124`, SHA-12 `bee6f618828d`.
- The report now says the `0.7991` artifact is a selected leaderboard artifact, not evidence of a new reportable training method.
- Next action: review/commit this report integration and keep further work focused on private robustness or final-selection documentation.
