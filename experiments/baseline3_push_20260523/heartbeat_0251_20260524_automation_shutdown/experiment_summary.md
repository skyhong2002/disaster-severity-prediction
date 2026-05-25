# 02:51 Automation Shutdown

- Live Kaggle check: Team 5 remains at public MAE `0.7991`, rank `5`, below Baseline 3 `0.8056`.
- No training, prediction, submission, or `missingness_shift_20260523` tmux job is running.
- Public-chase objective is complete and repeatedly verified.
- If the Kaggle final-selection UI allows selecting an existing submission, choose `baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` / Kaggle ref `52946620`.
- For report methodology, keep `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` / Kaggle ref `52698259` / public MAE `0.8124` as the reportable method lineage.
- Do not submit more public-chase or GRU variants.
- Recommendation: delete the hourly `train-next-kaggle-model` automation because the Baseline 3 crossing workflow has reached its terminal state.
