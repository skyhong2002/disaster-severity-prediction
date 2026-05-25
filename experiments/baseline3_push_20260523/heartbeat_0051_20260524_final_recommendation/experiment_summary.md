# 00:51 Final Selection Recommendation

- Live Kaggle check: Team 5 remains at public MAE `0.7991`, rank `5`, below Baseline 3 `0.8056`.
- No training, prediction, submission, or `missingness_shift_20260523` tmux job is running.
- Both final-selection and reportable-lineage CSVs passed sanity checks: 2248 rows, sample columns, region order, no NaN, predictions in `[0,5]`.
- If Kaggle final selection can choose an already submitted artifact, select `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` / Kaggle ref `52946620` (`0.7991`, SHA-12 `d550c9cbc465`).
- For report methodology, keep `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` (`0.8124`, SHA-12 `bee6f618828d`) as the reportable lineage.
- Do not submit more public-chase or GRU variants.
- Next action: commit/review the final recommendation and status documentation; use Kaggle UI only if final selection must be changed manually.
