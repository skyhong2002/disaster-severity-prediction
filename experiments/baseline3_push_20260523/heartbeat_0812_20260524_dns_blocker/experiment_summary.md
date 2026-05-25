# 2026-05-24 08:12 Daily Quota Check

- Kaggle live check: blocked by DNS. `curl` and Kaggle CLI both could not resolve `api.kaggle.com`.
- Submission action: no `--submit` run and no quota spent. Without live history and leaderboard, submitting would be unsafe.
- Last verified live snapshot remains `2026-05-24T02:51:32+08:00`: Team 5 public MAE `0.7991`, Baseline 3 public MAE `0.8056`, stop condition crossed.
- Local sanity rechecked:
  - `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv`: 2248 rows, sample columns, region order, no NaN, predictions in `[0,5]`, SHA-256 `d550c9cbc465d52f520756e3e97072f1e7bfe6a13150ba2ed8c6204632264a79`, label `public-chase`.
  - `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`: 2248 rows, sample columns, region order, no NaN, predictions in `[0,5]`, SHA-256 `bee6f618828db8d5263ba9df79eb81572c7c994b1b72abf184466078496653c9`, label `reportable`.
- `tmux ls` could not run in this sandbox, but `experiments/logs/missingness_shift_20260523.log` ends with `All missingness / shift experiments finished.`
- Next action: retry Kaggle live history and leaderboard after DNS recovers. If Team 5 still remains below `0.8056`, continue private robustness and reportable lineage cleanup, not public-chase quota spending.

## 2026-05-24 09:20 Follow-up Live Check

- DNS recovered: `nslookup api.kaggle.com 1.1.1.1` resolved `34.54.168.202`, and `curl -I --max-time 10 https://api.kaggle.com` returned an HTTP response instead of a DNS error.
- Kaggle CLI recovered: both live leaderboard and submission history commands completed.
- Live leaderboard remains crossed: Team 5 public MAE `0.7991`, Baseline 3 public MAE `0.8056`.
- Submission history still confirms `baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` is complete with public MAE `0.7991`.
- Action: no public-chase submission; continue private robustness and reportable lineage cleanup.
