# 2026-05-25 08:12 Daily Kaggle Quota Check

- Kaggle live check blocked by DNS: `api.kaggle.com` could not be resolved by `curl` or Kaggle CLI.
- `nslookup api.kaggle.com 1.1.1.1` also failed in this sandbox with `isc_socket_bind: Operation not permitted`, so it cannot disambiguate upstream DNS from sandbox socket restrictions.
- No `--submit` run was executed and no quota was spent. The intended submit command remains `uv run python scripts/run_baseline3_push.py --submit --allow-missing-features`, but it requires live history and leaderboard to be reachable first.
- No `missingness_shift_20260523` tmux session is visible.
- Last verified live snapshot remains `2026-05-24T10:49:42+08:00`: Team 5 public MAE `0.7930`, Baseline 3 public MAE `0.8056`, Team 5 rank `3`, stop condition crossed.
- Local sanity rechecked for the selected `public-chase` artifact, two private-hedge alternatives, and the `reportable` lineage: all have `2248` rows, sample columns, matching `region_id` order, no NaN, predictions in `[0,5]`, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

## Current Recommendation

- Do not submit while Kaggle live state is unreachable.
- When DNS recovers, fetch submissions and leaderboard first.
- If Team 5 remains below `0.8056`, keep public-chase closed and continue private robustness / final-selection review.
- If final selection needs a public-best artifact, use `submissions/baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8.csv` / Kaggle ref `52972132` / public MAE `0.7930` / SHA-12 `d2ba4500363e` as `public-chase`.
- For report methodology, keep `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` / Kaggle ref `52698259` / public MAE `0.8124` / SHA-12 `bee6f618828d` as `reportable`.
