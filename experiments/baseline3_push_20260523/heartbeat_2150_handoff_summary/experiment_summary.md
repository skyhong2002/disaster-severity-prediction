# 20260523 2150 Handoff Summary

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Crossed Baseline 3: `True`
- Team 5 rank: `5`

## Final Decision

- Stop public-chase: `true`
- Quota recommendation: `do_not_submit_more_public_chase_or_gru_today`
- Selected submitted artifact, if final selection is allowed: `submissions/baseline3_public_chase_v0_cat35_08124_alphap0p35.csv` (`0.7991`, SHA-12 `d550c9cbc465`, label `public-chase`)
- Reportable method lineage: `experiments/recovered_submissions_20260523/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv` (`0.8124`, SHA-12 `bee6f618828d`, label `reportable`)

## Do Not Do

- Do not submit more baseline3_public_chase alpha variants today.
- Do not submit GRU stack/raw or GRU-derived hedges as final/private anchors after public 0.9850/0.9916.
- Do not use restored_20260522_* or restored_unverified_* as legal/reportable sources.
- Do not claim the 0.7991 public-chase artifact as the reportable modeling method.

## Ready Artifacts

- `baseline3_status_md`: `docs/status/baseline3_push_20260523.md`
- `baseline3_status_json`: `docs/status/baseline3_push_20260523.json`
- `current_state_json`: `docs/status/current_state.json`
- `final_selection_matrix`: `experiments/baseline3_push_20260523/heartbeat_1650_final_selection/final_selection_matrix.json`
- `private_robustness_audit`: `experiments/baseline3_push_20260523/heartbeat_1750_private_robustness/private_robustness_audit.json`
- `final_private_selection_checklist`: `experiments/baseline3_push_20260523/heartbeat_1850_final_checklist/final_private_selection_checklist.json`
- `report_wording_package`: `experiments/baseline3_push_20260523/heartbeat_1950_report_wording/report_wording_package.json`
- `consistency_audit`: `experiments/baseline3_push_20260523/heartbeat_2050_consistency_audit/consistency_audit_report.json`

## Next Best Action

Prepare a focused commit/PR summary for the baseline3_push_20260523 artifacts and integrate the report wording into report/main.tex if final report editing is next.

## Live Inputs

- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
