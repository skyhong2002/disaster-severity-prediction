# 20260523 2050 Consistency Audit

## Live Gate

- Team 5 public MAE: `0.7991`
- Baseline 3 public MAE: `0.8056`
- Crossed Baseline 3: `True`
- Decision: public-chase remains closed; do not spend more quota.

## Fixes Applied

- `docs/status/current_state.json` `baseline3_push.mode`: `public-first` -> `final-report-after-public-cross`

## Consistency Checks

| File | stop policy | reportable anchor | restored block | stale public-first |
|---|---|---|---|---|
| `docs/status/current_state.json` | `True` | `True` | `False` | `False` |
| `docs/status/baseline3_push_20260523.json` | `True` | `True` | `True` | `False` |
| `docs/status/baseline3_push_20260523.md` | `True` | `True` | `True` | `False` |
| `experiments/baseline3_push_20260523/heartbeat_1950_report_wording/report_wording_package.json` | `True` | `True` | `True` | `False` |
| `experiments/baseline3_push_20260523/heartbeat_1850_final_checklist/final_private_selection_checklist.json` | `True` | `True` | `True` | `False` |

## Remaining Risk

- No stale `public-first` mode found in checked status/final-selection files.

## Next Action

If the leaderboard remains unchanged, stop creating new modeling artifacts and prepare a commit/summary of `baseline3_push_20260523` plus final report integration.

## Artifacts

- `consistency_audit_report.json`
- `kaggle_leaderboard_live.txt`
- `kaggle_submissions_live.txt`
