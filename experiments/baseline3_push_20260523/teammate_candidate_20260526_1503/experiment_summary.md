# Teammate Candidate Screen 2026-05-26

- Created Taipei: `2026-05-26T15:03:35+08:00`
- Candidate path: `/Users/skyhong/.codex/attachments/3a7c1688-1166-49c5-97c3-5a6b48128735/submission_20260526_145450_20260521_164434_lightgbm_two_stage_lgbm_v3_enhanced.csv`
- SHA-256: `66f7815e7ff30430fe543d1d1df0d3b1c0167594edbe2a34453418c25c1240cf`
- SHA-12: `66f7815e7ff3`
- Status: not submitted.
- Artifact label: `external-teammate-candidate`; `public-chase=true`; `reportable_method_claim=false`.
- Lineage status: `provenance_pending`. No matching run directory, exact filename, or exact local artifact was found in the current checkout.
- Staged submission path: `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv`.
- Queue update: per user directive on 2026-05-26, this candidate is now rank `1` for the first post-reset submission slot on `2026-05-27T08:00:00+08:00`, subject to live Kaggle and exact-file sanity gates.
- Kaggle result: already present in the 2026-05-27 live history as ref `53074655`, submitted by `veldahung`, public MAE `1.0685`.
- Decision: negative public readout; keep as `provenance_pending` and do not promote or use as reportable method contribution.

## Sanity Check

The file passed the standard submission-format gate: `2248` rows, sample columns, matching `region_id` order, no NaN predictions, prediction range in `[0,5]`, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.

| Metric | Value |
|---|---:|
| Prediction min | `0.000000` |
| Prediction max | `5.000000` |
| Prediction mean | `0.379247` |
| Prediction std | `0.823690` |
| Zero predictions | `2458` |
| Exact 5.0 predictions | `3` |

## Live Context

Live Kaggle checks at `2026-05-26T15:03:35+08:00` still showed Team 5 at public MAE `0.7917`, rank `3`, below Baseline 3 `0.8056`. The 2026-05-26 UTC quota had already been spent (`6/6`) on v2 private-hedge frontier submissions, so this candidate cannot be submitted before the `2026-05-27T08:00:00+08:00` reset.

## Distance To Current Artifacts

| Reference | Public MAE | Mean abs diff | RMSE diff | Flat corr |
|---|---:|---:|---:|---:|
| Current public-best v2 ref `53038031` | `0.7917` | `0.803732` | `1.052451` | `0.394512` |
| Smooth private hedge v2 ref `53038040` | `0.7929` | `0.789266` | `1.034856` | `0.415206` |
| Reportable clean anchor ref `52698259` | `0.8124` | `0.695846` | `0.944773` | `0.526317` |
| LGBM refit ref `52882449` | `0.9380` | `0.460938` | `0.664559` | `0.691746` |

The candidate is much lower in average prediction level than the currently successful public frontier (`0.379` versus roughly `0.94` to `0.95`) and has many more zero predictions. It is closer to the older single LGBM refit that scored public `0.9380` than to the recent public-best horizon hedges.

## Decision

Per user directive, this was the first 2026-05-27 queue item. It was already submitted by the time of the live gate and scored public MAE `1.0685`, so it is now a negative public readout. It remains a public-chase/provenance-pending teammate probe, not a reportable method claim.

Required before submission:

- the live history/leaderboard gate passes after the `08:00` Taipei reset;
- `submissions/teammate_first_queue_20260527_lightgbm_two_stage_lgbm_v3_enhanced.csv` is revalidated and the SHA remains `66f7815e7ff30430fe543d1d1df0d3b1c0167594edbe2a34453418c25c1240cf`;
- no evidence appears that the generation path used private labels, external answers, unauthorized data, `restored_20260522_*`, or `restored_unverified_*` sources.

After this first teammate probe, continue adjacent probes around the v2 frontier and keep refs `53038040`, `53038036`, and `53038033` as stronger private-risk alternatives.

Queue artifact: `experiments/baseline3_push_20260523/teammate_candidate_20260526_1503/next_submission_queue_20260527.json`.
