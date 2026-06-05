# Documentation Index

This directory is the project documentation hub. Use `status/` first when you
need the current decision state, then follow the dated experiment readouts for
lineage and rejected alternatives.

## Structure

| Directory | Purpose | Start here |
|---|---|---|
| `status/` | Current submission state, leaderboard interpretation, and private-LB plan. | `status/experiment_summary.md` |
| `experiments/` | Dated model-family readouts, Kaggle probes, and reproduction notes. | `experiments/anchor_family_probe_summary_20260522.md` |
| `validation/` | Blind backtests, drift checks, blend reports, and train/test distribution diagnostics. | `validation/validation_protocol.md` |
| `presentations/` | Marp progress-check decks and exported PDFs. | `presentations/progress-check-2026-05-21-5min.marp.zh.md` |
| `project/` | Assignment brief, planning notes, and external research references. | `project/DM_114_FinalProject.md` |
| `engineering/` | Implementation details that are broader than a single experiment. | `engineering/FEATURE_ENGINEERING_IMPLEMENTATION.md` |

## Current Sources Of Truth

- Current model-selection state: `status/experiment_summary.md`
- Machine-readable state guard: `status/current_state.json`
- Validation rules and leakage boundaries: `validation/validation_protocol.md`
- Report source: `../report/main.tex`
- Current legal public anchor: `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`, a LightGBM/XGBoost/CatBoost blend with public MAE `0.8124` and blind MAE `0.4038`.

## Submission Readout Policy

Record every Kaggle submission with:

- Exact file path and Kaggle timestamp.
- Public score and status.
- Source lineage and legality notes.
- The decision: promote, keep as diagnostic, reject, or defer.

Do not promote a candidate from local rolling-origin, blind, or LOO validation
alone. Public/private leaderboard evidence and reproducible lineage still
control the official model-selection narrative.
