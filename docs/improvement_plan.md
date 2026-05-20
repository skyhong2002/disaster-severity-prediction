# Historical Improvement Plan

Status: archived.

This file previously described an early plan for fixing train-test discrepancy by adding a separate score-estimation stage before horizon modeling. That plan is no longer the current implementation path.

The current project state is:

- Direct-horizon models for week 1 through week 5.
- LightGBM, XGBoost, and CatBoost model families.
- Leakage-aware 91-day-gapped score-history features.
- Feature profiles for memory control: `micro`, `lean`, and `full`.
- Three-model ensembling through `src/ensemble.py`.
- Current best legal public score recorded in the repo: `0.8124` from `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`.

Use `docs/experiment_summary.md` as the current source of truth for model selection and submission lineage.
