# Anchor Family Probe Summary - 2026-05-22

This note records the first batch of controlled probes after the Kaggle daily
submission limit increased from 3 to 6. The goal was to spend the extra quota
on explicit hypotheses rather than public-LB fishing.

## Inputs

The probe files were generated with `scripts/make_submission_blend.py`, using
the locally restored LGB/XGB/CatBoost component files plus, for two candidates,
the feature-fused TCN submission.

| Input | SHA-256 prefix | Note |
|---|---|---|
| `submissions/restored_20260522_lgb_component_for_20260516_anchor.csv` | `e286222e99b3` | Locally restored LGB component. |
| `submissions/restored_20260522_xgb_component_for_20260516_anchor.csv` | `ae42359e5494` | Locally restored XGB component. |
| `submissions/restored_20260522_cat_component_for_20260516_anchor.csv` | `38885424d0e8` | Locally restored CatBoost component. |
| `submissions/submission_20260522_023818_20260522_022815_tcn_tcn_feature_fused_tail1825_8ep_20260522_feature_fused.csv` | `d62f9e32e312` | Feature-fused TCN, public MAE `0.9450`. |

## Submitted Probes

| Candidate | Kaggle ref | Weights | Public MAE | SHA-256 prefix | Readout |
|---|---:|---|---:|---|---|
| `submissions/ensemble_20260522_lgb_xgb_cat2737_375_375_25.csv` | `52928386` | LGB/XGB/Cat `37.5/37.5/25` | `0.9546` | `795f7676a95d` | Negative; local restored components are not safe anchor sources. |
| `submissions/ensemble_20260522_lgb_xgb_cat2737_soft_cat_ramp.csv` | `52928403` | Cat ramp `25/27.5/30/32.5/35` | `0.9561` | `426d6e5f944e` | Negative; do not use restored components for anchor-family probes. |
| `submissions/ensemble_20260522_anchor_tcnf_cap5_global.csv` | `52928409` | Anchor-scaled LGB/XGB/Cat plus TCNF `5%` | `0.9509` | `452121d24b23` | Negative; TCNF cap cannot compensate for bad restored anchor source. |
| `submissions/ensemble_20260522_anchor_tcnf_late_ramp_cap10.csv` | `52928422` | TCNF late ramp `0/0/2.5/5/10%` | `0.9531` | `43fe42e39220` | Negative; same artifact-lineage issue. |

## Decision

The failed probes should not be interpreted as evidence that the original
`0.8124` LGB/XGB/CatBoost anchor is weak. They were generated from locally
restored component files, and their public scores show those restored artifacts
are not equivalent to the original successful submission lineage.

Rules going forward:

- Do not create new Kaggle submissions from `restored_20260522_*_component`
  files.
- Treat `restored_unverified_20260522_lgb_xgb_cat2737_35_35_30.csv` as
  non-submittable unless it is verified against the original Kaggle artifact.
- Continue to keep `ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`, Kaggle ref
  `52698259`, public MAE `0.8124`, as the current legal anchor in documentation.
- Extra daily quota should now be spent only on candidates generated from
  reproducible current runs or recovered exact historical CSVs, not reconstructed
  component guesses.
