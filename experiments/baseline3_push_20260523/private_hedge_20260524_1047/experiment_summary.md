# 2026-05-24 Manual Private-Hedge Push

- Created at Taipei time: `2026-05-24T10:49:42+08:00`
- Automation used: `false`
- Baseline 3 public MAE: `0.8056`
- Team 5 public MAE after batch: `0.7930`
- Quota spent today: `6` submissions
- Sanity: all submitted files have `2248` rows, sample columns, matching region order, no NaN, predictions in `[0,5]`, and no forbidden `restored_20260522_*` / `restored_unverified_*` filename pattern.

| Submission | Kaggle ref | Public MAE | SHA-12 | Anchor delta | Decision |
|---|---:|---:|---:|---:|---|
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p50.csv` | `52972055` | `0.7976` | `2dd02ae9e429` | `0.209374` | passed baseline3 and improved prior public best |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8.csv` | `52972132` | `0.7930` | `d2ba4500363e` | `0.194764` | best public and primary selected public chase candidate |
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p65.csv` | `52972150` | `0.7982` | `c30d494a8ba9` | `0.146562` | passed baseline3 but uniform anchor shift started to regress public |
| `submissions/baseline3_private_hedge_v0_cat35_08124_alphap0p80.csv` | `52972169` | `0.8022` | `994dd59ffee7` | `0.083750` | passed baseline3 as high anchor weight upper bound but not public best |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p4_0p45_0p55_0p7_0p85.csv` | `52972194` | `0.7933` | `a417100cad2b` | `0.173827` | near best public and stronger private hedge than primary |
| `submissions/baseline3_private_hedge_v0_cat35_horizon_0p45_0p55_0p65_0p85_1p0.csv` | `52972219` | `0.7945` | `419d11ce0a81` | `0.128243` | passed baseline3 and is most anchor tilted submitted horizon hedge |

Readout: the best selectable public artifact is now `submissions/baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8.csv` / Kaggle ref `52972132` with public MAE `0.7930`. For a more private-robust hedge, keep `submissions/baseline3_private_hedge_v0_cat35_horizon_0p4_0p45_0p55_0p7_0p85.csv` (`0.7933`) and `submissions/baseline3_private_hedge_v0_cat35_horizon_0p45_0p55_0p65_0p85_1p0.csv` (`0.7945`) in the final-selection discussion. The reportable method lineage remains the clean `0.8124` LGB/XGB/CatBoost anchor.
