---
marp: true
theme: default
paginate: true
size: 16:9
_class: lead
style: |
  section {
    font-family: "Inter", "Roboto", "Segoe UI", sans-serif;
  }
  h1, h2, h3, h4, h5, h6, strong {
    font-weight: 700;
  }
  code, pre {
    font-family: "Menlo", "Consolas", monospace;
  }
  section.compact h1 {
    font-size: 1.9rem;
  }
  section.compact p,
  section.compact li {
    font-size: 0.82rem;
  }
  section.compact table {
    font-size: 0.68rem;
  }
---

# Progress Check

## Natural Disaster Severity Prediction

**Group 5**  
Shin-Yu Chen, Velda Wei-Hsin Hung, Sky Shih-Kai Hong

May 21, 2026

<!-- Speaker timing: 0:00-0:15 -->

---

# Problem & Dataset

**Goal:** predict weekly drought severity for each region over a 5-week horizon after a 91-day blind gap.

| Item | Value |
|---|---:|
| Regions | 2,248 |
| Training records | 12,319,040 |
| Weekly labeled targets | 1,746,696 |
| Meteorological features | 14 |
| Metric | MAE, lower is better |

<!-- Speaker timing: 0:15-0:55 -->

---

# Key Challenges

- **Sparse labels:** `score` is only observed weekly, so most daily rows have no target.
- **Temporal mismatch:** daily weather signals must explain weekly drought severity.
- **91-day blind gap:** no ground-truth severity score is available before the test horizon.
- **Validation mismatch:** MAE values from different validation strategies are not directly comparable.

<!-- Speaker timing: 0:55-1:35 -->

---

# Current Implementation

The repository now supports three boosted-tree model families.

1. Build weather, seasonality, rolling, EWMA, climatology anomaly, and drought-proxy features.
2. Use 91-day-gapped historical score features to avoid touching the blind test window.
3. Train five direct horizon models: Week 1 through Week 5.
4. Blend LightGBM, XGBoost, and CatBoost outputs with `src/ensemble.py`.

**Engineering status:** `src/predict.py` can load LGB/XGB/CatBoost run directories.

<!-- Speaker timing: 1:35-2:35 -->

---

<!-- _class: compact -->

# Completed Experiments

| Experiment | Validation | Local MAE | Public MAE / Status |
|---|---|---:|---|
| LGBM v2 | Holdout | 0.6942 | LGB/XGB anchor |
| XGBoost v1 | Holdout | 0.7150 | LGB/XGB anchor |
| LGB/XGB 50/50 | - | - | 0.8232 |
| CatBoost tail2737 | Rolling origin | 0.2212 / 0.2192 rerun | ensemble member |
| LGBM micro 20260520 | Rolling origin | 0.2002 | diagnostic only |
| **LGB/XGB/Cat 35/35/30** | Blind backtest + Kaggle sanity | 0.4038 blind | **0.8124** |

**Current best public record:** `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`.

<!-- Speaker timing: 2:35-3:45 -->

---

# Main Insight

A single local validation score does not fully explain Kaggle generalization.

- Weather-only and long-term weather models looked strong locally but were weaker on the public leaderboard.
- 91-day-gapped score history still provides important region-specific context.
- CatBoost's native categorical handling added useful ensemble diversity and improved public MAE.
- CatBoost 35%, 40%, and horizon-ramp probes did not beat the 30% public anchor.
- Rolling-origin MAE is useful, but should not be compared directly against older holdout MAE.

The next decision should be based on private robustness, not only public score.

<!-- Speaker timing: 3:45-4:35 -->

---

# Next Steps

- Compare the LGB/XGB anchor and CatBoost blends on the 5/22 static private leaderboard.
- Treat the 5/20 CatBoost and LGBM reruns as post-readout blend inputs, not new anchors.
- Use `docs/status/current_state.json` and `scripts/check_current_state.py` to keep report, slides, and experiment records aligned.
- Finalize the IEEE report and keep code, experiments, and submission lineage consistent.

**Current status:** repo-recorded best legal public MAE = **0.8124**.

<!-- Speaker timing: 4:35-5:00 -->
