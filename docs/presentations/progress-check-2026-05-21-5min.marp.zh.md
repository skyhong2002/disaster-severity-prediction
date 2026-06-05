---
marp: true
theme: default
paginate: true
size: 16:9
_class: lead
style: |
  section {
    font-family: "Noto Sans CJK TC", "PingFang TC", "Microsoft JhengHei", "Inter", "Roboto", "Segoe UI", sans-serif;
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
    font-size: 0.66rem;
  }
---

# 進度檢查

## 天然災害嚴重度預測

**第 5 組**  
Shin-Yu Chen, Velda Wei-Hsin Hung, Sky Shih-Kai Hong

2026 年 5 月 21 日

<!-- Speaker timing: 0:00-0:15 -->

---

# 問題與資料集

**目標：** 在 91 天盲測缺口後，預測每個地區未來 5 週的每週乾旱嚴重度。

| 項目 | 數值 |
|---|---:|
| 地區數 | 2,248 |
| 訓練資料筆數 | 12,319,040 |
| 每週標記目標 | 1,746,696 |
| 氣象特徵 | 14 |
| 評估指標 | MAE，越低越好 |

<!-- Speaker timing: 0:15-0:55 -->

---

# 主要挑戰

- **標籤稀疏：** `score` 只有每週觀測一次，因此多數每日資料列沒有目標值。
- **時間尺度不一致：** 需要用每日氣象訊號解釋每週乾旱嚴重度。
- **91 天盲測缺口：** 測試 horizon 前沒有任何真實嚴重度分數可用。
- **驗證落差：** 不同 validation 設定的 MAE 不可直接比較，必須回到 Kaggle 與私榜風險檢查。

<!-- Speaker timing: 0:55-1:35 -->

---

# 目前已實作流程

目前 repo 已支援三個 boosted-tree model family。

1. 建立氣象、季節性、rolling、EWMA、climatology anomaly 與乾旱 proxy 特徵。
2. 使用 91-day gap 的 historical score features，避免直接碰到盲測區間。
3. 分別訓練 5 個 direct horizon 模型：第 1 週到第 5 週。
4. 支援 LightGBM、XGBoost、CatBoost，並用 `src/ensemble.py` 做三模型加權融合。

**工程狀態：** `src/predict.py` 可載入 LGB/XGB/CatBoost run directory 產生 submission。

<!-- Speaker timing: 1:35-2:35 -->

---

<!-- _class: compact -->

# 已完成實驗

| 實驗 | Validation | 本地 MAE | Public MAE / 狀態 |
|---|---:|---:|---|
| LGBM v2 | Holdout | 0.6942 | LGB/XGB anchor |
| XGBoost v1 | Holdout | 0.7150 | LGB/XGB anchor |
| LGB/XGB 50/50 | - | - | 0.8232 |
| CatBoost tail2737 | Rolling origin | 0.2212 / 0.2192 rerun | ensemble member |
| LGBM micro 20260520 | Rolling origin | 0.2002 | diagnostic only |
| **LGB/XGB/Cat 35/35/30** | Blind backtest + Kaggle sanity | 0.4038 blind | **0.8124** |

**目前 repo 紀錄的最佳 public：** `submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv`。

<!-- Speaker timing: 2:35-3:45 -->

---

# 目前洞察

單一 validation 分數不能直接代表 Kaggle 泛化能力。

- Weather-only 與 long-term weather 在本地看起來強，但 public LB 較弱。
- 91-day gap score history 仍然提供重要的地區性訊號。
- CatBoost 加入 native categorical handling 後，作為 diversity model 明顯改善 public score。
- CatBoost 35%、40% 與 horizon-ramp probes 都沒有超過 30% CatBoost public anchor。
- Rolling-origin 的低 MAE 很有參考價值，但和舊 holdout MAE 不可直接並列解讀。

下一步的重點不是追 public LB，而是檢查 private robustness。

<!-- Speaker timing: 3:45-4:35 -->

---

# 下一步

- 等 5/22 static private leaderboard，比較 LGB/XGB anchor 與 CatBoost blend 的私榜表現。
- 把 5/20 CatBoost 與 LGBM reruns 視為 post-readout blend inputs，不當成新 anchor。
- 用 `docs/status/current_state.json` 與 `scripts/check_current_state.py` 維持報告、簡報、實驗紀錄一致。
- 完成 IEEE 報告，並保持程式、實驗與提交紀錄一致。

**目前狀態：** repo 紀錄的最佳 legal public MAE = **0.8124**。

<!-- Speaker timing: 4:35-5:00 -->
