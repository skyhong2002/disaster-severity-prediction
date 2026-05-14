---
marp: true
theme: default
paginate: true
size: 16:9
_class: lead
style: |
  section {
    font-family: "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Heiti TC", sans-serif;
  }

  h1, h2, h3, h4, h5, h6, strong {
    font-family: "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Heiti TC", sans-serif;
    font-weight: 700;
  }

  code, pre {
    font-family: "Noto Sans Mono CJK TC", "SF Mono", Menlo, Consolas, monospace;
  }
---

# 期中進度報告 (Progress Check)

## 自然災害嚴重度預測 (Natural Disaster Severity Prediction)

**第 5 組 (Group 5)**  
Hsin-Yu Chen, Wei-Hsin Hung, Sky Shih-Kai Hong

2026 年 5 月 21 日

---

# 任務與數據集

預測每個地區在 91 天測試期之後**未來五週**的旱災嚴重程度。

| 屬性 | 數值 |
|---|---:|
| 預測地區數量 | 2,248 |
| 訓練集總資料筆數 | 12,319,040 |
| 可用 weekly label rows | 1,746,696 |
| 每個地區的測試天數 | 91 |
| 氣象特徵數量 | 14 |

評估指標：**MAE (平均絕對誤差)**，越低越好。

---

# 數據觀察與痛點

目標值 `score` 是以「週」為單位，但氣象資料是以「日」為單位。

- 每週約有 **6 天的 score 是 NaN**，標籤相對稀疏。
- 核心難題：將**每日天氣訊號**對齊到**每週旱災嚴重度**。
- 測試集 91 天完全沒有真實 `score`，不能直接使用近期分數。
- 因此任何 score-history 特徵都必須處理 train/test discrepancy。

---

# 目前的模型架構 (Pipeline)

目前已整理成可重現的 **Direct Horizon Forecasting Pipeline**。

1. 從天氣、日曆、climatology/anomaly 建立時間序列特徵。
2. 加入乾旱領域特徵，如 drought index 與 dryness index。
3. 針對未來 5 週，分別訓練獨立 horizon model。
4. 支援 LightGBM、XGBoost、Ensemble 與 versioned experiments。

---

# 特徵工程 (Feature Engineering)

目前模型輸入包含多種特徵 profile，可在準確率與記憶體間取捨：

- **天氣動態**：lags、rolling mean/std、EWM。
- **時間訊號**：month、week、seasonal cyclic encoding。
- **領域特徵**：drought index、dryness index、dewpoint / wetbulb spread。
- **Gap-aware score history**：只使用 91 天 blind gap 之前的歷史分數。
- **Profiles**：`micro`、`lean`、`full`；目前最新完整 run 使用 `lean`。

---

## 直接多步預測 (Direct Multi-Step Forecasting)

針對 `week1` 到 `week5` 分別訓練 5 個獨立模型，避免遞迴式誤差累積。

## 驗證策略 (Validation Strategy)

已從單一 holdout 擴充到 **rolling-origin validation**，用多個 forecast origin 模擬真實預測情境；最新 run 使用 3 folds。

---

# 本地驗證結果

目前保留的完整 experiments：

| Experiment | Validation | Features | Local MAE |
|---|---|---:|---:|
| `lgbm_v2` | holdout | 337 | 0.6942 |
| `xgb_v1` | holdout | 337 | 0.7150 |
| `lgbm_direct` | holdout | 318 | 0.6770 |
| `xgb_direct` | holdout | 318 | 0.7320 |
| `lean_v2` | rolling-origin | 240 | **0.1915** |

`lean_v2` 是目前最新完整 run，但尚未用 Kaggle 驗證泛化。

---

# Kaggle Public Leaderboard（**40%** 測試集資料）

| Method | MAE | Insight |
|---|---:|---|
| **Naive Persistence (Forward Fill)** | **1.0815** | 證明乾旱具備動態性，單純延續歷史最後一筆觀測值會導致嚴重的預測誤差 |
| **Incompatible Ensemble (Leaky + Clean)** | **0.8851** | 將「靜態洩漏模型」與「動態預測模型」強行平均，因優化流形 (Manifold) 衝突導致泛化崩潰 |
| **Team 5 Strategy B (純天氣 Direct)** | **0.8640** | 放棄歷史分數純看天氣。Local MAE 佳，但在 LB 泛化不足 |
| **Team 5 (長期氣象特徵 365天)** | **0.8604** | 即使給予長達一年的天氣累積量，依然無法取代隱含在歷史分數中的未觀察干擾變數 (Unobserved Confounders) |

---

# Kaggle Public Leaderboard (Cont.)

| Method | MAE | Insight |
|---|---:|---|
| **Team 5 Ensemble v1 (兩階段 Reconstructor)** | **0.8232 🏆** | **本組提議之最佳穩健模型**。透過兩階段機器學習架構，合理推斷盲區內的趨勢，完美結合氣象與歷史狀態 |
| **Team 5 初始提交 (Target Leakage Artifact)** | **0.8094** | 由於 `ffill` 的特徵漏洞與時間切分瑕疵，意外產生具備強大基準值的混合預測模型。**此為 Data Leakage，為維護學術嚴謹性已主動棄用**。 |

---

# 結論與學習 (The Kaggle Paradox & Ablation Study)

我們從這個深度的消融實驗中學到寶貴的一課。我們原本以為假分數的雜訊會傷害模型，因此嘗試用高達 365 天的真實天氣累積量來完全取代它。雖然這讓 Local MAE 暴降至 `0.488`，但 Kaggle 上依然只有 `0.8604`。
這證明了：**歷史旱災分數 (即使帶有雜訊)，依然包含了「純天氣數據」絕對無法捕捉的「未觀察干擾變數 (Unobserved Confounders)」**（如：當地的水利設施、土壤特性）。放棄這些資訊純看天氣會導致模型過擬合！因此，採用兩階段重建模型 (v1 融合) 的 `0.8232`，才是正確、合法、具備穩健泛化能力的最佳架構。

---

# 系統工程進度

程式碼庫已經支援穩定迭代與結果追蹤：

- 核心腳本已模組化：`src/train.py`, `src/predict.py`, `src/features.py`。
- 已新增 `src/train_xgb.py` 與 `src/ensemble.py` 供模型比較與融合。
- 每次訓練保存 `config.json`、`metrics.json`、模型檔與 submission metadata。
- 已清理只有 `config.json`、沒有 metrics/model 的 incomplete runs。

目前 `experiments/latest.txt` 指向完整的 `lean_v2` run。

---

# 關鍵發現與挑戰 (Discoveries & Challenges)

- **Data leakage 風險**：隨機切分或不嚴格時間切分會讓 local MAE 過度樂觀。
- **Train-test discrepancy**：測試期沒有真實 `score`，不能直接套用近期 score lags。
- **Direct baseline 更乾淨**：LGBM direct local MAE 0.6770，但 Kaggle 為 0.8615。
- **Gap-aware score history 很有潛力**：`lean_v2` rolling-origin MAE 降到 0.1915。

目前重點是確認 `lean_v2` 的 Kaggle 泛化，而不是只相信 local MAE。

---

# 下一步 (Next Steps)

在 Deadline 之前，我們計畫：

- 用 `lean_v2` 產生 submission，並在保留 quota 的前提下上傳 Kaggle。
- 若 public MAE 改善，將 gap-aware score-history 設為主模型方向。
- 若 public MAE 未改善，檢查 rolling-origin split 是否仍有樂觀偏差。
- 補齊報告中的模型比較：direct LGBM、XGBoost、Ensemble、gap-aware features。

---

# 總結 (Summary)

目前已完成：

- 建立可重現的 direct forecasting pipeline 與 experiment tracking。
- 修正早期 train/test discrepancy 與 validation 設計問題。
- 實作 LightGBM、XGBoost、Ensemble 與 gap-aware score-history。
- Kaggle clean baseline: **0.8232**；最新 local candidate: **0.1915**。

接下來的重點：驗證 `lean_v2` Kaggle 表現，並整理成最終報告方法論。
