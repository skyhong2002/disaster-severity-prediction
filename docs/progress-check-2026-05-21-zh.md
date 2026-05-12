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
| 訓練集總資料筆數 | 1,230 萬 |
| 每個地區的訓練天數 | 5,480 |
| 每個地區的測試天數 | 91 |
| 氣象特徵數量 | 14 |

評估指標：**MAE (平均絕對誤差)**，越低越好。

---

# 數據觀察與痛點

目標值 `score` (分數) 是以「週」為單位，但氣象數據是以「日」為單位。

- 導致每週約有 **6 天的 score 是 NaN**，相對於特徵來說標籤非常稀疏。
- 核心的建模難題：如何將**每日的天氣訊號**與**每週的旱災嚴重度**對齊。
- 測試集完全沒有 `score`，但過去的分數歷史通常對預測未來很有幫助。

---

# 目前的模型架構 (Pipeline)

我們實作了一個具備高重現性的 **直接預測融合模型 (Direct Forecasting Ensemble)**。

1. 從天氣、日曆以及地區統計數據中建立時間序列特徵。
2. 結合我們新引入的領域知識特徵（乾旱指數 Drought Index 與 乾燥指數 Dryness Index）。
3. 針對未來 5 週，分別訓練 **LightGBM** 與 **XGBoost** 兩個直接預測模型 (Horizon models)。
4. 將兩者的預測結果進行融合 (Ensemble)，產出穩健的 Kaggle 提交檔。

---

# 特徵工程 (Feature Engineering)

目前的模型輸入包含了多樣化的特徵組合：

- **天氣動態**：滯後特徵 (Lags)、滾動平均/標準差、指數加權平均 (EWM)。
- **時間與地區**：日曆編碼 (週期性特徵) 與地區歷史分數的統計特徵。
- **領域特徵 (Domain Features)**：乾旱指數 (溫度/降雨量) 與乾燥指數 (日溫差 * 最高溫)，用以明確捕捉容易發生旱災的氣候條件。
- *註*：因為會產生嚴重的訓練/測試集誤差，我們已經徹底移除了對歷史分數 (Score lags) 的依賴。

---

## 直接多步預測 (Direct Multi-Step Forecasting)

針對 `week1` 到 `week5` 分別訓練 5 個獨立的模型。每個模型專門預測未來某一週的分數。

## 融合策略 (Ensemble Strategy)

在完全相同的按時間排序特徵上，同時訓練 **LightGBM** (Leaf-wise 生長) 與 **XGBoost** (Depth-wise 生長)。最終的提交檔案採用 50/50 的比例平均融合，以最大化模型的多樣性 (Diversity)。

---

# 本地驗證結果 (時間序列切分)

我們修復了嚴重的**時間洩漏 (Time-Leakage)** Bug，改為嚴格的「時間序列保留法 (Chronological holdout)」（保留最後 20% 的時間作為驗證集）。

| 預測週數 | LGBM MAE | XGB MAE |
|---|---:|---:|
| 第 1 週 | 0.6772 | 0.7127 |
| 第 2 週 | 0.6729 | 0.7365 |
| 第 3 週 | 0.6795 | 0.7427 |
| 第 4 週 | 0.6761 | 0.7297 |
| 第 5 週 | 0.6795 | 0.7384 |
| **平均 MAE** | **0.6770** | **0.7320** |

---

# Kaggle 公開排行榜成績

Public Leaderboard 大約使用了 **40%** 的測試集資料。

| 方法 (Method) | Public MAE |
|---|---:|
| Baseline 3 | 0.8056 |
| Team 20 | 0.8062 |
| **Team 5 (v0 含有資料洩漏)** | **0.8094** |
| **Team 5 (v1 雙模型融合)**| **0.8232** |
| **Team 5 (策略 B 終極融合)** | **TBD (等待配額重置)** |
| Baseline 2 | 0.8623 |
| Baseline 1 | 0.9117 |

---

# 系統工程進度

程式碼庫 (Codebase) 已經為未來的迭代做好了準備：

- 核心腳本已模組化：`src/train.py`, `src/predict.py`, `src/features.py`。
- 每次訓練都會在 `experiments/<run_id>/` 下自動儲存 `config.json`、`metrics.json` 與提交檔 metadata。
- 龐大的模型檔與預測檔不會上傳 git，但輕量化的 metadata 會保留作為實驗紀錄。

這確保了我們在測試新演算法時，不會覆蓋掉原本穩定運作的 Baseline 模型。

---

# 關鍵發現與挑戰 (Discoveries & Challenges)

- **成功解決 Data Leakage**：起初採用「依地區隨機切分」，導致本地 MAE 虛低 (0.21) 但 Leaderboard 只有 0.80。改用「時間切分」後成功讓本地與線上 MAE 對齊。
- **訓練與測試的不一致 (Train-Test Discrepancy)**：嘗試用模型預測缺失的歷史分數，卻引入了高達 $\approx 0.74$ MAE 的雜訊，嚴重污染了特徵。
- 將依賴標籤的特徵 (score lags) 徹底拔除後（純天氣預測），我們獲得了更乾淨、記憶體用量更少且分數更準的架構。

---

# 下一步 (Next Steps)

在最終死線 (Deadline) 之前，我們計畫：

- 正式對 XGBoost 進行超參數調校 (Hyperparameter tuning)，縮小與 LGBM 的差距。
- 嘗試加入更長週期的滾動特徵 (Rolling windows) 與特徵篩選 (Feature selection)。
- 探索如 Temporal Fusion Transformer (TFT) 等深度學習序列模型。
- 將目前所有的實驗成果與方法論寫進 IEEE LaTeX 期末報告中。

---

# 總結 (Summary)

目前已完成：

- 解決了致命的資料洩漏 (Leakage) 與特徵不一致問題。
- 轉換為結合領域知識特徵的純直接預測架構 (Pure Direct Forecasting)。
- 實作並驗證了 XGBoost + LightGBM 的強大融合模型。
- Kaggle Public MAE: **0.8232** (乾淨、具重現性、無洩漏的版本)。

接下來的重點：超參數調校與模型精煉。

## 感謝聆聽 (∠·ω )⌒★
