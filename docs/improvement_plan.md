# 模型優化計畫 (Improvement Plan)

本計畫將針對目前災情嚴重度預測 (Disaster Severity Prediction) 專案進行 3 個階段的優化，以解決本地驗證與測試集表現不一致的根本原因，並提升模型準確率。

## 階段 1：解決訓練與預測的特徵不一致 (Train-Test Discrepancy) [優先實作]

**問題描述**：
在目前的 Two-Stage 架構中，`predict.py` 預測時無法獲得真實的歷史分數，所以使用 Stage 1 (Score Reconstructor) 預測出的「重建分數」來計算歷史特徵 (如 `score_lag`、`score_rmean`)。然而，在 `train.py` 訓練 Stage 2 時，餵給模型的卻是基於「真實歷史分數」計算的特徵。這導致模型在訓練階段過度依賴無誤差的真實歷史分數，但在推論階段遇到帶有誤差的重建分數時表現嚴重崩潰。

**實作方案**：
修改 `src/train.py` 的特徵生成流程：
1. 先僅計算天氣、日曆與地區統計特徵 (Basic Features)。
2. 利用 Basic Features 訓練 Stage 1 (Score Reconstructor)。
3. 使用訓練好的 Stage 1 模型，針對整個訓練集預測出 `score_reconstructed`。
4. **將訓練集中的歷史標籤替換為 `score_reconstructed`**。
5. 基於被替換過的分數，計算 `score_lag` 與 `score_rmean` 等歷史特徵。
6. 使用這些「帶有真實推論誤差」的特徵來訓練 Stage 2 (Horizon Models)。
如此可保證 Train/Test 的特徵分佈完全一致。

## 階段 2：加入乾旱領域知識的特徵 (Domain Features) [已完成]

**問題描述**：
單純仰賴演算法學習 327 個原始特徵的關聯效率較低，樹模型不易自動捕捉變數之間的除法等非線性交互作用。乾旱通常由「高溫」加「少雨」造成。

**實作方案 (已套用於 `src/features.py`)**：
* 新增 `drought_idx`：`tmp_rmean` / (`prec_rmean` + 0.01) (溫度與降雨量比例)。
* 新增 `dryness_idx`：`tmp_range_rmean` * `tmp_max_rmean` (日夜溫差與最高溫乘積)。

## 階段 3：引入 XGBoost 模型與 Ensemble

**問題描述**：
單一 LightGBM 模型容易在邊界資料上過擬合。引入不同機制的樹模型並進行 Ensemble (平均預測值) 能顯著提升穩健性。

**實作方案**：
1. 建立 `src/train_xgb.py`，將 LightGBM 替換為 XGBoost。
2. 寫一個 `src/ensemble.py` 腳本，讀取 LightGBM 與 XGBoost 各自的 submission，進行 0.5/0.5 的加權平均，產生最終上傳至 Kaggle 的 CSV。
