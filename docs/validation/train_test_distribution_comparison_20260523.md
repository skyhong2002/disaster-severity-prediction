# Train/Test Distribution Comparison

Generated: 2026-05-23T10:55:52

## 結論摘要

- 不能只用整個 train 跟 test 比，因為 test 是每個 region 訓練尾端之後的 91 天，季節位置不一致會製造很大的溫度 drift。這份報告改用數字化 synthetic date 排序，再加入同 region、同月日的歷史對照。
- 以最嚴格的 `同 region 同月日前一年` 來看，test 的 dry-day 比例變化為 `+0.085`，日降雨均值變化為 `-1.325`，90 天累積雨量變化為 `-53.966`。
- 因此「test 比較不會乾」沒有被主要乾旱指標支持。沒控季節時，test 的 humidity/dew point 會因為較暖而偏高；但控同 region 同月日前一年後，降雨下降、無雨日上升，daily dew-point depression 也增加 `+1.540`。
- 與最近 1095 天相比，test 平均氣溫高 `+5.349`；與 test 前 91 天相比，平均氣溫高 `+11.928`。這代表最大 drift 是季節/溫度 regime，不是單純 precipitation regime。

## 資料與日期口徑

- Train rows: `12,319,040`; test rows: `204,568`.
- Regions: `2,248`; test years span `3020` to `58063` because each region has its own synthetic timeline.
- Test months present across regions: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12`.
- Tail windows and previous-year matching use parsed numeric `(year, month, day)` ordinals, not raw string date sorting.

## Baseline Summary

| baseline | rows | adversarial AUC | avg PSI | avg KS | prec mean | dry-day share | tmp mean | dew-point spread | 90d precip | 90d dry days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 train | 12,319,040 | 0.8688 | 0.1573 | 0.1251 | 2.564 -> 1.925 | 0.471 -> 0.487 | 12.949 -> 18.723 | 6.167 -> 7.772 | 228.440 -> 223.710 | 42.051 -> 43.770 |
| 每區最近 1095 天 | 2,461,560 | 0.8703 | 0.1501 | 0.1286 | 2.696 -> 1.925 | 0.463 -> 0.487 | 13.374 -> 18.723 | 5.998 -> 7.772 | 244.326 -> 223.710 | 41.577 -> 43.770 |
| 每區最近 365 天 | 820,520 | 0.8866 | 0.2104 | 0.1502 | 2.945 -> 1.925 | 0.439 -> 0.487 | 12.922 -> 18.723 | 5.729 -> 7.772 | 247.406 -> 223.710 | 40.693 -> 43.770 |
| test 前 91 天 | 204,568 | 0.9832 | 0.5789 | 0.2653 | 3.168 -> 1.925 | 0.477 -> 0.487 | 6.795 -> 18.723 | 3.921 -> 7.772 | 299.707 -> 223.710 | 36.813 -> 43.770 |
| 同 region 同月日所有歷史年 | 3,066,927 | 0.8898 | 0.1434 | 0.1148 | 2.656 -> 1.925 | 0.464 -> 0.487 | 18.078 -> 18.723 | 7.092 -> 7.772 | 253.889 -> 223.710 | 37.842 -> 43.770 |
| 同 region 同月日近 5 年 | 901,000 | 0.8961 | 0.1800 | 0.1334 | 2.818 -> 1.925 | 0.449 -> 0.487 | 18.506 -> 18.723 | 6.685 -> 7.772 | 267.856 -> 223.710 | 36.609 -> 43.770 |
| 同 region 同月日前一年 | 204,568 | 0.9118 | 0.2282 | 0.1599 | 3.250 -> 1.925 | 0.403 -> 0.487 | 18.256 -> 18.723 | 6.231 -> 7.772 | 277.676 -> 223.710 | 35.592 -> 43.770 |

AUC 越接近 `0.5` 表示 train baseline 與 test 越難分開；PSI/KS 越低表示單變量分佈越接近。

## Dryness-Focused Checks

| feature | 全部 train -> test | 最近 1095 天 -> test | 同月日前一年 -> test |
|---|---:|---:|---:|
| daily precipitation | 2.564 -> 1.925 | 2.696 -> 1.925 | 3.250 -> 1.925 |
| dry-day share, prec < 0.1 | 0.471 -> 0.487 | 0.463 -> 0.487 | 0.403 -> 0.487 |
| rain-day share, prec >= 1 | 0.324 -> 0.293 | 0.335 -> 0.293 | 0.386 -> 0.293 |
| heavy-rain share, prec >= 10 | 0.077 -> 0.053 | 0.081 -> 0.053 | 0.096 -> 0.053 |
| 90-day precipitation sum | 228.440 -> 223.710 | 244.326 -> 223.710 | 277.676 -> 223.710 |
| 90-day dry days | 42.051 -> 43.770 | 41.577 -> 43.770 | 35.592 -> 43.770 |
| 90-day dry-day share | 0.472 -> 0.486 | 0.462 -> 0.486 | 0.395 -> 0.486 |
| max dry streak in prior 90d | 10.121 -> 10.910 | 10.345 -> 10.910 | 8.933 -> 10.910 |
| max temperature | 18.882 -> 25.148 | 19.223 -> 25.148 | 24.072 -> 25.148 |
| daily temperature range | 11.345 -> 12.264 | 11.167 -> 12.264 | 11.127 -> 12.264 |
| tmp - dew point | 6.167 -> 7.772 | 5.998 -> 7.772 | 6.231 -> 7.772 |
| tmp - wet bulb | 6.132 -> 7.890 | 5.980 -> 7.890 | 6.328 -> 7.890 |
| 90-day temp / precipitation proxy | 0.154 -> 0.100 | 0.166 -> 0.100 | 0.182 -> 0.100 |

## Most Shifted Raw Weather Features

### 全部 train vs test

| feature | train mean | test mean | delta | PSI | KS |
|---|---:|---:|---:|---:|---:|
| max temperature | 18.882 | 25.148 | 6.266 | 0.374 | 0.236 |
| mean temperature | 12.949 | 18.723 | 5.774 | 0.357 | 0.232 |
| surface temperature | 13.068 | 18.967 | 5.899 | 0.353 | 0.231 |
| min temperature | 7.537 | 12.884 | 5.347 | 0.342 | 0.231 |
| humidity | 7.800 | 9.553 | 1.753 | 0.227 | 0.185 |
| dew point temperature | 6.782 | 10.952 | 4.170 | 0.211 | 0.183 |
| wet bulb temperature | 6.817 | 10.833 | 4.016 | 0.204 | 0.178 |
| daily temperature range | 11.345 | 12.264 | 0.919 | 0.064 | 0.109 |
| wind | 3.647 | 3.376 | -0.270 | 0.025 | 0.056 |
| daily precipitation | 2.564 | 1.925 | -0.638 | 0.011 | 0.034 |

### test 前 91 天 vs test

| feature | train mean | test mean | delta | PSI | KS |
|---|---:|---:|---:|---:|---:|
| max temperature | 12.047 | 25.148 | 13.101 | 1.717 | 0.553 |
| surface temperature | 6.590 | 18.967 | 12.377 | 1.672 | 0.547 |
| mean temperature | 6.795 | 18.723 | 11.928 | 1.642 | 0.543 |
| min temperature | 2.405 | 12.884 | 10.479 | 1.409 | 0.511 |
| humidity | 5.749 | 9.553 | 3.804 | 0.898 | 0.421 |
| wet bulb temperature | 2.867 | 10.833 | 7.967 | 0.824 | 0.404 |
| dew point temperature | 2.874 | 10.952 | 8.077 | 0.823 | 0.405 |
| daily temperature range | 9.642 | 12.264 | 2.622 | 0.464 | 0.286 |
| wind | 3.630 | 3.376 | -0.254 | 0.042 | 0.076 |
| daily precipitation | 3.168 | 1.925 | -1.243 | 0.034 | 0.050 |

### 同 region 同月日前一年 vs test

| feature | train mean | test mean | delta | PSI | KS |
|---|---:|---:|---:|---:|---:|
| daily temperature range | 11.127 | 12.264 | 1.138 | 0.097 | 0.139 |
| daily precipitation | 3.250 | 1.925 | -1.325 | 0.054 | 0.095 |
| humidity | 10.282 | 9.553 | -0.730 | 0.054 | 0.099 |
| wet bulb temperature | 11.928 | 10.833 | -1.095 | 0.046 | 0.092 |
| dew point temperature | 12.025 | 10.952 | -1.073 | 0.044 | 0.090 |
| max temperature | 24.072 | 25.148 | 1.076 | 0.037 | 0.065 |
| surface temperature | 18.472 | 18.967 | 0.495 | 0.030 | 0.044 |
| mean temperature | 18.256 | 18.723 | 0.467 | 0.028 | 0.040 |
| min temperature | 12.945 | 12.884 | -0.061 | 0.013 | 0.034 |
| wind | 3.288 | 3.376 | 0.088 | 0.005 | 0.022 |

## Interpretation

- 沒控季節的比較中，`humidity` 和 `dp_tmp/wb_tmp` 在 test 較高，這是「看起來比較濕」的主要來源；但這同時伴隨 test 明顯更熱。
- 控同 region 同月日前一年後，test 的 `humidity`、`dp_tmp`、`wb_tmp` 反而較低，而 `tmp_range`、`tmp - dp_tmp`、`tmp - wb_tmp` 較高，表示大氣蒸散需求較強。
- 降雨訊號沒有支持「test 明顯比較不乾」：test 的日均降雨、90 天累積雨量、heavy-rain share 大多低於全 train、最近 1095 天、以及同月日前一年。
- Rolling 90 天特徵要另外讀：test 的 rolling context 包含 test 前 91 天與 test window 前段，因此它回答的是模型推論時看見的近期背景，不等於 test 當天的即時天氣。
- 對模型策略的含意：比起把 test 當成全面濕潤 regime，較合理的是處理溫度/季節 drift，並讓 rolling precipitation、dry-day streak、dew-point depression 這類 proxy 在驗證中被單獨檢查。
