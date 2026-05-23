# Drift Report

Generated: 2026-05-21T15:27:04

Lower PSI/KS/Wasserstein and lower adversarial AUC mean the train-tail candidate looks more like the Kaggle test window. AUC near 0.5 means the classifier struggles to separate train from test.

## Tail Candidate Summary

| tail_days | rows | auc_weather | auc_region_month | avg_psi | avg_ks | avg_q_wasserstein |
|---:|---:|---:|---:|---:|---:|---:|
| 1095 | 2461560 | 0.7432 | 0.7681 | 0.1389 | 0.1244 | 2.4219 |
| 1825 | 4102600 | 0.7719 | 0.7903 | 0.1506 | 0.1270 | 2.5744 |
| 2737 | 6152776 | 0.7863 | 0.7956 | 0.1437 | 0.1242 | 2.5062 |
| 3650 | 8205200 | 0.7998 | 0.8118 | 0.1586 | 0.1296 | 2.6196 |
| full | 12319040 | 0.8083 | 0.8186 | 0.1613 | 0.1330 | 2.6527 |

## Most Drifted Features

### tail_days=1095

| feature | psi | ks | mean_delta |
|---|---:|---:|---:|
| tmp_max | 0.3313 | 0.2191 | 5.9256 |
| surf_tmp | 0.3079 | 0.2197 | 5.4890 |
| tmp | 0.3057 | 0.2180 | 5.3490 |
| tmp_min | 0.2899 | 0.2139 | 4.8283 |
| humidity | 0.1905 | 0.1638 | 1.4417 |
| dp_tmp | 0.1758 | 0.1644 | 3.5750 |
| wb_tmp | 0.1736 | 0.1607 | 3.4393 |
| tmp_range | 0.0855 | 0.1245 | 1.0972 |

### tail_days=1825

| feature | psi | ks | mean_delta |
|---|---:|---:|---:|
| tmp_max | 0.3631 | 0.2261 | 6.1797 |
| surf_tmp | 0.3389 | 0.2256 | 5.7557 |
| tmp | 0.3379 | 0.2244 | 5.6144 |
| tmp_min | 0.3186 | 0.2218 | 5.1227 |
| humidity | 0.2058 | 0.1708 | 1.5372 |
| dp_tmp | 0.1915 | 0.1715 | 3.8110 |
| wb_tmp | 0.1890 | 0.1671 | 3.6643 |
| tmp_range | 0.0797 | 0.1201 | 1.0570 |

### tail_days=2737

| feature | psi | ks | mean_delta |
|---|---:|---:|---:|
| tmp_max | 0.3371 | 0.2217 | 5.9101 |
| surf_tmp | 0.3237 | 0.2206 | 5.5436 |
| tmp | 0.3223 | 0.2199 | 5.4171 |
| tmp_min | 0.3065 | 0.2184 | 4.9672 |
| humidity | 0.2047 | 0.1737 | 1.5962 |
| dp_tmp | 0.1921 | 0.1742 | 3.8707 |
| wb_tmp | 0.1871 | 0.1698 | 3.7200 |
| tmp_range | 0.0660 | 0.1096 | 0.9429 |

### tail_days=3650

| feature | psi | ks | mean_delta |
|---|---:|---:|---:|
| tmp_max | 0.3662 | 0.2311 | 6.2561 |
| surf_tmp | 0.3559 | 0.2290 | 5.8741 |
| tmp | 0.3546 | 0.2285 | 5.7480 |
| tmp_min | 0.3411 | 0.2273 | 5.2970 |
| humidity | 0.2275 | 0.1800 | 1.7004 |
| dp_tmp | 0.2123 | 0.1819 | 4.1083 |
| wb_tmp | 0.2070 | 0.1767 | 3.9564 |
| tmp_range | 0.0686 | 0.1114 | 0.9591 |

### tail_days=full

| feature | psi | ks | mean_delta |
|---|---:|---:|---:|
| tmp_max | 0.3747 | 0.2380 | 6.2660 |
| surf_tmp | 0.3618 | 0.2361 | 5.8990 |
| tmp | 0.3612 | 0.2358 | 5.7740 |
| tmp_min | 0.3467 | 0.2348 | 5.3466 |
| humidity | 0.2300 | 0.1864 | 1.7529 |
| dp_tmp | 0.2167 | 0.1866 | 4.1696 |
| wb_tmp | 0.2103 | 0.1816 | 4.0162 |
| tmp_range | 0.0663 | 0.1097 | 0.9195 |
