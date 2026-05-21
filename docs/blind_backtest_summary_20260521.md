# Blind Backtest Summary - 2026-05-21

## Protocol

- Branch: `refactor/validation-discipline`
- Draft PR: https://github.com/skyhong2002/disaster-severity-prediction/pull/2
- Leakage sentinel: pass for origins `5,13,26,39,52,78,104`, `history_tail_days=1100`, `feature_profile=lean`, `max_regions=256`.
- Drift grid: `tail_days=1095` is closest to test by adversarial AUC and PSI/KS; full-history is most separable.
- Blind origins: `5,13,26,39,52,78,104`
- LGBM/XGB blind context: `history_tail_days=1100`, matching `train_tail_days=1095`.
- CatBoost blind context: `history_tail_days=2737`, matching `train_tail_days=2737` for climatology/anomaly consistency.
- No Kaggle submission was made from these runs.

## Drift Summary

| tail days | rows | weather AUC | weather+region/month AUC | avg PSI | avg KS | read |
|---:|---:|---:|---:|---:|---:|---|
| 1095 | 2,461,560 | 0.7432 | 0.7681 | 0.1389 | 0.1244 | Closest overall candidate. |
| 1825 | 4,102,600 | 0.7719 | 0.7903 | 0.1506 | 0.1270 | More data, more drift. |
| 2737 | 6,152,776 | 0.7863 | 0.7956 | 0.1437 | 0.1242 | Useful CatBoost/public anchor, not closest by AUC. |
| 3650 | 8,205,200 | 0.7998 | 0.8118 | 0.1586 | 0.1296 | Drift increases. |
| full | 12,319,040 | 0.8083 | 0.8186 | 0.1613 | 0.1330 | Most separable; all-history lean LGBM was also killed with exit 137. |

Most visible drift features are temperature-related (`tmp`, `tmp_max`, `surf_tmp`, `dp_tmp`) plus humidity; this supports treating tail length and recency as first-class hyperparameters.

## Single Model Results

| model | train mode | feature/tail | local rolling MAE | blind MAE | w1 | w2 | w3 | w4 | w5 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LGBM | refit_full | lean, tail1095 | 0.2474 | 0.3549 | 0.2996 | 0.3067 | 0.3143 | 0.3842 | 0.4697 | Current pseudo-private anchor. |
| XGBoost | refit_full | lean, tail1095 | 0.2383 | 0.4420 | 0.4175 | 0.4201 | 0.4285 | 0.4434 | 0.5003 | Reject as dominant model; local metric overstates it. |
| CatBoost | refit_full | lean, tail2737, half-life1095 | 0.2192 | 0.4482 | 0.4184 | 0.4393 | 0.4506 | 0.4560 | 0.4769 | Reject as dominant model; possible late-horizon diversity only. |

Key read: rolling-origin MAE ranked CatBoost best and LGBM worst, while blind backtest ranked LGBM clearly best. This is the strongest evidence so far that local rolling MAE alone is not a safe promotion metric.

## Blend Results

| blend | weights | blind MAE | read |
|---|---|---:|---|
| Public anchor | LGB/XGB/Cat = 35/35/30 | 0.4038 | Not supported by blind backtest. |
| Regularized fit | w1 `88/6/6`, w2 `92/8/0`, w3 `92/8/0`, w4 `72/16/12`, w5 `44/14/42` | 0.3614 | Better than public anchor, still worse than LGBM alone. |
| Unregularized fit | w1-w4 nearly all LGBM, w5 `56/0/44` | 0.3536 | Slightly beats LGBM, but week-5 Cat/LGB bootstrap std is high at about 0.31. |

Bootstrap stability favors LGBM strongly for weeks 1-4. CatBoost only earns meaningful weight on week 5, and that weight is unstable across origin resamples.

## Promotion Decision

- Keep: LGBM `lean + tail1095 + refit_full` as the current pseudo-private anchor.
- Keep as diagnostic: unregularized horizon blend, especially week-5 LGBM/Cat split.
- Reject for submission now: XGBoost tail1095 as a dominant model, CatBoost tail2737 as a dominant model, and the public 35/35/30 anchor.
- Submit candidate: none yet.
- Next evidence to collect: LGBM tail grid (`1095/1825/2737`) under blind backtest, plus first feature ablations for `score_history`, `climatology`, `domain_indices`, and `long_drought_proxy`.

