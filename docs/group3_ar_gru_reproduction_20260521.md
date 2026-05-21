# Group 3 AR-GRU Reproduction - 2026-05-21

## Source Signals

This reproduction is based on the Group 3 progress presentation and classroom
discussion:

- Raw daily regional time series: `region_id`, `date`, `score`, and weather
  variables.
- Feature processing: data cleaning, grouping/windowing, and feature extraction.
- Region features: region encoding.
- Time features: year, month, and quarter.
- Model: 91-day weather sequence, static region/date context, 2-layer GRU
  encoder with hidden size 64, and autoregressive 5-horizon GRUCell decoder
  with previous-score feedback and teacher forcing during training.
- Reported observation: local validation for AR-GRU was strong, but validation
  and public leaderboard were not consistent.

## Implemented Scope

The reproduction lives in:

```bash
src/train_group3_ar_gru.py
```

It intentionally stays separate from the current LightGBM/XGBoost/CatBoost
submission path. The goal is to make Group 3's neural architecture testable
inside our validation discipline, not to replace the current tree ensemble.

Architecture mapping:

| Presentation item | Repo implementation |
|---|---|
| `91 x weather features` | `--seq-len 91` over the 14 meteorological columns |
| `region emb + date` | native region embedding plus year/month/quarter/week date features |
| `GRU Encoder, 2 layers, hidden=64` | default `--num-layers 2 --hidden-size 64` |
| `Initial State` | concat encoder output and static context through a linear layer |
| `AR Decoder` | `GRUCell x 5 horizons` |
| previous score feedback | decoder input starts from a 91-day-gapped visible score |
| teacher forcing | configurable with `--teacher-forcing-ratio` |

## Smoke Result

Command:

```bash
uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_smoke_20260521 \
  --max-rows 60000 \
  --max-regions 8 \
  --train-tail-days 730 \
  --epochs 1 \
  --batch-size 64 \
  --device cpu
```

Output run:

```text
experiments/20260521_185802_group3_ar_gru_group3_ar_gru_smoke_20260521
```

Observed smoke metrics:

| Setting | Value |
|---|---:|
| regions | 8 |
| sequence samples | 688 |
| train samples | 584 |
| validation samples | 64 |
| epochs | 1 |
| validation MAE | 3.5205 |

This MAE is not comparable to Group 3's reported local `0.7086` because the
smoke run uses only a tiny subset and one epoch. It only verifies that data
windowing, score feedback, teacher forcing, validation, and model persistence
work end to end.

## Suggested Formal Run

Use this only after current tree-model benchmark work is not occupying the
machine:

```bash
uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_tail2737_20ep_20260521 \
  --train-tail-days 2737 \
  --epochs 20 \
  --batch-size 256 \
  --teacher-forcing-ratio 0.5 \
  --device auto
```

For a lighter 16GB-RAM run:

```bash
uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_tail1825_10ep_20260521 \
  --train-tail-days 1825 \
  --epochs 10 \
  --batch-size 192 \
  --teacher-forcing-ratio 0.5 \
  --device auto
```

## Decision Rule

Do not submit this model to Kaggle directly after the first full run. Promote it
only if it shows one of these:

- Blind/backtest MAE competitive with current tree anchors.
- Stable late-horizon improvement without collapsing prediction distribution.
- Residual diversity versus LGB/XGB/CatBoost that justifies a low-weight blend.

The main risk is the same risk Group 3 reported: local neural-network validation
can look better than public leaderboard behavior.

## Formal Tail-1825 Run

Command:

```bash
PYTHONUNBUFFERED=1 uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_tail1825_10ep_20260521 \
  --train-tail-days 1825 \
  --epochs 10 \
  --batch-size 192 \
  --teacher-forcing-ratio 0.5 \
  --device auto
```

Output run:

```text
experiments/20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521
```

Run scale:

| Setting | Value |
|---|---:|
| device | MPS |
| regions | 2248 |
| raw rows after tailing | 4102600 |
| sequence samples | 544016 |
| train samples | 514792 |
| validation samples | 17984 |
| train tail days | 1825 |
| epochs | 10 |
| batch size | 192 |

Epoch curve:

| epoch | train MAE | validation MAE | pred mean | pred std |
|---:|---:|---:|---:|---:|
| 1 | 0.2623 | 0.2002 | 0.3925 | 0.8929 |
| 2 | 0.2104 | 0.2019 | 0.3402 | 0.7688 |
| 3 | 0.1977 | 0.1923 | 0.3484 | 0.8337 |
| 4 | 0.1895 | 0.1946 | 0.3653 | 0.8152 |
| 5 | 0.1849 | 0.1852 | 0.3090 | 0.7805 |
| 6 | 0.1796 | 0.2034 | 0.3128 | 0.7519 |
| 7 | 0.1747 | 0.2123 | 0.3741 | 0.8215 |
| 8 | 0.1728 | 0.2197 | 0.3718 | 0.8184 |
| 9 | 0.1686 | 0.2283 | 0.3655 | 0.7758 |
| 10 | 0.1680 | 0.2085 | 0.3513 | 0.8095 |

Best checkpoint:

| Metric | Value |
|---|---:|
| best epoch | 5 |
| best validation MAE | 0.1852 |
| week 1 MAE | 0.1933 |
| week 2 MAE | 0.1849 |
| week 3 MAE | 0.1825 |
| week 4 MAE | 0.1810 |
| week 5 MAE | 0.1845 |
| target mean | 0.3651 |
| target std | 0.8941 |
| prediction mean | 0.3090 |
| prediction std | 0.7805 |

Interpretation:

- The model is real and learns useful signal: the best validation MAE reaches
  `0.1852`, better than the one-epoch smoke and comparable to the scale of our
  rolling-origin tree diagnostics.
- It overfits quickly after epoch 5: train MAE keeps improving from `0.1849` to
  `0.1680`, while validation MAE worsens from `0.1852` to `0.2085`.
- The validation target distribution is very low severity (`target_mean=0.3651`),
  so this number is not comparable to Kaggle public MAE and must not be used as
  a direct submit signal.
- The next useful step is not Kaggle submission. The next step is to adapt
  blind-window backtesting or a submission writer for this neural checkpoint,
  then compare residual diversity against LGB/XGB/CatBoost.

## Kaggle Test-Window Inference

Inference script:

```bash
src/predict_group3_ar_gru.py
```

Command:

```bash
PYTHONUNBUFFERED=1 uv run python src/predict_group3_ar_gru.py \
  --run-dir experiments/20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521 \
  --device auto \
  --batch-size 512
```

Submission artifact:

```text
submissions/submission_20260521_193438_20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521_group3_ar_gru.csv
```

Sanity checks:

| Check | Value |
|---|---:|
| rows | 2248 |
| columns | 6 |
| NaN values | 0 |
| prediction min | 0.0000 |
| prediction max | 5.0000 |
| sha256 | `5ca7f847eba15e189aec400c14c1e00ae3e3427b2c4b84a28d4469032e065080` |

Prediction distribution:

| Horizon | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| week 1 | 0.6915 | 0.9282 | 0.0000 | 5.0000 |
| week 2 | 0.6914 | 0.9321 | 0.0000 | 5.0000 |
| week 3 | 0.6912 | 0.9358 | 0.0000 | 5.0000 |
| week 4 | 0.6921 | 0.9397 | 0.0000 | 5.0000 |
| week 5 | 0.6934 | 0.9442 | 0.0000 | 5.0000 |
| overall | 0.6919 | 0.9360 | 0.0000 | 5.0000 |

Interpretation:

- The submission writer is now reproducible and uses the same Kaggle-like
  blind setup as the official test data: 91 days of test weather are visible,
  but no test scores are used.
- The test-window prediction mean (`0.6919`) is much higher than the best local
  validation prediction mean (`0.3090`). This is plausible under distribution
  shift, but it also confirms that the local validation score is not enough to
  justify direct submission.
- Some rows saturate at both boundaries after clipping (`0` and `5`), so this
  model should be treated as a neural diversity candidate, not as a primary
  replacement for LGB/XGB/CatBoost.
- Promotion gate remains unchanged: run blind-window backtest or blend-residual
  analysis before spending a Kaggle slot on this artifact.

## Date Parsing Correction

While wiring the AR-GRU into blind backtesting, we found a hidden date parsing
bug. Competition dates use synthetic years with variable width, for example
`3004-12-31`, `23101-11-20`, and `58061-12-31`. The original parser assumed
fixed `YYYY-MM-DD` positions, so `23101-11-20` produced an invalid month from
the substring `-1`.

Fix:

- `features.parse_synthetic_date_parts()` now parses `year`, `month`, and `day`
  from dash-separated fields.
- `features.add_calendar_features()` uses that parser for global calendar
  features.
- `train_group3_ar_gru.add_date_parts()` uses the same parser for AR-GRU date
  context.
- `validation.evaluate_submission_like_predictions()` uses the parser for
  `mae_by_calendar_month`.

The pre-fix AR-GRU checkpoint is kept as a diagnostic artifact, but it should
not be promoted because its date context was malformed for variable-width
synthetic years.

## Corrected Formal Tail-1825 Run

Command:

```bash
PYTHONUNBUFFERED=1 uv run python src/train_group3_ar_gru.py \
  --experiment-name group3_ar_gru_tail1825_10ep_fixeddate_20260521 \
  --train-tail-days 1825 \
  --epochs 10 \
  --batch-size 192 \
  --teacher-forcing-ratio 0.5 \
  --device auto
```

Output run:

```text
experiments/20260521_194642_group3_ar_gru_group3_ar_gru_tail1825_10ep_fixeddate_20260521
```

Epoch curve:

| epoch | train MAE | validation MAE | pred mean | pred std |
|---:|---:|---:|---:|---:|
| 1 | 0.2634 | 0.1875 | 0.3425 | 0.8429 |
| 2 | 0.2116 | 0.2040 | 0.3512 | 0.7803 |
| 3 | 0.1985 | 0.1992 | 0.3661 | 0.8244 |
| 4 | 0.1906 | 0.1957 | 0.3499 | 0.8171 |
| 5 | 0.1854 | 0.2039 | 0.3468 | 0.7933 |
| 6 | 0.1792 | 0.2129 | 0.3140 | 0.7262 |
| 7 | 0.1757 | 0.2181 | 0.3365 | 0.7820 |
| 8 | 0.1731 | 0.2155 | 0.3740 | 0.8217 |
| 9 | 0.1693 | 0.2291 | 0.3649 | 0.7975 |
| 10 | 0.1678 | 0.2458 | 0.3825 | 0.7867 |

Best checkpoint:

| Metric | Value |
|---|---:|
| best epoch | 1 |
| best validation MAE | 0.1875 |
| target mean | 0.3651 |
| target std | 0.8941 |
| prediction mean | 0.3425 |
| prediction std | 0.8429 |

Interpretation:

- Correct date parsing did not improve the best local validation MAE versus the
  pre-fix run (`0.1875` vs `0.1852`), but it makes the experiment valid.
- The model overfits immediately after epoch 1: train MAE improves from
  `0.2634` to `0.1678`, while validation MAE worsens from `0.1875` to `0.2458`.
- This reinforces Group 3's observation that neural validation can be unstable
  and should not be promoted without blind-window evidence.

## Corrected Kaggle Test-Window Inference

Command:

```bash
PYTHONUNBUFFERED=1 uv run python src/predict_group3_ar_gru.py \
  --run-dir experiments/20260521_194642_group3_ar_gru_group3_ar_gru_tail1825_10ep_fixeddate_20260521 \
  --device auto \
  --batch-size 512
```

Submission artifact:

```text
submissions/submission_20260521_200832_20260521_194642_group3_ar_gru_group3_ar_gru_tail1825_10ep_fixeddate_20260521_group3_ar_gru.csv
```

Sanity checks:

| Check | Value |
|---|---:|
| rows | 2248 |
| columns | 6 |
| NaN values | 0 |
| prediction min | 0.0000 |
| prediction max | 5.0000 |
| sha256 | `0fbebc421de1644e2aa8211630253fbc833e014f821a1076fb5cf95bd0deae2e` |

Prediction distribution:

| Horizon | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| week 1 | 0.7088 | 0.9126 | 0.0000 | 5.0000 |
| week 2 | 0.7012 | 0.9058 | 0.0000 | 5.0000 |
| week 3 | 0.6944 | 0.8985 | 0.0000 | 5.0000 |
| week 4 | 0.6896 | 0.8917 | 0.0000 | 4.9804 |
| week 5 | 0.6856 | 0.8855 | 0.0000 | 4.9463 |
| overall | 0.6959 | 0.8989 | 0.0000 | 5.0000 |

## Corrected Blind Backtest

Command:

```bash
PYTHONUNBUFFERED=1 uv run python scripts/run_group3_ar_gru_blind_backtest.py \
  --run-dir experiments/20260521_194642_group3_ar_gru_group3_ar_gru_tail1825_10ep_fixeddate_20260521 \
  --origins "5,13,26,39,52,78,104" \
  --history-tail-days 1100 \
  --batch-size 512 \
  --device auto \
  --out-dir experiments/blind_20260521_group3_ar_gru_tail1825_fixeddate_h1100
```

Blind metrics:

| Model | Blind MAE | w1 | w2 | w3 | w4 | w5 |
|---|---:|---:|---:|---:|---:|---:|
| LGBM lean tail1095 refit_full | 0.3549 | 0.2996 | 0.3067 | 0.3143 | 0.3842 | 0.4697 |
| Group 3 AR-GRU fixed date | 0.3806 | 0.3485 | 0.3633 | 0.3749 | 0.3976 | 0.4185 |
| XGBoost lean tail1095 refit_full | 0.4420 | 0.4175 | 0.4201 | 0.4285 | 0.4434 | 0.5003 |
| CatBoost lean tail2737 refit_full | 0.4482 | 0.4184 | 0.4393 | 0.4506 | 0.4560 | 0.4769 |

Read:

- Corrected AR-GRU is not the best pseudo-private model; LGBM remains the
  strongest single-model anchor on this benchmark.
- AR-GRU is better than XGBoost and CatBoost in this specific blind matrix, and
  it has a different horizon shape: worse than LGBM on weeks 1-4, but better on
  week 5 (`0.4185` vs LGBM `0.4697`).
- The model should not be submitted as a standalone replacement. Its best use is
  a low-weight late-horizon diversity candidate, only after constrained blend
  testing confirms residual value.
