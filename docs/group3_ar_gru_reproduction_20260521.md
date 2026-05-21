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
