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
