"""
Reproduce Group 3's autoregressive GRU experiment.

This is an intentionally separate neural baseline, not part of the production
LightGBM/XGBoost/CatBoost inference path. It follows the public presentation:

- 91 x weather-feature input sequence
- static context from region embedding plus date features
- 2-layer GRU encoder with hidden size 64 by default
- autoregressive GRUCell decoder for five weekly horizons
- previous score feedback with teacher forcing during training

Example smoke run:
    uv run python src/train_group3_ar_gru.py \
      --experiment-name group3_ar_gru_smoke \
      --max-rows 60000 \
      --max-regions 8 \
      --train-tail-days 730 \
      --epochs 1 \
      --batch-size 64
"""
from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment_utils import create_run_dir, save_json
from features import METEO_COLS, parse_synthetic_date_parts

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

N_WEEKS = 5
DEFAULT_SEQ_LEN = 91
DEFAULT_SCORE_GAP_DAYS = 91
SCORE_SCALE = 5.0


@dataclass(frozen=True)
class PreparedPanel:
    weather_by_region: dict[int, np.ndarray]
    samples: pd.DataFrame
    region_to_code: dict[str, int]
    code_to_region: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Group-3-style autoregressive GRU baseline."
    )
    parser.add_argument("--experiment-name", default="group3_ar_gru")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--score-gap-days", type=int, default=DEFAULT_SCORE_GAP_DAYS)
    parser.add_argument("--train-tail-days", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="Read only the first N raw rows for smoke tests.")
    parser.add_argument("--max-regions", type=int, default=0, help="Keep only the first N regions for smoke tests.")
    parser.add_argument("--val-weeks", type=int, default=8, help="Number of final weekly origins per region used for validation.")
    parser.add_argument("--purge-weeks", type=int, default=N_WEEKS, help="Gap between train origins and validation origins.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-emb-dim", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    return parser.parse_args()


def natural_region_key(region_id: str) -> tuple[int, str]:
    suffix = str(region_id)[1:]
    if str(region_id).startswith("R") and suffix.isdigit():
        return int(suffix), str(region_id)
    return 10**9, str(region_id)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda")
    if raw == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_raw_train(max_rows: int) -> pd.DataFrame:
    usecols = ["region_id", "date", *METEO_COLS, "score"]
    nrows = max_rows if max_rows > 0 else None
    print(f"Loading train.csv{' first ' + str(max_rows) + ' rows' if nrows else ''} ...")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols, nrows=nrows)
    print(f"  raw shape: {train.shape}")
    return train


def clean_and_filter(train: pd.DataFrame, max_regions: int, train_tail_days: int) -> pd.DataFrame:
    """Apply the Group 3 cleaning idea: numeric coercion and imputation."""
    train = train.copy()
    train["region_id"] = train["region_id"].astype(str)
    train["date"] = train["date"].astype(str)
    for col in METEO_COLS:
        train[col] = pd.to_numeric(train[col], errors="coerce").astype(np.float32)
    train["score"] = pd.to_numeric(train["score"], errors="coerce").astype(np.float32)

    if max_regions > 0:
        regions = sorted(train["region_id"].dropna().unique(), key=natural_region_key)[:max_regions]
        train = train[train["region_id"].isin(regions)].copy()
        print(f"  kept first {len(regions)} regions for smoke test")

    train = train.sort_values(["region_id", "date"]).reset_index(drop=True)
    if train_tail_days > 0:
        train = (
            train.groupby("region_id", group_keys=False)
            .tail(train_tail_days)
            .reset_index(drop=True)
        )
        print(f"  kept latest {train_tail_days} daily rows per region")

    medians = train[METEO_COLS].median(numeric_only=True).astype(np.float32)
    train[METEO_COLS] = train.groupby("region_id")[METEO_COLS].transform(
        lambda col: col.ffill().bfill()
    )
    train[METEO_COLS] = train[METEO_COLS].fillna(medians).fillna(0.0).astype(np.float32)
    return train


def add_date_parts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    year, month, day = parse_synthetic_date_parts(frame["date"])
    frame["year"] = year.astype(np.float32)
    frame["month"] = month.astype(np.float32)
    frame["day"] = day.astype(np.float32)
    frame["quarter"] = (((frame["month"] - 1) // 3) + 1).astype(np.float32)
    frame["weekofyear"] = (((frame["month"] - 1) * 30 + frame["day"]) // 7).clip(0, 51).astype(np.float32)
    return frame


def make_date_feature_matrix(samples: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    samples = samples.copy()
    year_mean = float(samples.loc[train_mask, "year"].mean())
    year_std = float(samples.loc[train_mask, "year"].std() or 1.0)
    samples["year_z"] = ((samples["year"] - year_mean) / year_std).astype(np.float32)
    samples["month_sin"] = np.sin(2 * np.pi * samples["month"] / 12).astype(np.float32)
    samples["month_cos"] = np.cos(2 * np.pi * samples["month"] / 12).astype(np.float32)
    samples["quarter_sin"] = np.sin(2 * np.pi * samples["quarter"] / 4).astype(np.float32)
    samples["quarter_cos"] = np.cos(2 * np.pi * samples["quarter"] / 4).astype(np.float32)
    samples["week_sin"] = np.sin(2 * np.pi * samples["weekofyear"] / 52).astype(np.float32)
    samples["week_cos"] = np.cos(2 * np.pi * samples["weekofyear"] / 52).astype(np.float32)
    cols = ["year_z", "month_sin", "month_cos", "quarter_sin", "quarter_cos", "week_sin", "week_cos"]
    return samples, cols, {"year_mean": year_mean, "year_std": year_std}


def prepare_panel(train: pd.DataFrame, seq_len: int, score_gap_days: int) -> PreparedPanel:
    train = add_date_parts(train)
    regions = sorted(train["region_id"].unique(), key=natural_region_key)
    region_to_code = {region: idx for idx, region in enumerate(regions)}
    code_to_region = {idx: region for region, idx in region_to_code.items()}

    weather_by_region: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    global_score_median = train["score"].dropna().median()
    if pd.isna(global_score_median):
        global_score_median = 0.0
    global_score_median = float(global_score_median)

    for region_id, group in train.groupby("region_id", sort=False):
        code = region_to_code[region_id]
        group = group.sort_values("date").reset_index(drop=True)
        weather_by_region[code] = group[METEO_COLS].to_numpy(dtype=np.float32, copy=True)

        scores = group["score"].to_numpy(dtype=np.float32, copy=True)
        last_visible_score = (
            pd.Series(scores)
            .ffill()
            .fillna(global_score_median)
            .to_numpy(dtype=np.float32)
        )
        weekly_positions = np.flatnonzero(~np.isnan(scores))
        if len(weekly_positions) <= N_WEEKS:
            continue

        for weekly_rank in range(0, len(weekly_positions) - N_WEEKS):
            day_idx = int(weekly_positions[weekly_rank])
            if day_idx < seq_len - 1:
                continue
            target_positions = weekly_positions[weekly_rank + 1 : weekly_rank + N_WEEKS + 1]
            targets = scores[target_positions]
            if np.isnan(targets).any():
                continue

            visible_idx = day_idx - score_gap_days
            previous_score = (
                float(last_visible_score[visible_idx])
                if visible_idx >= 0
                else global_score_median
            )
            origin = group.iloc[day_idx]
            records.append(
                {
                    "region_id": region_id,
                    "region_code": code,
                    "day_idx": day_idx,
                    "week_idx": weekly_rank,
                    "date": origin["date"],
                    "year": float(origin["year"]),
                    "month": float(origin["month"]),
                    "quarter": float(origin["quarter"]),
                    "weekofyear": float(origin["weekofyear"]),
                    "previous_score": previous_score,
                    **{f"target_w{w}": float(targets[w - 1]) for w in range(1, N_WEEKS + 1)},
                }
            )

    samples = pd.DataFrame.from_records(records)
    if samples.empty:
        raise ValueError("No usable sequence samples. Increase --max-rows or --train-tail-days.")
    samples = samples.sort_values(["region_code", "week_idx"]).reset_index(drop=True)
    print(f"  sequence samples: {len(samples):,}")
    return PreparedPanel(weather_by_region, samples, region_to_code, code_to_region)


def validation_split(samples: pd.DataFrame, val_weeks: int, purge_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    samples = samples.copy()
    samples["sample_rank"] = samples.groupby("region_code").cumcount()
    samples["max_sample_rank"] = samples.groupby("region_code")["sample_rank"].transform("max")
    val_start = (samples["max_sample_rank"] - max(0, val_weeks - 1)).clip(lower=0)
    val_mask = samples["sample_rank"] >= val_start
    train_mask = samples["sample_rank"] + purge_weeks < val_start
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError(
            "Empty train/validation split. Use more rows, more tail days, or fewer validation weeks."
        )
    return train_mask.to_numpy(dtype=bool), val_mask.to_numpy(dtype=bool)


def weather_normalization(
    weather_by_region: dict[int, np.ndarray],
    samples: pd.DataFrame,
    train_mask: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(METEO_COLS), dtype=np.float64)
    sq_sums = np.zeros(len(METEO_COLS), dtype=np.float64)
    count = 0
    for row in samples.loc[train_mask, ["region_code", "day_idx"]].itertuples(index=False):
        seq = weather_by_region[int(row.region_code)][int(row.day_idx) - seq_len + 1 : int(row.day_idx) + 1]
        sums += seq.sum(axis=0)
        sq_sums += np.square(seq, dtype=np.float64).sum(axis=0)
        count += seq.shape[0]
    mean = sums / max(1, count)
    var = np.maximum(sq_sums / max(1, count) - mean**2, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class Group3SequenceDataset(Dataset):
    def __init__(
        self,
        weather_by_region: dict[int, np.ndarray],
        samples: pd.DataFrame,
        date_cols: list[str],
        weather_mean: np.ndarray,
        weather_std: np.ndarray,
        seq_len: int,
    ) -> None:
        self.weather_by_region = weather_by_region
        self.region_codes = samples["region_code"].to_numpy(dtype=np.int64)
        self.day_idxs = samples["day_idx"].to_numpy(dtype=np.int64)
        self.date_features = samples[date_cols].to_numpy(dtype=np.float32)
        self.previous_scores = (samples["previous_score"].to_numpy(dtype=np.float32) / SCORE_SCALE).reshape(-1, 1)
        target_cols = [f"target_w{w}" for w in range(1, N_WEEKS + 1)]
        self.targets = samples[target_cols].to_numpy(dtype=np.float32) / SCORE_SCALE
        self.weather_mean = weather_mean.reshape(1, -1)
        self.weather_std = np.maximum(weather_std.reshape(1, -1), 1e-6)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.region_codes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        region_code = int(self.region_codes[idx])
        day_idx = int(self.day_idxs[idx])
        seq = self.weather_by_region[region_code][day_idx - self.seq_len + 1 : day_idx + 1]
        seq = (seq - self.weather_mean) / self.weather_std
        return {
            "weather_seq": torch.as_tensor(seq, dtype=torch.float32),
            "region_code": torch.as_tensor(region_code, dtype=torch.long),
            "date_features": torch.as_tensor(self.date_features[idx], dtype=torch.float32),
            "previous_score": torch.as_tensor(self.previous_scores[idx], dtype=torch.float32),
            "targets": torch.as_tensor(self.targets[idx], dtype=torch.float32),
        }


class Group3AutoregressiveGRU(nn.Module):
    def __init__(
        self,
        n_weather: int,
        n_regions: int,
        date_feature_dim: int,
        hidden_size: int,
        num_layers: int,
        region_emb_dim: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=n_weather,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        self.static_mlp = nn.Sequential(
            nn.Linear(region_emb_dim + date_feature_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )
        self.initial_state = nn.Linear(hidden_size + context_dim, hidden_size)
        self.decoder_cell = nn.GRUCell(input_size=context_dim + 1, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size + context_dim, 1)

    def forward(
        self,
        weather_seq: torch.Tensor,
        region_code: torch.Tensor,
        date_features: torch.Tensor,
        previous_score: torch.Tensor,
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        _, hidden = self.encoder(weather_seq)
        encoded = hidden[-1]
        static = torch.cat([self.region_emb(region_code), date_features], dim=1)
        context = self.static_mlp(static)
        state = torch.tanh(self.initial_state(torch.cat([encoded, context], dim=1)))

        prev = previous_score
        outputs = []
        for horizon in range(N_WEEKS):
            state = self.decoder_cell(torch.cat([prev, context], dim=1), state)
            pred = self.output(torch.cat([state, context], dim=1))
            outputs.append(pred)
            if targets is not None and teacher_forcing_ratio > 0 and random.random() < teacher_forcing_ratio:
                prev = targets[:, horizon : horizon + 1]
            else:
                prev = pred
        return torch.cat(outputs, dim=1)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    losses = []
    criterion = nn.L1Loss()
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(
            batch["weather_seq"],
            batch["region_code"],
            batch["date_features"],
            batch["previous_score"],
            batch["targets"],
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss = criterion(preds, batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) * SCORE_SCALE


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc="valid", leave=False):
        batch = move_batch(batch, device)
        preds = model(
            batch["weather_seq"],
            batch["region_code"],
            batch["date_features"],
            batch["previous_score"],
            targets=None,
            teacher_forcing_ratio=0.0,
        )
        all_preds.append(torch.clamp(preds, 0.0, 1.0).cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
    preds_np = np.vstack(all_preds) * SCORE_SCALE
    targets_np = np.vstack(all_targets) * SCORE_SCALE
    abs_err = np.abs(preds_np - targets_np)
    horizon_mae = abs_err.mean(axis=0)
    return {
        "mae": float(abs_err.mean()),
        "mae_by_horizon": {f"week_{idx + 1}": float(value) for idx, value in enumerate(horizon_mae)},
        "pred_mean": float(preds_np.mean()),
        "pred_std": float(preds_np.std()),
        "pred_min": float(preds_np.min()),
        "pred_max": float(preds_np.max()),
        "target_mean": float(targets_np.mean()),
        "target_std": float(targets_np.std()),
        "target_min": float(targets_np.min()),
        "target_max": float(targets_np.max()),
    }


def subset_by_mask(samples: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return samples.loc[mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = create_run_dir("group3_ar_gru", args.experiment_name)

    print("=" * 72)
    print("  Group 3 Reproduction - Autoregressive GRU")
    print("=" * 72)
    print(f"Experiment run directory: {run_dir}")
    print(f"Device: {device}")

    raw = load_raw_train(args.max_rows)
    clean = clean_and_filter(raw, args.max_regions, args.train_tail_days)
    panel = prepare_panel(clean, seq_len=args.seq_len, score_gap_days=args.score_gap_days)
    train_mask, val_mask = validation_split(panel.samples, args.val_weeks, args.purge_weeks)
    print(f"  train samples: {int(train_mask.sum()):,}")
    print(f"  validation samples: {int(val_mask.sum()):,}")

    samples_with_dates, date_cols, date_norm = make_date_feature_matrix(panel.samples, train_mask)
    weather_mean, weather_std = weather_normalization(
        panel.weather_by_region,
        samples_with_dates,
        train_mask,
        args.seq_len,
    )

    train_samples = subset_by_mask(samples_with_dates, train_mask)
    val_samples = subset_by_mask(samples_with_dates, val_mask)
    train_ds = Group3SequenceDataset(
        panel.weather_by_region,
        train_samples,
        date_cols,
        weather_mean,
        weather_std,
        args.seq_len,
    )
    val_ds = Group3SequenceDataset(
        panel.weather_by_region,
        val_samples,
        date_cols,
        weather_mean,
        weather_std,
        args.seq_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Group3AutoregressiveGRU(
        n_weather=len(METEO_COLS),
        n_regions=len(panel.region_to_code),
        date_feature_dim=len(date_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        region_emb_dim=args.region_emb_dim,
        context_dim=args.context_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "group3_ar_gru",
        "experiment_name": args.experiment_name,
        "presentation_mapping": {
            "weather_sequence": f"{args.seq_len} x {len(METEO_COLS)} weather features",
            "static_context": "region embedding + year/month/quarter/week cyclical date features",
            "gru_encoder": f"{args.num_layers} layers, hidden={args.hidden_size}",
            "decoder": "GRUCell x 5 horizons with previous score feedback",
            "teacher_forcing": args.teacher_forcing_ratio,
        },
        "data_options": {
            "score_gap_days": args.score_gap_days,
            "train_tail_days": args.train_tail_days,
            "max_rows": args.max_rows,
            "max_regions": args.max_regions,
            "val_weeks": args.val_weeks,
            "purge_weeks": args.purge_weeks,
            "weather_columns": METEO_COLS,
            "date_feature_columns": date_cols,
            "date_normalization": date_norm,
        },
        "train_options": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "seed": args.seed,
        },
        "sample_counts": {
            "raw_rows_after_cleaning": int(len(clean)),
            "regions": int(len(panel.region_to_code)),
            "sequence_samples": int(len(panel.samples)),
            "train_samples": int(train_mask.sum()),
            "validation_samples": int(val_mask.sum()),
        },
    }
    save_json(run_dir / "config.json", config)

    best_mae = float("inf")
    best_epoch = 0
    history = []
    model_path = run_dir / "models" / "group3_ar_gru.pt"
    for epoch in range(1, args.epochs + 1):
        train_mae = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
        )
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mae": train_mae, **val_metrics})
        print(
            f"Epoch {epoch:03d}: train_mae={train_mae:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}, pred_mean={val_metrics['pred_mean']:.3f}"
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "region_to_code": panel.region_to_code,
                    "code_to_region": panel.code_to_region,
                    "weather_mean": weather_mean,
                    "weather_std": weather_std,
                    "date_feature_columns": date_cols,
                    "meteo_columns": METEO_COLS,
                },
                model_path,
            )

    final_metrics = {
        "model_family": "group3_ar_gru",
        "experiment_name": args.experiment_name,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "history": history,
        "model_path": str(model_path),
    }
    save_json(run_dir / "metrics.json", final_metrics)
    print(f"\nBest validation MAE: {best_mae:.4f} at epoch {best_epoch}")
    print(f"Model saved -> {model_path}")
    print(f"Metrics saved -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
