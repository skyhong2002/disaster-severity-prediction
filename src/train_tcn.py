"""
Train a lightweight Team-20-style TCN sequence baseline.

Two variants are supported:

- ``weather``: 91-day meteorological sequence + region/date context.
- ``feature_fused``: the same sequence plus time-safe score/region priors.

The goal is not to out-engineer the boosted-tree path immediately; it is to add
a genuine sequence-model signal that can be blind-tested and blended.
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment_utils import create_run_dir, save_json
from features import METEO_COLS
from train_group3_ar_gru import (
    DEFAULT_SCORE_GAP_DAYS,
    DEFAULT_SEQ_LEN,
    N_WEEKS,
    SCORE_SCALE,
    add_date_parts,
    clean_and_filter,
    natural_region_key,
    resolve_device,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

FUSION_FEATURE_COLS = [
    "visible_score_mean",
    "visible_score_std",
    "visible_score_max",
    "visible_score_eq0_ratio",
    "visible_score_ge3_ratio",
    "recent365_score_mean",
    "recent365_score_eq0_ratio",
    "recent730_score_mean",
    "recent730_score_ge3_ratio",
    "monthly_score_mean",
]


@dataclass(frozen=True)
class PreparedTCNPanel:
    weather_by_region: dict[int, np.ndarray]
    samples: pd.DataFrame
    region_to_code: dict[str, int]
    code_to_region: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight TCN sequence model.")
    parser.add_argument("--experiment-name", default="tcn")
    parser.add_argument("--variant", choices=["weather", "feature_fused"], default="weather")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--score-gap-days", type=int, default=DEFAULT_SCORE_GAP_DAYS)
    parser.add_argument("--train-tail-days", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--val-weeks", type=int, default=8)
    parser.add_argument("--purge-weeks", type=int, default=N_WEEKS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-emb-dim", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_train(max_rows: int) -> pd.DataFrame:
    usecols = ["region_id", "date", *METEO_COLS, "score"]
    nrows = max_rows if max_rows > 0 else None
    print(f"Loading train.csv{' first ' + str(max_rows) + ' rows' if nrows else ''} ...")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols, nrows=nrows)
    print(f"  raw shape: {train.shape}")
    return train


def _safe_stats(values: np.ndarray, global_score: float) -> dict[str, float]:
    if values.size == 0:
        values = np.asarray([global_score], dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "max": float(values.max()),
        "eq0_ratio": float((values == 0).mean()),
        "ge3_ratio": float((values >= 3).mean()),
    }


def visible_score_feature_row(
    scores: np.ndarray,
    months: np.ndarray,
    day_idx: int,
    score_gap_days: int,
    sample_month: int,
    global_score: float,
) -> dict[str, float]:
    """Compute score priors using only labels visible before the blind gap."""
    positions = np.arange(len(scores))
    visible_cutoff = day_idx - score_gap_days
    visible = (~np.isnan(scores)) & (positions <= visible_cutoff)
    visible_values = scores[visible].astype(np.float32)
    base = _safe_stats(visible_values, global_score)

    recent365 = visible & (positions >= visible_cutoff - 365 + 1)
    recent365_stats = _safe_stats(scores[recent365].astype(np.float32), global_score)
    recent730 = visible & (positions >= visible_cutoff - 730 + 1)
    recent730_stats = _safe_stats(scores[recent730].astype(np.float32), global_score)
    monthly = visible & (months == sample_month)
    monthly_stats = _safe_stats(scores[monthly].astype(np.float32), global_score)

    return {
        "visible_score_mean": base["mean"],
        "visible_score_std": base["std"],
        "visible_score_max": base["max"],
        "visible_score_eq0_ratio": base["eq0_ratio"],
        "visible_score_ge3_ratio": base["ge3_ratio"],
        "recent365_score_mean": recent365_stats["mean"],
        "recent365_score_eq0_ratio": recent365_stats["eq0_ratio"],
        "recent730_score_mean": recent730_stats["mean"],
        "recent730_score_ge3_ratio": recent730_stats["ge3_ratio"],
        "monthly_score_mean": monthly_stats["mean"],
    }


def prepare_tcn_panel(
    train: pd.DataFrame,
    seq_len: int,
    score_gap_days: int,
    variant: str,
) -> PreparedTCNPanel:
    train = add_date_parts(train)
    regions = sorted(train["region_id"].astype(str).unique(), key=natural_region_key)
    region_to_code = {region: idx for idx, region in enumerate(regions)}
    code_to_region = {idx: region for region, idx in region_to_code.items()}
    global_score = train["score"].dropna().median()
    if pd.isna(global_score):
        global_score = 0.0
    global_score = float(global_score)

    weather_by_region: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for region_id, group in train.groupby("region_id", sort=False):
        region = str(region_id)
        code = region_to_code[region]
        group = group.sort_values("date").reset_index(drop=True)
        weather_by_region[code] = group[METEO_COLS].to_numpy(dtype=np.float32, copy=True)
        scores = group["score"].to_numpy(dtype=np.float32, copy=True)
        months = group["month"].to_numpy(dtype=np.int16, copy=True)
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
            origin = group.iloc[day_idx]
            record = {
                "region_id": region,
                "region_code": code,
                "day_idx": day_idx,
                "week_idx": weekly_rank,
                "date": origin["date"],
                "year": float(origin["year"]),
                "month": float(origin["month"]),
                "quarter": float(origin["quarter"]),
                "weekofyear": float(origin["weekofyear"]),
                **{f"target_w{week}": float(targets[week - 1]) for week in range(1, N_WEEKS + 1)},
            }
            if variant == "feature_fused":
                record.update(
                    visible_score_feature_row(
                        scores,
                        months,
                        day_idx=day_idx,
                        score_gap_days=score_gap_days,
                        sample_month=int(origin["month"]),
                        global_score=global_score,
                    )
                )
            records.append(record)

    samples = pd.DataFrame.from_records(records)
    if samples.empty:
        raise ValueError("No usable TCN samples. Increase --max-rows or --train-tail-days.")
    samples = samples.sort_values(["region_code", "week_idx"]).reset_index(drop=True)
    print(f"  sequence samples: {len(samples):,}")
    return PreparedTCNPanel(weather_by_region, samples, region_to_code, code_to_region)


def validation_split(samples: pd.DataFrame, val_weeks: int, purge_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    samples = samples.copy()
    samples["sample_rank"] = samples.groupby("region_code").cumcount()
    samples["max_sample_rank"] = samples.groupby("region_code")["sample_rank"].transform("max")
    val_start = (samples["max_sample_rank"] - max(0, val_weeks - 1)).clip(lower=0)
    val_mask = samples["sample_rank"] >= val_start
    train_mask = samples["sample_rank"] + purge_weeks < val_start
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError("Empty TCN validation split.")
    return train_mask.to_numpy(dtype=bool), val_mask.to_numpy(dtype=bool)


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


def normalize_fusion_features(
    samples: pd.DataFrame,
    train_mask: np.ndarray,
    cols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    samples = samples.copy()
    norm: dict[str, dict[str, float]] = {}
    for col in cols:
        mean = float(samples.loc[train_mask, col].mean())
        std = float(samples.loc[train_mask, col].std() or 1.0)
        samples[f"{col}_z"] = ((samples[col] - mean) / std).astype(np.float32)
        norm[col] = {"mean": mean, "std": std}
    return samples, norm


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
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class TCNSequenceDataset(Dataset):
    def __init__(
        self,
        weather_by_region: dict[int, np.ndarray],
        samples: pd.DataFrame,
        date_cols: list[str],
        fusion_cols: list[str],
        weather_mean: np.ndarray,
        weather_std: np.ndarray,
        seq_len: int,
    ) -> None:
        self.weather_by_region = weather_by_region
        self.region_codes = samples["region_code"].to_numpy(dtype=np.int64)
        self.day_idxs = samples["day_idx"].to_numpy(dtype=np.int64)
        self.date_features = samples[date_cols].to_numpy(dtype=np.float32)
        self.fusion_features = (
            samples[fusion_cols].to_numpy(dtype=np.float32)
            if fusion_cols
            else np.zeros((len(samples), 0), dtype=np.float32)
        )
        self.targets = samples[[f"target_w{week}" for week in range(1, N_WEEKS + 1)]].to_numpy(dtype=np.float32) / SCORE_SCALE
        self.weather_mean = weather_mean.reshape(1, -1)
        self.weather_std = np.maximum(weather_std.reshape(1, -1), 1e-6)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.region_codes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        code = int(self.region_codes[idx])
        day_idx = int(self.day_idxs[idx])
        seq = self.weather_by_region[code][day_idx - self.seq_len + 1 : day_idx + 1]
        seq = (seq - self.weather_mean) / self.weather_std
        return {
            "weather_seq": torch.as_tensor(seq, dtype=torch.float32),
            "region_code": torch.as_tensor(code, dtype=torch.long),
            "date_features": torch.as_tensor(self.date_features[idx], dtype=torch.float32),
            "fusion_features": torch.as_tensor(self.fusion_features[idx], dtype=torch.float32),
            "targets": torch.as_tensor(self.targets[idx], dtype=torch.float32),
        }


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.pad(x, (self.pad, 0)))
        out = self.dropout(F.relu(out))
        out = self.conv2(F.pad(out, (self.pad, 0)))
        out = self.dropout(F.relu(out))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class TCNForecastModel(nn.Module):
    def __init__(
        self,
        n_weather: int,
        n_regions: int,
        date_feature_dim: int,
        fusion_feature_dim: int,
        channels: int,
        levels: int,
        kernel_size: int,
        region_emb_dim: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_weather
        for level in range(levels):
            blocks.append(TemporalBlock(in_channels, channels, kernel_size, dilation=2**level, dropout=dropout))
            in_channels = channels
        self.tcn = nn.Sequential(*blocks)
        self.region_emb = nn.Embedding(n_regions, region_emb_dim)
        static_dim = region_emb_dim + date_feature_dim + fusion_feature_dim
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + context_dim, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, N_WEEKS),
        )

    def forward(
        self,
        weather_seq: torch.Tensor,
        region_code: torch.Tensor,
        date_features: torch.Tensor,
        fusion_features: torch.Tensor,
    ) -> torch.Tensor:
        temporal = self.tcn(weather_seq.transpose(1, 2))[:, :, -1]
        static = torch.cat([self.region_emb(region_code), date_features, fusion_features], dim=1)
        context = self.static_mlp(static)
        return self.head(torch.cat([temporal, context], dim=1))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.L1Loss()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch["weather_seq"], batch["region_code"], batch["date_features"], batch["fusion_features"])
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
        preds = model(batch["weather_seq"], batch["region_code"], batch["date_features"], batch["fusion_features"])
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


def subset(samples: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return samples.loc[mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = create_run_dir("tcn", args.experiment_name)

    print("=" * 72)
    print("  Team 20 Reproduction - Lightweight TCN")
    print("=" * 72)
    print(f"Experiment run directory: {run_dir}")
    print(f"Variant: {args.variant}")
    print(f"Device: {device}")

    raw = load_raw_train(args.max_rows)
    clean = clean_and_filter(raw, args.max_regions, args.train_tail_days)
    panel = prepare_tcn_panel(clean, args.seq_len, args.score_gap_days, args.variant)
    train_mask, val_mask = validation_split(panel.samples, args.val_weeks, args.purge_weeks)
    print(f"  train samples: {int(train_mask.sum()):,}")
    print(f"  validation samples: {int(val_mask.sum()):,}")

    samples, date_cols, date_norm = make_date_feature_matrix(panel.samples, train_mask)
    raw_fusion_cols = FUSION_FEATURE_COLS if args.variant == "feature_fused" else []
    samples, fusion_norm = normalize_fusion_features(samples, train_mask, raw_fusion_cols)
    fusion_cols = [f"{col}_z" for col in raw_fusion_cols]
    weather_mean, weather_std = weather_normalization(panel.weather_by_region, samples, train_mask, args.seq_len)

    train_ds = TCNSequenceDataset(
        panel.weather_by_region,
        subset(samples, train_mask),
        date_cols,
        fusion_cols,
        weather_mean,
        weather_std,
        args.seq_len,
    )
    val_ds = TCNSequenceDataset(
        panel.weather_by_region,
        subset(samples, val_mask),
        date_cols,
        fusion_cols,
        weather_mean,
        weather_std,
        args.seq_len,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = TCNForecastModel(
        n_weather=len(METEO_COLS),
        n_regions=len(panel.region_to_code),
        date_feature_dim=len(date_cols),
        fusion_feature_dim=len(fusion_cols),
        channels=args.channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        region_emb_dim=args.region_emb_dim,
        context_dim=args.context_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "tcn",
        "experiment_name": args.experiment_name,
        "variant": args.variant,
        "data_options": {
            "seq_len": args.seq_len,
            "score_gap_days": args.score_gap_days,
            "train_tail_days": args.train_tail_days,
            "max_rows": args.max_rows,
            "max_regions": args.max_regions,
            "val_weeks": args.val_weeks,
            "purge_weeks": args.purge_weeks,
            "weather_columns": METEO_COLS,
            "date_feature_columns": date_cols,
            "date_normalization": date_norm,
            "fusion_feature_columns": raw_fusion_cols,
            "fusion_normalization": fusion_norm,
        },
        "model_options": {
            "channels": args.channels,
            "levels": args.levels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "region_emb_dim": args.region_emb_dim,
            "context_dim": args.context_dim,
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
    model_path = run_dir / "models" / "tcn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_mae = train_epoch(model, train_loader, optimizer, device)
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
                    "meteo_columns": METEO_COLS,
                },
                model_path,
            )

    metrics = {
        "model_family": "tcn",
        "variant": args.variant,
        "experiment_name": args.experiment_name,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "history": history,
        "model_path": str(model_path),
    }
    save_json(run_dir / "metrics.json", metrics)
    print(f"\nBest validation MAE: {best_mae:.4f} at epoch {best_epoch}")
    print(f"Model saved -> {model_path}")
    print(f"Metrics saved -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
