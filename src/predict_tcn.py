"""Generate Kaggle submissions from a lightweight TCN run."""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiment_utils import save_json
from features import METEO_COLS
from train_group3_ar_gru import SCORE_SCALE, add_date_parts, clean_and_filter, resolve_device
from train_tcn import (
    N_WEEKS,
    TCNForecastModel,
    visible_score_feature_row,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TCN Kaggle submission.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any]:
    path = run_dir / "models" / "tcn.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing TCN checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def build_model(checkpoint: dict[str, Any], device: torch.device) -> TCNForecastModel:
    config = checkpoint["config"]
    data_options = config["data_options"]
    model_options = config["model_options"]
    model = TCNForecastModel(
        n_weather=len(checkpoint.get("meteo_columns", METEO_COLS)),
        n_regions=len(checkpoint["region_to_code"]),
        date_feature_dim=len(data_options["date_feature_columns"]),
        fusion_feature_dim=len(data_options.get("fusion_feature_columns", [])),
        channels=int(model_options["channels"]),
        levels=int(model_options["levels"]),
        kernel_size=int(model_options["kernel_size"]),
        region_emb_dim=int(model_options["region_emb_dim"]),
        context_dim=int(model_options["context_dim"]),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def numeric_clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["region_id"] = frame["region_id"].astype(str)
    frame["date"] = frame["date"].astype(str)
    for col in METEO_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").astype(np.float32) if "score" in frame else np.nan
    return frame


def load_inference_panel(config: dict[str, Any]) -> pd.DataFrame:
    data_options = config.get("data_options", {})
    tail_days = int(data_options.get("train_tail_days", 0) or 0)
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["region_id", "date", *METEO_COLS, "score"])
    train = clean_and_filter(train, max_regions=0, train_tail_days=tail_days)
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["region_id", "date", *METEO_COLS])
    test = numeric_clean(test)
    test["is_test"] = True
    train["is_test"] = False
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.sort_values(["region_id", "date"]).reset_index(drop=True)
    medians = train[METEO_COLS].median(numeric_only=True).astype(np.float32)
    combined[METEO_COLS] = combined.groupby("region_id")[METEO_COLS].transform(lambda col: col.ffill().bfill())
    combined[METEO_COLS] = combined[METEO_COLS].fillna(medians).fillna(0.0).astype(np.float32)
    return add_date_parts(combined)


def date_features_for_rows(rows: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    rows = rows.copy()
    data_options = config["data_options"]
    norm = data_options["date_normalization"]
    rows["year_z"] = ((rows["year"] - float(norm["year_mean"])) / float(norm["year_std"] or 1.0)).astype(np.float32)
    rows["month_sin"] = np.sin(2 * np.pi * rows["month"] / 12).astype(np.float32)
    rows["month_cos"] = np.cos(2 * np.pi * rows["month"] / 12).astype(np.float32)
    rows["quarter_sin"] = np.sin(2 * np.pi * rows["quarter"] / 4).astype(np.float32)
    rows["quarter_cos"] = np.cos(2 * np.pi * rows["quarter"] / 4).astype(np.float32)
    rows["week_sin"] = np.sin(2 * np.pi * rows["weekofyear"] / 52).astype(np.float32)
    rows["week_cos"] = np.cos(2 * np.pi * rows["weekofyear"] / 52).astype(np.float32)
    return rows[data_options["date_feature_columns"]].to_numpy(dtype=np.float32)


def fusion_features_for_rows(
    combined: pd.DataFrame,
    final_indices: dict[str, int],
    config: dict[str, Any],
) -> np.ndarray:
    raw_cols = list(config["data_options"].get("fusion_feature_columns", []))
    if not raw_cols:
        return np.zeros((len(final_indices), 0), dtype=np.float32)
    score_gap_days = int(config["data_options"]["score_gap_days"])
    norms = config["data_options"]["fusion_normalization"]
    global_score = combined["score"].dropna().median()
    if pd.isna(global_score):
        global_score = 0.0
    rows = []
    for region_id, group in combined.groupby("region_id", sort=False):
        region = str(region_id)
        if region not in final_indices:
            continue
        group = group.sort_values("date").reset_index(drop=True)
        day_idx = int(final_indices[region])
        scores = group["score"].to_numpy(dtype=np.float32)
        months = group["month"].to_numpy(dtype=np.int16)
        sample_month = int(group.loc[day_idx, "month"])
        raw = visible_score_feature_row(scores, months, day_idx, score_gap_days, sample_month, float(global_score))
        rows.append([(raw[col] - norms[col]["mean"]) / (norms[col]["std"] or 1.0) for col in raw_cols])
    return np.asarray(rows, dtype=np.float32)


def build_prediction_arrays(
    combined: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = checkpoint["config"]
    seq_len = int(config["data_options"]["seq_len"])
    region_to_code = checkpoint["region_to_code"]
    weather_mean = np.asarray(checkpoint["weather_mean"], dtype=np.float32).reshape(1, -1)
    weather_std = np.maximum(np.asarray(checkpoint["weather_std"], dtype=np.float32).reshape(1, -1), 1e-6)

    region_ids: list[str] = []
    region_codes: list[int] = []
    weather_seqs: list[np.ndarray] = []
    final_rows = []
    final_indices: dict[str, int] = {}

    for region_id, group in combined.groupby("region_id", sort=False):
        region = str(region_id)
        if region not in region_to_code:
            continue
        group = group.sort_values("date").reset_index(drop=True)
        final_positions = np.flatnonzero(group["is_test"].to_numpy(dtype=bool))
        if len(final_positions) == 0:
            continue
        day_idx = int(final_positions[-1])
        if day_idx < seq_len - 1:
            raise ValueError(f"Region {region} has insufficient sequence history.")
        seq = group.loc[day_idx - seq_len + 1 : day_idx, METEO_COLS].to_numpy(dtype=np.float32)
        region_ids.append(region)
        region_codes.append(int(region_to_code[region]))
        weather_seqs.append((seq - weather_mean) / weather_std)
        final_rows.append(group.iloc[day_idx])
        final_indices[region] = day_idx

    final_frame = pd.DataFrame(final_rows)
    date_features = date_features_for_rows(final_frame, config)
    fusion_features = fusion_features_for_rows(combined, final_indices, config)
    return (
        region_ids,
        np.asarray(region_codes, dtype=np.int64),
        np.stack(weather_seqs).astype(np.float32),
        date_features,
        fusion_features,
    )


@torch.no_grad()
def predict_batches(
    model: TCNForecastModel,
    device: torch.device,
    region_codes: np.ndarray,
    weather_seqs: np.ndarray,
    date_features: np.ndarray,
    fusion_features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(region_codes), batch_size):
        end = min(start + batch_size, len(region_codes))
        preds = model(
            torch.as_tensor(weather_seqs[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(region_codes[start:end], dtype=torch.long, device=device),
            torch.as_tensor(date_features[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(fusion_features[start:end], dtype=torch.float32, device=device),
        )
        chunks.append(torch.clamp(preds, 0.0, 1.0).cpu().numpy() * SCORE_SCALE)
    return np.vstack(chunks)


def prediction_stats(preds: np.ndarray) -> dict[str, Any]:
    stats = {
        "overall": {
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        }
    }
    for idx in range(N_WEEKS):
        values = preds[:, idx]
        stats[f"pred_week{idx + 1}"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(run_dir, device)
    model = build_model(checkpoint, device)

    print("=" * 72)
    print("  Team 20 TCN - Inference")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    combined = load_inference_panel(checkpoint["config"])
    region_ids, region_codes, weather_seqs, date_features, fusion_features = build_prediction_arrays(combined, checkpoint)
    print(f"Prepared inference rows: {len(region_ids)}")
    preds = predict_batches(model, device, region_codes, weather_seqs, date_features, fusion_features, args.batch_size)

    pred_cols = [f"pred_week{week}" for week in range(1, N_WEEKS + 1)]
    result = pd.DataFrame({"region_id": region_ids})
    for idx, col in enumerate(pred_cols):
        result[col] = preds[:, idx]
    template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = template[["region_id"]].merge(result, on="region_id", how="left")
    if submission[pred_cols].isna().any().any():
        missing = submission.loc[submission[pred_cols].isna().any(axis=1), "region_id"].head(10).tolist()
        raise ValueError(f"Missing predictions for regions: {missing}")
    submission[pred_cols] = submission[pred_cols].clip(0, 5)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = checkpoint["config"].get("variant", "tcn")
    out_path = SUB_DIR / f"submission_{ts}_{run_dir.name}_{variant}.csv"
    submission.to_csv(out_path, index=False)
    run_submission_path = run_dir / "submissions" / out_path.name
    run_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(run_submission_path, index=False)

    stats = prediction_stats(submission[pred_cols].to_numpy(dtype=np.float32))
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "tcn",
        "variant": variant,
        "run_dir": str(run_dir),
        "global_submission_path": str(out_path),
        "run_submission_path": str(run_submission_path),
        "rows": int(len(submission)),
        "prediction_stats": stats,
        "sanity_checks": {
            "expected_rows": int(len(template)),
            "row_count_ok": bool(len(submission) == len(template)),
            "no_nan": bool(not submission[pred_cols].isna().any().any()),
            "range_min": float(submission[pred_cols].min().min()),
            "range_max": float(submission[pred_cols].max().max()),
        },
    }
    save_json(run_dir / "submission_metadata.json", metadata)
    print(f"Submission saved -> {out_path}")
    print(f"Run submission saved -> {run_submission_path}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(metadata["sanity_checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
