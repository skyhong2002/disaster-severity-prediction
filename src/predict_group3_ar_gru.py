"""
Generate a Kaggle submission from a Group-3-style autoregressive GRU run.

This script is separate from ``src/predict.py`` because the current production
path stores sklearn-style horizon models, while the AR-GRU checkpoint is a
single PyTorch sequence-to-sequence model.

Example:
    uv run python src/predict_group3_ar_gru.py \
      --run-dir experiments/20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521
"""
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
from train_group3_ar_gru import (
    DEFAULT_SCORE_GAP_DAYS,
    DEFAULT_SEQ_LEN,
    N_WEEKS,
    SCORE_SCALE,
    Group3AutoregressiveGRU,
    add_date_parts,
    clean_and_filter,
    resolve_device,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AR-GRU Kaggle submission.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment run directory containing models/group3_ar_gru.pt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = run_dir / "models" / "group3_ar_gru.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing AR-GRU checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def architecture_from_state(checkpoint: dict[str, Any]) -> dict[str, int]:
    state = checkpoint["model_state_dict"]
    hidden_size = int(state["encoder.weight_ih_l0"].shape[0] // 3)
    num_layers = sum(1 for key in state if key.startswith("encoder.weight_ih_l"))
    n_regions = int(state["region_emb.weight"].shape[0])
    region_emb_dim = int(state["region_emb.weight"].shape[1])
    context_dim = int(state["static_mlp.0.weight"].shape[0])
    date_feature_dim = len(checkpoint["date_feature_columns"])
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "n_regions": n_regions,
        "region_emb_dim": region_emb_dim,
        "context_dim": context_dim,
        "date_feature_dim": date_feature_dim,
        "n_weather": len(checkpoint.get("meteo_columns", METEO_COLS)),
    }


def build_model(checkpoint: dict[str, Any], device: torch.device) -> Group3AutoregressiveGRU:
    arch = architecture_from_state(checkpoint)
    model = Group3AutoregressiveGRU(
        n_weather=arch["n_weather"],
        n_regions=arch["n_regions"],
        date_feature_dim=arch["date_feature_dim"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        region_emb_dim=arch["region_emb_dim"],
        context_dim=arch["context_dim"],
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
    if "score" in frame.columns:
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce").astype(np.float32)
    else:
        frame["score"] = np.nan
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
    combined[METEO_COLS] = combined.groupby("region_id")[METEO_COLS].transform(
        lambda col: col.ffill().bfill()
    )
    combined[METEO_COLS] = combined[METEO_COLS].fillna(medians).fillna(0.0).astype(np.float32)
    return add_date_parts(combined)


def date_features_for_rows(rows: pd.DataFrame, config: dict[str, Any], date_cols: list[str]) -> np.ndarray:
    rows = rows.copy()
    norm = config["data_options"]["date_normalization"]
    year_mean = float(norm["year_mean"])
    year_std = float(norm["year_std"] or 1.0)
    rows["year_z"] = ((rows["year"] - year_mean) / year_std).astype(np.float32)
    rows["month_sin"] = np.sin(2 * np.pi * rows["month"] / 12).astype(np.float32)
    rows["month_cos"] = np.cos(2 * np.pi * rows["month"] / 12).astype(np.float32)
    rows["quarter_sin"] = np.sin(2 * np.pi * rows["quarter"] / 4).astype(np.float32)
    rows["quarter_cos"] = np.cos(2 * np.pi * rows["quarter"] / 4).astype(np.float32)
    rows["week_sin"] = np.sin(2 * np.pi * rows["weekofyear"] / 52).astype(np.float32)
    rows["week_cos"] = np.cos(2 * np.pi * rows["weekofyear"] / 52).astype(np.float32)
    return rows[date_cols].to_numpy(dtype=np.float32)


def build_inference_arrays(
    combined: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = checkpoint["config"]
    data_options = config.get("data_options", {})
    seq_len = int(data_options.get("seq_len", DEFAULT_SEQ_LEN) or DEFAULT_SEQ_LEN)
    score_gap_days = int(data_options.get("score_gap_days", DEFAULT_SCORE_GAP_DAYS) or DEFAULT_SCORE_GAP_DAYS)
    region_to_code = checkpoint["region_to_code"]
    date_cols = list(checkpoint["date_feature_columns"])
    weather_mean = np.asarray(checkpoint["weather_mean"], dtype=np.float32).reshape(1, -1)
    weather_std = np.maximum(np.asarray(checkpoint["weather_std"], dtype=np.float32).reshape(1, -1), 1e-6)

    global_score_median = combined["score"].dropna().median()
    if pd.isna(global_score_median):
        global_score_median = 0.0
    global_score_median = float(global_score_median)

    region_ids: list[str] = []
    region_codes: list[int] = []
    weather_seqs: list[np.ndarray] = []
    previous_scores: list[float] = []
    final_rows = []

    for region_id, group in combined.groupby("region_id", sort=False):
        region = str(region_id)
        if region not in region_to_code:
            continue
        group = group.sort_values("date").reset_index(drop=True)
        test_positions = np.flatnonzero(group["is_test"].to_numpy(dtype=bool))
        if len(test_positions) < seq_len:
            raise ValueError(f"Region {region} has only {len(test_positions)} test rows; need {seq_len}.")
        day_idx = int(test_positions[-1])
        seq = group.loc[day_idx - seq_len + 1 : day_idx, METEO_COLS].to_numpy(dtype=np.float32)
        if seq.shape != (seq_len, len(METEO_COLS)):
            raise ValueError(f"Region {region} sequence shape {seq.shape}, expected {(seq_len, len(METEO_COLS))}.")

        scores = group["score"].to_numpy(dtype=np.float32)
        last_visible_score = (
            pd.Series(scores)
            .ffill()
            .fillna(global_score_median)
            .to_numpy(dtype=np.float32)
        )
        visible_idx = day_idx - score_gap_days
        previous_score = float(last_visible_score[visible_idx]) if visible_idx >= 0 else global_score_median

        region_ids.append(region)
        region_codes.append(int(region_to_code[region]))
        weather_seqs.append((seq - weather_mean) / weather_std)
        previous_scores.append(previous_score / SCORE_SCALE)
        final_rows.append(group.iloc[day_idx])

    final_frame = pd.DataFrame(final_rows)
    date_features = date_features_for_rows(final_frame, config, date_cols)
    return (
        region_ids,
        np.asarray(region_codes, dtype=np.int64),
        np.stack(weather_seqs).astype(np.float32),
        date_features,
        np.asarray(previous_scores, dtype=np.float32).reshape(-1, 1),
    )


@torch.no_grad()
def predict_batches(
    model: Group3AutoregressiveGRU,
    device: torch.device,
    region_codes: np.ndarray,
    weather_seqs: np.ndarray,
    date_features: np.ndarray,
    previous_scores: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(region_codes), batch_size):
        end = min(start + batch_size, len(region_codes))
        preds = model(
            torch.as_tensor(weather_seqs[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(region_codes[start:end], dtype=torch.long, device=device),
            torch.as_tensor(date_features[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(previous_scores[start:end], dtype=torch.float32, device=device),
            targets=None,
            teacher_forcing_ratio=0.0,
        )
        chunks.append(torch.clamp(preds, 0.0, 1.0).cpu().numpy() * SCORE_SCALE)
    return np.vstack(chunks)


def prediction_stats(preds: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "overall": {
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        }
    }
    for idx in range(N_WEEKS):
        col = preds[:, idx]
        stats[f"pred_week{idx + 1}"] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
        }
    return stats


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(run_dir, device)
    model = build_model(checkpoint, device)

    print("=" * 72)
    print("  Group 3 AR-GRU - Inference")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    combined = load_inference_panel(checkpoint["config"])
    region_ids, region_codes, weather_seqs, date_features, previous_scores = build_inference_arrays(
        combined,
        checkpoint,
    )
    print(f"Prepared inference rows: {len(region_ids)}")

    preds = predict_batches(
        model,
        device,
        region_codes,
        weather_seqs,
        date_features,
        previous_scores,
        batch_size=args.batch_size,
    )

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
    out_path = SUB_DIR / f"submission_{ts}_{run_dir.name}_group3_ar_gru.csv"
    submission.to_csv(out_path, index=False)

    run_submission_path = run_dir / "submissions" / out_path.name
    run_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(run_submission_path, index=False)

    stats = prediction_stats(submission[pred_cols].to_numpy(dtype=np.float32))
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": "group3_ar_gru",
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "models" / "group3_ar_gru.pt"),
        "global_submission_path": str(out_path),
        "run_submission_path": str(run_submission_path),
        "rows": int(len(submission)),
        "columns": list(submission.columns),
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
    print("Prediction statistics:")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print("Sanity checks:")
    print(json.dumps(metadata["sanity_checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
