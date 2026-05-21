#!/usr/bin/env python3
"""Run Kaggle-like blind-window backtests for the Group 3 AR-GRU checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import get_latest_run_dir, save_json  # noqa: E402
from features import METEO_COLS  # noqa: E402
from predict_group3_ar_gru import (  # noqa: E402
    build_model,
    date_features_for_rows,
    load_checkpoint,
    predict_batches,
    prediction_stats,
)
from train_group3_ar_gru import (  # noqa: E402
    DEFAULT_SCORE_GAP_DAYS,
    DEFAULT_SEQ_LEN,
    SCORE_SCALE,
    add_date_parts,
    clean_and_filter,
    resolve_device,
)
from validation import (  # noqa: E402
    evaluate_submission_like_predictions,
    make_blind_backtest_origin,
    parse_origin_offsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Group 3 AR-GRU run with blind backtesting.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment run directory. Defaults to experiments/latest.txt.",
    )
    parser.add_argument(
        "--origins",
        default="5,13,26",
        help="Comma-separated week offsets from train tail. Negative values are accepted for readability.",
    )
    parser.add_argument("--blind-days", type=int, default=DEFAULT_SCORE_GAP_DAYS)
    parser.add_argument(
        "--history-tail-days",
        type=int,
        default=1100,
        help="Daily history rows per region retained before each pseudo origin.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run-dir>/validation_ar_gru.")
    return parser.parse_args()


def resolve_run_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    latest = get_latest_run_dir()
    if latest is None:
        raise FileNotFoundError("No --run-dir provided and experiments/latest.txt is missing.")
    return latest


def prepare_masked_panel(combined: pd.DataFrame) -> pd.DataFrame:
    """Apply the same numeric/imputation path as Group 3 training to masked rows."""
    cleaned = clean_and_filter(combined, max_regions=0, train_tail_days=0)
    return add_date_parts(cleaned)


def build_origin_arrays(
    combined: pd.DataFrame,
    targets: pd.DataFrame,
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

    groups = {
        str(region): group.sort_values("date").reset_index(drop=True)
        for region, group in combined.groupby("region_id")
    }
    region_ids: list[str] = []
    region_codes: list[int] = []
    weather_seqs: list[np.ndarray] = []
    previous_scores: list[float] = []
    final_rows = []

    for target in targets.itertuples(index=False):
        region = str(target.region_id)
        if region not in region_to_code:
            raise ValueError(f"Region {region} is not present in checkpoint region mapping.")
        group = groups.get(region)
        if group is None:
            raise ValueError(f"Region {region} is missing from the masked origin panel.")
        day_matches = np.flatnonzero(group["_day_idx"].to_numpy(dtype=np.int32) == int(target.origin_day_idx))
        if len(day_matches) != 1:
            raise ValueError(f"Expected one origin row for {region}, got {len(day_matches)}.")
        row_pos = int(day_matches[0])
        if row_pos < seq_len - 1:
            raise ValueError(f"Region {region} has only {row_pos + 1} rows before origin; need {seq_len}.")

        seq = group.loc[row_pos - seq_len + 1 : row_pos, METEO_COLS].to_numpy(dtype=np.float32)
        if seq.shape != (seq_len, len(METEO_COLS)):
            raise ValueError(f"Region {region} sequence shape {seq.shape}, expected {(seq_len, len(METEO_COLS))}.")

        scores = group["score"].to_numpy(dtype=np.float32)
        last_visible_score = (
            pd.Series(scores)
            .ffill()
            .fillna(global_score_median)
            .to_numpy(dtype=np.float32)
        )
        visible_idx = row_pos - score_gap_days
        previous_score = float(last_visible_score[visible_idx]) if visible_idx >= 0 else global_score_median

        region_ids.append(region)
        region_codes.append(int(region_to_code[region]))
        weather_seqs.append((seq - weather_mean) / weather_std)
        previous_scores.append(previous_score / SCORE_SCALE)
        final_rows.append(group.iloc[row_pos])

    final_frame = pd.DataFrame(final_rows)
    date_features = date_features_for_rows(final_frame, config, date_cols)
    return (
        region_ids,
        np.asarray(region_codes, dtype=np.int64),
        np.stack(weather_seqs).astype(np.float32),
        date_features,
        np.asarray(previous_scores, dtype=np.float32).reshape(-1, 1),
    )


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "validation_ar_gru"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(run_dir, device)
    model = build_model(checkpoint, device)
    origins = parse_origin_offsets(args.origins)

    print("=" * 72)
    print("  Group 3 AR-GRU - Kaggle-like Blind Backtest")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")

    print("\nLoading train.csv ...")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    print(f"  train shape: {train.shape}")

    prediction_frames = []
    target_frames = []
    for origin in origins:
        print(f"\n[Origin {origin.label}] Building masked pseudo-test window ...")
        combined_raw, _train_history, targets = make_blind_backtest_origin(
            train,
            origin,
            blind_days=args.blind_days,
            history_tail_days=args.history_tail_days,
        )
        combined = prepare_masked_panel(combined_raw)
        region_ids, region_codes, weather_seqs, date_features, previous_scores = build_origin_arrays(
            combined,
            targets,
            checkpoint,
        )
        preds_np = predict_batches(
            model,
            device,
            region_codes,
            weather_seqs,
            date_features,
            previous_scores,
            batch_size=args.batch_size,
        )
        pred_cols = [f"pred_week{week}" for week in range(1, 6)]
        preds = pd.DataFrame({"region_id": region_ids})
        for idx, col in enumerate(pred_cols):
            preds[col] = preds_np[:, idx]
        preds["origin"] = origin.label
        targets = targets.copy()
        targets["origin"] = origin.label
        if len(preds) != len(targets):
            raise ValueError(f"Prediction/target row mismatch for {origin.label}: {len(preds)} vs {len(targets)}")

        prediction_frames.append(preds)
        target_frames.append(targets)
        origin_pred_path = out_dir / f"blind_predictions_{origin.label}.csv"
        preds.to_csv(origin_pred_path, index=False)
        print(f"  rows: {len(preds)}")
        stats = prediction_stats(preds[pred_cols].to_numpy(dtype=np.float32))["overall"]
        print(f"  prediction stats: {json.dumps(stats, sort_keys=True)}")
        print(f"  saved predictions -> {origin_pred_path}")

    rows, metrics = evaluate_submission_like_predictions(prediction_frames, target_frames, train)
    metrics.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "model_family": "group3_ar_gru",
            "origins": [origin.label for origin in origins],
            "blind_days": args.blind_days,
            "history_tail_days": args.history_tail_days,
        }
    )

    rows_path = out_dir / "blind_backtest_rows.csv"
    metrics_path = out_dir / "blind_backtest_metrics.json"
    rows.to_csv(rows_path, index=False)
    save_json(metrics_path, metrics)

    print("\n--- Group 3 AR-GRU Blind Backtest Summary ---")
    print(f"Overall MAE: {metrics['overall_mae']:.5f}")
    for horizon, mae in metrics["mae_by_horizon"].items():
        print(f"  {horizon}: {mae:.5f}")
    print(f"Rows saved -> {rows_path}")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
