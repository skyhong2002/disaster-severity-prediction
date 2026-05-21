#!/usr/bin/env python3
"""Run Kaggle-like blind-window backtests for a lightweight TCN checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import save_json  # noqa: E402
from predict_tcn import (  # noqa: E402
    build_model,
    build_prediction_arrays,
    load_checkpoint,
    predict_batches,
)
from features import METEO_COLS  # noqa: E402
from train_group3_ar_gru import add_date_parts, resolve_device  # noqa: E402
from validation import (  # noqa: E402
    evaluate_submission_like_predictions,
    make_blind_backtest_origin,
    parse_origin_offsets,
)

N_WEEKS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TCN run with blind backtesting.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--origins", default="5,13,26")
    parser.add_argument("--blind-days", type=int, default=91)
    parser.add_argument("--history-tail-days", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def prepare_blind_panel(combined: pd.DataFrame) -> pd.DataFrame:
    """Prepare masked rows for TCN prediction; origin row acts as final test row."""
    panel = combined.copy()
    for col in METEO_COLS:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")
    panel["score"] = pd.to_numeric(panel["score"], errors="coerce").astype("float32")
    medians = panel[METEO_COLS].median(numeric_only=True).astype("float32")
    panel[METEO_COLS] = panel.groupby("region_id")[METEO_COLS].transform(lambda col: col.ffill().bfill())
    panel[METEO_COLS] = panel[METEO_COLS].fillna(medians).fillna(0.0).astype("float32")
    panel = add_date_parts(panel)
    panel["is_test"] = panel["_day_idx"] <= panel["_origin_day_idx"]
    panel["is_test"] = panel["is_test"] & (panel["_day_idx"] >= panel["_blind_start_idx"])
    return panel


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "validation_tcn"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(run_dir, device)
    model = build_model(checkpoint, device)
    config = checkpoint["config"]
    origins = parse_origin_offsets(args.origins)

    print("=" * 72)
    print("  Team 20 TCN - Kaggle-like Blind Backtest")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Variant: {config.get('variant')}")
    print(f"Device: {device}")

    train = pd.read_csv(ROOT / "data" / "train.csv")
    prediction_frames = []
    target_frames = []
    for origin in origins:
        print(f"\n[Origin {origin.label}] Building masked pseudo-test window ...")
        combined, _train_history, targets = make_blind_backtest_origin(
            train,
            origin,
            blind_days=args.blind_days,
            history_tail_days=args.history_tail_days,
        )
        panel = prepare_blind_panel(combined)
        region_ids, region_codes, weather_seqs, date_features, fusion_features = build_prediction_arrays(panel, checkpoint)
        preds_np = predict_batches(
            model,
            device,
            region_codes,
            weather_seqs,
            date_features,
            fusion_features,
            batch_size=args.batch_size,
        )
        preds = pd.DataFrame({"region_id": region_ids})
        for idx in range(N_WEEKS):
            preds[f"pred_week{idx + 1}"] = preds_np[:, idx]
        preds["origin"] = origin.label
        targets["origin"] = origin.label
        prediction_frames.append(preds)
        target_frames.append(targets)
        path = out_dir / f"blind_predictions_{origin.label}.csv"
        preds.to_csv(path, index=False)
        print(f"  saved predictions -> {path}")

    rows, metrics = evaluate_submission_like_predictions(prediction_frames, target_frames, train)
    metrics.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "model_family": "tcn",
            "variant": config.get("variant"),
            "origins": [origin.label for origin in origins],
            "blind_days": args.blind_days,
            "history_tail_days": args.history_tail_days,
        }
    )
    rows_path = out_dir / "blind_backtest_rows.csv"
    metrics_path = out_dir / "blind_backtest_metrics.json"
    rows.to_csv(rows_path, index=False)
    save_json(metrics_path, metrics)

    print("\n--- TCN Blind Backtest Summary ---")
    print(f"Overall MAE: {metrics['overall_mae']:.5f}")
    for horizon, mae in metrics["mae_by_horizon"].items():
        print(f"  {horizon}: {mae:.5f}")
    print(f"Rows saved -> {rows_path}")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
