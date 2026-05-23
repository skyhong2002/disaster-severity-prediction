#!/usr/bin/env python3
"""Run Kaggle-like blind-window backtests against a saved experiment run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import get_latest_run_dir, save_json  # noqa: E402
from features import required_context_days  # noqa: E402
from predict import load_feature_options, load_models, model_dir_for_run  # noqa: E402
from validation import (  # noqa: E402
    build_pseudo_test_window,
    evaluate_submission_like_predictions,
    parse_origin_offsets,
    predict_blind_origin,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a run with Kaggle-like blind backtesting.")
    parser.add_argument("--run-dir", default=None, help="Experiment run directory. Defaults to experiments/latest.txt.")
    parser.add_argument(
        "--origins",
        default="5,13,26",
        help="Comma-separated week offsets from train tail. Negative values are accepted for readability.",
    )
    parser.add_argument("--blind-days", type=int, default=91)
    parser.add_argument(
        "--history-tail-days",
        type=int,
        default=750,
        help="Daily history rows per region retained before each pseudo origin, matching predict.py by default.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run-dir>/validation.")
    return parser.parse_args()


def resolve_run_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    latest = get_latest_run_dir()
    if latest is None:
        raise FileNotFoundError("No --run-dir provided and experiments/latest.txt is missing.")
    return latest


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "validation"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Kaggle-like Blind Backtest")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        run_config = json.load(f)
    feature_options = load_feature_options(run_dir)
    print(f"Feature options: {feature_options}")
    required_days = required_context_days(
        str(feature_options.get("feature_profile", "micro")),
        score_gap_days=int(feature_options.get("score_gap_days", args.blind_days)),
        use_score_history=bool(feature_options.get("use_score_history", False)),
        max_score_lag_weeks=feature_options.get("max_score_lag_weeks"),
    )
    if args.history_tail_days < required_days:
        print(
            "WARNING: "
            f"history_tail_days={args.history_tail_days} is below required_context_days={required_days} "
            f"for feature_profile={feature_options.get('feature_profile', 'micro')}. "
            "This is fine for smoke tests but not valid for model selection."
        )

    print("\nLoading train.csv ...")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    print(f"  train shape: {train.shape}")

    models = load_models(model_dir_for_run(run_dir), run_dir)
    origins = parse_origin_offsets(args.origins)

    prediction_frames = []
    target_frames = []
    for origin in origins:
        print(f"\n[Origin {origin.label}] Building masked pseudo-test window ...")
        pseudo_last, targets = build_pseudo_test_window(
            train,
            origin,
            feature_options,
            blind_days=args.blind_days,
            history_tail_days=args.history_tail_days,
        )
        print(f"  pseudo rows: {len(pseudo_last)}, targets: {len(targets)}")
        preds = predict_blind_origin(models, pseudo_last)
        preds["origin"] = origin.label
        targets["origin"] = origin.label
        prediction_frames.append(preds)
        target_frames.append(targets)

        origin_pred_path = out_dir / f"blind_predictions_{origin.label}.csv"
        preds.to_csv(origin_pred_path, index=False)
        print(f"  saved predictions -> {origin_pred_path}")

    rows, metrics = evaluate_submission_like_predictions(prediction_frames, target_frames, train)
    metrics.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "model_family": run_config.get("model_family"),
            "origins": [origin.label for origin in origins],
            "blind_days": args.blind_days,
            "history_tail_days": args.history_tail_days,
            "feature_options": feature_options,
        }
    )

    rows_path = out_dir / "blind_backtest_rows.csv"
    metrics_path = out_dir / "blind_backtest_metrics.json"
    rows.to_csv(rows_path, index=False)
    save_json(metrics_path, metrics)

    print("\n--- Blind Backtest Summary ---")
    print(f"Overall MAE: {metrics['overall_mae']:.5f}")
    for horizon, mae in metrics["mae_by_horizon"].items():
        print(f"  {horizon}: {mae:.5f}")
    print(f"Rows saved -> {rows_path}")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
