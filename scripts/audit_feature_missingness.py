#!/usr/bin/env python3
"""Audit raw and engineered feature missingness for train/test pipelines."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import save_json  # noqa: E402
from features import METEO_COLS, build_features, get_feature_cols  # noqa: E402
from predict import apply_train_tail as apply_predict_train_tail  # noqa: E402
from predict import load_feature_options  # noqa: E402
from train import apply_train_tail, extract_weekly_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw and engineered feature missingness.")
    parser.add_argument(
        "--run-dir",
        default="experiments/20260521_153911_lightgbm_two_stage_lgbm_refit_full_lean_tail1095_20260521",
        help="Experiment run directory whose feature_options should be audited.",
    )
    parser.add_argument(
        "--out",
        default="docs/validation/feature_missingness_audit_20260523.md",
        help="Markdown report path.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON output path.")
    parser.add_argument("--top", type=int, default=40, help="Number of missing columns to show.")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def missing_summary(frame: pd.DataFrame, columns: list[str], top: int) -> tuple[dict, list[dict]]:
    clean = frame[columns].replace([np.inf, -np.inf], np.nan)
    missing = clean.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    rows = [
        {
            "feature": str(feature),
            "missing": int(count),
            "missing_rate": float(count / len(frame)),
        }
        for feature, count in missing.head(top).items()
    ]
    return (
        {
            "rows": int(len(frame)),
            "columns": int(len(columns)),
            "missing_feature_count": int(len(missing)),
            "missing_cell_count": int(missing.sum()),
            "max_missing_rate": float(missing.max() / len(frame)) if len(missing) else 0.0,
        },
        rows,
    )


def feature_options_for_report(feature_options: dict) -> dict:
    keys = [
        "feature_profile",
        "train_tail_days",
        "use_score_history",
        "score_gap_days",
        "max_score_lag_weeks",
        "drop_feature_nan_rows",
        "use_climatology",
        "use_region_stats",
        "drop_feature_groups",
    ]
    return {key: feature_options.get(key) for key in keys}


def build_markdown(metrics: dict, top: int) -> str:
    lines = [
        "# Feature Missingness Audit",
        "",
        f"Generated: {metrics['created_at']}",
        f"Run directory: `{metrics['run_dir']}`",
        "",
        "## Feature Options",
        "",
        "```json",
        json.dumps(metrics["feature_options"], indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| matrix | rows | columns | missing columns | missing cells | max missing rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("raw_train_weather", "raw train weather"),
        ("raw_test_weather", "raw test weather"),
        ("test_last_features", "inference test_last features"),
        ("train_weekly_features", "training weekly features"),
    ]:
        item = metrics[key]["summary"]
        lines.append(
            f"| {label} | {item['rows']:,} | {item['columns']:,} | "
            f"{item['missing_feature_count']:,} | {item['missing_cell_count']:,} | "
            f"{item['max_missing_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top Missing Engineered Train Features",
            "",
            f"Top {top} columns by missing rate on supervised weekly training rows.",
            "",
            "| feature | missing | missing rate |",
            "|---|---:|---:|",
        ]
    )
    for row in metrics["train_weekly_features"]["top_missing"]:
        lines.append(f"| `{row['feature']}` | {row['missing']:,} | {row['missing_rate']:.4f} |")
    if not metrics["train_weekly_features"]["top_missing"]:
        lines.append("| none | 0 | 0.0000 |")

    lines.extend(
        [
            "",
            "## Top Missing Inference Features",
            "",
            "| feature | missing | missing rate |",
            "|---|---:|---:|",
        ]
    )
    for row in metrics["test_last_features"]["top_missing"]:
        lines.append(f"| `{row['feature']}` | {row['missing']:,} | {row['missing_rate']:.4f} |")
    if not metrics["test_last_features"]["top_missing"]:
        lines.append("| none | 0 | 0.0000 |")

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Raw weather columns have no missing values in the current train/test files.",
            "- Most engineered missingness comes from warmup windows: long score-history lags and long meteorological lags at the beginning of each region history.",
            "- The final Kaggle-style inference row per region should have no missing values; `src/predict.py` now audits and fails fast if that changes.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    out_path = resolve_path(args.out)
    json_path = resolve_path(args.json_out) if args.json_out else out_path.with_suffix(".json")

    feature_options = load_feature_options(run_dir)
    usecols = ["region_id", "date"] + METEO_COLS
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols + ["score"])
    test = pd.read_csv(ROOT / "data" / "test.csv", usecols=usecols)

    raw_train_summary, raw_train_top = missing_summary(train, usecols, args.top)
    raw_test_summary, raw_test_top = missing_summary(test, usecols, args.top)

    test_with_score = test.copy()
    test_with_score["score"] = np.nan
    train_ref = apply_predict_train_tail(train, int(feature_options.get("train_tail_days", 0) or 0))
    train_recent = train_ref.groupby("region_id").tail(750).copy()
    combined = pd.concat([train_recent, test_with_score], ignore_index=True)
    test_feature_frame = build_features(
        combined,
        train_ref,
        is_train=False,
        use_score_history=bool(feature_options.get("use_score_history", False)),
        score_gap_days=int(feature_options.get("score_gap_days", 91)),
        use_climatology=bool(feature_options.get("use_climatology", True)),
        use_region_stats=bool(feature_options.get("use_region_stats", False)),
        feature_profile=str(feature_options.get("feature_profile", "micro")),
        max_score_lag_weeks=feature_options.get("max_score_lag_weeks"),
        drop_feature_groups=feature_options.get("drop_feature_groups", []),
    )
    test_dates = set(test["date"].unique())
    test_rows = test_feature_frame[test_feature_frame["date"].isin(test_dates)].copy()
    test_last = test_rows.sort_values(["region_id", "date"]).groupby("region_id").last().reset_index()
    test_feature_cols = get_feature_cols(test_rows)
    test_summary, test_top = missing_summary(test_last, test_feature_cols, args.top)

    train_tail = apply_train_tail(train, int(feature_options.get("train_tail_days", 0) or 0))
    train_feature_frame = build_features(
        train_tail,
        train_tail,
        is_train=True,
        use_score_history=bool(feature_options.get("use_score_history", False)),
        score_gap_days=int(feature_options.get("score_gap_days", 91)),
        use_climatology=bool(feature_options.get("use_climatology", True)),
        use_region_stats=bool(feature_options.get("use_region_stats", False)),
        feature_profile=str(feature_options.get("feature_profile", "micro")),
        max_score_lag_weeks=feature_options.get("max_score_lag_weeks"),
        drop_feature_groups=feature_options.get("drop_feature_groups", []),
    )
    weekly = extract_weekly_labels(train_feature_frame)
    train_feature_cols = get_feature_cols(train_feature_frame)
    merged = weekly[["region_id", "date", "target_w1"]].merge(
        train_feature_frame[["region_id", "date"] + train_feature_cols],
        on=["region_id", "date"],
        how="left",
    )
    train_summary, train_top = missing_summary(merged, train_feature_cols, args.top)

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "feature_options": feature_options_for_report(feature_options),
        "raw_train_weather": {"summary": raw_train_summary, "top_missing": raw_train_top},
        "raw_test_weather": {"summary": raw_test_summary, "top_missing": raw_test_top},
        "test_last_features": {"summary": test_summary, "top_missing": test_top},
        "train_weekly_features": {"summary": train_summary, "top_missing": train_top},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(metrics, args.top), encoding="utf-8")
    save_json(json_path, metrics)
    print(f"Markdown report saved -> {out_path}")
    print(f"JSON metrics saved -> {json_path}")


if __name__ == "__main__":
    main()
