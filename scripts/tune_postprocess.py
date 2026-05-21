#!/usr/bin/env python3
"""Fit and apply simple validation-driven prediction postprocessing."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
N_WEEKS = 5


def parse_grid(raw: str) -> list[float]:
    """Parse ``start:stop:step`` inclusive-ish or comma-separated values."""
    raw = raw.strip()
    if ":" not in raw:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    start, stop, step = (float(part) for part in raw.split(":"))
    values = []
    current = start
    while current <= stop + step / 2:
        values.append(round(current, 10))
        current += step
    return values


def fit_horizon_affine(
    rows: pd.DataFrame,
    scales: list[float],
    biases: list[float],
    lambda_reg: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Fit independent clipped affine transforms per horizon."""
    params: dict[str, dict[str, float]] = {}
    metrics: dict[str, float] = {}
    baseline_errors = []
    tuned_errors = []

    for week in range(1, N_WEEKS + 1):
        pred_col = f"pred_week{week}"
        target_col = f"target_w{week}"
        y = pd.to_numeric(rows[target_col], errors="coerce").to_numpy(dtype=np.float64)
        pred = pd.to_numeric(rows[pred_col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(pred)
        y = y[mask]
        pred = pred[mask]

        base_mae = mean_absolute_error(y, pred)
        baseline_errors.append(np.abs(pred - y))
        best = (float("inf"), 1.0, 0.0, base_mae)
        for scale in scales:
            for bias in biases:
                adjusted = np.clip(pred * scale + bias, 0.0, 5.0)
                mae = mean_absolute_error(y, adjusted)
                objective = mae + lambda_reg * ((scale - 1.0) ** 2 + bias**2)
                if objective < best[0]:
                    best = (objective, scale, bias, mae)

        _, scale, bias, tuned_mae = best
        adjusted = np.clip(pred * scale + bias, 0.0, 5.0)
        tuned_errors.append(np.abs(adjusted - y))
        params[pred_col] = {"scale": float(scale), "bias": float(bias)}
        metrics[f"{pred_col}_baseline_mae"] = float(base_mae)
        metrics[f"{pred_col}_tuned_mae"] = float(tuned_mae)

    metrics["baseline_overall_mae"] = float(np.concatenate(baseline_errors).mean())
    metrics["tuned_overall_mae"] = float(np.concatenate(tuned_errors).mean())
    metrics["overall_delta"] = metrics["tuned_overall_mae"] - metrics["baseline_overall_mae"]
    return params, metrics


def apply_horizon_affine(submission: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Apply clipped horizon-wise affine calibration to a submission-like frame."""
    adjusted = submission.copy()
    for week in range(1, N_WEEKS + 1):
        col = f"pred_week{week}"
        scale = float(params[col]["scale"])
        bias = float(params[col]["bias"])
        adjusted[col] = np.clip(pd.to_numeric(adjusted[col], errors="coerce") * scale + bias, 0.0, 5.0)
    return adjusted


def submission_stats(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for week in range(1, N_WEEKS + 1):
        col = f"pred_week{week}"
        values = pd.to_numeric(frame[col], errors="coerce")
        stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune horizon-wise postprocessing from blind rows.")
    parser.add_argument("--blind-rows", required=True, help="CSV with pred_week* and target_w* columns.")
    parser.add_argument("--submission", default=None, help="Optional submission CSV to transform.")
    parser.add_argument("--scale-grid", default="0.75:1.10:0.025")
    parser.add_argument("--bias-grid", default="-0.30:0.15:0.025")
    parser.add_argument("--lambda-reg", type=float, default=0.02)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-submission", default=None)
    args = parser.parse_args()

    blind_path = Path(args.blind_rows)
    if not blind_path.is_absolute():
        blind_path = ROOT / blind_path
    rows = pd.read_csv(blind_path)
    scales = parse_grid(args.scale_grid)
    biases = parse_grid(args.bias_grid)
    params, metrics = fit_horizon_affine(rows, scales, biases, args.lambda_reg)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "horizon_affine",
        "blind_rows": str(blind_path),
        "scale_grid": scales,
        "bias_grid": biases,
        "lambda_reg": args.lambda_reg,
        "params": params,
        "metrics": metrics,
    }

    if args.submission:
        submission_path = Path(args.submission)
        if not submission_path.is_absolute():
            submission_path = ROOT / submission_path
        submission = pd.read_csv(submission_path)
        adjusted = apply_horizon_affine(submission, params)
        out_submission = Path(args.out_submission) if args.out_submission else (
            ROOT / "submissions" / f"{submission_path.stem}_postprocessed.csv"
        )
        if not out_submission.is_absolute():
            out_submission = ROOT / out_submission
        out_submission.parent.mkdir(parents=True, exist_ok=True)
        adjusted.to_csv(out_submission, index=False)
        payload["submission"] = str(submission_path)
        payload["out_submission"] = str(out_submission)
        payload["submission_stats_before"] = submission_stats(submission)
        payload["submission_stats_after"] = submission_stats(adjusted)

    out_json = Path(args.out_json) if args.out_json else ROOT / "experiments" / "postprocess_tuning.json"
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Postprocess config saved -> {out_json}")
    print(f"Baseline MAE: {metrics['baseline_overall_mae']:.5f}")
    print(f"Tuned MAE:    {metrics['tuned_overall_mae']:.5f}")
    print(f"Delta:        {metrics['overall_delta']:.5f}")
    for week in range(1, N_WEEKS + 1):
        col = f"pred_week{week}"
        print(
            f"  {col}: scale={params[col]['scale']:.3f}, bias={params[col]['bias']:.3f}, "
            f"MAE {metrics[col + '_baseline_mae']:.5f}->{metrics[col + '_tuned_mae']:.5f}"
        )
    if "out_submission" in payload:
        print(f"Adjusted submission saved -> {payload['out_submission']}")


if __name__ == "__main__":
    main()
