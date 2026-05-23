#!/usr/bin/env python3
"""Build a GRU-family consensus by minimizing origin-level worst-case risk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]
CANDIDATES = {
    "stack": {
        "name": "gru_family_constrained_stack_secondary",
        "path": "submissions/submission_20260522_gru_family_constrained_stack.csv",
        "loo_col": "stack_mae",
        "anchor": 0.45,
        "cap": 0.60,
    },
    "raw": {
        "name": "raw_gru_primary_probe",
        "path": "submissions/submission_20260521_193438_20260521_190454_group3_ar_gru_group3_ar_gru_tail1825_10ep_20260521_group3_ar_gru.csv",
        "loo_col": "raw_gru_mae",
        "anchor": 0.35,
        "cap": 0.60,
    },
    "affine": {
        "name": "affine_calibrated_gru_tertiary",
        "path": "submissions/submission_20260522_gru_affine_calibrated.csv",
        "loo_col": "affine_calibrated_mae",
        "anchor": 0.15,
        "cap": 0.35,
    },
    "calendar": {
        "name": "calendar_gated_gru_secondary_high_variance",
        "path": "submissions/submission_20260522_calendar_regime_gated_gru_loo_robust.csv",
        "loo_col": "calendar_gated_mae",
        "anchor": 0.05,
        "cap": 0.20,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = ["region_id", *PRED_COLS]
    if list(frame.columns) != expected:
        raise ValueError(f"{path} columns {list(frame.columns)} != {expected}")
    if len(frame) != len(sample):
        raise ValueError(f"{path} rows {len(frame)} != sample rows {len(sample)}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{path} region_id order mismatch")
    if frame[PRED_COLS].isna().any().any():
        raise ValueError(f"{path} has NaN predictions")
    return frame


def simplex_grid(names: list[str], step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--grid-step must evenly divide 1.0")
    out: list[dict[str, float]] = []
    for raw in product(range(units + 1), repeat=len(names) - 1):
        total = sum(raw)
        if total > units:
            continue
        values = list(raw) + [units - total]
        weights = {name: value / units for name, value in zip(names, values)}
        if all(weights[name] <= CANDIDATES[name]["cap"] + 1e-12 for name in names):
            out.append(weights)
    return out


def score_weights(weights: dict[str, float], loo_rows: list[dict[str, str]], lambda_anchor: float) -> dict[str, object]:
    anchor_penalty = sum((weights[name] - float(CANDIDATES[name]["anchor"])) ** 2 for name in weights)
    origin_rows = []
    weighted_values = []
    for row in loo_rows:
        value = 0.0
        components = {}
        for name, weight in weights.items():
            component = float(row[str(CANDIDATES[name]["loo_col"])])
            components[f"{name}_mae"] = component
            value += weight * component
        weighted_values.append(value)
        origin_rows.append({"holdout_origin": row["holdout_origin"], "weighted_mae_proxy": value, **components})

    values = np.array(weighted_values, dtype=np.float64)
    worst = float(values.max())
    mean = float(values.mean())
    std = float(values.std())
    objective = worst + 0.25 * mean + 0.10 * std + lambda_anchor * anchor_penalty
    return {
        "objective": objective,
        "worst_origin_proxy": worst,
        "mean_origin_proxy": mean,
        "std_origin_proxy": std,
        "anchor_penalty": anchor_penalty,
        "origin_rows": origin_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo-scores", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.02)
    args = parser.parse_args()

    with args.loo_scores.open(newline="") as file:
        loo_rows = list(csv.DictReader(file))

    names = list(CANDIDATES)
    best = None
    for weights in simplex_grid(names, args.grid_step):
        scored = score_weights(weights, loo_rows, args.lambda_anchor)
        if best is None or scored["objective"] < best["score"]["objective"]:
            best = {"weights": weights, "score": scored}
    if best is None:
        raise ValueError("No feasible minimax weights found")

    sample = pd.read_csv(args.sample_submission)
    frames = {name: load_submission(Path(str(CANDIDATES[name]["path"])), sample) for name in names}
    output = sample[["region_id"]].copy()
    for col in PRED_COLS:
        output[col] = sum(frames[name][col] * best["weights"][name] for name in names).clip(0, 5)

    args.out_submission.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_submission, index=False)
    digest = sha256(args.out_submission)
    component_deltas = {}
    for name, frame in frames.items():
        delta = (output[PRED_COLS] - frame[PRED_COLS]).abs()
        component_deltas[name] = {
            "mean_abs_delta": float(delta.to_numpy().mean()),
            "max_abs_delta": float(delta.to_numpy().max()),
        }

    report = {
        "hypothesis": "A capped minimax GRU consensus may reduce pseudo-private worst-origin risk compared with pooled-MAE or softmax-risk weighting.",
        "weights": best["weights"],
        "score": best["score"],
        "candidate_metadata": CANDIDATES,
        "component_deltas": component_deltas,
        "sanity": {
            "submission_path": str(args.out_submission),
            "sha256": digest,
            "sha12": digest[:12],
            "rows": int(len(output)),
            "columns_ok": list(output.columns) == ["region_id", *PRED_COLS],
            "region_order_ok": output["region_id"].equals(sample["region_id"]),
            "nan_count": int(output[PRED_COLS].isna().sum().sum()),
            "min_prediction": float(output[PRED_COLS].min().min()),
            "max_prediction": float(output[PRED_COLS].max().max()),
            "mean_by_horizon": {col: float(output[col].mean()) for col in PRED_COLS},
        },
        "recommendation": "Use only as a deferred minimax hedge after stack/raw GRU receive public feedback; this is proxy-optimized, not true row-level OOF optimized.",
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"submission": str(args.out_submission), "sha12": digest[:12], "weights": best["weights"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
