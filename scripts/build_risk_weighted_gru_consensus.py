#!/usr/bin/env python3
"""Build a risk-weighted GRU-family consensus submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_cols = ["region_id", *PRED_COLS]
    if list(frame.columns) != expected_cols:
        raise ValueError(f"{path} has columns {list(frame.columns)}, expected {expected_cols}")
    if len(frame) != len(sample):
        raise ValueError(f"{path} has {len(frame)} rows, expected {len(sample)}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{path} region_id order does not match sample submission")
    if frame[PRED_COLS].isna().any().any():
        raise ValueError(f"{path} contains NaN predictions")
    return frame


def normalize_weights(scored: list[dict[str, object]], temperature: float, max_weight: float) -> dict[str, float]:
    scores = np.array([float(row["risk_adjusted_score"]) for row in scored], dtype=np.float64)
    shifted = scores - scores.min()
    raw = np.exp(-shifted / temperature)
    weights = raw / raw.sum()

    for _ in range(20):
        overflow = weights > max_weight
        if not overflow.any():
            break
        fixed_total = float(np.sum(np.where(overflow, max_weight, 0.0)))
        free = ~overflow
        if not free.any():
            break
        free_weights = weights[free]
        free_weights = free_weights / free_weights.sum() * (1.0 - fixed_total)
        weights = np.where(overflow, max_weight, weights)
        weights[free] = free_weights

    weights = weights / weights.sum()
    return {str(row["name"]): float(weight) for row, weight in zip(scored, weights)}


def weighted_origin_proxy(
    loo_scores_path: Path,
    candidate_to_column: dict[str, str],
    weights: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with loo_scores_path.open(newline="") as file:
        for row in csv.DictReader(file):
            proxy = 0.0
            components: dict[str, float] = {}
            for name, column in candidate_to_column.items():
                value = float(row[column])
                components[name] = value
                proxy += weights[name] * value
            rows.append(
                {
                    "holdout_origin": row["holdout_origin"],
                    "weighted_mae_proxy": proxy,
                    **{f"{name}_mae": value for name, value in components.items()},
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-ranking", type=Path, required=True)
    parser.add_argument("--loo-scores", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.002)
    parser.add_argument("--max-weight", type=float, default=0.50)
    args = parser.parse_args()

    report = json.loads(args.risk_ranking.read_text())
    scored = sorted(report["ranked_candidates"], key=lambda row: int(row["risk_rank"]))[: args.top_n]
    weights = normalize_weights(scored, args.temperature, args.max_weight)

    sample = pd.read_csv(args.sample_submission)
    frames = {str(row["name"]): load_frame(Path(str(row["path"])), sample) for row in scored}
    output = sample[["region_id"]].copy()
    for col in PRED_COLS:
        values = sum(frames[name][col] * weight for name, weight in weights.items())
        output[col] = values.clip(0, 5)

    args.out_submission.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_submission, index=False)

    component_deltas = {}
    for name, frame in frames.items():
        delta = (output[PRED_COLS] - frame[PRED_COLS]).abs()
        component_deltas[name] = {
            "mean_abs_delta": float(delta.to_numpy().mean()),
            "max_abs_delta": float(delta.to_numpy().max()),
        }

    proxy_rows = weighted_origin_proxy(
        args.loo_scores,
        {
            "gru_family_constrained_stack_secondary": "stack_mae",
            "raw_gru_primary_probe": "raw_gru_mae",
            "affine_calibrated_gru_tertiary": "affine_calibrated_mae",
        },
        weights,
    )
    proxy_values = [float(row["weighted_mae_proxy"]) for row in proxy_rows]
    output_sha = file_sha256(args.out_submission)
    sanity = {
        "submission_path": str(args.out_submission),
        "sha256": output_sha,
        "sha12": output_sha[:12],
        "rows": int(len(output)),
        "columns_ok": list(output.columns) == ["region_id", *PRED_COLS],
        "region_order_ok": output["region_id"].equals(sample["region_id"]),
        "nan_count": int(output[PRED_COLS].isna().sum().sum()),
        "min_prediction": float(output[PRED_COLS].min().min()),
        "max_prediction": float(output[PRED_COLS].max().max()),
        "mean_by_horizon": {col: float(output[col].mean()) for col in PRED_COLS},
    }

    final_report = {
        "hypothesis": "A capped risk-weighted consensus of the top GRU-family candidates can reduce single-postprocess fragility while staying inside the validated GRU prediction family.",
        "inputs": {
            "risk_ranking": str(args.risk_ranking),
            "loo_scores": str(args.loo_scores),
        },
        "weights": weights,
        "component_deltas": component_deltas,
        "origin_weighted_mae_proxy": {
            "mean": float(np.mean(proxy_values)),
            "std": float(np.std(proxy_values)),
            "worst": float(np.max(proxy_values)),
            "rows": proxy_rows,
            "note": "This is a convex combination of component origin MAEs, not true row-level blend MAE.",
        },
        "sanity": sanity,
        "recommendation": "Treat as a conservative fourth-slot or blend-back candidate; do not submit before stack and raw GRU receive public feedback.",
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(final_report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"submission": str(args.out_submission), "sha12": output_sha[:12], "weights": weights}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
