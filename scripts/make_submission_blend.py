#!/usr/bin/env python3
"""Create a convex blend from two or more Kaggle submission CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_named_paths(raw: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        path = Path(value.strip())
        parsed[name.strip()] = path if path.is_absolute() else ROOT / path
    if len(parsed) < 2:
        raise ValueError("--inputs must contain at least two name=path entries.")
    return parsed


def parse_weights(raw: str, names: list[str], n_horizons: int) -> dict[str, list[float]]:
    weights: dict[str, list[float]] = {}
    for part in raw.split(";"):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        values = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
        if len(values) == 1:
            values *= n_horizons
        if len(values) != n_horizons:
            raise ValueError(f"{name} has {len(values)} weights; expected 1 or {n_horizons}.")
        weights[name.strip()] = values

    missing = [name for name in names if name not in weights]
    extra = [name for name in weights if name not in names]
    if missing:
        raise ValueError(f"--weights is missing entries for: {', '.join(missing)}")
    if extra:
        raise ValueError(f"--weights has unknown entries: {', '.join(extra)}")

    for idx in range(n_horizons):
        total = sum(weights[name][idx] for name in names)
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"Weights for horizon {idx + 1} sum to {total:.8f}; expected 1.0.")
    return weights


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Comma-separated name=csv files.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Semicolon-separated weights, e.g. lgb=0.35;xgb=0.35;cat=0.30. Values may be per-horizon CSVs.",
    )
    parser.add_argument("--out", required=True, help="Output submission CSV.")
    parser.add_argument("--sample", default="data/sample_submission.csv", help="Sample submission for row/order checks.")
    parser.add_argument("--out-json", default=None, help="Optional JSON metadata path.")
    args = parser.parse_args()

    input_paths = parse_named_paths(args.inputs)
    frames = {}
    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frames[name] = pd.read_csv(path)

    names = list(frames)
    first = frames[names[0]]
    pred_cols = [col for col in first.columns if col.startswith("pred_week")]
    if not pred_cols:
        raise ValueError("No pred_week columns found.")
    weights = parse_weights(args.weights, names, len(pred_cols))

    sample_path = Path(args.sample)
    sample_path = sample_path if sample_path.is_absolute() else ROOT / sample_path
    sample = pd.read_csv(sample_path)

    for name, frame in frames.items():
        if list(frame.columns) != list(first.columns):
            raise ValueError(f"{name} columns differ from first input.")
        if len(frame) != len(first):
            raise ValueError(f"{name} row count differs from first input.")
        if not frame["region_id"].equals(first["region_id"]):
            raise ValueError(f"{name} region order differs from first input.")
        if frame[pred_cols].isna().any().any():
            raise ValueError(f"{name} contains NaN predictions.")

    if len(first) != len(sample) or not first["region_id"].equals(sample["region_id"]):
        raise ValueError("Input region order does not match sample submission.")

    blended = first[["region_id"]].copy()
    for idx, col in enumerate(pred_cols):
        blended[col] = sum(frames[name][col] * weights[name][idx] for name in names).clip(0, 5)

    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out_path, index=False)

    stats = {
        "out": str(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path),
        "sha12": sha12(out_path),
        "rows": int(len(blended)),
        "prediction_min": float(blended[pred_cols].min().min()),
        "prediction_max": float(blended[pred_cols].max().max()),
        "prediction_mean": float(blended[pred_cols].mean().mean()),
        "inputs": {name: str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) for name, path in input_paths.items()},
        "input_sha12": {name: sha12(path) for name, path in input_paths.items()},
        "weights": weights,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json = out_json if out_json.is_absolute() else ROOT / out_json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
