#!/usr/bin/env python3
"""Fit constrained blend weights from OOF or blind-backtest predictions."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blend import (  # noqa: E402
    apply_weights,
    bootstrap_constrained_weights,
    fit_constrained_weights,
    parse_anchor,
    parse_caps,
    parse_named_paths,
)
from experiment_utils import save_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Fit validation-driven constrained blend weights.")
    parser.add_argument("--preds", required=True, help="Comma-separated name=csv prediction files.")
    parser.add_argument("--target", required=True, help="CSV with target_week or target_w columns.")
    parser.add_argument("--anchor", default=None, help="Semicolon anchor, e.g. lgb=0.35;xgb=0.35;cat=0.30.")
    parser.add_argument("--caps", default=None, help="Optional semicolon caps, e.g. lgbm_micro=0.15.")
    parser.add_argument("--lambda-reg", type=float, default=0.05)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap draws for weight stability. 0 disables.")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-json", default="experiments/blend_weights.json")
    parser.add_argument("--out-preds", default=None, help="Optional blended prediction CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = parse_named_paths(args.preds)
    frames = {}
    for name, path in paths.items():
        print(f"Loading {name}: {path}")
        frames[name] = pd.read_csv(ROOT / path if not Path(path).is_absolute() else path)
    target_path = Path(args.target)
    target = pd.read_csv(ROOT / target_path if not target_path.is_absolute() else target_path)

    model_names = list(frames)
    anchor = parse_anchor(args.anchor, model_names)
    caps = parse_caps(args.caps, model_names)
    weights, metrics = fit_constrained_weights(
        frames,
        target,
        anchor,
        grid_step=args.grid_step,
        lambda_reg=args.lambda_reg,
        caps=caps,
    )
    bootstrap = None
    if args.bootstrap > 0:
        bootstrap = bootstrap_constrained_weights(
            frames,
            target,
            anchor,
            grid_step=args.grid_step,
            lambda_reg=args.lambda_reg,
            caps=caps,
            n_bootstrap=args.bootstrap,
            random_state=args.bootstrap_seed,
        )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_names": model_names,
        "anchor": anchor,
        "caps": caps,
        "lambda_reg": args.lambda_reg,
        "grid_step": args.grid_step,
        "weights": weights,
        "metrics": metrics,
        "bootstrap": bootstrap,
    }

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    save_json(out_json, payload)
    print(f"Weights saved -> {out_json}")

    if args.out_preds:
        blended = apply_weights(frames, weights)
        out_preds = Path(args.out_preds)
        if not out_preds.is_absolute():
            out_preds = ROOT / out_preds
        out_preds.parent.mkdir(parents=True, exist_ok=True)
        blended.to_csv(out_preds, index=False)
        print(f"Blended predictions saved -> {out_preds}")

    print("\nWeights:")
    pred_cols = [col for col in next(iter(frames.values())).columns if col.startswith("pred_week")]
    for idx, pred_col in enumerate(pred_cols):
        line = ", ".join(f"{name}={weights[name][idx]:.2f}" for name in model_names)
        print(f"  {pred_col}: {line}  MAE={metrics['mae_by_horizon'][pred_col]:.5f}")
        if bootstrap:
            stability = ", ".join(
                f"{name}={bootstrap['summary'][pred_col][name]['mean']:.2f}"
                f"+/-{bootstrap['summary'][pred_col][name]['std']:.2f}"
                for name in model_names
            )
            print(f"    bootstrap: {stability}")


if __name__ == "__main__":
    main()
