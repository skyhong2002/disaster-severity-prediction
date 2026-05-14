"""
predict.py
Two-stage inference pipeline:
  Stage 1: Reconstruct weekly scores for the test 91-day window using
            meteorological features (no score leakage).
  Stage 2: Use reconstructed scores as score-history features to predict
            the next 5 weekly scores.

Usage:
    python3 src/predict.py
    python3 src/predict.py --run-dir experiments/<run_id>
Output:
    submissions/submission_<timestamp>.csv
"""
import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features, get_feature_cols
from experiment_utils import get_latest_run_dir, save_json

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUB_DIR   = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)

N_WEEKS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment run directory. Defaults to experiments/latest.txt.",
    )
    return parser.parse_args()


def resolve_run_dir(run_dir_arg: str | None) -> Path | None:
    if run_dir_arg:
        path = Path(run_dir_arg)
        return path if path.is_absolute() else ROOT / path
    return get_latest_run_dir()


def model_dir_for_run(run_dir: Path | None) -> Path:
    if run_dir is not None and (run_dir / "models" / "lgbm_models.pkl").exists():
        return run_dir / "models"
    return MODEL_DIR


def load_models(model_dir: Path):
    model_path = model_dir / "lgbm_models.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} horizon models from {model_path}")
    return models


def model_feature_columns(models: dict, fallback_cols: list[str]) -> list[str]:
    """Use persisted model feature names when available for legacy compatibility."""
    first_model = models[min(models.keys())]
    names = getattr(first_model, "feature_name_", None)
    if names is None:
        names = getattr(first_model, "feature_names_in_", None)
    return list(names) if names is not None else fallback_cols


def load_feature_options(run_dir: Path | None) -> dict:
    """Read feature-engineering options recorded by train.py."""
    defaults = {
        "use_score_history": False,
        "score_gap_days": 91,
        "use_climatology": True,
        "use_region_stats": True,
        "feature_profile": "full",
    }
    if run_dir is None:
        return defaults
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return defaults
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {**defaults, **config.get("feature_options", {})}






def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    model_dir = model_dir_for_run(run_dir)

    print("=" * 60)
    print("  Natural Disaster Severity Prediction — Inference")
    print("=" * 60)
    if run_dir:
        print(f"Experiment run directory: {run_dir}")
    print(f"Model directory: {model_dir}")
    feature_options = load_feature_options(run_dir)
    print(f"Feature options: {feature_options}")

    # ── 1. Load raw data ──────────────────────────────────────────
    print("\nLoading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["date"])
    sub_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    # ── 2. Build features for test window ────────────
    print("\n[Stage 1] Building features for forecast ...")
    test["score"] = np.nan
    
    # Prevent OOM by only concatenating the last 750 days of train for lag calculations
    # 'date' column has format 'XXXX-09-18' so Timedelta fails. Since each region has 1 row/day, use tail(750)
    train_recent = train.groupby("region_id").tail(750).copy()
    combined = pd.concat([train_recent, test], ignore_index=True)
    
    from features import build_features
    combined_feat = build_features(
        combined,
        train,
        is_train=False,
        use_score_history=feature_options.get("use_score_history", False),
        score_gap_days=int(feature_options.get("score_gap_days", 91)),
        use_climatology=feature_options.get("use_climatology", True),
        use_region_stats=feature_options.get("use_region_stats", True),
        feature_profile=feature_options.get("feature_profile", "full"),
    )
    
    test_dates = test["date"].unique()
    test_feat = combined_feat[combined_feat["date"].isin(test_dates)].copy()

    # For each region, take the LAST row of the 91-day test window
    test_last = (
        test_feat
        .sort_values(["region_id", "date"])
        .groupby("region_id")
        .last()
        .reset_index()
    )

    print("\n[Stage 2] Forecasting weeks 1–5 ...")
    models    = load_models(model_dir)
    fallback_cols = [c for c in get_feature_cols(test_feat) if c in test_last.columns]
    expected_cols = model_feature_columns(models, fallback_cols)
    missing_cols = [c for c in expected_cols if c not in test_last.columns]
    if missing_cols:
        raise ValueError(f"Missing {len(missing_cols)} model feature columns, e.g. {missing_cols[:10]}")
    feat_cols = expected_cols
    X_test    = test_last[feat_cols]

    preds = {}
    for week in range(1, N_WEEKS + 1):
        raw = models[week].predict(X_test)
        preds[f"pred_week{week}"] = np.clip(raw, 0, 5)
        print(f"  Week {week}: mean={preds[f'pred_week{week}'].mean():.3f}")

    # ── 5. Build and save submission ──────────────────────────────
    result = test_last[["region_id"]].copy()
    for week in range(1, N_WEEKS + 1):
        result[f"pred_week{week}"] = preds[f"pred_week{week}"]

    submission = sub_template[["region_id"]].merge(result, on="region_id", how="left")
    pred_cols  = [f"pred_week{w}" for w in range(1, N_WEEKS + 1)]
    submission[pred_cols] = submission[pred_cols].fillna(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = run_dir.name if run_dir else "legacy"
    out_path = SUB_DIR / f"submission_{ts}_{run_suffix}.csv"
    submission.to_csv(out_path, index=False)

    run_submission_path = None
    if run_dir:
        run_submission_path = run_dir / "submissions" / out_path.name
        run_submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(run_submission_path, index=False)

    print(f"\nSubmission saved → {out_path}")
    if run_submission_path:
        print(f"Run submission saved → {run_submission_path}")
    print(submission.head(10).to_string(index=False))

    print("\nPrediction statistics:")
    pred_stats = {}
    for col in pred_cols:
        pred_stats[col] = {
            "mean": submission[col].mean(),
            "std": submission[col].std(),
            "min": submission[col].min(),
            "max": submission[col].max(),
        }
        print(f"  {col}: mean={submission[col].mean():.3f}, "
              f"std={submission[col].std():.3f}, "
              f"min={submission[col].min():.3f}, "
              f"max={submission[col].max():.3f}")

    if run_dir:
        save_json(
            run_dir / "submission_metadata.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_dir": model_dir,
                "global_submission_path": out_path,
                "run_submission_path": run_submission_path,
                "rows": len(submission),
                "prediction_stats": pred_stats,
            },
        )
        print(f"Submission metadata saved → {run_dir / 'submission_metadata.json'}")


if __name__ == "__main__":
    main()
