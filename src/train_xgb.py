"""
train.py
Two-stage LightGBM training:
  Stage 1: Train a score reconstructor (meteo features → weekly score).
           Used at inference to fill score-history features for test data
           without leaking real labels.
  Stage 2: Train one LightGBM model per horizon (week 1–5) using the full
           feature set (meteo + score history + calendar).

Usage:
    python3 src/train.py
    python3 src/train.py --experiment-name lgbm_v1
"""
import argparse
import pickle
import warnings
import gc
from datetime import datetime
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import build_features, get_feature_cols, METEO_COLS, reduce_mem_usage
from experiment_utils import create_run_dir, save_json, write_latest_run

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FAMILY = "xgboost_two_stage"

# ── LightGBM config ────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "reg:absoluteerror",
    "eval_metric":      "mae",
    "max_depth":        7,
    "learning_rate":    0.05,
    "colsample_bytree": 0.8,
    "subsample":        0.8,
    "n_estimators":     3000,
    "n_jobs":           -1,
    "random_state":     42,
    "tree_method":      "hist",
    "early_stopping_rounds": 100,
}

N_WEEKS = 5       # predict weeks 1–5
N_FOLDS = 5       # time-series CV folds


def parse_args():
    parser = argparse.ArgumentParser(description="Train disaster severity models.")
    parser.add_argument(
        "--experiment-name",
        default="current",
        help="Readable name appended to experiments/<run_id>/.",
    )
    parser.add_argument(
        "--model-family",
        default=MODEL_FAMILY,
        help="Model family label for experiment tracking.",
    )
    return parser.parse_args()


def load_data():
    print("Loading train.csv ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    train = reduce_mem_usage(train)
    print(f"  train shape: {train.shape}")
    return train


def extract_weekly_labels(train: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows with non-NaN scores (one per week per region).
    Each row becomes one training sample; we attach the NEXT 5 weekly scores
    as multi-output targets.
    """
    # Sort and index weekly scores
    weekly = (
        train.dropna(subset=["score"])
        .sort_values(["region_id", "date"])
        .reset_index(drop=True)
    )

    # For each week row, attach score_week1 … score_week5 (future targets)
    for w in range(1, N_WEEKS + 1):
        weekly[f"target_w{w}"] = (
            weekly.groupby("region_id")["score"].shift(-w)
        )

    # Drop rows where any future target is missing (last N_WEEKS of train)
    target_cols = [f"target_w{w}" for w in range(1, N_WEEKS + 1)]
    weekly = weekly.dropna(subset=target_cols).reset_index(drop=True)
    print(f"  Weekly label rows (with all 5 future targets): {len(weekly)}")
    return weekly


def time_series_cv_split(df: pd.DataFrame, n_splits: int = 5):
    """
    Yield (train_idx, val_idx) for time-ordered splits per region.
    Validation is always the last chunk.
    """
    regions = df["region_id"].unique()
    # Use global date ordering
    dates  = df["date"].sort_values().unique()
    fold_size = len(dates) // (n_splits + 1)

    for fold in range(n_splits):
        cutoff = dates[fold_size * (fold + 1)]
        train_mask = df["date"] < cutoff
        val_mask   = (df["date"] >= cutoff) & (df["date"] < dates[min(
            fold_size * (fold + 2), len(dates) - 1
        )])
        yield df[train_mask].index, df[val_mask].index


def train_one_horizon(
    feature_df: pd.DataFrame,
    weekly_df:  pd.DataFrame,
    week: int,
) -> xgb.XGBRegressor:
    """Train a single XGBoost model for horizon `week`."""
    print(f"\n  --- Training horizon: week {week} ---")
    target_col = f"target_w{week}"

    # Merge features onto weekly label rows
    feat_cols = get_feature_cols(feature_df)
    merged = weekly_df[["region_id", "date", target_col]].merge(
        feature_df[["region_id", "date"] + feat_cols],
        on=["region_id", "date"],
        how="left",
    )

    X = merged[feat_cols]
    y = merged[target_col]

    # Chronological train/val split (last 20% of dates as val)
    dates = merged["date"].sort_values().unique()
    val_cutoff = dates[-int(len(dates) * 0.2)]
    
    train_mask = merged["date"] < val_cutoff
    val_mask = merged["date"] >= val_cutoff
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val   = X[val_mask]
    y_val   = y[val_mask]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )

    val_pred = model.predict(X_val)
    val_mae  = mean_absolute_error(y_val, val_pred)
    print(f"  Val MAE (week {week}): {val_mae:.4f}")

    return model, val_mae




def main():
    args = parse_args()
    run_dir = create_run_dir(args.model_family, args.experiment_name)

    print("=" * 60)
    print("  Natural Disaster Severity Prediction — Training (Optimized)")
    print("=" * 60)
    print(f"Experiment run directory: {run_dir}")

    save_json(
        run_dir / "config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_family": args.model_family,
            "experiment_name": args.experiment_name,
            "n_weeks": N_WEEKS,
            "validation_strategy": "chronological_holdout_last_20_percent",
            "meteorological_columns": METEO_COLS,
            "xgboost_params": XGB_PARAMS,
            "pipeline": [
                "load train.csv",
                "build temporal features",
                "train one direct XGBoost model per horizon",
                "save versioned models and metrics",
            ],
        },
    )

    # 1. Load data
    train = load_data()
    train_shape = train.shape

    from features import (
        reduce_mem_usage, add_calendar_features, add_meteo_features,
        add_region_stats, build_features
    )

    # 2. Basic Feature engineering
    print("\nBuilding features ...")
    feat_df = build_features(train, train, is_train=True)
    del train
    gc.collect()

    # 3. Extract weekly training rows
    print("\nExtracting weekly labels ...")
    weekly_df = extract_weekly_labels(feat_df)

    # 5. Stage 2: Train one model per horizon
    models   = {}
    val_maes = {}
    for week in range(1, N_WEEKS + 1):
        model, val_mae   = train_one_horizon(feat_df, weekly_df, week)
        models[week]     = model
        val_maes[week]   = val_mae

    # 6. Save horizon models
    model_path = run_dir / "models" / "lgbm_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nHorizon models saved → {model_path}")

    legacy_model_path = MODEL_DIR / "lgbm_models.pkl"
    with open(legacy_model_path, "wb") as f:
        pickle.dump(models, f)

    # Summary
    print("\n--- Validation MAE per horizon ---")
    for w, mae in val_maes.items():
        print(f"  Week {w}: {mae:.4f}")
    avg = np.mean(list(val_maes.values()))
    print(f"  Average: {avg:.4f}")
    metrics = {
        "model_family": args.model_family,
        "experiment_name": args.experiment_name,
        "train_shape": train_shape,
        "feature_columns": len(get_feature_cols(feat_df)),
        "weekly_label_rows": len(weekly_df),
        "horizon_val_mae": {f"week_{w}": mae for w, mae in val_maes.items()},
        "average_val_mae": avg,
        "model_paths": {
            "run_horizon_models": model_path,
            "latest_horizon_models": legacy_model_path,
        },
    }
    save_json(run_dir / "metrics.json", metrics)
    write_latest_run(run_dir)
    print(f"Metrics saved → {run_dir / 'metrics.json'}")
    print(f"Latest run pointer updated → {run_dir.name}")
    print("\nDone! Run src/predict.py to generate submission.")


if __name__ == "__main__":
    main()
