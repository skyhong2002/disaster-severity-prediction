"""
train.py
Train LightGBM models (one per prediction horizon week 1–5).
Saves models to models/ directory.

Usage:
    python3 src/train.py
"""
import os
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from features import build_features, get_feature_cols

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── LightGBM config ────────────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":        "regression_l1",   # optimize MAE directly
    "metric":           "mae",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "min_child_samples": 20,
    "n_estimators":     3000,
    "early_stopping_rounds": 100,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}

N_WEEKS = 5       # predict weeks 1–5
N_FOLDS = 5       # time-series CV folds


def load_data():
    print("Loading train.csv ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
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
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model for horizon `week`."""
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

    # Simple time-based train/val split (last 20% as val)
    n_val   = int(len(merged) * 0.2)
    X_train = X.iloc[:-n_val]
    y_train = y.iloc[:-n_val]
    X_val   = X.iloc[-n_val:]
    y_val   = y.iloc[-n_val:]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100, verbose=False)],
    )

    val_pred = model.predict(X_val)
    val_mae  = mean_absolute_error(y_val, val_pred)
    print(f"  Val MAE (week {week}): {val_mae:.4f}")

    return model, val_mae


def main():
    print("=" * 60)
    print("  Natural Disaster Severity Prediction — Training")
    print("=" * 60)

    # 1. Load data
    train = load_data()

    # 2. Feature engineering
    print("\nBuilding features ...")
    feat_df   = build_features(train, train, is_train=True)

    # 3. Extract weekly training rows
    print("\nExtracting weekly labels ...")
    weekly_df = extract_weekly_labels(feat_df)

    # 4. Train one model per horizon
    models   = {}
    val_maes = {}
    for week in range(1, N_WEEKS + 1):
        model, val_mae   = train_one_horizon(feat_df, weekly_df, week)
        models[week]     = model
        val_maes[week]   = val_mae

    # 5. Save models
    model_path = MODEL_DIR / "lgbm_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved → {model_path}")

    # Summary
    print("\n--- Validation MAE per horizon ---")
    for w, mae in val_maes.items():
        print(f"  Week {w}: {mae:.4f}")
    avg = np.mean(list(val_maes.values()))
    print(f"  Average: {avg:.4f}")
    print("\nDone! Run src/predict.py to generate submission.")


if __name__ == "__main__":
    main()
