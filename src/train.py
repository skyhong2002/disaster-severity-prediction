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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import (
    SCORE_GAP_DAYS,
    build_features,
    get_feature_cols,
    METEO_COLS,
    reduce_mem_usage,
)
from experiment_utils import create_run_dir, save_json, write_latest_run

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FAMILY = "lightgbm_two_stage"

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
    parser.add_argument(
        "--no-score-history",
        action="store_true",
        help="Disable 91-day-gapped historical score features.",
    )
    parser.add_argument(
        "--score-gap-days",
        type=int,
        default=SCORE_GAP_DAYS,
        help="Blind-window gap before score-history features become available.",
    )
    parser.add_argument(
        "--no-climatology",
        action="store_true",
        help="Disable region-month climatology and anomaly features.",
    )
    parser.add_argument(
        "--use-global-region-stats",
        action="store_true",
        help="Add full-train region score stats. Useful for submissions, but optimistic for local validation.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=0,
        help="Use only weekly training rows from the most recent N days. 0 uses all rows.",
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=0,
        help="Apply exponential sample weights by recency. 0 disables weighting.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["holdout", "rolling_origin"],
        default="holdout",
        help="Validation strategy. rolling_origin simulates multiple forecast origins.",
    )
    parser.add_argument(
        "--rolling-folds",
        type=int,
        default=3,
        help="Number of rolling-origin folds when --validation-mode=rolling_origin.",
    )
    parser.add_argument(
        "--regularized",
        action="store_true",
        help="Use a more conservative LightGBM preset for lower overfit risk.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["micro", "lean", "full"],
        default="micro",
        help="Feature set size. micro is intended for 32GB RAM machines.",
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
    weekly["week_idx"] = weekly.groupby("region_id").cumcount().astype(np.int32)
    weekly["max_week_idx"] = weekly.groupby("region_id")["week_idx"].transform("max").astype(np.int32)

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


def get_lgb_params(regularized: bool = False) -> dict:
    """Return the requested LightGBM parameter preset."""
    params = LGB_PARAMS.copy()
    if regularized:
        params.update(
            {
                "num_leaves": 63,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "min_child_samples": 80,
                "lambda_l1": 0.1,
                "lambda_l2": 2.0,
                "extra_trees": True,
            }
        )
    return params


def apply_recent_filter(merged: pd.DataFrame, recent_days: int) -> pd.Series:
    """Return a mask selecting only recent samples if requested."""
    if recent_days <= 0:
        return pd.Series(True, index=merged.index)
    date_rank = merged["date"].map({d: i for i, d in enumerate(sorted(merged["date"].unique()))})
    max_rank = date_rank.max()
    recent_steps = max(1, int(recent_days / 7))
    return date_rank >= max_rank - recent_steps


def make_recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray | None:
    """Create exponential recency weights for training rows."""
    if half_life_days <= 0:
        return None
    date_rank = dates.map({d: i for i, d in enumerate(sorted(dates.unique()))}).astype(np.float32)
    age_days = (date_rank.max() - date_rank) * 7.0
    weights = np.power(0.5, age_days / half_life_days).astype(np.float32)
    return np.clip(weights / np.mean(weights), 0.05, 20.0)


def validation_masks(merged: pd.DataFrame, mode: str, rolling_folds: int, horizon: int):
    """Yield train/validation masks for the selected validation strategy."""
    if mode == "holdout":
        if {"week_idx", "max_week_idx"}.issubset(merged.columns):
            val_start = (merged["max_week_idx"] * 0.8).astype(np.int32)
            train_mask = merged["week_idx"] + horizon < val_start
            val_mask = merged["week_idx"] >= val_start
            yield train_mask, val_mask, "last_20pct_per_region"
        else:
            dates = np.array(sorted(merged["date"].unique()))
            val_cutoff = dates[-int(len(dates) * 0.2)]
            yield merged["date"] < val_cutoff, merged["date"] >= val_cutoff, str(val_cutoff)
        return

    if not {"week_idx", "max_week_idx"}.issubset(merged.columns):
        raise ValueError("rolling_origin validation requires week_idx and max_week_idx columns.")

    # Validate at the same relative forecast-origin week inside every region.
    # Negative offsets leave room for target_w1..target_w5 and keep fold sizes
    # close to the number of regions instead of depending on synthetic dates.
    offsets = np.linspace(rolling_folds + 4, 5, rolling_folds).round().astype(int)
    for offset in offsets:
        val_idx = merged["max_week_idx"] - offset
        # Purge training rows whose future target would fall on or after the
        # validation origin. Without this, horizon models can train on labels
        # from the validation future via nearby earlier origins.
        train_mask = merged["week_idx"] + horizon < val_idx
        val_mask = merged["week_idx"] == val_idx
        yield train_mask, val_mask, f"relative_week_-{offset}"


def train_one_horizon(
    feature_df: pd.DataFrame,
    weekly_df:  pd.DataFrame,
    week: int,
    lgb_params: dict,
    validation_mode: str,
    rolling_folds: int,
    recent_days: int,
    recency_half_life_days: float,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model for horizon `week`."""
    print(f"\n  --- Training horizon: week {week} ---")
    target_col = f"target_w{week}"

    # Merge features onto weekly label rows
    feat_cols = get_feature_cols(feature_df)
    index_cols = ["region_id", "date", "week_idx", "max_week_idx", target_col]
    merged = weekly_df[index_cols].merge(
        feature_df[["region_id", "date"] + feat_cols],
        on=["region_id", "date"],
        how="left",
    )

    X = merged[feat_cols]
    y = merged[target_col]
    usable_mask = apply_recent_filter(merged, recent_days)

    fold_maes = []
    final_model = None
    for train_mask, val_mask, fold_name in validation_masks(merged, validation_mode, rolling_folds, week):
        train_mask = train_mask & usable_mask
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        print(f"  Fold {fold_name}: train_rows={int(train_mask.sum())}, val_rows={int(val_mask.sum())}")

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val   = X[val_mask]
        y_val   = y[val_mask]
        sample_weight = make_recency_weights(merged.loc[train_mask, "date"], recency_half_life_days)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100, verbose=False)],
        )

        val_pred = model.predict(X_val)
        val_mae  = mean_absolute_error(y_val, val_pred)
        fold_maes.append(val_mae)
        final_model = model
        print(f"  Val MAE (week {week}, {fold_name}): {val_mae:.4f}")

    if not fold_maes or final_model is None:
        raise RuntimeError(f"No valid validation fold for week {week}.")

    avg_mae = float(np.mean(fold_maes))
    print(f"  Val MAE (week {week}, average): {avg_mae:.4f}")
    return final_model, avg_mae




def main():
    args = parse_args()
    run_dir = create_run_dir(args.model_family, args.experiment_name)
    lgb_params = get_lgb_params(args.regularized)

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
            "validation_strategy": args.validation_mode,
            "rolling_folds": args.rolling_folds,
            "meteorological_columns": METEO_COLS,
            "lightgbm_params": lgb_params,
            "feature_options": {
                "use_score_history": not args.no_score_history,
                "score_gap_days": args.score_gap_days,
                "use_climatology": not args.no_climatology,
                "use_region_stats": args.use_global_region_stats,
                "feature_profile": args.feature_profile,
                "recent_days": args.recent_days,
                "recency_half_life_days": args.recency_half_life_days,
                "regularized": args.regularized,
            },
            "pipeline": [
                "load train.csv",
                "build temporal, anomaly, and optional score-gap features",
                "train one direct LightGBM model per horizon",
                "save versioned models and metrics",
            ],
        },
    )

    # 1. Load data
    train = load_data()
    train_shape = train.shape

    # 2. Basic Feature engineering
    print("\nBuilding features ...")
    feat_df = build_features(
        train,
        train,
        is_train=True,
        use_score_history=not args.no_score_history,
        score_gap_days=args.score_gap_days,
        use_climatology=not args.no_climatology,
        use_region_stats=args.use_global_region_stats,
        feature_profile=args.feature_profile,
    )
    del train
    gc.collect()

    # 3. Extract weekly training rows
    print("\nExtracting weekly labels ...")
    weekly_df = extract_weekly_labels(feat_df)

    # 5. Stage 2: Train one model per horizon
    models   = {}
    val_maes = {}
    for week in range(1, N_WEEKS + 1):
        model, val_mae   = train_one_horizon(
            feat_df,
            weekly_df,
            week,
            lgb_params,
            args.validation_mode,
            args.rolling_folds,
            args.recent_days,
            args.recency_half_life_days,
        )
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
        "feature_options": {
            "use_score_history": not args.no_score_history,
            "score_gap_days": args.score_gap_days,
            "use_climatology": not args.no_climatology,
            "use_region_stats": args.use_global_region_stats,
            "feature_profile": args.feature_profile,
            "recent_days": args.recent_days,
            "recency_half_life_days": args.recency_half_life_days,
            "regularized": args.regularized,
        },
        "validation_strategy": args.validation_mode,
        "rolling_folds": args.rolling_folds,
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
