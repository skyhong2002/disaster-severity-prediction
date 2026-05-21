"""
train_xgb.py
Direct-horizon XGBoost training for disaster severity prediction.

The current pipeline builds leakage-aware temporal features, including optional
91-day-gapped score-history features, then trains one independent XGBoost
regressor for each forecast horizon (week 1-5). The historical separate
score-estimation approach is kept only in old experiment artifacts, not in
this training flow.

Usage:
    python3 src/train_xgb.py
    python3 src/train_xgb.py --experiment-name xgb_v1
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

from features import (
    SCORE_GAP_DAYS,
    build_features,
    get_feature_cols,
    METEO_COLS,
    parse_drop_feature_groups,
    reduce_mem_usage,
)
from experiment_utils import create_run_dir, save_json, write_latest_run
from model_wrappers import AveragingRegressor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FAMILY = "xgboost_two_stage"  # Kept for compatibility with existing runs.

# ── XGBoost config ─────────────────────────────────────────────────────────────
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
        "--train-tail-days",
        type=int,
        default=0,
        help="Use only the most recent N raw daily rows per region before feature building. 0 uses all rows.",
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
        help="Use a more conservative XGBoost preset for lower overfit risk.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["micro", "lean", "full"],
        default="micro",
        help="Feature set size. micro is intended for 32GB RAM machines.",
    )
    parser.add_argument(
        "--final-train-mode",
        choices=["last_fold", "fold_ensemble", "refit_full"],
        default="refit_full",
        help="How to produce the saved submission model after validation.",
    )
    parser.add_argument(
        "--drop-feature-groups",
        default="",
        help="Comma-separated coarse feature groups to remove for ablation.",
    )
    return parser.parse_args()


def load_data():
    print("Loading train.csv ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    train = reduce_mem_usage(train)
    print(f"  train shape: {train.shape}")
    return train


def apply_train_tail(train: pd.DataFrame, tail_days: int) -> pd.DataFrame:
    """Trim raw daily rows per region before expensive feature construction."""
    if tail_days <= 0:
        return train
    trimmed = (
        train.sort_values(["region_id", "date"])
        .groupby("region_id", group_keys=False)
        .tail(tail_days)
        .reset_index(drop=True)
    )
    print(f"  using latest {tail_days} daily rows per region: {trimmed.shape}")
    return trimmed


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


def get_xgb_params(regularized: bool = False) -> dict:
    """Return the requested XGBoost parameter preset."""
    params = XGB_PARAMS.copy()
    if regularized:
        params.update(
            {
                "max_depth": 4,
                "min_child_weight": 10,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.7,
                "subsample": 0.7,
            }
        )
    return params


def apply_recent_filter(merged: pd.DataFrame, recent_days: int) -> pd.Series:
    if recent_days <= 0:
        return pd.Series(True, index=merged.index)
    date_rank = merged["date"].map({d: i for i, d in enumerate(sorted(merged["date"].unique()))})
    max_rank = date_rank.max()
    recent_steps = max(1, int(recent_days / 7))
    return date_rank >= max_rank - recent_steps


def make_recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray | None:
    if half_life_days <= 0:
        return None
    date_rank = dates.map({d: i for i, d in enumerate(sorted(dates.unique()))}).astype(np.float32)
    age_days = (date_rank.max() - date_rank) * 7.0
    weights = np.power(0.5, age_days / half_life_days).astype(np.float32)
    return np.clip(weights / np.mean(weights), 0.05, 20.0)


def best_xgb_iteration(model: xgb.XGBRegressor, fallback: int) -> int:
    """Return a usable one-based best iteration for final refit."""
    best = getattr(model, "best_iteration", None)
    if best is not None and best >= 0:
        return int(best) + 1
    best = getattr(model, "n_estimators", None)
    if best is None or best <= 0:
        best = fallback
    return int(best)


def refit_xgb_params(params: dict, n_estimators: int) -> dict:
    """Remove validation-only settings and fix tree count for full refit."""
    refit = params.copy()
    refit["n_estimators"] = int(max(1, n_estimators))
    refit.pop("early_stopping_rounds", None)
    return refit


def validation_masks(merged: pd.DataFrame, mode: str, rolling_folds: int, horizon: int):
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

    offsets = np.linspace(rolling_folds + 4, 5, rolling_folds).round().astype(int)
    for offset in offsets:
        val_idx = merged["max_week_idx"] - offset
        train_mask = merged["week_idx"] + horizon < val_idx
        val_mask = merged["week_idx"] == val_idx
        yield train_mask, val_mask, f"relative_week_-{offset}"


def train_one_horizon(
    feature_df: pd.DataFrame,
    weekly_df:  pd.DataFrame,
    week: int,
    xgb_params: dict,
    validation_mode: str,
    rolling_folds: int,
    recent_days: int,
    recency_half_life_days: float,
    final_train_mode: str,
) -> tuple[xgb.XGBRegressor | AveragingRegressor, float, list[float], dict]:
    """Train a single XGBoost model for horizon `week`."""
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
    fold_models = []
    best_iterations = []
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

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )

        val_pred = model.predict(X_val)
        val_mae  = mean_absolute_error(y_val, val_pred)
        fold_maes.append(float(val_mae))
        fold_models.append(model)
        best_iterations.append(best_xgb_iteration(model, xgb_params.get("n_estimators", 3000)))
        final_model = model
        print(f"  Val MAE (week {week}, {fold_name}): {val_mae:.4f}")

    if not fold_maes or final_model is None:
        raise RuntimeError(f"No valid validation fold for week {week}.")

    avg_mae = float(np.mean(fold_maes))
    print(f"  Val MAE (week {week}, average): {avg_mae:.4f}")
    final_info = {
        "final_train_mode": final_train_mode,
        "fold_count": len(fold_models),
        "fold_best_iterations": best_iterations,
    }

    if final_train_mode == "fold_ensemble":
        final_info["ensemble_size"] = len(fold_models)
        return AveragingRegressor(fold_models, feature_names=feat_cols), avg_mae, fold_maes, final_info

    if final_train_mode == "refit_full":
        final_n_estimators = int(np.median(best_iterations))
        final_info["final_n_estimators"] = final_n_estimators
        final_mask = usable_mask
        sample_weight = make_recency_weights(merged.loc[final_mask, "date"], recency_half_life_days)
        print(f"  Refit full model: train_rows={int(final_mask.sum())}, n_estimators={final_n_estimators}")
        refit_model = xgb.XGBRegressor(**refit_xgb_params(xgb_params, final_n_estimators))
        refit_model.fit(X.loc[final_mask], y.loc[final_mask], sample_weight=sample_weight, verbose=False)
        return refit_model, avg_mae, fold_maes, final_info

    return final_model, avg_mae, fold_maes, final_info




def main():
    args = parse_args()
    run_dir = create_run_dir(args.model_family, args.experiment_name)
    xgb_params = get_xgb_params(args.regularized)
    drop_groups = parse_drop_feature_groups(args.drop_feature_groups)

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
            "xgboost_params": xgb_params,
            "feature_options": {
                "use_score_history": not args.no_score_history,
                "score_gap_days": args.score_gap_days,
                "use_climatology": not args.no_climatology,
                "use_region_stats": args.use_global_region_stats,
                "feature_profile": args.feature_profile,
                "recent_days": args.recent_days,
                "train_tail_days": args.train_tail_days,
                "recency_half_life_days": args.recency_half_life_days,
                "regularized": args.regularized,
                "final_train_mode": args.final_train_mode,
                "drop_feature_groups": drop_groups,
            },
            "pipeline": [
                "load train.csv",
                "build temporal, anomaly, and optional score-gap features",
                "validate one direct XGBoost model per horizon",
                "save refit_full, fold_ensemble, or last_fold final horizon models",
                "save versioned models and metrics",
            ],
        },
    )

    # 1. Load data
    train = load_data()
    train = apply_train_tail(train, args.train_tail_days)
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
        drop_feature_groups=drop_groups,
    )
    del train
    gc.collect()

    # 3. Extract weekly training rows
    print("\nExtracting weekly labels ...")
    weekly_df = extract_weekly_labels(feat_df)

    # 5. Stage 2: Train one model per horizon
    models   = {}
    val_maes = {}
    fold_maes = {}
    final_model_info = {}
    for week in range(1, N_WEEKS + 1):
        model, val_mae, week_fold_maes, week_final_info = train_one_horizon(
            feat_df,
            weekly_df,
            week,
            xgb_params,
            args.validation_mode,
            args.rolling_folds,
            args.recent_days,
            args.recency_half_life_days,
            args.final_train_mode,
        )
        models[week]     = model
        val_maes[week]   = val_mae
        fold_maes[week] = week_fold_maes
        final_model_info[week] = week_final_info

    # 6. Save horizon models
    model_path = run_dir / "models" / "xgb_models.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nHorizon models saved → {model_path}")

    legacy_model_path = MODEL_DIR / "xgb_models.pkl"
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
            "train_tail_days": args.train_tail_days,
            "recency_half_life_days": args.recency_half_life_days,
            "regularized": args.regularized,
            "final_train_mode": args.final_train_mode,
            "drop_feature_groups": drop_groups,
        },
        "validation_strategy": args.validation_mode,
        "rolling_folds": args.rolling_folds,
        "weekly_label_rows": len(weekly_df),
        "horizon_val_mae": {f"week_{w}": mae for w, mae in val_maes.items()},
        "horizon_fold_mae": {f"week_{w}": maes for w, maes in fold_maes.items()},
        "final_model_info": {f"week_{w}": info for w, info in final_model_info.items()},
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
