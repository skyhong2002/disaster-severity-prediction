"""
train_catboost.py
Direct-horizon CatBoost training for disaster severity prediction.

Usage:
    python3 src/train_catboost.py
    python3 src/train_catboost.py --experiment-name catboost_rolling
"""
import argparse
import gc
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

from experiment_utils import create_run_dir, save_json, write_latest_run
from features import (
    METEO_COLS,
    SCORE_GAP_DAYS,
    build_features,
    get_feature_cols,
    reduce_mem_usage,
)
from train import (
    N_WEEKS,
    apply_recent_filter,
    extract_weekly_labels,
    make_recency_weights,
    validation_masks,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FAMILY = "catboost_two_stage"

CAT_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "has_time": True,
    "od_type": "Iter",
    "od_wait": 100,
    "allow_writing_files": False,
    "thread_count": -1,
    "verbose": 200,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost disaster severity models.")
    parser.add_argument("--experiment-name", default="current")
    parser.add_argument("--model-family", default=MODEL_FAMILY)
    parser.add_argument("--no-score-history", action="store_true")
    parser.add_argument("--score-gap-days", type=int, default=SCORE_GAP_DAYS)
    parser.add_argument("--no-climatology", action="store_true")
    parser.add_argument(
        "--use-global-region-stats",
        action="store_true",
        help="Add full-train region score stats. Useful for submissions, but optimistic for local validation.",
    )
    parser.add_argument("--recent-days", type=int, default=0)
    parser.add_argument(
        "--train-tail-days",
        type=int,
        default=0,
        help="Use only the most recent N raw daily rows per region before feature building. 0 uses all rows.",
    )
    parser.add_argument("--recency-half-life-days", type=float, default=0)
    parser.add_argument(
        "--validation-mode",
        choices=["holdout", "rolling_origin"],
        default="rolling_origin",
    )
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override CatBoost iterations for quick smoke runs.",
    )
    parser.add_argument("--regularized", action="store_true")
    parser.add_argument(
        "--feature-profile",
        choices=["micro", "lean", "full"],
        default="micro",
        help="Feature set size. micro is intended for 32GB RAM machines.",
    )
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    print("Loading train.csv ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    train = reduce_mem_usage(train)
    print(f"  train shape: {train.shape}")
    return train


def apply_train_tail(train: pd.DataFrame, tail_days: int) -> pd.DataFrame:
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


def get_cat_params(regularized: bool = False) -> dict:
    params = CAT_PARAMS.copy()
    if regularized:
        params.update(
            {
                "depth": 6,
                "learning_rate": 0.04,
                "l2_leaf_reg": 12.0,
                "random_strength": 1.0,
                "bagging_temperature": 0.5,
            }
        )
    return params


def apply_param_overrides(params: dict, args: argparse.Namespace) -> dict:
    params = params.copy()
    if args.iterations is not None:
        params["iterations"] = args.iterations
    return params


def get_catboost_feature_cols(feature_df: pd.DataFrame) -> list[str]:
    """Include region_id as a native categorical feature for CatBoost."""
    cols = get_feature_cols(feature_df)
    if "region_id" not in cols:
        cols = ["region_id"] + cols
    return cols


def get_cat_cols(columns: list[str]) -> list[str]:
    candidates = {"region_id", "month", "quarter", "weekofyear"}
    return [col for col in columns if col in candidates]


def train_one_horizon(
    feature_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    week: int,
    cat_params: dict,
    validation_mode: str,
    rolling_folds: int,
    recent_days: int,
    recency_half_life_days: float,
) -> tuple[CatBoostRegressor, float, list[float]]:
    print(f"\n  --- Training horizon: week {week} ---")
    target_col = f"target_w{week}"
    feat_cols = get_catboost_feature_cols(feature_df)
    cat_cols = get_cat_cols(feat_cols)

    index_cols = ["region_id", "date", "week_idx", "max_week_idx", target_col]
    merged = weekly_df[index_cols].merge(
        feature_df[["region_id", "date"] + [c for c in feat_cols if c != "region_id"]],
        on=["region_id", "date"],
        how="left",
    )
    merged = merged.sort_values(["date", "region_id"]).reset_index(drop=True)

    for col in cat_cols:
        merged[col] = merged[col].astype(str).fillna("__missing__")

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

        sample_weight = make_recency_weights(merged.loc[train_mask, "date"], recency_half_life_days)
        train_pool = Pool(
            X.loc[train_mask],
            y.loc[train_mask],
            cat_features=cat_cols,
            weight=sample_weight,
        )
        val_pool = Pool(X.loc[val_mask], y.loc[val_mask], cat_features=cat_cols)

        model = CatBoostRegressor(**cat_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict(val_pool)
        val_mae = mean_absolute_error(y.loc[val_mask], val_pred)
        fold_maes.append(float(val_mae))
        final_model = model
        print(f"  Val MAE (week {week}, {fold_name}): {val_mae:.4f}")

    if not fold_maes or final_model is None:
        raise RuntimeError(f"No valid validation fold for week {week}.")

    avg_mae = float(np.mean(fold_maes))
    print(f"  Val MAE (week {week}, average): {avg_mae:.4f}")
    return final_model, avg_mae, fold_maes


def main():
    args = parse_args()
    run_dir = create_run_dir(args.model_family, args.experiment_name)
    cat_params = apply_param_overrides(get_cat_params(args.regularized), args)

    print("=" * 60)
    print("  Natural Disaster Severity Prediction - CatBoost Training")
    print("=" * 60)
    print(f"Experiment run directory: {run_dir}")

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_family": args.model_family,
        "experiment_name": args.experiment_name,
        "n_weeks": N_WEEKS,
        "validation_strategy": args.validation_mode,
        "rolling_folds": args.rolling_folds,
        "meteorological_columns": METEO_COLS,
        "catboost_params": cat_params,
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
            "iterations_override": args.iterations,
        },
        "pipeline": [
            "load train.csv",
            "build temporal, anomaly, and optional score-gap features",
            "train one direct CatBoost model per horizon with has_time=True",
            "save versioned models and metrics",
        ],
    }
    save_json(run_dir / "config.json", config)

    train = load_data()
    train = apply_train_tail(train, args.train_tail_days)
    train_shape = train.shape

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

    print("\nExtracting weekly labels ...")
    weekly_df = extract_weekly_labels(feat_df)

    models = {}
    val_maes = {}
    fold_maes = {}
    for week in range(1, N_WEEKS + 1):
        model, val_mae, week_fold_maes = train_one_horizon(
            feat_df,
            weekly_df,
            week,
            cat_params,
            args.validation_mode,
            args.rolling_folds,
            args.recent_days,
            args.recency_half_life_days,
        )
        models[week] = model
        val_maes[week] = val_mae
        fold_maes[week] = week_fold_maes

    model_path = run_dir / "models" / "catboost_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nHorizon models saved -> {model_path}")

    legacy_model_path = MODEL_DIR / "catboost_models.pkl"
    with open(legacy_model_path, "wb") as f:
        pickle.dump(models, f)

    print("\n--- Validation MAE per horizon ---")
    for w, mae in val_maes.items():
        print(f"  Week {w}: {mae:.4f}")
    avg = float(np.mean(list(val_maes.values())))
    print(f"  Average: {avg:.4f}")

    metrics = {
        "model_family": args.model_family,
        "experiment_name": args.experiment_name,
        "train_shape": train_shape,
        "feature_columns": len(get_catboost_feature_cols(feat_df)),
        "categorical_features": get_cat_cols(get_catboost_feature_cols(feat_df)),
        "feature_options": config["feature_options"],
        "validation_strategy": args.validation_mode,
        "rolling_folds": args.rolling_folds,
        "weekly_label_rows": len(weekly_df),
        "horizon_val_mae": {f"week_{w}": mae for w, mae in val_maes.items()},
        "horizon_fold_mae": {f"week_{w}": maes for w, maes in fold_maes.items()},
        "average_val_mae": avg,
        "model_paths": {
            "run_horizon_models": model_path,
            "latest_horizon_models": legacy_model_path,
        },
    }
    save_json(run_dir / "metrics.json", metrics)
    write_latest_run(run_dir)
    print(f"Metrics saved -> {run_dir / 'metrics.json'}")
    print(f"Latest run pointer updated -> {run_dir.name}")
    print("\nDone! Run src/predict.py to generate submission.")


if __name__ == "__main__":
    main()
