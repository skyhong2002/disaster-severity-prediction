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


def load_score_reconstructor(model_dir: Path):
    """
    Load Stage-1 model that predicts score from meteorological features.
    Trained during train.py (saved as lgbm_score_reconstructor.pkl).
    Falls back to None if not found.
    """
    path = model_dir / "lgbm_score_reconstructor.pkl"
    if path.exists():
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded score reconstructor from {path}")
        return model
    print("⚠  No score reconstructor found — score history features will be 0.")
    return None


def reconstruct_test_scores(
    test_feat: pd.DataFrame,
    reconstructor,
) -> pd.DataFrame:
    """
    Stage 1: Fill in weekly score estimates for the test window.
    Uses the same feature set as training: all meteo + calendar features
    (no score-history features).
    """
    if reconstructor is None:
        test_feat["score_reconstructed"] = 0.0
        return test_feat

    exclude = {"region_id", "date", "score"}
    score_lag_prefixes = ("score_lag", "score_rmean", "score_rstd", "score_reconstructed")
    feat_cols = [
        c for c in test_feat.columns
        if c not in exclude and not c.startswith(score_lag_prefixes)
    ]
    test_feat = test_feat.copy()
    test_feat["score_reconstructed"] = np.clip(
        reconstructor.predict(test_feat[feat_cols].fillna(0)), 0, 5
    )
    return test_feat


def inject_reconstructed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace score_ff (used for score-history features) with reconstructed
    values so that lag/rolling score features don't carry real labels.
    """
    df = df.copy()
    # Overwrite score_ff column used in feature engineering
    # (score itself stays NaN so it's not used as a label)
    if "score_reconstructed" in df.columns:
        df["score"] = df["score_reconstructed"]
    return df


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

    # ── 1. Load raw data ──────────────────────────────────────────
    print("\nLoading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["date"])
    sub_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    # ── 2. Stage 1: Reconstruct scores for test window ────────────
    print("\n[Stage 1] Reconstructing scores for test window ...")
    reconstructor = load_score_reconstructor(model_dir)

    # Build minimal features (meteo only, no score history) for reconstructor
    test["score"] = np.nan
    combined_raw = pd.concat([train, test], ignore_index=True).sort_values(
        ["region_id", "date"]
    )
    # Add basic meteo features for reconstruction
    from features import add_calendar_features, add_meteo_features, add_region_stats
    combined_meteo = add_meteo_features(add_calendar_features(combined_raw))
    combined_meteo = add_region_stats(combined_meteo, train)
    test_meteo = combined_meteo[combined_meteo["date"].isin(test["date"].unique())].copy()
    test_meteo = reconstruct_test_scores(test_meteo, reconstructor)

    # Inject reconstructed scores back so score-history features
    # can be computed without leakage
    test["score"] = test_meteo.set_index(["region_id", "date"])["score_reconstructed"] \
        .reindex(pd.MultiIndex.from_frame(test[["region_id", "date"]])).values

    # ── 3. Build full features with reconstructed scores ──────────
    print("\n[Stage 2] Building full features for forecast ...")
    combined = pd.concat([train, test], ignore_index=True)
    combined_feat = build_features(combined, train, is_train=False)
    test_feat = combined_feat[combined_feat["date"].isin(test["date"].unique())].copy()

    # For each region, take the LAST row of the 91-day test window
    test_last = (
        test_feat
        .sort_values(["region_id", "date"])
        .groupby("region_id")
        .last()
        .reset_index()
    )

    # ── 4. Load horizon models and predict ────────────────────────
    print("\n[Stage 2] Forecasting weeks 1–5 ...")
    models    = load_models(model_dir)
    feat_cols = [c for c in get_feature_cols(test_feat) if c in test_last.columns]
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
