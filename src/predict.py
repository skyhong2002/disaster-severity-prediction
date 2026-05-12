"""
predict.py
Two-stage inference pipeline:
  Stage 1: Reconstruct weekly scores for the test 91-day window using
            meteorological features (no score leakage).
  Stage 2: Use reconstructed scores as score-history features to predict
            the next 5 weekly scores.

Usage:
    python3 src/predict.py
Output:
    submissions/submission_<timestamp>.csv
"""
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features, get_feature_cols, METEO_COLS

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUB_DIR   = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)

N_WEEKS = 5


def load_models():
    model_path = MODEL_DIR / "lgbm_models.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} horizon models from {model_path}")
    return models


def load_score_reconstructor():
    """
    Load Stage-1 model that predicts score from meteorological features.
    Trained during train.py (saved as lgbm_score_reconstructor.pkl).
    Falls back to None if not found.
    """
    path = MODEL_DIR / "lgbm_score_reconstructor.pkl"
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
    Only applied to rows that correspond to the weekly scoring day
    (we pick one row per 7-day block per region, i.e. every 7th row).
    """
    if reconstructor is None:
        test_feat["score_reconstructed"] = 0.0
        return test_feat

    feat_cols = [c for c in METEO_COLS if c in test_feat.columns]
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
    print("=" * 60)
    print("  Natural Disaster Severity Prediction — Inference")
    print("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────────
    print("\nLoading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["date"])
    sub_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    # ── 2. Stage 1: Reconstruct scores for test window ────────────
    print("\n[Stage 1] Reconstructing scores for test window ...")
    reconstructor = load_score_reconstructor()

    # Build minimal features (meteo only, no score history) for reconstructor
    test["score"] = np.nan
    combined_raw = pd.concat([train, test], ignore_index=True).sort_values(
        ["region_id", "date"]
    )
    # Add basic meteo features for reconstruction
    from features import add_calendar_features, add_meteo_features
    combined_meteo = add_meteo_features(add_calendar_features(combined_raw))
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
    models    = load_models()
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

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUB_DIR / f"submission_{ts}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head(10).to_string(index=False))

    print("\nPrediction statistics:")
    for col in pred_cols:
        print(f"  {col}: mean={submission[col].mean():.3f}, "
              f"std={submission[col].std():.3f}, "
              f"min={submission[col].min():.3f}, "
              f"max={submission[col].max():.3f}")


if __name__ == "__main__":
    main()
