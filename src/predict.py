"""
predict.py
Load trained models and generate Kaggle submission.

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

from features import build_features, get_feature_cols

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUB_DIR   = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)

N_WEEKS = 5


def main():
    print("=" * 60)
    print("  Natural Disaster Severity Prediction — Inference")
    print("=" * 60)

    # 1. Load data
    print("\nLoading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["date"])
    sub_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    # 2. Build features for test (using full train as reference for region stats)
    print("\nBuilding test features ...")
    # Concatenate for proper lag computation, then split
    test["score"] = np.nan   # placeholder
    combined = pd.concat([train, test], ignore_index=True)
    combined_feat = build_features(combined, train, is_train=False)
    test_feat = combined_feat[combined_feat["date"].isin(test["date"].unique())].copy()

    # For each region, take the LAST row of the 91-day test window as the
    # "current state" from which we forecast weeks 1–5
    test_last = (
        test_feat
        .sort_values(["region_id", "date"])
        .groupby("region_id")
        .last()
        .reset_index()
    )

    # 3. Load models
    model_path = MODEL_DIR / "lgbm_models.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    print(f"\nLoaded {len(models)} models from {model_path}")

    # 4. Predict
    feat_cols = get_feature_cols(test_feat)
    # Keep only cols present in test_last
    feat_cols = [c for c in feat_cols if c in test_last.columns]

    X_test = test_last[feat_cols]
    preds  = {}
    for week in range(1, N_WEEKS + 1):
        raw = models[week].predict(X_test)
        # Clip to valid score range
        preds[f"pred_week{week}"] = np.clip(raw, 0, 5)

    # 5. Build submission
    result = test_last[["region_id"]].copy()
    for week in range(1, N_WEEKS + 1):
        result[f"pred_week{week}"] = preds[f"pred_week{week}"]

    # Align with template order
    submission = sub_template[["region_id"]].merge(result, on="region_id", how="left")

    # Fill any missing regions with 0
    pred_cols = [f"pred_week{w}" for w in range(1, N_WEEKS + 1)]
    submission[pred_cols] = submission[pred_cols].fillna(0)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUB_DIR / f"submission_{ts}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head(10).to_string(index=False))

    # Stats
    print("\nPrediction statistics:")
    for col in pred_cols:
        print(f"  {col}: mean={submission[col].mean():.3f}, "
              f"std={submission[col].std():.3f}, "
              f"min={submission[col].min():.3f}, "
              f"max={submission[col].max():.3f}")


if __name__ == "__main__":
    main()
