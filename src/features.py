"""
features.py
Temporal feature engineering for disaster severity prediction.
"""
import numpy as np
import pandas as pd
from typing import List

METEO_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

LAG_DAYS   = [7, 14, 21, 28, 35, 42, 49]
ROLL_WINS  = [7, 14, 21, 28, 35]
SCORE_LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12]   # in weeks


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year, week, month, season features."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["month"]      = dt.dt.month
    df["quarter"]    = dt.dt.quarter
    # Cyclical encoding
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["sin_mon"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_mon"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling statistics for each meteorological column."""
    df = df.copy()
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    for col in METEO_COLS:
        grp = df.groupby("region_id")[col]

        # Lag features
        for lag in LAG_DAYS:
            df[f"{col}_lag{lag}"] = grp.shift(lag)

        # Rolling mean & std
        for win in ROLL_WINS:
            df[f"{col}_rmean{win}"] = grp.transform(
                lambda x: x.shift(1).rolling(win, min_periods=1).mean()
            )
            df[f"{col}_rstd{win}"]  = grp.transform(
                lambda x: x.shift(1).rolling(win, min_periods=1).std()
            )

        # EWM
        df[f"{col}_ewm3"]  = grp.transform(lambda x: x.shift(1).ewm(span=3).mean())
        df[f"{col}_ewm7"]  = grp.transform(lambda x: x.shift(1).ewm(span=7).mean())
        df[f"{col}_ewm14"] = grp.transform(lambda x: x.shift(1).ewm(span=14).mean())

    return df


def add_score_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged weekly score features.
    Score is only non-NaN 1 day per week; we forward-fill within each 7-day
    window so that lag arithmetic works correctly on weekly cadence.
    """
    df = df.copy()
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # Forward-fill score within each region (for feature construction only)
    df["score_ff"] = df.groupby("region_id")["score"].transform(
        lambda x: x.fillna(method="ffill")
    )

    grp_score = df.groupby("region_id")["score_ff"]
    for lag_weeks in SCORE_LAGS:
        lag_days = lag_weeks * 7
        df[f"score_lag{lag_weeks}w"] = grp_score.shift(lag_days)

    # Rolling score stats (over past N weeks)
    for win_weeks in [4, 8, 12, 26, 52]:
        win_days = win_weeks * 7
        df[f"score_rmean{win_weeks}w"] = grp_score.transform(
            lambda x: x.shift(7).rolling(win_days, min_periods=1).mean()
        )
        df[f"score_rstd{win_weeks}w"] = grp_score.transform(
            lambda x: x.shift(7).rolling(win_days, min_periods=1).std()
        )

    df = df.drop(columns=["score_ff"])
    return df


def add_region_stats(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-region historical score statistics (from training data only)."""
    region_stats = (
        train_df.dropna(subset=["score"])
        .groupby("region_id")["score"]
        .agg(
            region_mean="mean",
            region_std="std",
            region_median="median",
            region_max="max",
            region_min="min",
        )
        .reset_index()
    )
    df = df.merge(region_stats, on="region_id", how="left")
    return df


def build_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print(f"  Building features for {'train' if is_train else 'test'} set...")
    df = add_calendar_features(df)
    df = add_meteo_features(df)
    df = add_score_history_features(df)
    df = add_region_stats(df, train_df)
    print(f"  → {len(df.columns)} columns total")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all feature columns (exclude id/date/target)."""
    exclude = {"region_id", "date", "score"}
    return [c for c in df.columns if c not in exclude]
