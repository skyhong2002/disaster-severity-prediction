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


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to float32/int32 to save memory."""
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year, week, month, season features."""
    # Use format='mixed' + errors='coerce' to handle unusual dates (year 3000+)
    dt = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    dt = dt.fillna(pd.Timestamp("2001-01-01"))
    df["month"]      = dt.dt.month.astype(np.int8)
    df["quarter"]    = dt.dt.quarter.astype(np.int8)
    df["dayofmonth"] = dt.dt.day.astype(np.int8)
    # Approximate day-of-year and week from month/day to avoid leap-year issues
    doy = (dt.dt.month - 1) * 30 + dt.dt.day
    df["dayofyear"]  = doy.astype(np.int16)
    df["weekofyear"] = (doy // 7).clip(0, 51).astype(np.int8)
    # Cyclical encoding
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365).astype(np.float32)
    df["sin_mon"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["cos_mon"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling statistics for each meteorological column."""
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    for col in METEO_COLS:
        grp = df.groupby("region_id")[col]

        # Lag features
        for lag in LAG_DAYS:
            df[f"{col}_lag{lag}"] = grp.shift(lag).astype(np.float32)

        # Rolling mean & std
        # Use more efficient transform + built-in rolling
        for win in ROLL_WINS:
            df[f"{col}_rmean{win}"] = grp.shift(1).rolling(window=win, min_periods=1).mean().astype(np.float32)
            df[f"{col}_rstd{win}"]  = grp.shift(1).rolling(window=win, min_periods=1).std().astype(np.float32)

        # EWM
        df[f"{col}_ewm3"]  = grp.shift(1).ewm(span=3).mean().astype(np.float32)
        df[f"{col}_ewm7"]  = grp.shift(1).ewm(span=7).mean().astype(np.float32)
        df[f"{col}_ewm14"] = grp.shift(1).ewm(span=14).mean().astype(np.float32)

    return df


def add_score_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged weekly score features.
    Score is only non-NaN 1 day per week; we forward-fill within each 7-day
    window so that lag arithmetic works correctly on weekly cadence.
    """
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # Forward-fill score within each region (for feature construction only)
    score_ff = df.groupby("region_id")["score"].ffill()

    for lag_weeks in SCORE_LAGS:
        lag_days = lag_weeks * 7
        df[f"score_lag{lag_weeks}w"] = score_ff.groupby(df["region_id"]).shift(lag_days).astype(np.float32)

    # Rolling score stats (over past N weeks)
    for win_weeks in [4, 8, 12, 26, 52]:
        win_days = win_weeks * 7
        df[f"score_rmean{win_weeks}w"] = score_ff.groupby(df["region_id"]).shift(7).rolling(win_days, min_periods=1).mean().astype(np.float32)
        df[f"score_rstd{win_weeks}w"]  = score_ff.groupby(df["region_id"]).shift(7).rolling(win_days, min_periods=1).std().astype(np.float32)

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
        .astype(np.float32)
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
    df = reduce_mem_usage(df)
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
