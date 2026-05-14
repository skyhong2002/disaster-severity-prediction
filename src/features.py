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
EWM_SPANS = [3, 7, 14]
SCORE_LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12]   # in weeks
SCORE_GAP_DAYS = 91
SCORE_HISTORY_LAGS = [0, 1, 2, 3, 4, 6, 8, 12, 26, 52]  # in weeks after gap
SCORE_HISTORY_WINS = [28, 91, 182, 365, 728]

FEATURE_PROFILES = {
    "full": {
        "meteo_cols": METEO_COLS,
        "lag_days": LAG_DAYS,
        "roll_wins": ROLL_WINS,
        "ewm_spans": EWM_SPANS,
        "domain_wins": ROLL_WINS,
        "long_wins": [30, 60, 90, 180, 270, 365, 540, 730],
        "score_lags": SCORE_HISTORY_LAGS,
        "score_wins": SCORE_HISTORY_WINS,
    },
    "lean": {
        "meteo_cols": METEO_COLS,
        "lag_days": LAG_DAYS,
        "roll_wins": ROLL_WINS,
        "ewm_spans": EWM_SPANS,
        "domain_wins": ROLL_WINS,
        "long_wins": [91, 180],
        "score_lags": [0, 1, 2, 4, 8, 12, 26, 52],
        "score_wins": [91, 182, 365],
    },
    "micro": {
        "meteo_cols": ["prec", "humidity", "tmp", "dp_tmp", "tmp_max", "tmp_range", "surf_pre"],
        "lag_days": [28, 49],
        "roll_wins": [28, 35],
        "ewm_spans": [14],
        "domain_wins": [35],
        "long_wins": [91],
        "score_lags": [0, 4, 12, 26, 52],
        "score_wins": [91, 365],
    },
}


def get_feature_profile(name: str) -> dict:
    """Return a predefined feature profile."""
    if name not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {name}")
    return FEATURE_PROFILES[name]


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
    # Dates use synthetic years beyond pandas' timestamp bounds. Parse month/day
    # directly from YYYY-MM-DD strings so seasonal features remain meaningful.
    date_str = df["date"].astype(str)
    month = pd.to_numeric(date_str.str.slice(5, 7), errors="coerce").fillna(1).astype(np.int8)
    day = pd.to_numeric(date_str.str.slice(8, 10), errors="coerce").fillna(1).astype(np.int8)
    df["month"]      = month
    df["quarter"]    = (((month - 1) // 3) + 1).astype(np.int8)
    df["dayofmonth"] = day
    # Approximate day-of-year and week from month/day to avoid leap-year issues.
    doy = (month.astype(np.int16) - 1) * 30 + day.astype(np.int16)
    df["dayofyear"]  = doy.astype(np.int16)
    df["weekofyear"] = (doy // 7).clip(0, 51).astype(np.int8)
    # Cyclical encoding
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365).astype(np.float32)
    df["sin_mon"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["cos_mon"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    return df


def add_meteo_features(df: pd.DataFrame, profile: dict | None = None) -> pd.DataFrame:
    """Add lag and rolling statistics for each meteorological column."""
    profile = profile or get_feature_profile("full")
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    for col in profile["meteo_cols"]:
        grp = df.groupby("region_id")[col]
        shifted = grp.shift(1)

        # Lag features
        for lag in profile["lag_days"]:
            df[f"{col}_lag{lag}"] = grp.shift(lag).astype(np.float32)

        # Rolling mean & std. Group a second time after shifting so windows do
        # not cross region boundaries.
        for win in profile["roll_wins"]:
            roll = shifted.groupby(df["region_id"]).rolling(window=win, min_periods=1)
            df[f"{col}_rmean{win}"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
            df[f"{col}_rstd{win}"]  = roll.std().reset_index(level=0, drop=True).astype(np.float32)

        # EWM
        for span in profile["ewm_spans"]:
            df[f"{col}_ewm{span}"] = (
                shifted.groupby(df["region_id"])
                .ewm(span=span, adjust=False)
                .mean()
                .reset_index(level=0, drop=True)
                .astype(np.float32)
            )

    # --- Domain Features: Drought Approximations ---
    # Combine rolling temperature and precipitation (High Temp + Low Rain = High Drought Risk)
    for win in profile["domain_wins"]:
        # Avoid division by zero with + 0.01
        if {f"tmp_rmean{win}", f"prec_rmean{win}"}.issubset(df.columns):
            df[f"drought_idx_r{win}"] = (df[f"tmp_rmean{win}"] / (df[f"prec_rmean{win}"] + 0.01)).astype(np.float32)
        # Daily temp range can also indicate dry air (higher range = drier air)
        if {f"tmp_range_rmean{win}", f"tmp_max_rmean{win}"}.issubset(df.columns):
            df[f"dryness_idx_r{win}"] = (df[f"tmp_range_rmean{win}"] * df[f"tmp_max_rmean{win}"]).astype(np.float32)

        if {f"tmp_max_rmean{win}", f"humidity_rmean{win}"}.issubset(df.columns):
            df[f"heat_humidity_idx_r{win}"] = (df[f"tmp_max_rmean{win}"] - df[f"humidity_rmean{win}"]).astype(np.float32)
        if {f"tmp_rmean{win}", f"dp_tmp_rmean{win}"}.issubset(df.columns):
            df[f"dewpoint_spread_r{win}"] = (df[f"tmp_rmean{win}"] - df[f"dp_tmp_rmean{win}"]).astype(np.float32)
        if {f"tmp_rmean{win}", f"wb_tmp_rmean{win}"}.issubset(df.columns):
            df[f"wetbulb_spread_r{win}"] = (df[f"tmp_rmean{win}"] - df[f"wb_tmp_rmean{win}"]).astype(np.float32)

    for win in profile["long_wins"]:
        df[f"prec_sum{win}"] = (
            df.groupby("region_id")["prec"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )
        df[f"dry_days{win}"] = (
            (df["prec"] < 0.1)
            .astype(np.float32)
            .groupby(df["region_id"])
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )
        df[f"hot_days{win}"] = (
            (df["tmp_max"] >= 30)
            .astype(np.float32)
            .groupby(df["region_id"])
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )
        df[f"tmp_mean_long{win}"] = (
            df.groupby("region_id")["tmp"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )
        # Long-term drought proxy: average temp over accumulated precipitation
        df[f"long_drought_idx{win}"] = (
            df[f"tmp_mean_long{win}"] / (df[f"prec_sum{win}"] + 0.1)
        ).astype(np.float32)

    return df


def add_climatology_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Add region-month weather anomalies computed from training weather only."""
    clim_cols = ["prec", "tmp", "tmp_max", "humidity", "tmp_range"]
    if "month" not in df.columns:
        df = add_calendar_features(df)
    train_base = train_df.copy()
    if "month" not in train_base.columns:
        train_base = add_calendar_features(train_base)

    stats = (
        train_base.groupby(["region_id", "month"])[clim_cols]
        .agg(["mean", "std"])
        .astype(np.float32)
    )
    stats.columns = [f"{col}_clim_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    df = df.merge(stats, on=["region_id", "month"], how="left")

    for col in clim_cols:
        denom = df[f"{col}_clim_std"].replace(0, np.nan)
        df[f"{col}_anom"] = ((df[col] - df[f"{col}_clim_mean"]) / denom).astype(np.float32)
    return df


def add_score_history_features(
    df: pd.DataFrame,
    gap_days: int = SCORE_GAP_DAYS,
    profile: dict | None = None,
) -> pd.DataFrame:
    """
    Add score history available after a blind test window.

    For Kaggle inference the final forecast row is at the end of a 91-day test
    window with no score labels. A feature at lag 0 therefore means the last
    known score as of 91 days earlier, not yesterday's label.
    """
    if "score" not in df.columns:
        return df

    profile = profile or get_feature_profile("full")
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)
    score_grp = df.groupby("region_id")["score"]
    known_score = score_grp.ffill()

    for lag_week in profile["score_lags"]:
        lag_days = gap_days + lag_week * 7
        df[f"score_gap_lag{lag_week}w"] = (
            known_score.groupby(df["region_id"]).shift(lag_days).astype(np.float32)
        )

    shifted_score = score_grp.shift(gap_days)
    for win in profile["score_wins"]:
        roll = shifted_score.groupby(df["region_id"]).rolling(window=win, min_periods=1)
        df[f"score_gap_mean{win}d"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f"score_gap_std{win}d"] = roll.std().reset_index(level=0, drop=True).astype(np.float32)
        df[f"score_gap_max{win}d"] = roll.max().reset_index(level=0, drop=True).astype(np.float32)

    if {"score_gap_mean28d", "score_gap_mean182d"}.issubset(df.columns):
        df["score_gap_trend_4w_26w"] = (
            df["score_gap_mean28d"] - df["score_gap_mean182d"]
        ).astype(np.float32)
    if {"score_gap_mean91d", "score_gap_mean365d"}.issubset(df.columns):
        df["score_gap_trend_13w_52w"] = (
            df["score_gap_mean91d"] - df["score_gap_mean365d"]
        ).astype(np.float32)
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
    use_score_history: bool = False,
    score_gap_days: int = SCORE_GAP_DAYS,
    use_climatology: bool = True,
    use_region_stats: bool = True,
    feature_profile: str = "full",
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print(f"  Building features for {'train' if is_train else 'test'} set...")
    profile = get_feature_profile(feature_profile)
    df = reduce_mem_usage(df)
    df = add_calendar_features(df)
    df = add_meteo_features(df, profile=profile)
    if use_climatology:
        df = add_climatology_features(df, train_df)
    if use_score_history:
        df = add_score_history_features(df, gap_days=score_gap_days, profile=profile)
    if use_region_stats:
        df = add_region_stats(df, train_df)
    df = df.copy()
    print(f"  → {len(df.columns)} columns total")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all feature columns (exclude id/date/target)."""
    exclude = {"region_id", "date", "score"}
    return [c for c in df.columns if c not in exclude]
