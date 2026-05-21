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
PILLAR_CONTEXT_WINS = [14, 30, 60, 90, 180]

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

FEATURE_GROUPS = {
    "calendar",
    "short_lag",
    "rolling",
    "ewm",
    "long_drought_proxy",
    "domain_indices",
    "climatology",
    "score_history",
    "region_stats",
    "region_id",
}


def get_feature_profile(name: str) -> dict:
    """Return a predefined feature profile."""
    if name not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {name}")
    return FEATURE_PROFILES[name]


def required_context_days(
    feature_profile: str,
    score_gap_days: int = SCORE_GAP_DAYS,
    use_score_history: bool = False,
) -> int:
    """Return the minimum history window needed to fully populate a profile."""
    profile = get_feature_profile(feature_profile)
    required = 1
    for key in ("lag_days", "roll_wins", "domain_wins", "long_wins", "ewm_spans"):
        values = profile.get(key, [])
        if values:
            required = max(required, int(max(values)))
    required = max(required, max(PILLAR_CONTEXT_WINS))
    if use_score_history:
        score_lag_days = [score_gap_days + int(lag_week) * 7 for lag_week in profile.get("score_lags", [])]
        score_win_days = [score_gap_days + int(win) for win in profile.get("score_wins", [])]
        if score_lag_days:
            required = max(required, max(score_lag_days))
        if score_win_days:
            required = max(required, max(score_win_days))
    return int(required)


def parse_drop_feature_groups(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize comma-separated feature groups for ablation runs."""
    if raw is None:
        return []
    if isinstance(raw, str):
        groups = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        groups = [str(part).strip() for part in raw if str(part).strip()]
    unknown = sorted(set(groups) - FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {', '.join(unknown)}")
    return groups


def columns_for_feature_group(columns: list[str], group: str) -> list[str]:
    """Return columns belonging to one coarse feature group."""
    if group == "calendar":
        names = {"month", "quarter", "dayofmonth", "dayofyear", "weekofyear", "sin_doy", "cos_doy", "sin_mon", "cos_mon"}
        return [col for col in columns if col in names]
    if group == "short_lag":
        return [col for col in columns if "_lag" in col and not col.startswith("score_gap")]
    if group == "rolling":
        prefixes = ("tmp_range_std_", "wind_std_")
        return [col for col in columns if "_rmean" in col or "_rstd" in col or col.startswith(prefixes)]
    if group == "ewm":
        return [col for col in columns if "_ewm" in col]
    if group == "long_drought_proxy":
        prefixes = (
            "prec_sum",
            "dry_days",
            "hot_days",
            "tmp_mean_long",
            "long_drought_idx",
            "consecutive_dry_days",
            "cdd_rolling",
            "hot_days_above",
            "heat_stress_sum",
        )
        return [col for col in columns if col.startswith(prefixes)]
    if group == "domain_indices":
        prefixes = (
            "drought_idx",
            "dryness_idx",
            "heat_humidity_idx",
            "dewpoint_spread",
            "wetbulb_spread",
            "surf_air_temp_diff",
            "dew_point_depression",
            "wet_bulb_depression",
            "surf_air_diff_mean",
            "dew_depression_mean",
            "wet_bulb_depression_mean",
        )
        return [col for col in columns if col.startswith(prefixes)]
    if group == "climatology":
        return [col for col in columns if "_clim_" in col or col.endswith("_anom")]
    if group == "score_history":
        prefixes = ("score_gap", "last_known_score", "score_velocity", "score_momentum")
        return [col for col in columns if col.startswith(prefixes)]
    if group == "region_stats":
        return [col for col in columns if col.startswith("region_") and col != "region_id"]
    if group == "region_id":
        return ["region_id"] if "region_id" in columns else []
    raise ValueError(f"Unknown feature group: {group}")


def drop_feature_group_columns(df: pd.DataFrame, groups: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    """Drop coarse feature groups for validation-driven ablation."""
    groups = parse_drop_feature_groups(groups)
    if not groups:
        return df
    drop_cols: set[str] = set()
    columns = list(df.columns)
    for group in groups:
        if group == "region_id":
            continue
        drop_cols.update(columns_for_feature_group(columns, group))
    if drop_cols:
        df = df.drop(columns=sorted(drop_cols), errors="ignore")
        print(f"  Dropped {len(drop_cols)} columns from feature groups: {', '.join(groups)}")
    return df


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


def build_consecutive_dry_days(df: pd.DataFrame, dry_threshold: float = 0.1) -> pd.DataFrame:
    """
    Build consecutive dry days (CDD) feature: the maximum number of consecutive days
    with precipitation < dry_threshold.

    CDD is a golden feature for drought prediction, capturing persistent water deficit.
    """
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # Mark dry days (prec < threshold)
    df["is_dry"] = (df["prec"] < dry_threshold).astype(np.int32)

    # Shift by region to avoid cross-region leakage
    df["dry_block"] = (
        (df["is_dry"] == 0)
        .groupby(df["region_id"])
        .cumsum()
        .astype(np.int32)
    )

    # Calculate consecutive dry days within each block
    consecutive_dry = (
        df.groupby(["region_id", "dry_block"])["is_dry"]
        .cumsum()
        .astype(np.int32)
    )

    # The CDD is the max consecutive dry within each block, broadcasted to all rows
    df["consecutive_dry_days"] = (
        consecutive_dry
        .groupby(df["region_id"])
        .transform("max")
        .astype(np.float32)
    )

    # Rolling CDD: track max consecutive dry in past N days (golden indicator)
    for win in [30, 60, 90]:
        df[f"cdd_rolling{win}"] = (
            df.groupby("region_id")["consecutive_dry_days"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

    df = df.drop(columns=["is_dry", "dry_block"])
    return df


def build_heat_accumulation_features(df: pd.DataFrame, hot_threshold: float = 35.0) -> pd.DataFrame:
    """
    Build heat accumulation features: count of days exceeding temperature threshold.

    Represents prolonged high-temperature stress that compounds water deficit and
    increases evaporation demand on soil.
    """
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    for win in [30, 60, 90, 180]:
        # Count days where tmp_max >= hot_threshold
        df[f"hot_days_above{int(hot_threshold)}_{win}d"] = (
            (df["tmp_max"] >= hot_threshold)
            .astype(np.float32)
            .groupby(df["region_id"])
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

    # Sum of positive temperature anomalies (heat stress accumulation)
    for win in [30, 60, 90]:
        # Use 30°C as baseline for anomaly
        baseline = 30.0
        heat_stress = (df["tmp_max"] - baseline).clip(lower=0)
        df[f"heat_stress_sum_{win}d"] = (
            heat_stress
            .groupby(df["region_id"])
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

    return df


def build_temperature_instability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temperature and wind instability features.

    High variability (std dev) in tmp_range and wind indicates chaotic/dry climate system.
    """
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    for win in [14, 30, 60]:
        # Standard deviation of daily temp range (high = drier air)
        df[f"tmp_range_std_{win}d"] = (
            df.groupby("region_id")["tmp_range"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        # Standard deviation of wind (high = more unstable)
        df[f"wind_std_{win}d"] = (
            df.groupby("region_id")["wind"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

    return df


def build_physical_vapor_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build domain-specific physical/thermodynamic proxy features for evaporation & drought.

    Cross-field features derived from tmp, surf_tmp, dp_tmp (dew point), wb_tmp (wet bulb):
    - surf_air_temp_diff: when soil is dry, surface cannot cool via evaporation → surf_tmp >> tmp
    - dew_point_depression: tmp - dp_tmp; larger = drier air = faster evaporation
    - wet_bulb_depression: tmp - wb_tmp; larger = air strongly desires moisture
    """
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # 1. Surface-Air Temperature Difference: high when soil is dry (cannot evaporate to cool)
    df["surf_air_temp_diff"] = (df["surf_tmp"] - df["tmp"]).astype(np.float32)

    # 2. Dew Point Depression (tmp - dp_tmp): larger = drier, faster evaporation
    df["dew_point_depression"] = (df["tmp"] - df["dp_tmp"]).astype(np.float32)

    # 3. Wet Bulb Depression (tmp - wb_tmp): larger = air can absorb more moisture
    df["wet_bulb_depression"] = (df["tmp"] - df["wb_tmp"]).astype(np.float32)

    # Rolling averages of these physical indicators to capture persistence
    for win in [14, 30]:
        df[f"surf_air_diff_mean_{win}d"] = (
            df.groupby("region_id")["surf_air_temp_diff"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        df[f"dew_depression_mean_{win}d"] = (
            df.groupby("region_id")["dew_point_depression"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        df[f"wet_bulb_depression_mean_{win}d"] = (
            df.groupby("region_id")["wet_bulb_depression"]
            .shift(1)
            .groupby(df["region_id"])
            .rolling(window=win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

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
    Add score history features available after blind test window.

    Includes:
    1. Last Effective Score: the most recent known score in the 91-day window
    2. Score Velocity: change rate (latest - previous week score)
    3. Historical rolling statistics: mean, std, max of past scores
    """
    if "score" not in df.columns:
        return df

    profile = profile or get_feature_profile("full")
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)
    score_grp = df.groupby("region_id")["score"]

    # Forward fill historical labels, then shift by the blind-window gap so
    # train-time score dynamics match Kaggle/test-time score availability.
    known_score = score_grp.ffill()
    visible_score = known_score.groupby(df["region_id"]).shift(gap_days)

    # 1. Last Effective Score: the final known score visible at prediction time (91 days)
    df["last_known_score"] = visible_score.astype(np.float32)

    # 2. Score Velocity: change from 7 days ago (previous week)
    score_7d_ago = known_score.groupby(df["region_id"]).shift(gap_days + 7)
    df["score_velocity_1w"] = (visible_score - score_7d_ago).astype(np.float32)

    # 3. Score Momentum: longer-term trends (14d, 28d)
    for period in [14, 28]:
        score_n_ago = known_score.groupby(df["region_id"]).shift(gap_days + period)
        df[f"score_momentum_{period}d"] = (visible_score - score_n_ago).astype(np.float32)

    # Original gapped score lags (after 91-day blind window)
    for lag_week in profile["score_lags"]:
        lag_days = gap_days + lag_week * 7
        df[f"score_gap_lag{lag_week}w"] = (
            known_score.groupby(df["region_id"]).shift(lag_days).astype(np.float32)
        )

    # Gapped rolling statistics on lagged scores
    shifted_score = score_grp.shift(gap_days)
    for win in profile["score_wins"]:
        roll = shifted_score.groupby(df["region_id"]).rolling(window=win, min_periods=1)
        df[f"score_gap_mean{win}d"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f"score_gap_std{win}d"] = roll.std().reset_index(level=0, drop=True).astype(np.float32)
        df[f"score_gap_max{win}d"] = roll.max().reset_index(level=0, drop=True).astype(np.float32)

    # Trend composites
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
    """
    Add per-region historical score statistics (from training data only).

    These are "region profiles" that capture long-term characteristics:
    - region_score_mean: average severity (baseline risk)
    - region_score_max: extreme exposure (worst-case scenario)
    - region_score_median: typical severity distribution

    Prevents data leakage by computing stats only on train set.
    """
    region_stats = (
        train_df.dropna(subset=["score"])
        .groupby("region_id")["score"]
        .agg(
            region_score_mean="mean",
            region_score_std="std",
            region_score_median="median",
            region_score_max="max",
            region_score_min="min",
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
    drop_feature_groups: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline with three core pillars:

    1. Temporal Accumulation & Instability (build_weather_rolling_features)
       - Consecutive dry days (CDD) - the golden feature for drought
       - Heat accumulation - prolonged high-temperature stress
       - Temperature/wind instability - chaos indicator

    2. Domain-Specific Cross Features (build_domain_proxy_features)
       - Physical thermodynamic proxies (surf_air_diff, dew_point_depression, wet_bulb_depression)
       - Captures soil moisture deficit and atmospheric evaporation demand

    3. Historical Score & Regional Baseline (add_score_history_features, add_region_stats)
       - Last known score, score velocity, momentum
       - Region profile statistics (mean, max, baseline)
    """
    print(f"  Building features for {'train' if is_train else 'test'} set...")
    profile = get_feature_profile(feature_profile)

    # Memory-efficient type reduction
    df = reduce_mem_usage(df)

    # ─── Core Feature Engineering ───────────────────────────────────────────
    # 1. Calendar features for seasonality
    df = add_calendar_features(df)

    # 2. Standard meteorological lag/rolling features
    df = add_meteo_features(df, profile=profile)

    # 3. === PILLAR 1: Temporal Accumulation & Instability ===
    print("  → Adding consecutive dry days (CDD) feature...")
    df = build_consecutive_dry_days(df, dry_threshold=0.1)

    print("  → Adding heat accumulation features...")
    df = build_heat_accumulation_features(df, hot_threshold=35.0)

    print("  → Adding temperature & wind instability features...")
    df = build_temperature_instability_features(df)

    # 4. === PILLAR 2: Domain-Specific Physical Cross Features ===
    print("  → Adding physical vapor pressure proxy features...")
    df = build_physical_vapor_proxy_features(df)

    # 5. Climatology anomalies (region-month statistics)
    if use_climatology:
        print("  → Adding climatology anomaly features...")
        df = add_climatology_features(df, train_df)

    # 6. === PILLAR 3: Score History & Region Baseline ===
    if use_score_history:
        print("  → Adding score history features (last known state, velocity, momentum)...")
        df = add_score_history_features(df, gap_days=score_gap_days, profile=profile)

    # 7. Region-level baseline statistics (from training data)
    if use_region_stats:
        print("  → Adding per-region historical baseline statistics...")
        df = add_region_stats(df, train_df)
    df = drop_feature_group_columns(df, drop_feature_groups)

    df = df.copy()
    print(f"  → {len(df.columns)} columns total")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all feature columns (exclude id/date/target)."""
    exclude = {"region_id", "date", "score"}
    return [
        c
        for c in df.columns
        if c not in exclude and not c.startswith("_") and not c.startswith("target_")
    ]
