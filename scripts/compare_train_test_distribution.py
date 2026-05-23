#!/usr/bin/env python3
"""Compare train and test weather/drought feature distributions.

This script is intentionally stricter than the lightweight drift report:
synthetic dates are parsed into numeric ordinals before any tail-window or
season-matched selection, because the date year has variable width.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import save_json  # noqa: E402
from features import METEO_COLS  # noqa: E402

MONTH_STARTS = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int16)

RAW_REPORT_COLS = [
    "prec",
    "humidity",
    "tmp",
    "dp_tmp",
    "wb_tmp",
    "tmp_max",
    "tmp_min",
    "tmp_range",
    "surf_tmp",
    "wind",
]

DERIVED_COLS = [
    "dry_day",
    "rain_day_ge_1",
    "heavy_precip_ge_10",
    "dew_point_depression",
    "wet_bulb_depression",
    "surf_air_temp_diff",
    "dryness_idx_daily",
    "heat_humidity_idx_daily",
]

ROLLING_COLS = [
    "prec_sum_30d",
    "prec_sum_90d",
    "prec_sum_180d",
    "dry_days_30d",
    "dry_days_90d",
    "dry_share_90d",
    "rain_days_90d",
    "hot_days_90d",
    "tmp_mean_90d",
    "humidity_mean_90d",
    "dew_depression_mean_90d",
    "wet_bulb_depression_mean_90d",
    "max_dry_streak_90d",
    "rolling_drought_idx_90d",
]

ANALYSIS_COLS = RAW_REPORT_COLS + DERIVED_COLS + ROLLING_COLS

FEATURE_LABELS = {
    "prec": "daily precipitation",
    "humidity": "humidity",
    "tmp": "mean temperature",
    "dp_tmp": "dew point temperature",
    "wb_tmp": "wet bulb temperature",
    "tmp_max": "max temperature",
    "tmp_min": "min temperature",
    "tmp_range": "daily temperature range",
    "surf_tmp": "surface temperature",
    "wind": "wind",
    "dry_day": "dry-day share, prec < 0.1",
    "rain_day_ge_1": "rain-day share, prec >= 1",
    "heavy_precip_ge_10": "heavy-rain share, prec >= 10",
    "dew_point_depression": "tmp - dew point",
    "wet_bulb_depression": "tmp - wet bulb",
    "surf_air_temp_diff": "surface temp - air temp",
    "dryness_idx_daily": "tmp_range * tmp_max",
    "heat_humidity_idx_daily": "tmp_max - humidity",
    "prec_sum_30d": "30-day precipitation sum",
    "prec_sum_90d": "90-day precipitation sum",
    "prec_sum_180d": "180-day precipitation sum",
    "dry_days_30d": "30-day dry days",
    "dry_days_90d": "90-day dry days",
    "dry_share_90d": "90-day dry-day share",
    "rain_days_90d": "90-day rain days, prec >= 1",
    "hot_days_90d": "90-day hot days, tmp_max >= 30",
    "tmp_mean_90d": "90-day mean temperature",
    "humidity_mean_90d": "90-day mean humidity",
    "dew_depression_mean_90d": "90-day mean tmp-dew point spread",
    "wet_bulb_depression_mean_90d": "90-day mean tmp-wet bulb spread",
    "max_dry_streak_90d": "max dry streak in prior 90d",
    "rolling_drought_idx_90d": "90-day temp / precipitation proxy",
}

BASELINE_LABELS = {
    "full_train": "全部 train",
    "tail_1095d": "每區最近 1095 天",
    "tail_365d": "每區最近 365 天",
    "adjacent_prev_91d": "test 前 91 天",
    "same_md_all_years": "同 region 同月日所有歷史年",
    "same_md_recent_5y": "同 region 同月日近 5 年",
    "prev_year_same_md": "同 region 同月日前一年",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a detailed train/test distribution comparison report.")
    parser.add_argument("--max-samples", type=int, default=120000, help="Rows per side for KS/PSI/AUC sampling.")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for continuous PSI.")
    parser.add_argument(
        "--out",
        default="docs/validation/train_test_distribution_comparison_20260523.md",
        help="Markdown report path.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON metrics path.")
    parser.add_argument("--csv-out", default=None, help="Optional per-feature metrics CSV path.")
    return parser.parse_args()


def resolve_path(raw: str | None, fallback: Path) -> Path:
    path = Path(raw) if raw else fallback
    return path if path.is_absolute() else ROOT / path


def load_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols_train = ["region_id", "date"] + METEO_COLS
    dtype = {col: "float32" for col in METEO_COLS}
    print("Loading train/test weather columns ...")
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols_train, dtype=dtype)
    test = pd.read_csv(ROOT / "data" / "test.csv", usecols=usecols_train, dtype=dtype)
    print(f"  train={train.shape}, test={test.shape}")
    return train, test


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["date"].astype("string").str.split("-", n=2, expand=True)
    year = pd.to_numeric(parts[0], downcast="integer")
    month = pd.to_numeric(parts[1], downcast="integer")
    day = pd.to_numeric(parts[2], downcast="integer")
    doy = MONTH_STARTS[month.to_numpy(dtype=np.int16)] + day.to_numpy(dtype=np.int16)
    df["_year"] = year.astype("int32")
    df["_month"] = month.astype("int16")
    df["_day"] = day.astype("int16")
    df["_doy"] = pd.Series(doy, index=df.index).astype("int16")
    df["_md"] = (df["_month"].astype("int16") * 100 + df["_day"].astype("int16")).astype("int16")
    df["_ordinal"] = (df["_year"].astype("int64") * 366 + df["_doy"].astype("int64")).astype("int64")
    return df


def add_daily_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["dry_day"] = (df["prec"] < 0.1).astype("float32")
    df["rain_day_ge_1"] = (df["prec"] >= 1.0).astype("float32")
    df["heavy_precip_ge_10"] = (df["prec"] >= 10.0).astype("float32")
    df["dew_point_depression"] = (df["tmp"] - df["dp_tmp"]).astype("float32")
    df["wet_bulb_depression"] = (df["tmp"] - df["wb_tmp"]).astype("float32")
    df["surf_air_temp_diff"] = (df["surf_tmp"] - df["tmp"]).astype("float32")
    df["dryness_idx_daily"] = (df["tmp_range"] * df["tmp_max"]).astype("float32")
    df["heat_humidity_idx_daily"] = (df["tmp_max"] - df["humidity"]).astype("float32")
    return df


def rolling_by_region(df: pd.DataFrame, values: pd.Series, window: int, agg: str) -> pd.Series:
    rolled = (
        values.groupby(df["region_id"], sort=False)
        .rolling(window=window, min_periods=1)
        .agg(agg)
        .reset_index(level=0, drop=True)
    )
    return rolled.astype("float32")


def add_rolling_features(combined: pd.DataFrame) -> pd.DataFrame:
    print("Building shifted rolling drought features ...")
    combined = combined.sort_values(["region_id", "_ordinal"]).reset_index(drop=True)
    grp = combined.groupby("region_id", sort=False)

    shifted_prec = grp["prec"].shift(1)
    shifted_tmp = grp["tmp"].shift(1)
    shifted_humidity = grp["humidity"].shift(1)
    shifted_dew_spread = grp["dew_point_depression"].shift(1)
    shifted_wet_spread = grp["wet_bulb_depression"].shift(1)
    dry_flag = (combined["prec"] < 0.1).astype("float32")
    rain_flag = (combined["prec"] >= 1.0).astype("float32")
    hot_flag = (combined["tmp_max"] >= 30.0).astype("float32")
    shifted_dry = dry_flag.groupby(combined["region_id"], sort=False).shift(1)
    shifted_rain = rain_flag.groupby(combined["region_id"], sort=False).shift(1)
    shifted_hot = hot_flag.groupby(combined["region_id"], sort=False).shift(1)
    shifted_count = pd.Series(1.0, index=combined.index, dtype="float32").groupby(combined["region_id"], sort=False).shift(1)

    for window in (30, 90, 180):
        combined[f"prec_sum_{window}d"] = rolling_by_region(combined, shifted_prec, window, "sum")
        combined[f"dry_days_{window}d"] = rolling_by_region(combined, shifted_dry, window, "sum")
        combined[f"hist_days_{window}d"] = rolling_by_region(combined, shifted_count, window, "sum")

    combined["dry_share_90d"] = (combined["dry_days_90d"] / combined["hist_days_90d"].replace(0, np.nan)).astype("float32")
    combined["rain_days_90d"] = rolling_by_region(combined, shifted_rain, 90, "sum")
    combined["hot_days_90d"] = rolling_by_region(combined, shifted_hot, 90, "sum")
    combined["tmp_mean_90d"] = rolling_by_region(combined, shifted_tmp, 90, "mean")
    combined["humidity_mean_90d"] = rolling_by_region(combined, shifted_humidity, 90, "mean")
    combined["dew_depression_mean_90d"] = rolling_by_region(combined, shifted_dew_spread, 90, "mean")
    combined["wet_bulb_depression_mean_90d"] = rolling_by_region(combined, shifted_wet_spread, 90, "mean")

    dry_block = (dry_flag == 0).groupby(combined["region_id"], sort=False).cumsum()
    dry_streak = dry_flag.groupby([combined["region_id"], dry_block], sort=False).cumsum().astype("float32")
    combined["dry_streak_before"] = dry_streak.groupby(combined["region_id"], sort=False).shift(1).fillna(0).astype("float32")
    combined["max_dry_streak_90d"] = rolling_by_region(combined, combined["dry_streak_before"], 90, "max")
    combined["rolling_drought_idx_90d"] = (combined["tmp_mean_90d"] / (combined["prec_sum_90d"] + 0.1)).astype("float32")
    return combined


def build_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = add_daily_derived(add_date_parts(train.copy()))
    test = add_daily_derived(add_date_parts(test.copy()))
    train["_split"] = "train"
    test["_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True, copy=False)
    combined = add_rolling_features(combined)
    train_features = combined.loc[combined["_split"] == "train"].copy()
    test_features = combined.loc[combined["_split"] == "test"].copy()
    print(f"  feature frames: train={train_features.shape}, test={test_features.shape}")
    return train_features, test_features


def select_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("Selecting comparison baselines ...")
    baselines: dict[str, pd.DataFrame] = {"full_train": train}
    grouped = train.groupby("region_id", sort=False, group_keys=False)
    baselines["tail_1095d"] = grouped.tail(1095)
    baselines["tail_365d"] = grouped.tail(365)
    baselines["adjacent_prev_91d"] = grouped.tail(91)

    test_md_keys = test[["region_id", "_md"]].drop_duplicates()
    same_md_all = train.merge(test_md_keys, on=["region_id", "_md"], how="inner", sort=False)
    baselines["same_md_all_years"] = same_md_all

    test_start = test.groupby("region_id", sort=False)["_ordinal"].min().rename("_test_start")
    same_md_recent = same_md_all.merge(test_start, on="region_id", how="left", sort=False)
    same_md_recent = same_md_recent.loc[same_md_recent["_ordinal"] >= same_md_recent["_test_start"] - 366 * 5].drop(
        columns=["_test_start"]
    )
    baselines["same_md_recent_5y"] = same_md_recent

    test_year_keys = test[["region_id", "_md", "_year"]].drop_duplicates().rename(columns={"_year": "_test_year"})
    prev_candidates = train.merge(test_year_keys, on=["region_id", "_md"], how="inner", sort=False)
    prev_candidates = prev_candidates.loc[prev_candidates["_year"] < prev_candidates["_test_year"]]
    idx = prev_candidates.groupby(["region_id", "_md", "_test_year"], sort=False)["_year"].idxmax()
    baselines["prev_year_same_md"] = prev_candidates.loc[idx].drop(columns=["_test_year"])

    for name, frame in baselines.items():
        print(f"  {name}: {len(frame):,} rows")
    return baselines


def sample_series(series: pd.Series, max_samples: int, random_state: int) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= max_samples:
        return clean
    return clean.sample(max_samples, random_state=random_state)


def psi(expected: pd.Series, actual: pd.Series, bins: int) -> float:
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    unique = pd.concat([expected, actual], ignore_index=True).nunique(dropna=True)
    if unique <= 20:
        cats = np.union1d(expected.unique(), actual.unique())
        expected_pct = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        actual_pct = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected.to_numpy(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0] = -np.inf
        edges[-1] = np.inf
        expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    a_values = np.sort(a.replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    b_values = np.sort(b.replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    if len(a_values) == 0 or len(b_values) == 0:
        return float("nan")
    values = np.sort(np.concatenate([a_values, b_values]))
    cdf_a = np.searchsorted(a_values, values, side="right") / len(a_values)
    cdf_b = np.searchsorted(b_values, values, side="right") / len(b_values)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def feature_metric(train_frame: pd.DataFrame, test_frame: pd.DataFrame, col: str, bins: int, max_samples: int) -> dict:
    train_values = train_frame[col].replace([np.inf, -np.inf], np.nan)
    test_values = test_frame[col].replace([np.inf, -np.inf], np.nan)
    train_sample = sample_series(train_values, max_samples, 100 + len(col))
    test_sample = sample_series(test_values, max_samples, 200 + len(col))
    train_mean = float(train_values.mean())
    test_mean = float(test_values.mean())
    train_std = float(train_values.std())
    return {
        "feature": col,
        "feature_label": FEATURE_LABELS.get(col, col),
        "train_mean": train_mean,
        "test_mean": test_mean,
        "mean_delta": test_mean - train_mean,
        "std_mean_delta": (test_mean - train_mean) / train_std if train_std > 0 else float("nan"),
        "train_p10": float(train_values.quantile(0.10)),
        "test_p10": float(test_values.quantile(0.10)),
        "train_p50": float(train_values.quantile(0.50)),
        "test_p50": float(test_values.quantile(0.50)),
        "train_p90": float(train_values.quantile(0.90)),
        "test_p90": float(test_values.quantile(0.90)),
        "psi": psi(train_sample, test_sample, bins),
        "ks": ks_stat(train_sample, test_sample),
        "train_non_null": int(train_values.notna().sum()),
        "test_non_null": int(test_values.notna().sum()),
    }


def adversarial_auc(train_frame: pd.DataFrame, test_frame: pd.DataFrame, max_samples: int) -> float:
    auc_cols = [
        "prec",
        "humidity",
        "tmp",
        "dp_tmp",
        "tmp_max",
        "tmp_range",
        "wind",
        "dry_day",
        "dew_point_depression",
        "prec_sum_90d",
        "dry_days_90d",
        "tmp_mean_90d",
        "humidity_mean_90d",
        "max_dry_streak_90d",
    ]
    left = train_frame[auc_cols].sample(min(len(train_frame), max_samples), random_state=31)
    right = test_frame[auc_cols].sample(min(len(test_frame), max_samples), random_state=32)
    X = pd.concat([left, right], ignore_index=True).replace([np.inf, -np.inf], np.nan).astype("float64")
    # The ratio-style drought proxy is intentionally sensitive when rainfall is
    # tiny. Clip only for the auxiliary classifier so AUC is numerical, while
    # leaving the reported univariate metrics untouched.
    lower = X.quantile(0.005)
    upper = X.quantile(0.995)
    X = X.clip(lower=lower, upper=upper, axis=1)
    y = np.array([0] * len(left) + [1] * len(right))
    model = HistGradientBoostingClassifier(
        max_iter=80,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=41,
    )
    model.fit(X, y)
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def summarize_baseline(name: str, train_frame: pd.DataFrame, test_frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    print(f"Summarizing {name} ...")
    rows = [feature_metric(train_frame, test_frame, col, args.bins, args.max_samples) for col in ANALYSIS_COLS]
    avg_psi = float(np.nanmean([row["psi"] for row in rows]))
    avg_ks = float(np.nanmean([row["ks"] for row in rows]))
    auc = adversarial_auc(train_frame, test_frame, args.max_samples)
    summary = {
        "name": name,
        "label": BASELINE_LABELS.get(name, name),
        "rows": int(len(train_frame)),
        "adversarial_auc": auc,
        "avg_psi": avg_psi,
        "avg_ks": avg_ks,
    }
    return summary, rows


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def metric_lookup(feature_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["baseline"], row["feature"]): row for row in feature_rows}


def feature_value(lookup: dict[tuple[str, str], dict], baseline: str, feature: str, key: str = "test_mean") -> float:
    return float(lookup[(baseline, feature)][key])


def top_feature_table(feature_rows: list[dict], baseline: str, cols: list[str], limit: int = 12) -> list[str]:
    rows = [row for row in feature_rows if row["baseline"] == baseline and row["feature"] in cols]
    rows = sorted(rows, key=lambda row: (row["psi"], abs(row["std_mean_delta"])), reverse=True)[:limit]
    lines = ["| feature | train mean | test mean | delta | PSI | KS |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['feature_label']} | {fmt(row['train_mean'])} | {fmt(row['test_mean'])} | "
            f"{fmt(row['mean_delta'])} | {fmt(row['psi'])} | {fmt(row['ks'])} |"
        )
    return lines


def format_report(metrics: dict, feature_rows: list[dict], test: pd.DataFrame) -> str:
    lookup = metric_lookup(feature_rows)
    summary = metrics["baselines"]
    prev = "prev_year_same_md"
    tail = "tail_1095d"
    adjacent = "adjacent_prev_91d"
    full = "full_train"

    prev_dry_delta = feature_value(lookup, prev, "dry_day", "mean_delta")
    prev_prec_delta = feature_value(lookup, prev, "prec", "mean_delta")
    prev_dew_delta = feature_value(lookup, prev, "dew_point_depression", "mean_delta")
    prev_prec90_delta = feature_value(lookup, prev, "prec_sum_90d", "mean_delta")
    tail_temp_delta = feature_value(lookup, tail, "tmp", "mean_delta")
    adjacent_temp_delta = feature_value(lookup, adjacent, "tmp", "mean_delta")

    test_year_min = int(test["_year"].min())
    test_year_max = int(test["_year"].max())
    test_months = ", ".join(str(int(month)) for month in sorted(test["_month"].unique()))
    generated = metrics["created_at"]

    lines = [
        "# Train/Test Distribution Comparison",
        "",
        f"Generated: {generated}",
        "",
        "## 結論摘要",
        "",
        (
            "- 不能只用整個 train 跟 test 比，因為 test 是每個 region 訓練尾端之後的 91 天，"
            "季節位置不一致會製造很大的溫度 drift。這份報告改用數字化 synthetic date 排序，"
            "再加入同 region、同月日的歷史對照。"
        ),
        (
            f"- 以最嚴格的 `同 region 同月日前一年` 來看，test 的 dry-day 比例變化為 "
            f"`{prev_dry_delta:+.3f}`，日降雨均值變化為 `{prev_prec_delta:+.3f}`，"
            f"90 天累積雨量變化為 `{prev_prec90_delta:+.3f}`。"
        ),
        (
            f"- 因此「test 比較不會乾」沒有被主要乾旱指標支持。沒控季節時，test 的 humidity/"
            f"dew point 會因為較暖而偏高；但控同 region 同月日前一年後，降雨下降、無雨日上升，"
            f"daily dew-point depression 也增加 `{prev_dew_delta:+.3f}`。"
        ),
        (
            f"- 與最近 1095 天相比，test 平均氣溫高 `{tail_temp_delta:+.3f}`；與 test 前 91 天相比，"
            f"平均氣溫高 `{adjacent_temp_delta:+.3f}`。這代表最大 drift 是季節/溫度 regime，"
            "不是單純 precipitation regime。"
        ),
        "",
        "## 資料與日期口徑",
        "",
        f"- Train rows: `{metrics['train_rows']:,}`; test rows: `{metrics['test_rows']:,}`.",
        f"- Regions: `{metrics['regions']:,}`; test years span `{test_year_min}` to `{test_year_max}` because each region has its own synthetic timeline.",
        f"- Test months present across regions: `{test_months}`.",
        "- Tail windows and previous-year matching use parsed numeric `(year, month, day)` ordinals, not raw string date sorting.",
        "",
        "## Baseline Summary",
        "",
        "| baseline | rows | adversarial AUC | avg PSI | avg KS | prec mean | dry-day share | tmp mean | dew-point spread | 90d precip | 90d dry days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        name = item["name"]
        lines.append(
            f"| {item['label']} | {item['rows']:,} | {fmt(item['adversarial_auc'], 4)} | "
            f"{fmt(item['avg_psi'], 4)} | {fmt(item['avg_ks'], 4)} | "
            f"{fmt(feature_value(lookup, name, 'prec', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'prec'))} | "
            f"{fmt(feature_value(lookup, name, 'dry_day', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'dry_day'))} | "
            f"{fmt(feature_value(lookup, name, 'tmp', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'tmp'))} | "
            f"{fmt(feature_value(lookup, name, 'dew_point_depression', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'dew_point_depression'))} | "
            f"{fmt(feature_value(lookup, name, 'prec_sum_90d', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'prec_sum_90d'))} | "
            f"{fmt(feature_value(lookup, name, 'dry_days_90d', 'train_mean'))} -> {fmt(feature_value(lookup, name, 'dry_days_90d'))} |"
        )

    lines.extend(
        [
            "",
            "AUC 越接近 `0.5` 表示 train baseline 與 test 越難分開；PSI/KS 越低表示單變量分佈越接近。",
            "",
            "## Dryness-Focused Checks",
            "",
            "| feature | 全部 train -> test | 最近 1095 天 -> test | 同月日前一年 -> test |",
            "|---|---:|---:|---:|",
        ]
    )
    dryness_features = [
        "prec",
        "dry_day",
        "rain_day_ge_1",
        "heavy_precip_ge_10",
        "prec_sum_90d",
        "dry_days_90d",
        "dry_share_90d",
        "max_dry_streak_90d",
        "tmp_max",
        "tmp_range",
        "dew_point_depression",
        "wet_bulb_depression",
        "rolling_drought_idx_90d",
    ]
    for feature in dryness_features:
        lines.append(
            f"| {FEATURE_LABELS[feature]} | "
            f"{fmt(feature_value(lookup, full, feature, 'train_mean'))} -> {fmt(feature_value(lookup, full, feature))} | "
            f"{fmt(feature_value(lookup, tail, feature, 'train_mean'))} -> {fmt(feature_value(lookup, tail, feature))} | "
            f"{fmt(feature_value(lookup, prev, feature, 'train_mean'))} -> {fmt(feature_value(lookup, prev, feature))} |"
        )

    lines.extend(
        [
            "",
            "## Most Shifted Raw Weather Features",
            "",
            "### 全部 train vs test",
            "",
            *top_feature_table(feature_rows, full, RAW_REPORT_COLS),
            "",
            "### test 前 91 天 vs test",
            "",
            *top_feature_table(feature_rows, adjacent, RAW_REPORT_COLS),
            "",
            "### 同 region 同月日前一年 vs test",
            "",
            *top_feature_table(feature_rows, prev, RAW_REPORT_COLS),
            "",
            "## Interpretation",
            "",
            "- 沒控季節的比較中，`humidity` 和 `dp_tmp/wb_tmp` 在 test 較高，這是「看起來比較濕」的主要來源；但這同時伴隨 test 明顯更熱。",
            "- 控同 region 同月日前一年後，test 的 `humidity`、`dp_tmp`、`wb_tmp` 反而較低，而 `tmp_range`、`tmp - dp_tmp`、`tmp - wb_tmp` 較高，表示大氣蒸散需求較強。",
            "- 降雨訊號沒有支持「test 明顯比較不乾」：test 的日均降雨、90 天累積雨量、heavy-rain share 大多低於全 train、最近 1095 天、以及同月日前一年。",
            "- Rolling 90 天特徵要另外讀：test 的 rolling context 包含 test 前 91 天與 test window 前段，因此它回答的是模型推論時看見的近期背景，不等於 test 當天的即時天氣。",
            "- 對模型策略的含意：比起把 test 當成全面濕潤 regime，較合理的是處理溫度/季節 drift，並讓 rolling precipitation、dry-day streak、dew-point depression 這類 proxy 在驗證中被單獨檢查。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    out_path = resolve_path(
        args.out,
        ROOT / "docs" / "validation" / "train_test_distribution_comparison_20260523.md",
    )
    json_path = resolve_path(args.json_out, out_path.with_suffix(".json"))
    csv_path = resolve_path(args.csv_out, out_path.with_name(out_path.stem + "_feature_metrics.csv"))

    train_raw, test_raw = load_weather()
    train_features, test_features = build_feature_frames(train_raw, test_raw)
    baselines = select_baselines(train_features, test_features)

    summaries = []
    feature_rows = []
    for name, frame in baselines.items():
        summary, rows = summarize_baseline(name, frame, test_features, args)
        summaries.append(summary)
        for row in rows:
            row["baseline"] = name
            row["baseline_label"] = BASELINE_LABELS.get(name, name)
            feature_rows.append(row)

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "regions": int(test_features["region_id"].nunique()),
        "max_samples": args.max_samples,
        "bins": args.bins,
        "baselines": summaries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_report(metrics, feature_rows, test_features), encoding="utf-8")
    pd.DataFrame(feature_rows).to_csv(csv_path, index=False)
    save_json(json_path, {"metrics": metrics, "feature_rows": feature_rows})
    print(f"Markdown report saved -> {out_path}")
    print(f"Feature metrics saved -> {csv_path}")
    print(f"JSON metrics saved -> {json_path}")


if __name__ == "__main__":
    main()
