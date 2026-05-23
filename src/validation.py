"""Kaggle-like blind-window validation utilities.

The Kaggle test setup provides 91 days of meteorological observations with no
scores, then asks for five future weekly scores.  These helpers reproduce that
shape inside the historical training set by masking score labels in a 91-day
window before rebuilding features and predicting from the final window row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from features import METEO_COLS, SCORE_GAP_DAYS, build_features, get_feature_cols, parse_synthetic_date_parts
from model_wrappers import get_model_feature_names, predict_model_or_ensemble


N_WEEKS = 5


@dataclass(frozen=True)
class BlindBacktestOrigin:
    """One pseudo Kaggle forecast origin, expressed as weeks from train tail."""

    offset_weeks: int
    label: str


def parse_origin_offsets(raw: str) -> list[BlindBacktestOrigin]:
    """Parse comma-separated offsets such as ``5,13,26`` or ``-5,-13``."""
    origins: list[BlindBacktestOrigin] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        offset = abs(int(item.strip()))
        if offset < N_WEEKS:
            raise ValueError(f"Origin offset must be >= {N_WEEKS} weeks, got {offset}.")
        origins.append(BlindBacktestOrigin(offset_weeks=offset, label=f"tail_minus_{offset}w"))
    if not origins:
        raise ValueError("At least one blind-backtest origin is required.")
    return origins


def add_weekly_targets(train: pd.DataFrame) -> pd.DataFrame:
    """Return weekly score rows with future week targets and daily row indices."""
    work = train.sort_values(["region_id", "date"]).copy()
    work["_day_idx"] = work.groupby("region_id").cumcount().astype(np.int32)

    weekly = (
        work.dropna(subset=["score"])
        .sort_values(["region_id", "date"])
        .reset_index(drop=True)
    )
    weekly["week_idx"] = weekly.groupby("region_id").cumcount().astype(np.int32)
    weekly["max_week_idx"] = weekly.groupby("region_id")["week_idx"].transform("max").astype(np.int32)
    for week in range(1, N_WEEKS + 1):
        weekly[f"target_w{week}"] = weekly.groupby("region_id")["score"].shift(-week)
    return weekly.dropna(subset=[f"target_w{week}" for week in range(1, N_WEEKS + 1)])


def blind_score_mask(
    train: pd.DataFrame,
    origin: BlindBacktestOrigin,
    blind_days: int = SCORE_GAP_DAYS,
) -> pd.Series:
    """Return rows whose scores should be hidden for one pseudo origin."""
    work = train.sort_values(["region_id", "date"]).copy()
    work["_source_index"] = work.index
    work["_day_idx"] = work.groupby("region_id").cumcount().astype(np.int32)
    weekly = add_weekly_targets(work)

    origin_rows = weekly[weekly["week_idx"] == weekly["max_week_idx"] - origin.offset_weeks].copy()
    if origin_rows.empty:
        raise ValueError(f"No weekly rows found for origin {origin.label}.")

    origin_lookup = origin_rows[["region_id", "_day_idx"]].rename(columns={"_day_idx": "_origin_day_idx"})
    origin_lookup["_blind_start_idx"] = origin_lookup["_origin_day_idx"] - blind_days + 1
    merged = work.merge(origin_lookup, on="region_id", how="inner")
    masked_source_indices = merged.loc[
        (merged["_day_idx"] >= merged["_blind_start_idx"])
        & (merged["_day_idx"] <= merged["_origin_day_idx"]),
        "_source_index",
    ]
    mask = pd.Series(False, index=train.index)
    mask.loc[masked_source_indices.to_numpy()] = True
    return mask


def make_blind_backtest_origin(
    train: pd.DataFrame,
    origin: BlindBacktestOrigin,
    blind_days: int = SCORE_GAP_DAYS,
    history_tail_days: int = 750,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build masked raw rows, time-safe stats rows, and future targets for one origin."""
    if history_tail_days < blind_days:
        raise ValueError("history_tail_days must be >= blind_days.")

    work = train.sort_values(["region_id", "date"]).copy()
    work["_day_idx"] = work.groupby("region_id").cumcount().astype(np.int32)
    weekly = add_weekly_targets(work)

    origin_rows = weekly[weekly["week_idx"] == weekly["max_week_idx"] - origin.offset_weeks].copy()
    if origin_rows.empty:
        raise ValueError(f"No weekly rows found for origin {origin.label}.")

    origin_lookup = origin_rows[["region_id", "date", "_day_idx"]].rename(
        columns={"date": "_origin_date", "_day_idx": "_origin_day_idx"}
    )
    origin_lookup["_blind_start_idx"] = origin_lookup["_origin_day_idx"] - blind_days + 1
    if (origin_lookup["_blind_start_idx"] < 0).any():
        raise ValueError(f"Origin {origin.label} does not have {blind_days} prior daily rows.")

    merged = work.merge(origin_lookup, on="region_id", how="inner")
    start_idx = merged["_origin_day_idx"] - history_tail_days + 1
    history_mask = (merged["_day_idx"] <= merged["_origin_day_idx"]) & (merged["_day_idx"] >= start_idx)
    combined = merged.loc[history_mask].copy()

    blind_mask = combined["_day_idx"] >= combined["_blind_start_idx"]
    combined.loc[blind_mask, "score"] = np.nan
    train_history = combined.loc[~blind_mask].copy()

    target_cols = ["region_id", "date", "_day_idx"] + [f"target_w{week}" for week in range(1, N_WEEKS + 1)]
    targets = origin_rows[target_cols].rename(columns={"date": "origin_date", "_day_idx": "origin_day_idx"})
    return combined, train_history, targets


def build_pseudo_test_window(
    train: pd.DataFrame,
    origin: BlindBacktestOrigin,
    feature_options: dict,
    blind_days: int = SCORE_GAP_DAYS,
    history_tail_days: int = 750,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build final-row features and targets for one pseudo Kaggle origin."""
    combined, train_history, targets = make_blind_backtest_origin(
        train,
        origin,
        blind_days=blind_days,
        history_tail_days=history_tail_days,
    )
    feature_df = build_features(
        combined.drop(columns=[c for c in combined.columns if c.startswith("_origin") or c == "_blind_start_idx"]),
        train_history.drop(columns=[c for c in train_history.columns if c.startswith("_origin") or c == "_blind_start_idx"]),
        is_train=False,
        use_score_history=bool(feature_options.get("use_score_history", False)),
        score_gap_days=int(feature_options.get("score_gap_days", blind_days)),
        use_climatology=bool(feature_options.get("use_climatology", True)),
        use_region_stats=bool(feature_options.get("use_region_stats", False)),
        feature_profile=str(feature_options.get("feature_profile", "micro")),
        max_score_lag_weeks=feature_options.get("max_score_lag_weeks"),
        drop_feature_groups=feature_options.get("drop_feature_groups", []),
    )

    row_keys = targets[["region_id", "origin_day_idx"]].rename(columns={"origin_day_idx": "_day_idx"})
    if "_day_idx" not in feature_df.columns:
        feature_df = feature_df.merge(
            combined[["region_id", "date", "_day_idx"]],
            on=["region_id", "date"],
            how="left",
        )
    pseudo_last = feature_df.merge(row_keys, on=["region_id", "_day_idx"], how="inner")
    return pseudo_last, targets


def model_feature_columns(model, fallback_cols: list[str]) -> list[str]:
    """Return feature columns expected by a single model or fold ensemble wrapper."""
    names = get_model_feature_names(model)
    return list(names) if names is not None else fallback_cols


def predict_blind_origin(models: dict, pseudo_last: pd.DataFrame) -> pd.DataFrame:
    """Predict five future weeks from pseudo-test final rows."""
    fallback_cols = [c for c in get_feature_cols(pseudo_last) if c in pseudo_last.columns]
    result = pseudo_last[["region_id"]].copy()
    for week in range(1, N_WEEKS + 1):
        model = models[week]
        feat_cols = model_feature_columns(model, fallback_cols)
        missing = [col for col in feat_cols if col not in pseudo_last.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} feature columns for week {week}: {missing[:10]}")
        feature_na = pseudo_last[feat_cols].replace([np.inf, -np.inf], np.nan).isna().sum()
        feature_na = feature_na[feature_na > 0].sort_values(ascending=False)
        if len(feature_na):
            raise ValueError(
                f"Pseudo-test features contain missing/non-finite values for week {week}: "
                f"{feature_na.head(10).to_dict()}"
            )
        result[f"pred_week{week}"] = np.clip(predict_model_or_ensemble(model, pseudo_last[feat_cols]), 0, 5)
    return result


def compute_region_clusters(train: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Create coarse climate/score clusters for stress reporting only."""
    score_stats = (
        train.dropna(subset=["score"])
        .groupby("region_id")["score"]
        .agg(score_mean="mean", score_zero_ratio=lambda s: float((s == 0).mean()))
    )
    meteo_stats = train.groupby("region_id")[METEO_COLS].mean()
    stats = meteo_stats.join(score_stats, how="left").fillna(0)
    n_clusters = max(1, min(n_clusters, len(stats)))
    scaled = StandardScaler().fit_transform(stats)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    return pd.DataFrame({"region_id": stats.index, "region_cluster": labels.astype(int)})


def _stats(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _flattened_segment_mae(all_rows: pd.DataFrame, predicate) -> float | None:
    errors = []
    for week in range(1, N_WEEKS + 1):
        target = all_rows[f"target_w{week}"]
        mask = predicate(target)
        if mask.any():
            errors.append(np.abs(all_rows.loc[mask, f"pred_week{week}"] - target.loc[mask]).to_numpy())
    if not errors:
        return None
    return float(np.concatenate(errors).mean())


def evaluate_submission_like_predictions(
    prediction_frames: Iterable[pd.DataFrame],
    target_frames: Iterable[pd.DataFrame],
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return row-level predictions plus aggregate blind-backtest metrics."""
    rows = []
    clusters = compute_region_clusters(train)

    for preds, targets in zip(prediction_frames, target_frames):
        merge_keys = ["region_id"]
        if "origin" in preds.columns and "origin" in targets.columns:
            merge_keys.append("origin")
        merged = preds.merge(targets, on=merge_keys, how="inner").merge(clusters, on="region_id", how="left")
        _, origin_month, _ = parse_synthetic_date_parts(merged["origin_date"])
        merged["origin_month"] = origin_month.astype(int)
        rows.append(merged)

    all_rows = pd.concat(rows, ignore_index=True)
    pred_cols = [f"pred_week{week}" for week in range(1, N_WEEKS + 1)]
    target_cols = [f"target_w{week}" for week in range(1, N_WEEKS + 1)]

    metrics: dict = {
        "rows": int(len(all_rows)),
        "overall_mae": float(mean_absolute_error(all_rows[target_cols].to_numpy().ravel(), all_rows[pred_cols].to_numpy().ravel())),
        "mae_by_horizon": {},
        "mae_by_origin": {},
        "mae_by_region_cluster": {},
        "mae_by_region": {},
        "mae_by_calendar_month": {},
        "target_segment_mae": {},
        "prediction_stats": {},
        "target_stats": {},
    }
    for week in range(1, N_WEEKS + 1):
        metrics["mae_by_horizon"][f"week_{week}"] = float(
            mean_absolute_error(all_rows[f"target_w{week}"], all_rows[f"pred_week{week}"])
        )
        metrics["prediction_stats"][f"pred_week{week}"] = _stats(all_rows[f"pred_week{week}"])
        metrics["target_stats"][f"target_w{week}"] = _stats(all_rows[f"target_w{week}"])

    abs_err = np.vstack(
        [np.abs(all_rows[f"pred_week{week}"] - all_rows[f"target_w{week}"]) for week in range(1, N_WEEKS + 1)]
    ).mean(axis=0)
    all_rows["_row_mae"] = abs_err
    if "origin" in all_rows.columns:
        metrics["mae_by_origin"] = {
            str(key): float(value)
            for key, value in all_rows.groupby("origin")["_row_mae"].mean().sort_index().items()
        }
    metrics["mae_by_region"] = {
        str(key): float(value)
        for key, value in all_rows.groupby("region_id")["_row_mae"].mean().sort_index().items()
    }
    metrics["mae_by_region_cluster"] = {
        str(key): float(value)
        for key, value in all_rows.groupby("region_cluster")["_row_mae"].mean().sort_index().items()
    }
    metrics["mae_by_calendar_month"] = {
        str(key): float(value)
        for key, value in all_rows.groupby("origin_month")["_row_mae"].mean().sort_index().items()
    }
    metrics["target_segment_mae"] = {
        "zero_score": _flattened_segment_mae(all_rows, lambda s: s == 0),
        "nonzero_score": _flattened_segment_mae(all_rows, lambda s: s > 0),
        "high_severity_ge3": _flattened_segment_mae(all_rows, lambda s: s >= 3),
    }
    return all_rows.drop(columns=["_row_mae"]), metrics
