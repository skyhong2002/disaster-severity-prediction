"""Validation-driven constrained blending helpers."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def parse_named_paths(raw: str) -> dict[str, str]:
    """Parse ``name=path,name2=path2`` into a dictionary."""
    parsed = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, path = part.split("=", 1)
        parsed[name.strip()] = path.strip()
    if len(parsed) < 2:
        raise ValueError("At least two prediction files are required.")
    return parsed


def parse_anchor(raw: str | None, model_names: list[str]) -> dict[str, float]:
    """Parse ``lgb=0.35;xgb=0.35;cat=0.30`` or return equal weights."""
    if raw is None:
        equal = 1.0 / len(model_names)
        return {name: equal for name in model_names}
    parsed = {}
    for part in raw.split(";"):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        parsed[name.strip()] = float(value)
    missing = [name for name in model_names if name not in parsed]
    if missing:
        raise ValueError(f"Anchor missing weights for: {', '.join(missing)}")
    total = sum(parsed.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Anchor weights sum to {total:.6f}, expected 1.0.")
    return parsed


def parse_caps(raw: str | None, model_names: list[str]) -> dict[str, float]:
    """Parse optional max-weight caps, e.g. ``lgbm_micro=0.15``."""
    caps = {name: 1.0 for name in model_names}
    if raw is None:
        return caps
    for part in raw.split(";"):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in model_names:
            raise ValueError(f"Unknown model cap key: {name}")
        caps[name] = float(value)
    return caps


def simplex_grid(n_models: int, step: float, caps: np.ndarray | None = None):
    """Yield non-negative weights summing to one."""
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--grid-step must evenly divide 1.0, e.g. 0.01, 0.02, 0.05.")
    caps = np.ones(n_models, dtype=np.float64) if caps is None else caps
    for cuts in itertools.product(range(units + 1), repeat=n_models - 1):
        if sum(cuts) > units:
            continue
        weights = list(cuts) + [units - sum(cuts)]
        candidate = np.array(weights, dtype=np.float64) / units
        if np.all(candidate <= caps + 1e-12):
            yield candidate


def align_prediction_frames(frames: dict[str, pd.DataFrame], target: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Align predictions and target by region plus origin when available."""
    first = next(iter(frames.values()))
    keys = ["region_id"]
    if "origin" in first.columns and "origin" in target.columns:
        keys.append("origin")

    base = target[keys].drop_duplicates()
    aligned_target = base.merge(target, on=keys, how="left")
    aligned = {}
    for name, frame in frames.items():
        aligned[name] = base.merge(frame, on=keys, how="left")
        if aligned[name].filter(like="pred_week").isna().any().any():
            raise ValueError(f"{name} predictions have missing rows after alignment.")
    return aligned, aligned_target


def target_col_for_prediction(pred_col: str, target: pd.DataFrame) -> str:
    week_suffix = pred_col.replace("pred_week", "")
    target_candidates = [pred_col.replace("pred_", "target_"), f"target_w{week_suffix}"]
    target_col = next((col for col in target_candidates if col in target.columns), None)
    if target_col is None:
        raise ValueError(f"Target file missing one of: {', '.join(target_candidates)}.")
    return target_col


def fit_constrained_weights_from_aligned(
    aligned: dict[str, pd.DataFrame],
    aligned_target: pd.DataFrame,
    model_names: list[str],
    anchor: dict[str, float],
    grid_step: float = 0.02,
    lambda_reg: float = 0.05,
    caps: dict[str, float] | None = None,
    row_indices: np.ndarray | None = None,
) -> tuple[dict[str, list[float]], dict]:
    """Fit weights from already aligned frames, optionally on sampled rows."""
    anchor_vec = np.array([anchor[name] for name in model_names], dtype=np.float64)
    caps_vec = np.array([(caps or {}).get(name, 1.0) for name in model_names], dtype=np.float64)
    pred_cols = [col for col in aligned[model_names[0]].columns if col.startswith("pred_week")]
    row_indices = np.arange(len(aligned_target)) if row_indices is None else row_indices

    weights = {name: [] for name in model_names}
    metrics = {"mae_by_horizon": {}, "objective_by_horizon": {}, "residual_correlation": {}}
    grid = list(simplex_grid(len(model_names), grid_step, caps=caps_vec))
    if not grid:
        raise ValueError("No blend weights satisfy the requested caps.")

    for pred_col in pred_cols:
        target_col = target_col_for_prediction(pred_col, aligned_target)
        y = aligned_target[target_col].iloc[row_indices].to_numpy()
        pred_matrix = np.vstack([aligned[name][pred_col].iloc[row_indices].to_numpy() for name in model_names])

        best = (float("inf"), None, None)
        for candidate in grid:
            pred = candidate @ pred_matrix
            mae = mean_absolute_error(y, pred)
            objective = mae + lambda_reg * float(np.sum((candidate - anchor_vec) ** 2))
            if objective < best[0]:
                best = (objective, candidate, mae)

        assert best[1] is not None and best[2] is not None
        for name, value in zip(model_names, best[1]):
            weights[name].append(float(value))
        metrics["mae_by_horizon"][pred_col] = float(best[2])
        metrics["objective_by_horizon"][pred_col] = float(best[0])

        residuals = {
            name: aligned[name][pred_col].iloc[row_indices].to_numpy() - y
            for name in model_names
        }
        corr = pd.DataFrame(residuals).corr().fillna(0.0)
        metrics["residual_correlation"][pred_col] = corr.to_dict()

    return weights, metrics


def fit_constrained_weights(
    frames: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    anchor: dict[str, float],
    grid_step: float = 0.02,
    lambda_reg: float = 0.05,
    caps: dict[str, float] | None = None,
) -> tuple[dict[str, list[float]], dict]:
    """Grid-search non-negative per-horizon weights with anchor regularization."""
    model_names = list(frames)
    aligned, aligned_target = align_prediction_frames(frames, target)
    return fit_constrained_weights_from_aligned(
        aligned,
        aligned_target,
        model_names,
        anchor,
        grid_step=grid_step,
        lambda_reg=lambda_reg,
        caps=caps,
    )


def bootstrap_constrained_weights(
    frames: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    anchor: dict[str, float],
    grid_step: float = 0.02,
    lambda_reg: float = 0.05,
    caps: dict[str, float] | None = None,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict:
    """Bootstrap blend weights by origin when available, otherwise by row."""
    model_names = list(frames)
    aligned, aligned_target = align_prediction_frames(frames, target)
    rng = np.random.default_rng(random_state)

    if "origin" in aligned_target.columns:
        group_values = aligned_target["origin"].astype(str).to_numpy()
        unique_groups = np.unique(group_values)
        group_indices = {group: np.flatnonzero(group_values == group) for group in unique_groups}
        sampler_size = len(unique_groups)
        def sample_indices():
            sampled = rng.choice(unique_groups, size=sampler_size, replace=True)
            return np.concatenate([group_indices[group] for group in sampled])
    else:
        n_rows = len(aligned_target)
        def sample_indices():
            return rng.integers(0, n_rows, size=n_rows)

    pred_cols = [col for col in aligned[model_names[0]].columns if col.startswith("pred_week")]
    draws = []
    for _ in range(n_bootstrap):
        row_indices = sample_indices()
        weights, _ = fit_constrained_weights_from_aligned(
            aligned,
            aligned_target,
            model_names,
            anchor,
            grid_step=grid_step,
            lambda_reg=lambda_reg,
            caps=caps,
            row_indices=row_indices,
        )
        draws.append(weights)

    summary = {}
    for horizon_idx, pred_col in enumerate(pred_cols):
        summary[pred_col] = {}
        for name in model_names:
            values = np.array([draw[name][horizon_idx] for draw in draws], dtype=np.float64)
            summary[pred_col][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
            }
    return {
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
        "resample_unit": "origin" if "origin" in aligned_target.columns else "row",
        "summary": summary,
    }


def apply_weights(frames: dict[str, pd.DataFrame], weights: dict[str, list[float]]) -> pd.DataFrame:
    """Apply per-horizon weights to aligned submission-like prediction frames."""
    model_names = list(frames)
    first = frames[model_names[0]]
    result_cols = ["region_id"] + (["origin"] if "origin" in first.columns else [])
    result = first[result_cols].copy()
    pred_cols = [col for col in first.columns if col.startswith("pred_week")]
    for idx, col in enumerate(pred_cols):
        result[col] = sum(frames[name][col] * weights[name][idx] for name in model_names).clip(0, 5)
    return result
