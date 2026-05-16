import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error


def parse_weights(raw: str | None, default: float, n_cols: int) -> list[float]:
    """Parse either one global weight or one weight per horizon."""
    if raw is None:
        return [default] * n_cols
    weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(weights) == 1:
        return weights * n_cols
    if len(weights) != n_cols:
        raise ValueError(f"Expected 1 or {n_cols} weights, got {len(weights)}.")
    return weights


def parse_model_weights(raw: str | None, model_names: list[str], n_cols: int) -> dict[str, list[float]]:
    """Parse semicolon-separated per-model weights, e.g. lgb=0.4;xgb=0.4;cat=0.2."""
    if raw is None:
        equal = 1.0 / len(model_names)
        return {name: [equal] * n_cols for name in model_names}

    parsed: dict[str, list[float]] = {}
    for part in raw.split(";"):
        if not part.strip():
            continue
        name, values = part.split("=", 1)
        name = name.strip()
        if name not in model_names:
            raise ValueError(f"Unknown model weight key: {name}")
        parsed[name] = parse_weights(values, 0.0, n_cols)

    missing = [name for name in model_names if name not in parsed]
    if missing:
        raise ValueError(f"Missing weights for: {', '.join(missing)}")

    for idx in range(n_cols):
        total = sum(parsed[name][idx] for name in model_names)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights for horizon {idx + 1} sum to {total:.6f}, expected 1.0.")
    return parsed


def search_weights(
    df_lgb: pd.DataFrame,
    df_xgb: pd.DataFrame,
    df_target: pd.DataFrame,
    pred_cols: list[str],
    grid_step: float,
) -> list[float]:
    """Find the best LightGBM blend weight for each horizon by validation MAE."""
    weights = []
    grid = np.arange(0, 1 + grid_step / 2, grid_step)
    for col in pred_cols:
        target_col = col.replace("pred_", "target_")
        if target_col not in df_target.columns:
            target_col = col
        if target_col not in df_target.columns:
            raise ValueError(f"Target file needs column {target_col} or {col}.")

        best_weight = 0.5
        best_mae = float("inf")
        y = df_target[target_col]
        for weight in grid:
            pred = df_lgb[col] * weight + df_xgb[col] * (1.0 - weight)
            mae = mean_absolute_error(y, pred)
            if mae < best_mae:
                best_mae = mae
                best_weight = float(weight)
        print(f"Best {col}: LGBM={best_weight:.2f}, XGB={1-best_weight:.2f}, MAE={best_mae:.5f}")
        weights.append(best_weight)
    return weights


def search_three_weights(
    dfs: dict[str, pd.DataFrame],
    df_target: pd.DataFrame,
    pred_cols: list[str],
    grid_step: float,
) -> dict[str, list[float]]:
    """Grid-search non-negative three-model weights summing to one per horizon."""
    model_names = list(dfs)
    if len(model_names) != 3:
        raise ValueError("search_three_weights expects exactly three models.")

    weights = {name: [] for name in model_names}
    grid = np.arange(0, 1 + grid_step / 2, grid_step)
    for col in pred_cols:
        target_col = col.replace("pred_", "target_")
        if target_col not in df_target.columns:
            target_col = col
        if target_col not in df_target.columns:
            raise ValueError(f"Target file needs column {target_col} or {col}.")

        best = (float("inf"), None)
        y = df_target[target_col]
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                w2 = max(0.0, w2)
                pred = (
                    dfs[model_names[0]][col] * w0
                    + dfs[model_names[1]][col] * w1
                    + dfs[model_names[2]][col] * w2
                )
                mae = mean_absolute_error(y, pred)
                if mae < best[0]:
                    best = (mae, (float(w0), float(w1), float(w2)))

        assert best[1] is not None
        print(
            f"Best {col}: "
            + ", ".join(f"{name}={weight:.2f}" for name, weight in zip(model_names, best[1]))
            + f", MAE={best[0]:.5f}"
        )
        for name, weight in zip(model_names, best[1]):
            weights[name].append(weight)
    return weights

def main():
    parser = argparse.ArgumentParser(description="Ensemble LightGBM, XGBoost, and optionally CatBoost submissions")
    parser.add_argument("--lgb", required=True, help="Path to LightGBM submission CSV")
    parser.add_argument("--xgb", required=True, help="Path to XGBoost submission CSV")
    parser.add_argument("--cat", default=None, help="Optional path to CatBoost submission CSV")
    parser.add_argument("--out", default="submissions/ensemble.csv", help="Output path")
    parser.add_argument("--lgb-weight", type=float, default=0.5, help="Weight for LightGBM")
    parser.add_argument(
        "--lgb-weights",
        default=None,
        help="Comma-separated LightGBM weights. Provide one value or one per pred_week column.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Semicolon-separated model weights, e.g. lgb=0.4;xgb=0.4;cat=0.2. Values can be per-horizon CSVs.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional validation target CSV for grid-searching per-horizon blend weights.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Grid step used with --target.",
    )
    args = parser.parse_args()

    print(f"Loading LightGBM from {args.lgb}")
    df_lgb = pd.read_csv(args.lgb)
    
    print(f"Loading XGBoost from {args.xgb}")
    df_xgb = pd.read_csv(args.xgb)
    dfs = {"lgb": df_lgb, "xgb": df_xgb}

    if args.cat:
        print(f"Loading CatBoost from {args.cat}")
        dfs["cat"] = pd.read_csv(args.cat)

    for name, df in dfs.items():
        assert len(df_lgb) == len(df), f"{name} submission must have the same number of rows!"
        assert all(df_lgb["region_id"] == df["region_id"]), f"{name} region IDs must match!"

    pred_cols = [c for c in df_lgb.columns if c.startswith("pred_week")]

    if args.target:
        df_target = pd.read_csv(args.target)
        assert all(df_lgb["region_id"] == df_target["region_id"]), "Target region IDs must match!"
        if args.cat:
            model_weights = search_three_weights(dfs, df_target, pred_cols, args.grid_step)
        else:
            lgb_weights = search_weights(df_lgb, df_xgb, df_target, pred_cols, args.grid_step)
            model_weights = {"lgb": lgb_weights, "xgb": [1.0 - w for w in lgb_weights]}
    elif args.cat:
        model_weights = parse_model_weights(args.weights, list(dfs), len(pred_cols))
    else:
        lgb_weights = parse_weights(args.lgb_weights, args.lgb_weight, len(pred_cols))
        model_weights = {"lgb": lgb_weights, "xgb": [1.0 - w for w in lgb_weights]}
    
    df_ens = df_lgb[["region_id"]].copy()

    print("Ensembling with weights:")
    for idx, col in enumerate(pred_cols):
        weight_line = ", ".join(f"{name}={model_weights[name][idx]:.2f}" for name in dfs)
        print(f"  {col}: {weight_line}")
        df_ens[col] = sum(dfs[name][col] * model_weights[name][idx] for name in dfs)
        df_ens[col] = df_ens[col].clip(0, 5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ens.to_csv(out_path, index=False)
    
    print(f"Ensemble saved to {out_path}")
    print(df_ens.head())

if __name__ == "__main__":
    main()
