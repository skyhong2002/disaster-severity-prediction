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

def main():
    parser = argparse.ArgumentParser(description="Ensemble LightGBM and XGBoost submissions")
    parser.add_argument("--lgb", required=True, help="Path to LightGBM submission CSV")
    parser.add_argument("--xgb", required=True, help="Path to XGBoost submission CSV")
    parser.add_argument("--out", default="submissions/ensemble.csv", help="Output path")
    parser.add_argument("--lgb-weight", type=float, default=0.5, help="Weight for LightGBM")
    parser.add_argument(
        "--lgb-weights",
        default=None,
        help="Comma-separated LightGBM weights. Provide one value or one per pred_week column.",
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

    assert len(df_lgb) == len(df_xgb), "Submissions must have the same number of rows!"
    assert all(df_lgb["region_id"] == df_xgb["region_id"]), "Region IDs must match!"

    pred_cols = [c for c in df_lgb.columns if c.startswith("pred_week")]

    if args.target:
        df_target = pd.read_csv(args.target)
        assert all(df_lgb["region_id"] == df_target["region_id"]), "Target region IDs must match!"
        lgb_weights = search_weights(df_lgb, df_xgb, df_target, pred_cols, args.grid_step)
    else:
        lgb_weights = parse_weights(args.lgb_weights, args.lgb_weight, len(pred_cols))
    
    df_ens = df_lgb[["region_id"]].copy()

    print("Ensembling with weights:")
    for col, lgb_weight in zip(pred_cols, lgb_weights):
        xgb_weight = 1.0 - lgb_weight
        print(f"  {col}: LGBM={lgb_weight:.2f}, XGB={xgb_weight:.2f}")
        df_ens[col] = df_lgb[col] * lgb_weight + df_xgb[col] * xgb_weight
        df_ens[col] = df_ens[col].clip(0, 5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ens.to_csv(out_path, index=False)
    
    print(f"Ensemble saved to {out_path}")
    print(df_ens.head())

if __name__ == "__main__":
    main()
