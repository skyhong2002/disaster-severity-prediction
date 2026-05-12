import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Ensemble LightGBM and XGBoost submissions")
    parser.add_argument("--lgb", required=True, help="Path to LightGBM submission CSV")
    parser.add_argument("--xgb", required=True, help="Path to XGBoost submission CSV")
    parser.add_argument("--out", default="submissions/ensemble.csv", help="Output path")
    parser.add_argument("--lgb-weight", type=float, default=0.5, help="Weight for LightGBM")
    args = parser.parse_args()

    print(f"Loading LightGBM from {args.lgb}")
    df_lgb = pd.read_csv(args.lgb)
    
    print(f"Loading XGBoost from {args.xgb}")
    df_xgb = pd.read_csv(args.xgb)

    assert len(df_lgb) == len(df_xgb), "Submissions must have the same number of rows!"
    assert all(df_lgb["region_id"] == df_xgb["region_id"]), "Region IDs must match!"

    pred_cols = [c for c in df_lgb.columns if c.startswith("pred_week")]
    
    df_ens = df_lgb[["region_id"]].copy()
    xgb_weight = 1.0 - args.lgb_weight

    print(f"Ensembling with weights: LGBM={args.lgb_weight}, XGB={xgb_weight}")
    for col in pred_cols:
        df_ens[col] = df_lgb[col] * args.lgb_weight + df_xgb[col] * xgb_weight

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ens.to_csv(out_path, index=False)
    
    print(f"Ensemble saved to {out_path}")
    print(df_ens.head())

if __name__ == "__main__":
    main()
