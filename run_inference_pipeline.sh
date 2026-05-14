#!/bin/bash
echo "Waiting for XGBoost training to complete..."
while pgrep -f "train_xgb.py --experiment-name xgb_long_term_weather" > /dev/null; do
    sleep 10
done

echo "Running Inference for LightGBM..."
uv run python src/predict.py --run-dir experiments/20260514_023028_lightgbm_two_stage_lgbm_long_term_weather

echo "Running Inference for XGBoost..."
uv run python src/predict.py --run-dir experiments/20260514_023744_xgboost_two_stage_xgb_long_term_weather

echo "Ensembling..."
LGB_SUB=$(ls experiments/20260514_023028_lightgbm_two_stage_lgbm_long_term_weather/submissions/*.csv | tail -n 1)
XGB_SUB=$(ls experiments/20260514_023744_xgboost_two_stage_xgb_long_term_weather/submissions/*.csv | tail -n 1)

uv run python src/ensemble.py --lgb "$LGB_SUB" --xgb "$XGB_SUB" --out submissions/ensemble_strategy_b_long_term.csv

echo "Pipeline Complete! File ready at submissions/ensemble_strategy_b_long_term.csv"
