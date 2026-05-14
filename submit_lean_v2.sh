#!/bin/bash
echo "Waiting for predict.py to finish..."
while pgrep -f "src/predict.py --run-dir experiments/20260513_165547" > /dev/null; do
    sleep 10
done

echo "Finding submission file..."
SUB_FILE=$(ls -t experiments/20260513_165547_lightgbm_two_stage_lgbm_gap_anomaly_regularized_lean_v2/submissions/*.csv | head -n 1)

if [ -n "$SUB_FILE" ]; then
    echo "Submitting $SUB_FILE to Kaggle..."
    kaggle competitions submit -c data-mining-2026-final-project -f "$SUB_FILE" -m "lean_v2 rolling-origin (Local MAE 0.19)"
    echo "Waiting for Kaggle to score..."
    sleep 20
    kaggle competitions submissions -c data-mining-2026-final-project
else
    echo "Error: Submission file not found."
fi
