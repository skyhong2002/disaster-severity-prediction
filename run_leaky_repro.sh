#!/bin/bash
echo "Waiting for leaky LGBM training to complete..."
while pgrep -f "train.py --experiment-name lgbm_leaky_repro" > /dev/null; do
    sleep 10
done

echo "Running Inference for Leaky Model..."
uv run python src/predict.py --run-dir experiments/20260514_121452_lightgbm_two_stage_lgbm_leaky_repro

echo "Leaky Reproduction Complete! You can find the submission in the latest folder."
