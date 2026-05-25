#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="20260523"
LOG_DIR="$ROOT/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/missingness_shift_${TS}.log"
MANIFEST="$ROOT/docs/missingness_shift_experiments_${TS}_commands.md"
SUMMARY="$ROOT/docs/missingness_shift_experiments_${TS}_summary.md"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "# Missingness / Shift Experiments: ${TS}" > "$MANIFEST"
echo "" >> "$MANIFEST"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
echo "" >> "$MANIFEST"

echo "# Missingness / Shift Experiment Summary: ${TS}" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| Experiment | Run dir | Rolling avg MAE | Blind MAE | Blind output |" >> "$SUMMARY"
echo "|---|---|---:|---:|---|" >> "$SUMMARY"

BASE_ARGS=(
  --feature-profile lean
  --validation-mode rolling_origin
  --rolling-folds 3
  --final-train-mode refit_full
  --regularized
  --train-tail-days 1095
  --recency-half-life-days 1095
)

run_experiment() {
  local label="$1"
  shift
  local cmd=(uv run python src/train.py --experiment-name "${label}" "${BASE_ARGS[@]}" "$@")

  echo "" >> "$MANIFEST"
  echo "## ${label}" >> "$MANIFEST"
  echo "" >> "$MANIFEST"
  printf '```bash\n' >> "$MANIFEST"
  printf '%q ' "${cmd[@]}" >> "$MANIFEST"
  printf '\n```\n' >> "$MANIFEST"

  echo "============================================================"
  echo "Running ${label}"
  echo "============================================================"
  "${cmd[@]}"

  local run_id
  run_id="$(cat experiments/latest.txt)"
  local run_dir="experiments/${run_id}"
  local blind_dir="experiments/blind_${TS}_${label}_h1100"

  uv run python scripts/run_blind_backtest.py \
    --run-dir "$run_dir" \
    --origins 5,13,26,39,52,78,104 \
    --history-tail-days 1100 \
    --out-dir "$blind_dir"

  local rolling_mae
  local blind_mae
  rolling_mae="$(uv run python - "$run_dir/metrics.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(f"{json.load(f)['average_val_mae']:.5f}")
PY
)"
  blind_mae="$(uv run python - "$blind_dir/blind_backtest_metrics.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(f"{json.load(f)['overall_mae']:.5f}")
PY
)"
  echo "| \`${label}\` | \`${run_dir}\` | \`${rolling_mae}\` | \`${blind_mae}\` | \`${blind_dir}\` |" >> "$SUMMARY"
}

run_experiment "missingness_lgbm_lean_tail1095_drop_feature_nan_rows_${TS}" \
  --drop-feature-nan-rows

run_experiment "missingness_lgbm_lean_tail1095_score_lag26_${TS}" \
  --max-score-lag-weeks 26

run_experiment "shift_lgbm_lean_tail1095_recency365_${TS}" \
  --recency-half-life-days 365

run_experiment "shift_lgbm_lean_tail1095_seasonmatch2_${TS}" \
  --season-match-weight 2.0

run_experiment "shift_lgbm_lean_tail365_lag26_recency365_${TS}" \
  --train-tail-days 365 \
  --recency-half-life-days 365 \
  --max-score-lag-weeks 26

echo "" >> "$SUMMARY"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
echo "All missingness / shift experiments finished."
