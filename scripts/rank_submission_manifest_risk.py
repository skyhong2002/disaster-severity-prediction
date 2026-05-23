#!/usr/bin/env python3
"""Risk-adjust GRU-family submission order using origin-level LOO evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


LOO_COLUMNS = {
    "raw_gru_primary_probe": "raw_gru_mae",
    "gru_family_constrained_stack_secondary": "stack_mae",
    "calendar_gated_gru_secondary_high_variance": "calendar_gated_mae",
    "affine_calibrated_gru_tertiary": "affine_calibrated_mae",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def sha12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def score_candidate(name: str, values: list[float], raw_values: list[float]) -> dict[str, float | int]:
    regrets = [value - raw for value, raw in zip(values, raw_values)]
    wins_vs_raw = sum(value < raw for value, raw in zip(values, raw_values))
    ties_vs_raw = sum(abs(value - raw) <= 1e-12 for value, raw in zip(values, raw_values))
    mean_mae = mean(values)
    std_mae = pstdev(values)
    worst_mae = max(values)
    worst_regret = max(regrets)
    mean_regret = mean(regrets)
    effective_wins = wins_vs_raw + ties_vs_raw
    win_penalty = max(0, 4 - effective_wins) * 0.0015
    variance_penalty = 0.20 * std_mae
    tail_penalty = 0.05 * worst_mae
    regret_penalty = max(0.0, worst_regret) * 0.50 + max(0.0, mean_regret) * 0.25
    complexity_penalty = 0.0
    if "calendar" in name:
        complexity_penalty = 0.0030
    elif "affine" in name:
        complexity_penalty = 0.0015
    elif "stack" in name:
        complexity_penalty = 0.0008

    risk_adjusted_score = (
        mean_mae
        + variance_penalty
        + tail_penalty
        + regret_penalty
        + win_penalty
        + complexity_penalty
    )
    return {
        "loo_mean_mae": mean_mae,
        "loo_std_mae": std_mae,
        "loo_worst_mae": worst_mae,
        "mean_regret_vs_raw": mean_regret,
        "worst_regret_vs_raw": worst_regret,
        "wins_vs_raw": wins_vs_raw,
        "ties_vs_raw": ties_vs_raw,
        "origins": len(values),
        "risk_adjusted_score": risk_adjusted_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--loo-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    loo_rows = read_csv(args.loo_scores)
    raw_values = [float(row["raw_gru_mae"]) for row in loo_rows]

    by_name = {item["name"]: item for item in manifest}
    scored: list[dict[str, object]] = []
    for name, column in LOO_COLUMNS.items():
        item = by_name[name]
        values = [float(row[column]) for row in loo_rows]
        metrics = score_candidate(name, values, raw_values)
        submission_path = Path(item["path"])
        scored.append(
            {
                "name": name,
                "old_rank": item["rank"],
                "path": item["path"],
                "sha12_manifest": item["sha12"],
                "sha12_actual": sha12(submission_path),
                "sha_match": item["sha12"] == sha12(submission_path),
                "blind_mae": item["blind_mae"],
                **metrics,
            }
        )

    scored.sort(key=lambda row: (float(row["risk_adjusted_score"]), int(row["old_rank"])))
    for rank, row in enumerate(scored, start=1):
        row["risk_rank"] = rank

    risk_manifest = []
    for row in scored:
        source = dict(by_name[str(row["name"])])
        source["original_rank"] = source["rank"]
        source["rank"] = row["risk_rank"]
        source["risk_rank"] = row["risk_rank"]
        source["risk_adjusted_score"] = row["risk_adjusted_score"]
        source["risk_metrics"] = {
            key: row[key]
            for key in [
                "loo_mean_mae",
                "loo_std_mae",
                "loo_worst_mae",
                "mean_regret_vs_raw",
                "worst_regret_vs_raw",
                "wins_vs_raw",
                "ties_vs_raw",
                "origins",
            ]
        }
        if row["risk_rank"] <= 3:
            source["status"] = "approved_for_submission_order"
            source["slot_policy"] = "submit_by_risk_rank_if_quota_available_and_not_already_submitted"
        else:
            source["status"] = "defer_after_top3_risk_adjusted"
            source["slot_policy"] = "defer_until_top3_risk_adjusted_are_submitted_or_rejected"
        risk_manifest.append(source)

    report = {
        "hypothesis": "Submission order should penalize origin variance, worst-origin fragility, regret vs raw GRU, and rule complexity instead of ranking only by pooled blind MAE.",
        "formula": "loo_mean + 0.20*loo_std + 0.05*loo_worst + 0.50*positive_worst_regret + 0.25*positive_mean_regret + win_penalty + complexity_penalty",
        "inputs": {
            "manifest": str(args.manifest),
            "loo_scores": str(args.loo_scores),
        },
        "ranked_candidates": scored,
        "submit_order_top3": [row["name"] for row in scored[:3]],
        "recommendation": "Use risk_rank for quota order unless live Kaggle history shows one of these SHA12s has already been submitted.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "risk_adjusted_ranking_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "risk_adjusted_manifest.json").write_text(
        json.dumps(risk_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "risk_adjusted_ranking.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
