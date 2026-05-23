#!/usr/bin/env python3
"""Audit candidate submissions against locally available public-LB failures."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]


def parse_live_scores(path: Path) -> dict[str, dict[str, object]]:
    scores: dict[str, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if "SubmissionStatus.COMPLETE" not in line:
            continue
        match = re.match(
            r"^(\S+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+"
            r"(.*?)\s+SubmissionStatus\.COMPLETE\s+([0-9.]+)?\s*$",
            line,
        )
        if not match:
            continue
        file_name, submitted_at, description, public_score = match.groups()
        scores[file_name] = {
            "date": submitted_at,
            "description": " ".join(description.split()),
            "public_score": float(public_score) if public_score else None,
        }
    return scores


def load_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_cols = ["region_id", *PRED_COLS]
    if list(frame.columns) != expected_cols:
        raise ValueError(f"{path} columns {list(frame.columns)} != {expected_cols}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{path} region order does not match sample")
    if frame[PRED_COLS].isna().any().any():
        raise ValueError(f"{path} contains NaN")
    return frame


def prediction_distance(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    diff = (left[PRED_COLS] - right[PRED_COLS]).abs()
    return {
        "mae_distance": float(diff.to_numpy().mean()),
        "max_distance": float(diff.to_numpy().max()),
        **{f"{col}_mae_distance": float(diff[col].mean()) for col in PRED_COLS},
    }


def distribution_summary(frame: pd.DataFrame) -> dict[str, float]:
    values = frame[PRED_COLS].to_numpy(dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "zero_fraction": float((values <= 1e-12).mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--live-history", type=Path, required=True)
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--bad-threshold", type=float, default=0.90)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_submission)
    live = parse_live_scores(args.live_history)
    candidates = json.loads(args.candidate_manifest.read_text())
    public_refs = []
    for file_name, meta in live.items():
        public_score = meta["public_score"]
        path = args.submissions_dir / file_name
        if public_score is None or not path.exists():
            continue
        frame = load_submission(path, sample)
        public_refs.append(
            {
                "file_name": file_name,
                "path": str(path),
                "public_score": public_score,
                "public_bucket": "bad_public_failure" if public_score >= args.bad_threshold else "better_public_reference",
                "frame": frame,
                "distribution": distribution_summary(frame),
            }
        )

    candidate_rows = []
    pair_rows = []
    for candidate in candidates:
        candidate_path = Path(candidate["path"])
        candidate_frame = load_submission(candidate_path, sample)
        distances = []
        for ref in public_refs:
            distance = prediction_distance(candidate_frame, ref["frame"])
            row = {
                "candidate_name": candidate["name"],
                "candidate_path": str(candidate_path),
                "candidate_sha12": candidate.get("sha12"),
                "reference_file": ref["file_name"],
                "reference_public_score": ref["public_score"],
                "reference_bucket": ref["public_bucket"],
                **distance,
            }
            pair_rows.append(row)
            distances.append(row)

        bad_distances = [row for row in distances if row["reference_bucket"] == "bad_public_failure"]
        nearest_bad = min(bad_distances, key=lambda row: row["mae_distance"]) if bad_distances else None
        nearest_any = min(distances, key=lambda row: row["mae_distance"]) if distances else None
        candidate_rows.append(
            {
                "candidate_name": candidate["name"],
                "candidate_path": str(candidate_path),
                "candidate_sha12": candidate.get("sha12"),
                "manifest_status": candidate.get("status"),
                "distribution": distribution_summary(candidate_frame),
                "nearest_public_reference": nearest_any,
                "nearest_bad_public_failure": nearest_bad,
                "bad_failure_refs_compared": len(bad_distances),
                "all_public_refs_compared": len(distances),
                "risk_flag": (
                    "near_bad_failure_family"
                    if nearest_bad and nearest_bad["mae_distance"] < 0.05
                    else "not_near_local_bad_failure_family"
                ),
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "Candidates close in prediction space to recent public-LB failures should be deprioritized even if local GRU validation looks good.",
        "inputs": {
            "candidate_manifest": str(args.candidate_manifest),
            "live_history": str(args.live_history),
            "bad_threshold": args.bad_threshold,
        },
        "public_reference_count": len(public_refs),
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "limitations": [
            "The strong 0.8124 public anchor and 0.8094 historical best CSVs are not available locally, so this is a negative-control audit only.",
            "Prediction-space distance is a risk signal, not a validation score.",
        ],
    }
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.out_csv.open("w", newline="") as file:
        fieldnames = [
            "candidate_name",
            "candidate_path",
            "candidate_sha12",
            "reference_file",
            "reference_public_score",
            "reference_bucket",
            "mae_distance",
            "max_distance",
            *[f"{col}_mae_distance" for col in PRED_COLS],
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
