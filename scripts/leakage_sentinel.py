#!/usr/bin/env python3
"""Poison blind-window scores and assert pseudo-test features do not change."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import METEO_COLS, get_feature_cols, required_context_days  # noqa: E402
from validation import blind_score_mask, build_pseudo_test_window, parse_origin_offsets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Detect score leakage from blind-window labels.")
    parser.add_argument("--origin", default="5", help="One week offset from train tail, e.g. 5 or -5.")
    parser.add_argument("--origins", default=None, help="Comma-separated week offsets. Overrides --origin.")
    parser.add_argument("--blind-days", type=int, default=91)
    parser.add_argument("--history-tail-days", type=int, default=1100)
    parser.add_argument("--feature-profile", choices=["micro", "lean", "full"], default="micro")
    parser.add_argument("--score-gap-days", type=int, default=91)
    parser.add_argument("--max-regions", type=int, default=32)
    parser.add_argument("--sentinel-value", type=float, default=999.0)
    parser.add_argument("--out-dir", default=None, help="Optional directory for leakage_sentinel_summary.json.")
    return parser.parse_args()


def load_subset(max_regions: int, raw_tail_days: int) -> pd.DataFrame:
    usecols = ["region_id", "date", "score"] + METEO_COLS
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    regions = train["region_id"].drop_duplicates().head(max_regions)
    subset = train[train["region_id"].isin(regions)].copy()
    subset = (
        subset.sort_values(["region_id", "date"])
        .groupby("region_id", group_keys=False)
        .tail(raw_tail_days)
        .reset_index(drop=True)
    )
    return subset


def comparable_numeric_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in get_feature_cols(df):
        if col in {"region_id", "date"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def run_one_origin(train: pd.DataFrame, origin, args) -> dict:
    feature_options = {
        "use_score_history": True,
        "score_gap_days": args.score_gap_days,
        "use_climatology": True,
        "use_region_stats": False,
        "feature_profile": args.feature_profile,
    }

    print(f"\n[Origin {origin.label}] Building baseline pseudo-test features ...")
    baseline, _ = build_pseudo_test_window(
        train,
        origin,
        feature_options,
        blind_days=args.blind_days,
        history_tail_days=args.history_tail_days,
    )

    blind_mask = blind_score_mask(train, origin, blind_days=args.blind_days)
    poisoned = train.copy()
    poison_mask = blind_mask & poisoned["score"].notna()
    poisoned.loc[poison_mask, "score"] = args.sentinel_value
    print(f"Poisoned {int(poison_mask.sum())} weekly score rows inside the blind window.")

    print(f"[Origin {origin.label}] Building poisoned pseudo-test features ...")
    candidate, _ = build_pseudo_test_window(
        poisoned,
        origin,
        feature_options,
        blind_days=args.blind_days,
        history_tail_days=args.history_tail_days,
    )

    keys = ["region_id", "_day_idx"]
    feature_cols = [col for col in comparable_numeric_features(baseline) if col in candidate.columns]
    merged = baseline[keys + feature_cols].merge(
        candidate[keys + feature_cols],
        on=keys,
        suffixes=("_baseline", "_poisoned"),
        how="inner",
    )
    if len(merged) != len(baseline):
        return {
            "origin": origin.label,
            "status": "fail",
            "reason": f"Feature alignment changed after poisoning: {len(merged)} vs {len(baseline)} rows.",
            "poisoned_score_rows": int(poison_mask.sum()),
            "regions": int(len(baseline)),
        }

    mismatches = []
    for col in feature_cols:
        left = merged[f"{col}_baseline"].to_numpy()
        right = merged[f"{col}_poisoned"].to_numpy()
        equal = np.isclose(left, right, equal_nan=True, rtol=1e-7, atol=1e-7)
        if not bool(equal.all()):
            diff = np.nanmax(np.abs(left - right))
            mismatches.append((col, float(diff), int((~equal).sum())))

    if mismatches:
        print("FAILED: blind-window score poisoning changed final-row features.")
        for col, diff, count in sorted(mismatches, key=lambda item: item[1], reverse=True)[:20]:
            print(f"  {col}: max_abs_diff={diff:.6g}, changed_rows={count}")
        return {
            "origin": origin.label,
            "status": "fail",
            "reason": "blind-window score poisoning changed final-row features",
            "poisoned_score_rows": int(poison_mask.sum()),
            "regions": int(len(merged)),
            "feature_count": int(len(feature_cols)),
            "mismatches": [
                {"feature": col, "max_abs_diff": diff, "changed_rows": count}
                for col, diff, count in sorted(mismatches, key=lambda item: item[1], reverse=True)
            ],
        }

    print(
        "PASSED: blind-window score poisoning did not change final-row features "
        f"({len(feature_cols)} numeric features, {len(merged)} regions)."
    )
    return {
        "origin": origin.label,
        "status": "pass",
        "poisoned_score_rows": int(poison_mask.sum()),
        "regions": int(len(merged)),
        "feature_count": int(len(feature_cols)),
    }


def main():
    args = parse_args()
    origins = parse_origin_offsets(args.origins or args.origin)
    required_days = required_context_days(
        args.feature_profile,
        score_gap_days=args.score_gap_days,
        use_score_history=True,
    )
    if args.history_tail_days < required_days:
        print(
            "WARNING: "
            f"history_tail_days={args.history_tail_days} is below required_context_days={required_days}; "
            "sentinel still runs, but this is not a formal model-selection context."
        )

    max_origin_offset = max(origin.offset_weeks for origin in origins)
    raw_tail_days = args.history_tail_days + max_origin_offset * 7 + 35
    print(f"Loading subset: regions={args.max_regions}, raw_tail_days={raw_tail_days}")
    train = load_subset(args.max_regions, raw_tail_days)

    results = [run_one_origin(train, origin, args) for origin in origins]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pass" if all(result["status"] == "pass" for result in results) else "fail",
        "feature_profile": args.feature_profile,
        "history_tail_days": args.history_tail_days,
        "required_context_days": required_days,
        "blind_days": args.blind_days,
        "score_gap_days": args.score_gap_days,
        "max_regions": args.max_regions,
        "sentinel_value": args.sentinel_value,
        "origins": [origin.label for origin in origins],
        "results": results,
    }
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "leakage_sentinel_summary.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSummary saved -> {out_path}")

    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
