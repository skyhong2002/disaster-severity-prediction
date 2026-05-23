#!/usr/bin/env python3
"""Guard Kaggle submission candidates against live history and block rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_live_history(path: Path) -> dict[str, dict[str, object]]:
    live: dict[str, dict[str, object]] = {}
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
        live[file_name] = {
            "fileName": file_name,
            "date": submitted_at,
            "description": " ".join(description.split()),
            "publicScore": float(public_score) if public_score else None,
        }
    return live


def classify_candidate(item: dict[str, object], live: dict[str, dict[str, object]]) -> dict[str, object]:
    path = str(item["path"])
    basename = Path(path).name
    live_record = live.get(basename)
    seen = live_record is not None
    manifest_status = str(item.get("status", ""))
    approved = manifest_status == "approved_for_submission_order"
    blocked = manifest_status.startswith("blocked_") or item.get("slot_policy") == "do_not_submit"
    deferred = manifest_status.startswith("defer_")

    if approved and not seen:
        decision = "ready_for_next_quota"
        action = "可在 quota 可用時依 rank 提交"
    elif approved and seen:
        decision = "already_submitted_skip"
        action = "已提交，避免重複燒 quota"
    elif blocked and seen:
        decision = "blocked_but_already_submitted_public_bad_skip_forever"
        action = "已被提交且 public score 明顯差，永久禁止再送"
    elif blocked:
        decision = "blocked_do_not_submit"
        action = "封鎖，不可提交"
    elif deferred:
        decision = "deferred_do_not_submit_now"
        action = "暫緩，等高順位候選送完或被拒絕後再評估"
    else:
        decision = "manual_review"
        action = "需要人工檢查"

    return {
        "rank": item.get("rank"),
        "name": item.get("name"),
        "path": path,
        "basename": basename,
        "sha12": item.get("sha12"),
        "manifest_status": manifest_status,
        "slot_policy": item.get("slot_policy"),
        "live_seen": seen,
        "live_date": live_record["date"] if live_record else "",
        "live_public_score": live_record["publicScore"] if live_record else "",
        "dry_run_decision": decision,
        "recommended_action": action,
        "blind_mae": item.get("blind_mae"),
        "loo_mean_mae": item.get("loo_mean_mae"),
    }


def sort_key(row: dict[str, object]) -> tuple[int, str]:
    rank = row.get("rank")
    return (int(rank) if isinstance(rank, int) else 9999, str(row.get("name", "")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--live-history", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    live = parse_live_history(args.live_history)
    rows = [classify_candidate(item, live) for item in manifest]
    ready = sorted((r for r in rows if r["dry_run_decision"] == "ready_for_next_quota"), key=sort_key)
    blocked_seen = sorted(
        (r for r in rows if r["dry_run_decision"] == "blocked_but_already_submitted_public_bad_skip_forever"),
        key=lambda row: str(row["live_date"]),
    )

    report = {
        "inputs": {
            "manifest": str(args.manifest),
            "live_history": str(args.live_history),
        },
        "rules": [
            "Only submit manifest entries with status=approved_for_submission_order.",
            "Submit approved candidates in ascending rank only if basename is absent from live Kaggle history.",
            "Never submit blocked_unverified_restore or blocked_unverified_restore_dependency entries.",
        ],
        "summary": {
            "manifest_entries": len(manifest),
            "live_submissions_parsed": len(live),
            "ready_for_next_quota_count": len(ready),
            "blocked_already_submitted_count": len(blocked_seen),
            "top_ready_order": [
                {"rank": r["rank"], "name": r["name"], "path": r["path"], "sha12": r["sha12"]}
                for r in ready
            ],
            "blocked_public_failures": [
                {
                    "name": r["name"],
                    "path": r["path"],
                    "sha12": r["sha12"],
                    "publicScore": r["live_public_score"],
                }
                for r in blocked_seen
            ],
        },
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
