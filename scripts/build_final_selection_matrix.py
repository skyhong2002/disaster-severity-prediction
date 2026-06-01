#!/usr/bin/env python3
"""Build a reproducible public/private final-selection matrix.

This script does not submit to Kaggle. It consolidates submitted public-chase
hedges and private-robust alternatives into one readout so the final-selection
decision can be reproduced without re-litigating the whole history from memory.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE3_PUBLIC = 0.8056
PUBLIC_TOLERANCE = 0.0020
ANCHOR_DELTA_SCALE = 0.18


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def path_name(value: str | None) -> str:
    return Path(value or "").name


def add_candidate(
    rows: list[dict[str, Any]],
    *,
    source_key: str,
    path: str,
    kaggle_ref: int | None,
    public_mae: float | None,
    sha12: str | None,
    alphas: list[float] | None,
    delta_to_anchor: float | None,
    label: str,
    reportable_method_claim: bool,
    status: str,
    decision: str,
) -> None:
    if not path:
        return
    rows.append(
        {
            "source_key": source_key,
            "path": path,
            "basename": path_name(path),
            "kaggle_ref": kaggle_ref,
            "public_mae": public_mae,
            "sha12": sha12,
            "alphas": alphas,
            "mean_abs_delta_to_reportable_anchor": delta_to_anchor,
            "artifact_label": label,
            "reportable_method_claim": reportable_method_claim,
            "status": status,
            "decision": decision,
        }
    )


def collect_submitted(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for key in [
        "private_hedge_curve_20260601",
        "private_hedge_curve_20260531",
        "private_hedge_curve_20260530",
        "private_hedge_curve_20260528",
        "private_hedge_curve_20260527",
        "private_hedge_curve_20260526",
        "private_hedge_curve_20260525",
        "private_hedge_curve_20260524",
    ]:
        for item in ledger.get(key, []):
            path = item.get("submission") or item.get("path")
            kaggle_ref = item.get("kaggle_ref")
            identity = (str(path), int(kaggle_ref) if kaggle_ref is not None else None)
            if identity in seen:
                continue
            seen.add(identity)
            add_candidate(
                rows,
                source_key=key,
                path=str(path),
                kaggle_ref=int(kaggle_ref) if kaggle_ref is not None else None,
                public_mae=as_float(item.get("public_mae")),
                sha12=item.get("sha12"),
                alphas=item.get("alphas"),
                delta_to_anchor=as_float(item.get("mean_abs_delta_to_reportable_anchor")),
                label=item.get("label", "public-chase"),
                reportable_method_claim=bool(item.get("reportable_method_claim", False)),
                status="submitted",
                decision=item.get("decision", ""),
            )
    return rows


def score_submitted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    submitted = [row for row in rows if row["public_mae"] is not None]
    best_public = min(float(row["public_mae"]) for row in submitted)
    for row in submitted:
        public_mae = float(row["public_mae"])
        delta = row["mean_abs_delta_to_reportable_anchor"]
        public_gap = public_mae - best_public
        row["public_gap_to_best"] = public_gap
        row["baseline3_margin"] = BASELINE3_PUBLIC - public_mae
        row["within_public_0p0005"] = public_gap <= 0.0005 + 1e-12
        row["within_public_0p0020"] = public_gap <= PUBLIC_TOLERANCE + 1e-12
        if delta is None:
            row["private_robust_score"] = None
        else:
            row["private_robust_score"] = 0.55 * (public_gap / PUBLIC_TOLERANCE) + 0.45 * (
                float(delta) / ANCHOR_DELTA_SCALE
            )
    return submitted


def collect_queue_items(queue: dict[str, Any], backup: dict[str, Any]) -> dict[str, Any]:
    manual_items = queue.get("submission_order", [])
    v5_items = [
        item
        for item in manual_items
        if str(item.get("artifact_label")) == "public-chase" and str(item.get("name", "")).startswith("baseline3")
    ]
    teammate = manual_items[0] if manual_items else None
    v6_items = backup.get("submission_order", [])
    return {
        "teammate_gate": teammate,
        "v5_submit_order_if_teammate_skips": v5_items,
        "v6_backup_order_after_v5_readout": v6_items,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rec = payload["recommendations"]
    static_rec = rec["static_private_selection"]
    queue = payload["queue_readout"]
    lines = [
        "# Final Selection Matrix 2026-06-01",
        "",
        f"- Created UTC: `{payload['created_at_utc']}`",
        f"- Live Team 5 public MAE: `{payload['live_context']['team5_public_mae']}`",
        f"- Baseline 3 public MAE: `{payload['live_context']['baseline3_public_mae']}`",
        f"- Team 5 public rank: `{payload['live_context']['team5_rank']}`",
        f"- Static private snapshot: Team 5 rank `{payload['static_private_context']['team5_rank']}` at `{payload['static_private_context']['last_released_time_displayed']}`.",
        "- Role: decision support for public/private final selection; not a reportable method claim.",
        "",
        "## Recommendations",
        "",
        f"- Select 1, public-best artifact: `{static_rec['select_1_public_best']['path']}` / ref `{static_rec['select_1_public_best']['kaggle_ref']}` / public `{static_rec['select_1_public_best']['public_mae']}`.",
        f"- Select 2, static/private hedge: `{static_rec['select_2_private_robust_hedge']['path']}` / ref `{static_rec['select_2_private_robust_hedge']['kaggle_ref']}` / public `{static_rec['select_2_private_robust_hedge']['public_mae']}` / delta `{static_rec['select_2_private_robust_hedge']['mean_abs_delta_to_reportable_anchor']:.6f}`.",
        f"- Public-biased alternate if the second slot must stay closer to public-best: `{static_rec['public_biased_alternate_hedge']['path']}` / ref `{static_rec['public_biased_alternate_hedge']['kaggle_ref']}` / public `{static_rec['public_biased_alternate_hedge']['public_mae']}` / delta `{static_rec['public_biased_alternate_hedge']['mean_abs_delta_to_reportable_anchor']:.6f}`.",
        f"- Stronger private fallback by the matrix score: `{rec['stronger_private_fallback']['path']}` / ref `{rec['stronger_private_fallback']['kaggle_ref']}` / public `{rec['stronger_private_fallback']['public_mae']}` / delta `{rec['stronger_private_fallback']['mean_abs_delta_to_reportable_anchor']:.6f}`.",
        "",
        "## Selected Submitted Frontier",
        "",
        "| Role | File | Ref | Public MAE | Delta | SHA-12 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role, row in [
        ("select_1_public_best", static_rec["select_1_public_best"]),
        ("select_2_private_robust_hedge", static_rec["select_2_private_robust_hedge"]),
        ("public_biased_alternate", static_rec["public_biased_alternate_hedge"]),
        ("stronger_private_fallback", rec["stronger_private_fallback"]),
    ]:
        lines.append(
            f"| `{role}` | `{row['path']}` | `{row['kaggle_ref']}` | `{row['public_mae']}` | "
            f"`{row['mean_abs_delta_to_reportable_anchor']:.6f}` | `{row['sha12']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Quota Rules",
            "",
        ]
    )
    for rule in payload["next_quota_rules"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## Queue Pointers",
            "",
            f"- Teammate gate: `{queue['teammate_gate']['path'] if queue.get('teammate_gate') else 'not_applicable'}`",
            f"- Latest public-best selected: `{static_rec['select_1_public_best']['path']}`",
            f"- Latest static/private hedge selected: `{static_rec['select_2_private_robust_hedge']['path']}`",
            f"- First v6/private-robust backup if future quota is reopened: `{queue['v6_backup_order_after_v5_readout'][0]['path'] if queue.get('v6_backup_order_after_v5_readout') else 'not_applicable'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "docs" / "status" / "baseline3_push_20260523.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT
        / "experiments"
        / "baseline3_push_20260523"
        / "private_hedge_frontier_20260529_queue_20260528_1555"
        / "next_submission_queue_20260529.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "baseline3_push_20260523" / "final_selection_matrix_20260601_2335",
    )
    args = parser.parse_args()

    ledger = read_json(args.ledger)
    queue = read_json(args.queue)
    backup = ledger["private_hedge_frontier_20260530_backup"]
    submitted = score_submitted(collect_submitted(ledger))
    public_best = min(
        submitted,
        key=lambda row: (
            float(row["public_mae"]),
            float(row["mean_abs_delta_to_reportable_anchor"] or 9),
        ),
    )
    submitted_keys = [row["source_key"] for row in submitted]
    active_frontier_key = next(
        key
        for key in [
            "private_hedge_curve_20260601",
            "private_hedge_curve_20260531",
            "private_hedge_curve_20260530",
            "private_hedge_curve_20260528",
            "private_hedge_curve_20260527",
            "private_hedge_curve_20260526",
            "private_hedge_curve_20260525",
            "private_hedge_curve_20260524",
        ]
        if key in submitted_keys
    )
    same_day_candidates = [
        row
        for row in submitted
        if row["source_key"] == active_frontier_key and row["within_public_0p0005"]
    ]
    same_day_private = min(
        same_day_candidates,
        key=lambda row: (
            float(row["mean_abs_delta_to_reportable_anchor"] or 9),
            float(row["public_mae"]),
        ),
    )
    public_biased_alternate_pool = [
        row
        for row in same_day_candidates
        if row["kaggle_ref"] != public_best["kaggle_ref"]
    ]
    public_biased_alternate = min(
        public_biased_alternate_pool,
        key=lambda row: (
            float(row["public_mae"]),
            float(row["mean_abs_delta_to_reportable_anchor"] or 9),
        ),
    )
    robust_pool = [row for row in submitted if row["within_public_0p0020"]]
    stronger_private = min(
        robust_pool,
        key=lambda row: (
            float(row["mean_abs_delta_to_reportable_anchor"] or 9),
            float(row["public_mae"]),
        ),
    )
    queue_readout = collect_queue_items(queue, backup)
    quota_context = (
        ledger.get("private_hedge_frontier_20260601_quota")
        or ledger.get("private_hedge_frontier_20260531_quota")
        or ledger.get("private_hedge_frontier_20260530_quota", {})
    )
    static_private_context = ledger.get("static_private_snapshot_20260529", {})
    payload = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "ledger": rel(args.ledger),
            "queue": rel(args.queue),
            "v6_backup_readout": backup["report"],
        },
        "live_context": {
            "team5_public_mae": ledger["team5_public_mae"],
            "baseline3_public_mae": ledger["baseline3_public_mae"],
            "team5_rank": quota_context.get("team5_rank", ledger["private_hedge_frontier_20260528_queue"]["team5_rank"]),
            "quota_status": quota_context.get(
                "quota_used_20260530_utc",
                ledger["private_hedge_frontier_20260530_backup"]["quota_used_20260528_utc"],
            ),
            "next_quota_reset_taipei": quota_context.get(
                "next_quota_reset_taipei",
                ledger["private_hedge_frontier_20260530_backup"]["next_quota_reset_taipei"],
            ),
        },
        "static_private_context": static_private_context,
        "scoring_policy": {
            "private_robust_score": "0.55 * public_gap_to_best / 0.0020 + 0.45 * mean_abs_delta_to_reportable_anchor / 0.18",
            "interpretation": "Lower score is better among submitted public-chase artifacts that still pass Baseline 3.",
        },
        "submitted_rankings": {
            "by_public": sorted(submitted, key=lambda row: (float(row["public_mae"]), float(row["mean_abs_delta_to_reportable_anchor"] or 9))),
            "by_private_robust_score": sorted(
                [row for row in submitted if row["private_robust_score"] is not None],
                key=lambda row: float(row["private_robust_score"]),
            ),
        },
        "recommendations": {
            "active_frontier_key": active_frontier_key,
            "public_best": public_best,
            "same_day_private_hedge": same_day_private,
            "stronger_private_fallback": stronger_private,
            "static_private_selection": {
                "select_1_public_best": public_best,
                "select_2_private_robust_hedge": same_day_private,
                "public_biased_alternate_hedge": public_biased_alternate,
            },
            "reportable_method_lineage": ledger["reportable_legal_anchor"],
        },
        "queue_readout": queue_readout,
        "next_quota_rules": [
            "Current 2026-06-01 UTC quota is 10/10 used after the v9 quota-10 frontier.",
            "For Static Private / final-selection UI, manually select refs 53204258 and 53259683; do not rely only on auto-selection.",
            "Live-gate Kaggle submission history and leaderboard after 2026-06-02T08:00:00+08:00 Taipei before spending any new quota.",
            "Do not resubmit the teammate file: live history already confirms ref 53074655 / public 1.0685 for the same artifact.",
            "Use the v8 pair refs 53204258/53204319 or the v7 pair refs 53186508/53186571 as fallback if the UI or team policy prefers a previous manually selected Static Private pair.",
            "Do not describe any v5/v6/v7/v8/v9 public-chase artifact as a reportable method claim.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "final_selection_matrix.json"
    out_md = args.out_dir / "experiment_summary.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"json": rel(out_json), "summary": rel(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
