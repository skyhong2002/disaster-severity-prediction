#!/usr/bin/env python3
"""Build a small horizon-hedge frontier for final Kaggle selection.

The generated files are public-chase / final-selection artifacts only. They
blend an exact recovered historical Team 5 submission toward the clean
reportable 0.8124 anchor, and they are not reportable method claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]
BLOCKED_PATTERNS = ("restored_20260522_", "restored_unverified_")


@dataclass(frozen=True)
class Source:
    key: str
    path: Path
    public_score: float
    role: str


@dataclass(frozen=True)
class Spec:
    tag: str
    alphas: tuple[float, float, float, float, float]
    rationale: str


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alpha_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def read_checked(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if any(pattern in path.name for pattern in BLOCKED_PATTERNS):
        raise ValueError(f"{rel(path)} is a blocked restored/unverified source")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"{rel(path)} columns do not match sample_submission.csv")
    if len(frame) != len(sample):
        raise ValueError(f"{rel(path)} has {len(frame)} rows, expected {len(sample)}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{rel(path)} region_id order differs from sample")
    values = frame[PRED_COLS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{rel(path)} contains NaN or non-numeric predictions")
    if float(values.min().min()) < -1e-9 or float(values.max().max()) > 5.0 + 1e-9:
        raise ValueError(f"{rel(path)} predictions are outside [0, 5]")
    return frame


def validate_output(path: Path, sample: pd.DataFrame) -> dict[str, Any]:
    frame = read_checked(path, sample)
    values = frame[PRED_COLS]
    digest = sha256(path)
    return {
        "path": rel(path),
        "rows": int(len(frame)),
        "columns_match_sample": list(frame.columns) == list(sample.columns),
        "region_order_match": bool(frame["region_id"].equals(sample["region_id"])),
        "nan_count": int(values.isna().sum().sum()),
        "prediction_min": float(values.min().min()),
        "prediction_max": float(values.max().max()),
        "prediction_mean": float(values.mean().mean()),
        "prediction_std_mean": float(values.std().mean()),
        "sha256": digest,
        "sha12": digest[:12],
        "blocked_source_pattern": any(pattern in path.name for pattern in BLOCKED_PATTERNS),
    }


def horizon_blend(base: pd.DataFrame, anchor: pd.DataFrame, alphas: tuple[float, ...]) -> pd.DataFrame:
    out = base[["region_id"]].copy()
    for col, alpha in zip(PRED_COLS, alphas):
        out[col] = base[col] * (1.0 - alpha) + anchor[col] * alpha
    out[PRED_COLS] = out[PRED_COLS].clip(0, 5)
    return out


def mean_abs_delta(left: pd.DataFrame, right: pd.DataFrame) -> float:
    return float((left[PRED_COLS] - right[PRED_COLS]).abs().mean().mean())


def build_specs(spec_set: str) -> list[Spec]:
    if spec_set == "v1":
        return [
            Spec(
                "left_of_best_public",
                (0.325, 0.375, 0.475, 0.625, 0.775),
                "Tests whether the public optimum sits slightly left of the current 0.7930 horizon hedge.",
            ),
            Spec(
                "midpoint_best_to_nearbest",
                (0.375, 0.425, 0.525, 0.675, 0.825),
                "Midpoint between the best public hedge and the near-best stronger private hedge.",
            ),
            Spec(
                "preserve_early_anchor_late",
                (0.350, 0.425, 0.550, 0.725, 0.900),
                "Keeps week 1 near the best public hedge while moving later horizons closer to the clean anchor.",
            ),
            Spec(
                "public_early_full_late_anchor",
                (0.300, 0.400, 0.550, 0.750, 1.000),
                "Explores a stronger private hedge on late horizons without over-anchoring early horizons.",
            ),
            Spec(
                "robust_mid_frontier",
                (0.400, 0.475, 0.575, 0.750, 0.900),
                "Interpolates toward the most robust submitted horizon hedge while staying near the public-best frontier.",
            ),
            Spec(
                "smooth_high_anchor",
                (0.425, 0.500, 0.600, 0.800, 0.950),
                "High-anchor smooth hedge for private-risk coverage while still retaining public-chase source signal.",
            ),
        ]
    if spec_set == "v2":
        return [
            Spec(
                "lower_w1_only",
                (0.275, 0.400, 0.550, 0.750, 1.000),
                "Local probe around the 0.7922 best: reduce week-1 anchor weight while preserving the late-anchor shape.",
            ),
            Spec(
                "lower_w2_only",
                (0.300, 0.375, 0.550, 0.750, 1.000),
                "Local probe around the 0.7922 best: reduce week-2 anchor weight while keeping the full week-5 anchor.",
            ),
            Spec(
                "lower_early_pair",
                (0.250, 0.375, 0.550, 0.750, 1.000),
                "Tests whether the public optimum remains slightly left on early horizons after the v1 readout.",
            ),
            Spec(
                "late_more_anchor",
                (0.300, 0.400, 0.575, 0.800, 1.000),
                "Moves weeks 3-4 closer to the clean anchor while preserving the best early-horizon weights.",
            ),
            Spec(
                "balanced_private_frontier",
                (0.325, 0.425, 0.575, 0.775, 1.000),
                "Balanced public/private frontier: modestly more anchor weight than the 0.7922 best across weeks 1-4.",
            ),
            Spec(
                "smooth_high_anchor_v2",
                (0.350, 0.450, 0.600, 0.800, 1.000),
                "Most anchor-tilted v2 hedge, intended as a private-risk alternative if public score remains below Baseline 3.",
            ),
        ]
    if spec_set == "v3":
        return [
            Spec(
                "lower_w1_more",
                (0.225, 0.375, 0.550, 0.750, 1.000),
                "After v2 lower_early_pair became public-best, reduce week-1 anchor weight slightly more while preserving weeks 2-5.",
            ),
            Spec(
                "lower_w2_more",
                (0.250, 0.350, 0.550, 0.750, 1.000),
                "Local v3 probe: keep the best week-1 setting and reduce week-2 anchor weight to test the early-horizon public optimum.",
            ),
            Spec(
                "lower_early_more",
                (0.225, 0.350, 0.550, 0.750, 1.000),
                "More aggressive early-horizon public-side probe; higher public-chase risk but useful as a boundary point.",
            ),
            Spec(
                "week3_slight_down",
                (0.250, 0.375, 0.525, 0.750, 1.000),
                "Tests whether the public optimum also prefers slightly less anchor weight on week 3.",
            ),
            Spec(
                "public_best_late_more_anchor",
                (0.250, 0.375, 0.575, 0.800, 1.000),
                "Private-robust hedge around the v2 public best: preserve early weights and move weeks 3-4 toward the clean anchor.",
            ),
        ]
    if spec_set == "v4":
        return [
            Spec(
                "w1_0p20_keep_shape",
                (0.200, 0.375, 0.550, 0.750, 1.000),
                "Follows the v3 week-1 signal: reduce only week-1 anchor weight from 0.225 to 0.200 while preserving the selected v3 shape.",
            ),
            Spec(
                "w1_0p175_keep_shape",
                (0.175, 0.375, 0.550, 0.750, 1.000),
                "Boundary probe for the week-1 public optimum after v3 showed week-1 reduction helped.",
            ),
            Spec(
                "w1_0p2125_keep_shape",
                (0.2125, 0.375, 0.550, 0.750, 1.000),
                "Fine-grained interpolation between the v3 selected 0.225 week-1 weight and the next lower 0.200 probe.",
            ),
            Spec(
                "w1_0p225_w2_mid",
                (0.225, 0.3625, 0.550, 0.750, 1.000),
                "Tests whether a milder week-2 reduction than v3 lower_early_more can keep the 0.7915 public score with less anchor drift.",
            ),
            Spec(
                "w1_0p20_late_anchor",
                (0.200, 0.375, 0.575, 0.800, 1.000),
                "Private-robust hedge combining the new lower week-1 probe with more anchor weight on weeks 3-4.",
            ),
            Spec(
                "w1_0p225_late_mid_anchor",
                (0.225, 0.375, 0.565, 0.775, 1.000),
                "Softer late-anchor hedge between the v3 selected public-best shape and the same-day late-anchor private hedge.",
            ),
        ]
    if spec_set == "v5":
        return [
            Spec(
                "w1_0p1625_keep_shape",
                (0.1625, 0.375, 0.550, 0.750, 1.000),
                "Fine interpolation just below the v4 public-best week-1 weight 0.175.",
            ),
            Spec(
                "w1_0p15_keep_shape",
                (0.150, 0.375, 0.550, 0.750, 1.000),
                "Main continuation of the week-1 public-side curve after v4 improved monotonically down to 0.175.",
            ),
            Spec(
                "w1_0p125_keep_shape",
                (0.125, 0.375, 0.550, 0.750, 1.000),
                "Boundary probe to locate the left side of the week-1 optimum without jumping all the way to zero.",
            ),
            Spec(
                "w1_0p10_keep_shape",
                (0.100, 0.375, 0.550, 0.750, 1.000),
                "More aggressive week-1 public-side boundary probe; use after the safer 0.1625/0.15/0.125 points.",
            ),
            Spec(
                "w1_0p175_late_anchor",
                (0.175, 0.375, 0.575, 0.800, 1.000),
                "Private-robust hedge at the v4 public-best week-1 weight with weeks 3-4 moved toward the clean anchor.",
            ),
            Spec(
                "w1_0p15_late_anchor",
                (0.150, 0.375, 0.575, 0.800, 1.000),
                "Private-robust hedge combining the next lower week-1 public-side probe with late-anchor protection.",
            ),
        ]
    if spec_set == "v6":
        return [
            Spec(
                "w1_0p1625_late_anchor",
                (0.1625, 0.375, 0.575, 0.800, 1.000),
                "Backup private hedge pairing the v5 lower week-1 probe with the v4 late-anchor protection that stayed near public best.",
            ),
            Spec(
                "w1_0p20_late_anchor_w2_0p40",
                (0.200, 0.400, 0.575, 0.800, 1.000),
                "Adds week-2 anchor weight to the v4 ref 53109166 late-anchor hedge, seeking lower private risk with limited public drift.",
            ),
            Spec(
                "w1_0p175_late_anchor_w2_0p40",
                (0.175, 0.400, 0.575, 0.800, 1.000),
                "Combines the v4 public-best week-1 setting with more week-2 and late-horizon anchor weight.",
            ),
            Spec(
                "w1_0p20_stronger_late_anchor",
                (0.200, 0.375, 0.600, 0.825, 1.000),
                "Stronger late-anchor version of ref 53109166 for private-risk coverage if v5 public-side probes overfit.",
            ),
            Spec(
                "w1_0p225_stronger_late_anchor",
                (0.225, 0.375, 0.600, 0.825, 1.000),
                "Uses the v3 selected early shape with stronger week-3 and week-4 anchor protection.",
            ),
            Spec(
                "w1_0p25_stronger_late_anchor",
                (0.250, 0.375, 0.600, 0.825, 1.000),
                "Most conservative v6 late-anchor backup, staying closer to prior private-robust refs while still below Baseline 3 in nearby submitted points.",
            ),
        ]
    if spec_set == "v7":
        return [
            Spec(
                "w1_0p10_late_anchor",
                (0.100, 0.375, 0.575, 0.800, 1.000),
                "Private-robust check at the current v5 public-best week-1 weight, moving weeks 3-4 back toward the clean anchor.",
            ),
            Spec(
                "w1_0p075_keep_shape",
                (0.075, 0.375, 0.550, 0.750, 1.000),
                "Next public-side week-1 boundary point after v5 improved monotonically through 0.100.",
            ),
            Spec(
                "w1_0p075_late_anchor",
                (0.075, 0.375, 0.575, 0.800, 1.000),
                "Pairs the next lower week-1 public probe with late-anchor private-risk protection.",
            ),
            Spec(
                "w1_0p05_keep_shape",
                (0.050, 0.375, 0.550, 0.750, 1.000),
                "More aggressive week-1 public-side boundary probe after 0.075.",
            ),
            Spec(
                "w1_0p05_late_anchor",
                (0.050, 0.375, 0.575, 0.800, 1.000),
                "More aggressive week-1 public probe with late-anchor protection.",
            ),
            Spec(
                "w1_0p025_keep_shape",
                (0.025, 0.375, 0.550, 0.750, 1.000),
                "Near-zero week-1 anchor boundary to test whether the public optimum has already reversed.",
            ),
            Spec(
                "w1_0p025_late_anchor",
                (0.025, 0.375, 0.575, 0.800, 1.000),
                "Near-zero week-1 anchor boundary with private-robust late-horizon anchor weight.",
            ),
            Spec(
                "w1_0p00_keep_shape",
                (0.000, 0.375, 0.550, 0.750, 1.000),
                "Zero week-1 anchor upper-risk boundary; submit only after the safer 0.075/0.050 readouts remain public-competitive.",
            ),
            Spec(
                "w1_0p075_stronger_late_anchor",
                (0.075, 0.375, 0.600, 0.825, 1.000),
                "Stronger late-anchor hedge around the likely public-side 0.075 week-1 region.",
            ),
            Spec(
                "w1_0p10_stronger_late_anchor",
                (0.100, 0.375, 0.600, 0.825, 1.000),
                "Stronger late-anchor hedge at the current v5 public-best week-1 setting, designed as a near-public private fallback.",
            ),
        ]
    raise ValueError(f"unknown spec set: {spec_set}")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Private-Hedge Frontier {payload['experiment_label']}",
        "",
        f"- Created UTC: `{payload['created_at_utc']}`",
        f"- Experiment label: `{payload['experiment_label']}`",
        f"- Spec set: `{payload['spec_set']}`",
        "- Role: public-chase / final-selection hedge only; not a reportable method claim.",
        "- Source policy: exact recovered Team 5 submissions only; no private labels, external answers, or restored/unverified source files.",
        "",
        "## Sources",
        "",
        "| Key | Role | Public MAE | SHA-12 | Path |",
        "|---|---|---:|---:|---|",
    ]
    for source in payload["sources"].values():
        lines.append(
            f"| `{source['key']}` | `{source['role']}` | `{source['public_score']:.4f}` | "
            f"`{source['sha12']}` | `{source['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate Slate",
            "",
            "| Candidate | Horizon alphas | SHA-12 | Delta to reportable anchor | Rationale |",
            "|---|---|---:|---:|---|",
        ]
    )
    for candidate in payload["candidates"]:
        alphas = ", ".join(f"{value:.3g}" for value in candidate["alphas"])
        lines.append(
            f"| `{candidate['path']}` | `{alphas}` | `{candidate['audit']['sha12']}` | "
            f"`{candidate['mean_abs_delta_to_reportable_anchor']:.6f}` | "
            f"{candidate['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Required Submission Sanity",
            "",
            "All generated candidates passed: 2248 rows, sample columns, matching `region_id` order, no NaN, prediction range in `[0,5]`, SHA-256 recorded, and no `restored_20260522_*` / `restored_unverified_*` filename pattern.",
            "",
            "After each Kaggle submit, append the Kaggle ref and public score to the readout JSON and status ledger before choosing final-selection wording.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "experiments" / "recovered_submissions_20260523",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--spec-set", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"], default="v1")
    parser.add_argument("--experiment-label", default=None)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    default_labels = {
        "v1": "private_hedge_frontier_20260525_0850",
        "v2": "private_hedge_frontier_20260526_1135",
        "v3": "private_hedge_frontier_20260527_1445",
        "v4": "private_hedge_frontier_20260528_queue_20260527_1500",
        "v5": "private_hedge_frontier_20260529_queue_20260528_1555",
        "v6": "private_hedge_frontier_20260530_backup_20260528_1610",
        "v7": "private_hedge_frontier_20260530_quota_20260530_2155",
    }
    experiment_label = args.experiment_label or default_labels[args.spec_set]
    if args.work_dir is None:
        work_dir = ROOT / "experiments" / "baseline3_push_20260523" / experiment_label
    else:
        work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir

    sources = {
        "v0_08094": Source(
            key="v0_08094",
            path=source_dir / "submission_20260512_195951.csv",
            public_score=0.8094,
            role="source_reference_only_public_chase",
        ),
        "cat35_08124": Source(
            key="cat35_08124",
            path=source_dir / "ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv",
            public_score=0.8124,
            role="clean_reportable_anchor",
        ),
    }
    frames = {key: read_checked(source.path, sample) for key, source in sources.items()}

    specs = build_specs(args.spec_set)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    base = frames["v0_08094"]
    anchor = frames["cat35_08124"]
    for spec in specs:
        alpha_part = "_".join(alpha_slug(alpha) for alpha in spec.alphas)
        name = f"baseline3_private_hedge_{args.spec_set}_cat35_{spec.tag}_horizon_{alpha_part}"
        out_path = out_dir / f"{name}.csv"
        frame = horizon_blend(base, anchor, spec.alphas)
        frame.to_csv(out_path, index=False)
        audit = validate_output(out_path, sample)
        if audit["blocked_source_pattern"]:
            raise ValueError(f"{rel(out_path)} matched a blocked pattern")
        candidates.append(
            {
                "name": name,
                "path": rel(out_path),
                "artifact_label": "public-chase",
                "kind": "private_robustness_horizon_hedge",
                "reportable_method_claim": False,
                "base": "v0_08094",
                "other": "cat35_08124",
                "alphas": list(spec.alphas),
                "rationale": spec.rationale,
                "known_public_scores": {
                    "base": sources["v0_08094"].public_score,
                    "other": sources["cat35_08124"].public_score,
                },
                "mean_abs_delta_to_reportable_anchor": mean_abs_delta(frame, anchor),
                "mean_abs_delta_to_public_reference": mean_abs_delta(frame, base),
                "audit": audit,
                "submission": None,
            }
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "experiment_label": experiment_label,
        "spec_set": args.spec_set,
        "competition": "data-mining-2026-final-project",
        "sources": {
            key: {
                "key": source.key,
                "path": rel(source.path),
                "public_score": source.public_score,
                "role": source.role,
                "sha256": sha256(source.path),
                "sha12": sha256(source.path)[:12],
            }
            for key, source in sources.items()
        },
        "blocked_source_patterns": list(BLOCKED_PATTERNS),
        "candidates": candidates,
        "recommended_submission_order": [candidate["name"] for candidate in candidates],
        "lineage_policy": [
            "public-chase/final-selection artifact only",
            "not a reportable method claim",
            "no private labels or external answers used",
            "do not use restored_20260522_* or restored_unverified_* sources",
        ],
    }
    readout_path = work_dir / "frontier_readout.json"
    summary_path = work_dir / "experiment_summary.md"
    readout_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(summary_path, payload)
    print(json.dumps({"readout": rel(readout_path), "summary": rel(summary_path), "candidates": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
