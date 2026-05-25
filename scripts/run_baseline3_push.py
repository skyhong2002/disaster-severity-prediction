#!/usr/bin/env python3
"""Public-first Baseline 3 push orchestrator.

The script turns completed experiment runs into sanity-checked Kaggle
submission candidates, writes an auditable ledger, and optionally submits a
quota-aware slate. It is intentionally conservative about side effects:
Kaggle submission only happens when --submit is passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"
PRED_COLS = [f"pred_week{week}" for week in range(1, 6)]
DEFAULT_COMPETITION = "data-mining-2026-final-project"
DEFAULT_TARGET_SCORE = 0.8056
DEFAULT_TEAM_NAME = "Team 5"
LEGAL_ANCHOR_SUBMISSION = "submissions/ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv"
LEGAL_ANCHOR_SCORE = "0.8124"
LEGAL_ANCHOR_BLEND = "35% LightGBM / 35% XGBoost / 30% CatBoost"
DEFAULT_SUMMARY = ROOT / "docs" / "missingness_shift_experiments_20260523_summary.md"
DEFAULT_LEDGER_JSON = ROOT / "docs" / "status" / "baseline3_push_20260523.json"
DEFAULT_LEDGER_MD = ROOT / "docs" / "status" / "baseline3_push_20260523.md"
DEFAULT_WORK_DIR = ROOT / "experiments" / "baseline3_push_20260523"
BLOCKED_SUBMISSION_PATTERNS = ("restored_20260522_", "restored_unverified_")


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ExperimentRow:
    experiment: str
    run_dir: Path
    rolling_mae: float | None
    blind_mae: float | None
    blind_output: Path | None


@dataclass
class Candidate:
    name: str
    path: Path
    kind: str
    labels: list[str]
    source: str
    priority: int
    metrics: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    submit_message: str = ""
    blocked_reason: str = ""
    submitted: bool = False
    submit_result: dict[str, Any] | None = None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_to_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_cmd(
    cmd: list[str],
    *,
    check: bool = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result = CommandResult(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {command_to_text(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return result


def kaggle_cmd(args: list[str], *, check: bool = False, timeout: int | None = 120) -> CommandResult:
    return run_cmd(
        ["uv", "run", "kaggle", *args],
        check=check,
        timeout=timeout,
        env={"UV_CACHE_DIR": ".uv-cache"},
    )


def tmux_session_active(session: str) -> bool:
    result = run_cmd(["tmux", "has-session", "-t", session])
    return result.returncode == 0


def wait_for_tmux(session: str, poll_seconds: int, timeout_minutes: int) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + timeout_minutes * 60 if timeout_minutes > 0 else None
    while tmux_session_active(session):
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "session": session,
                "active": True,
                "waited_seconds": int(time.monotonic() - started),
                "timed_out": True,
            }
        time.sleep(poll_seconds)
    return {
        "session": session,
        "active": False,
        "waited_seconds": int(time.monotonic() - started),
        "timed_out": False,
    }


def parse_submission_history(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^(\S+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+"
        r"(.*?)\s+SubmissionStatus\.COMPLETE\s+([0-9.]+)?(?:\s+([0-9.]+)?)?\s*$"
    )
    for line in raw.splitlines():
        if "SubmissionStatus.COMPLETE" not in line:
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        file_name, submitted_at, description, public_score, private_score = match.groups()
        rows.append(
            {
                "fileName": file_name,
                "date": submitted_at,
                "description": " ".join(description.split()),
                "publicScore": float(public_score) if public_score else None,
                "privateScore": float(private_score) if private_score else None,
            }
        )
    return rows


def parse_leaderboard(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(\d+)\s+(.+?)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+([0-9.]+)\s*$"
    )
    for line in raw.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        team_id, team_name, submitted_at, score = match.groups()
        rows.append(
            {
                "teamId": int(team_id),
                "teamName": " ".join(team_name.split()),
                "submissionDate": submitted_at,
                "score": float(score),
            }
        )
    return rows


def fetch_live_state(competition: str, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    submissions_result = kaggle_cmd(["competitions", "submissions", "-c", competition], timeout=180)
    leaderboard_result = kaggle_cmd(["competitions", "leaderboard", competition, "-s"], timeout=180)
    (work_dir / "kaggle_submissions_live.txt").write_text(
        submissions_result.stdout + submissions_result.stderr,
        encoding="utf-8",
    )
    (work_dir / "kaggle_leaderboard_live.txt").write_text(
        leaderboard_result.stdout + leaderboard_result.stderr,
        encoding="utf-8",
    )
    return {
        "submissions_command": {
            "cmd": submissions_result.cmd,
            "returncode": submissions_result.returncode,
            "stderr": submissions_result.stderr,
        },
        "leaderboard_command": {
            "cmd": leaderboard_result.cmd,
            "returncode": leaderboard_result.returncode,
            "stderr": leaderboard_result.stderr,
        },
        "submission_history": parse_submission_history(submissions_result.stdout),
        "leaderboard": parse_leaderboard(leaderboard_result.stdout),
    }


def parse_float(value: str) -> float | None:
    value = value.strip().strip("`")
    if not value or value.upper() in {"N/A", "NA", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_summary_table(path: Path) -> list[ExperimentRow]:
    if not path.exists():
        return []
    rows: list[ExperimentRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        parts = [part.strip().strip("`") for part in stripped.strip("|").split("|")]
        if len(parts) < 5:
            continue
        experiment, run_dir_raw, rolling_raw, blind_raw, blind_raw_path = parts[:5]
        run_dir = ROOT / run_dir_raw
        blind_output = ROOT / blind_raw_path if blind_raw_path else None
        rows.append(
            ExperimentRow(
                experiment=experiment,
                run_dir=run_dir,
                rolling_mae=parse_float(rolling_raw),
                blind_mae=parse_float(blind_raw),
                blind_output=blind_output,
            )
        )
    return rows


def submission_from_metadata(run_dir: Path) -> Path | None:
    metadata_path = run_dir / "submission_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in ("global_submission_path", "run_submission_path"):
        value = metadata.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
    return None


def generate_submission(row: ExperimentRow, allow_missing_features: bool) -> Path | None:
    existing = submission_from_metadata(row.run_dir)
    if existing:
        return existing
    if not (row.run_dir / "models").exists():
        return None
    cmd = ["uv", "run", "python", "src/predict.py", "--run-dir", rel(row.run_dir)]
    if allow_missing_features:
        cmd.append("--allow-missing-features")
    result = run_cmd(cmd, timeout=3600)
    if result.returncode != 0:
        return None
    return submission_from_metadata(row.run_dir)


def validate_submission(path: Path) -> dict[str, Any]:
    sample = pd.read_csv(SAMPLE_PATH)
    frame = pd.read_csv(path)
    expected_cols = list(sample.columns)
    pred_cols = [col for col in expected_cols if col.startswith("pred_week")]
    if list(frame.columns) != expected_cols:
        raise ValueError(f"{rel(path)} columns do not match sample submission")
    if len(frame) != len(sample):
        raise ValueError(f"{rel(path)} has {len(frame)} rows, expected {len(sample)}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{rel(path)} region_id order differs from sample submission")
    if frame[pred_cols].isna().any().any():
        raise ValueError(f"{rel(path)} contains NaN predictions")
    pred_values = frame[pred_cols].apply(pd.to_numeric, errors="coerce")
    if pred_values.isna().any().any():
        raise ValueError(f"{rel(path)} contains non-numeric predictions")
    pred_min = float(pred_values.min().min())
    pred_max = float(pred_values.max().max())
    if pred_min < -1e-9 or pred_max > 5.0 + 1e-9:
        raise ValueError(f"{rel(path)} prediction range [{pred_min}, {pred_max}] is outside [0, 5]")
    digest = sha256(path)
    return {
        "path": rel(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": digest,
        "sha12": digest[:12],
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "prediction_mean": float(pred_values.mean().mean()),
        "prediction_std_mean": float(pred_values.std().mean()),
    }


def add_candidate(candidates: list[Candidate], candidate: Candidate) -> None:
    if any(pattern in candidate.path.name for pattern in BLOCKED_SUBMISSION_PATTERNS):
        candidate.blocked_reason = "blocked_restored_or_unverified_historical_artifact"
    try:
        candidate.audit = validate_submission(candidate.path)
    except Exception as exc:  # noqa: BLE001 - ledger should explain failed candidates
        candidate.blocked_reason = f"failed_sanity_check: {exc}"
    candidates.append(candidate)


def build_single_candidates(
    rows: list[ExperimentRow],
    allow_missing_features: bool,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in rows:
        if not row.run_dir.exists():
            continue
        path = generate_submission(row, allow_missing_features)
        if path is None:
            continue
        metrics = {
            "rolling_mae": row.rolling_mae,
            "blind_mae": row.blind_mae,
            "blind_output": rel(row.blind_output) if row.blind_output else None,
        }
        candidate = Candidate(
            name=f"single_{row.experiment}",
            path=path,
            kind="single_model",
            labels=["public-chase", "reportable"],
            source=row.experiment,
            priority=20,
            metrics=metrics,
            submit_message=f"Baseline3 public-first single {row.experiment}",
        )
        add_candidate(candidates, candidate)
    return candidates


def candidate_score(candidate: Candidate) -> tuple[float, float, int, str]:
    blind = candidate.metrics.get("blind_mae")
    rolling = candidate.metrics.get("rolling_mae")
    return (
        float(blind) if blind is not None else 999.0,
        float(rolling) if rolling is not None else 999.0,
        candidate.priority,
        candidate.name,
    )


def create_affine_variant(
    base: Candidate,
    variant: str,
    blind_rows: Path,
    scale_grid: str,
    bias_grid: str,
    lambda_reg: float,
    work_dir: Path,
) -> Candidate | None:
    out_path = ROOT / "submissions" / f"baseline3_{base.path.stem}_{variant}.csv"
    out_json = work_dir / f"{out_path.stem}.json"
    result = run_cmd(
        [
            "uv",
            "run",
            "python",
            "scripts/tune_postprocess.py",
            "--blind-rows",
            rel(blind_rows),
            "--submission",
            rel(base.path),
            "--scale-grid",
            scale_grid,
            "--bias-grid",
            bias_grid,
            "--lambda-reg",
            str(lambda_reg),
            "--out-json",
            rel(out_json),
            "--out-submission",
            rel(out_path),
        ],
        timeout=900,
    )
    if result.returncode != 0 or not out_path.exists():
        return None
    metrics: dict[str, Any] = {
        "base_candidate": base.name,
        "variant": variant,
        "postprocess_json": rel(out_json),
    }
    try:
        metrics.update(json.loads(out_json.read_text(encoding="utf-8")).get("metrics", {}))
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    candidate = Candidate(
        name=f"{base.name}_{variant}",
        path=out_path,
        kind="affine_postprocess",
        labels=["public-chase"],
        source=base.name,
        priority=10,
        metrics=metrics,
        submit_message=f"Baseline3 public-first affine {variant} from {base.name}",
    )
    return candidate


def build_affine_candidates(base_candidates: list[Candidate], work_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    usable = [c for c in base_candidates if not c.blocked_reason]
    usable.sort(key=candidate_score)
    for base in usable[:1]:
        blind_output = base.metrics.get("blind_output")
        if not blind_output:
            continue
        blind_rows = ROOT / blind_output / "blind_backtest_rows.csv"
        if not blind_rows.exists():
            continue
        specs = [
            ("affine_mild", "0.90:1.05:0.025", "-0.10:0.10:0.025", 0.05),
            ("affine_aggressive", "0.80:1.15:0.025", "-0.25:0.20:0.025", 0.01),
        ]
        for variant, scale_grid, bias_grid, lambda_reg in specs:
            candidate = create_affine_variant(
                base,
                variant,
                blind_rows,
                scale_grid,
                bias_grid,
                lambda_reg,
                work_dir,
            )
            if candidate is not None:
                add_candidate(candidates, candidate)
    return candidates


def blend_frames(inputs: list[Candidate], weights: list[float], out_path: Path) -> None:
    frames = [pd.read_csv(candidate.path) for candidate in inputs]
    first = frames[0]
    blended = first[["region_id"]].copy()
    for col in PRED_COLS:
        blended[col] = sum(frame[col] * weight for frame, weight in zip(frames, weights)).clip(0, 5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out_path, index=False)


def inverse_mae_weights(candidates: list[Candidate]) -> list[float]:
    values = []
    for candidate in candidates:
        score = candidate.metrics.get("blind_mae") or candidate.metrics.get("rolling_mae")
        if score is None or float(score) <= 0:
            values.append(1.0)
        else:
            values.append(1.0 / float(score))
    total = sum(values)
    return [value / total for value in values]


def build_blend_candidates(base_candidates: list[Candidate]) -> list[Candidate]:
    candidates: list[Candidate] = []
    usable = [c for c in base_candidates if not c.blocked_reason]
    usable.sort(key=candidate_score)
    if len(usable) < 2:
        return candidates
    top_two = usable[:2]
    weights = inverse_mae_weights(top_two)
    out_path = ROOT / "submissions" / "baseline3_blend_top2_inverse_mae_20260523.csv"
    blend_frames(top_two, weights, out_path)
    candidate = Candidate(
        name="blend_top2_inverse_mae",
        path=out_path,
        kind="convex_blend",
        labels=["public-chase", "reportable"],
        source="+".join(c.name for c in top_two),
        priority=15,
        metrics={
            "inputs": [c.name for c in top_two],
            "input_paths": [rel(c.path) for c in top_two],
            "weights": weights,
        },
        submit_message="Baseline3 public-first top2 inverse-MAE blend",
    )
    add_candidate(candidates, candidate)
    return candidates


def count_used_today_utc(history: list[dict[str, Any]]) -> int:
    today = utc_now().date().isoformat()
    return sum(1 for row in history if str(row.get("date", "")).startswith(today))


def previous_submitted_sha(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    shas: set[str] = set()
    for candidate in payload.get("candidates", []):
        audit = candidate.get("audit") or {}
        if candidate.get("submitted") and audit.get("sha256"):
            shas.add(str(audit["sha256"]))
    return shas


def choose_slate(
    candidates: list[Candidate],
    history: list[dict[str, Any]],
    ledger_json: Path,
    slate_size: int,
    ignore_quota_count: bool,
) -> list[Candidate]:
    seen_names = {row["fileName"] for row in history}
    seen_sha = previous_submitted_sha(ledger_json)
    usable: list[Candidate] = []
    for candidate in candidates:
        if candidate.blocked_reason:
            continue
        if candidate.path.name in seen_names:
            candidate.blocked_reason = "already_seen_in_kaggle_history"
            continue
        if candidate.audit.get("sha256") in seen_sha:
            candidate.blocked_reason = "already_submitted_same_sha_in_ledger"
            continue
        usable.append(candidate)

    used_today = count_used_today_utc(history)
    remaining = slate_size if ignore_quota_count else max(0, slate_size - used_today)
    usable.sort(key=candidate_score)

    reportable = [c for c in usable if "reportable" in c.labels]
    public_only = [c for c in usable if "reportable" not in c.labels]
    slate: list[Candidate] = []
    if reportable:
        slate.append(reportable[0])
    for candidate in [*public_only, *reportable[1:]]:
        if candidate not in slate:
            slate.append(candidate)
        if len(slate) >= remaining:
            break
    return slate[:remaining]


def submit_candidate(candidate: Candidate, competition: str) -> dict[str, Any]:
    message = candidate.submit_message[:80]
    result = kaggle_cmd(
        [
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            rel(candidate.path),
            "-m",
            message,
        ],
        timeout=300,
    )
    payload = {
        "cmd": result.cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    candidate.submitted = result.returncode == 0
    candidate.submit_result = payload
    if result.returncode != 0:
        candidate.blocked_reason = f"submit_failed_returncode_{result.returncode}"
    return payload


def leaderboard_score(rows: list[dict[str, Any]], team_name: str) -> float | None:
    for row in rows:
        if row.get("teamName") == team_name:
            return float(row["score"])
    return None


def candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "path": rel(candidate.path),
        "kind": candidate.kind,
        "labels": candidate.labels,
        "source": candidate.source,
        "priority": candidate.priority,
        "metrics": candidate.metrics,
        "audit": candidate.audit,
        "submit_message": candidate.submit_message,
        "blocked_reason": candidate.blocked_reason,
        "submitted": candidate.submitted,
        "submit_result": candidate.submit_result,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    live = payload["live_state"]
    after = payload.get("live_state_after_submit") or live
    team_score = payload.get("team_score_after_submit", payload.get("team_score"))
    crossed = team_score is not None and team_score < payload["target_score"]
    lines = [
        "# Baseline 3 Push Status: 2026-05-23",
        "",
        f"- Updated: `{payload['created_at_utc']}` UTC",
        f"- Mode: `public-first`, daily slate size `{payload['slate_size']}`",
        f"- Baseline 3 target: `< {payload['target_score']:.4f}` public MAE",
        f"- Team score: `{team_score if team_score is not None else 'unknown'}`",
        f"- Current reportable legal anchor: `{LEGAL_ANCHOR_SUBMISSION}`, public MAE `{LEGAL_ANCHOR_SCORE}`, {LEGAL_ANCHOR_BLEND}",
        f"- Stop condition crossed: `{crossed}`",
        f"- Tmux session `{payload['tmux']['session']}` active: `{payload['tmux']['active']}`",
        f"- Submit flag used: `{payload['submit_enabled']}`",
        "",
        "## Live Leaderboard",
        "",
        "| Rank | Team | Score | Submission date |",
        "|---:|---|---:|---|",
    ]
    for idx, row in enumerate(after.get("leaderboard", [])[:20], start=1):
        lines.append(f"| {idx} | {row['teamName']} | `{row['score']:.4f}` | `{row['submissionDate']}` |")
    lines.extend(["", "## Candidate Slate", "", "| Candidate | Kind | Labels | SHA-12 | Status |", "|---|---|---|---|---|"])
    slate_names = {item["name"] for item in payload.get("slate", [])}
    for candidate in payload["candidates"]:
        status = "selected" if candidate["name"] in slate_names else "not selected"
        if candidate["blocked_reason"]:
            status = candidate["blocked_reason"]
        if candidate["submitted"]:
            status = "submitted"
        labels = ", ".join(candidate["labels"])
        sha12 = candidate.get("audit", {}).get("sha12", "")
        lines.append(
            f"| `{candidate['name']}` | `{candidate['kind']}` | `{labels}` | `{sha12}` | {status} |"
        )
    lines.extend(["", "## Current Readout", ""])
    if crossed:
        lines.append("- Baseline 3 has been crossed on the public leaderboard. Switch follow-up work back to private robustness and reportable lineage cleanup.")
    elif payload["tmux"]["active"]:
        lines.append("- The missingness/shift queue is still active. Resume after it finishes, then generate the remaining candidates.")
    elif not payload.get("slate"):
        lines.append("- No quota-ready slate was selected. Recover exact historical artifacts or wait for more completed runs.")
    elif payload["submit_enabled"]:
        lines.append("- Submitted the selected slate; refetch live history before choosing the next batch.")
    else:
        lines.append("- Dry run only. Re-run with `--submit` to spend quota on the selected slate.")
    lines.extend(
        [
            "",
            "## Useful Commands",
            "",
            "```bash",
            "tmux attach -t missingness_shift_20260523",
            "tail -f experiments/logs/missingness_shift_20260523.log",
            "UV_CACHE_DIR=.uv-cache uv run kaggle competitions leaderboard data-mining-2026-final-project -s",
            "uv run python scripts/run_baseline3_push.py --submit --allow-missing-features",
            "```",
            "",
            "## Artifact Labels",
            "",
            "- `public-chase`: optimized for immediate public leaderboard feedback.",
            "- `reportable`: reproducible, non-leaky lineage suitable for final-method claims if later evidence supports it.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--team-name", default=DEFAULT_TEAM_NAME)
    parser.add_argument("--slate-size", type=int, default=6)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ledger-json", type=Path, default=DEFAULT_LEDGER_JSON)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER_MD)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--tmux-session", default="missingness_shift_20260523")
    parser.add_argument("--wait-for-tmux", action="store_true")
    parser.add_argument("--tmux-timeout-minutes", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--ignore-quota-count", action="store_true")
    args = parser.parse_args()

    summary = args.summary if args.summary.is_absolute() else ROOT / args.summary
    ledger_json = args.ledger_json if args.ledger_json.is_absolute() else ROOT / args.ledger_json
    ledger_md = args.ledger_md if args.ledger_md.is_absolute() else ROOT / args.ledger_md
    work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir

    if args.wait_for_tmux:
        tmux_info = wait_for_tmux(args.tmux_session, args.poll_seconds, args.tmux_timeout_minutes)
    else:
        tmux_info = {
            "session": args.tmux_session,
            "active": tmux_session_active(args.tmux_session),
            "waited_seconds": 0,
            "timed_out": False,
        }

    live_state = fetch_live_state(args.competition, work_dir)
    team_score = leaderboard_score(live_state["leaderboard"], args.team_name)
    target_crossed = team_score is not None and team_score < args.target_score

    summary_rows = parse_summary_table(summary)
    candidates: list[Candidate] = []
    if not target_crossed and not tmux_info["active"]:
        single_candidates = build_single_candidates(summary_rows, args.allow_missing_features)
        candidates.extend(single_candidates)
        candidates.extend(build_affine_candidates(single_candidates, work_dir))
        candidates.extend(build_blend_candidates(single_candidates))
    elif target_crossed:
        pass
    else:
        # Queue is still active; write a monitor ledger without generating more work.
        pass

    slate = choose_slate(
        candidates,
        live_state["submission_history"],
        ledger_json,
        args.slate_size,
        args.ignore_quota_count,
    )

    submit_results = []
    if args.submit and not target_crossed and slate:
        for candidate in slate:
            submit_results.append(submit_candidate(candidate, args.competition))
            time.sleep(3)

    live_state_after = fetch_live_state(args.competition, work_dir) if submit_results else None
    team_score_after = (
        leaderboard_score(live_state_after["leaderboard"], args.team_name)
        if live_state_after is not None
        else team_score
    )

    payload = {
        "created_at_utc": utc_now().isoformat(timespec="seconds"),
        "competition": args.competition,
        "target_score": args.target_score,
        "team_name": args.team_name,
        "team_score": team_score,
        "team_score_after_submit": team_score_after,
        "target_crossed_before": target_crossed,
        "target_crossed_after": team_score_after is not None and team_score_after < args.target_score,
        "slate_size": args.slate_size,
        "submit_enabled": args.submit,
        "ignore_quota_count": args.ignore_quota_count,
        "used_today_utc_before": count_used_today_utc(live_state["submission_history"]),
        "tmux": tmux_info,
        "summary": rel(summary),
        "summary_rows": [
            {
                "experiment": row.experiment,
                "run_dir": rel(row.run_dir),
                "rolling_mae": row.rolling_mae,
                "blind_mae": row.blind_mae,
                "blind_output": rel(row.blind_output) if row.blind_output else None,
            }
            for row in summary_rows
        ],
        "live_state": live_state,
        "live_state_after_submit": live_state_after,
        "candidates": [candidate_to_json(candidate) for candidate in candidates],
        "slate": [candidate_to_json(candidate) for candidate in slate],
        "submit_results": submit_results,
        "historical_recovery_notes": [
            "Exact historical files such as the 0.8124 and 0.8094 artifacts are used only if present locally or recovered exactly.",
            "Files whose basename starts with restored_20260522_ or restored_unverified_ are blocked from submission candidate status.",
        ],
    }

    write_json(ledger_json, payload)
    write_markdown(ledger_md, payload)
    print(f"Ledger JSON -> {rel(ledger_json)}")
    print(f"Ledger Markdown -> {rel(ledger_md)}")
    print(f"Team score: {team_score_after if team_score_after is not None else 'unknown'}")
    print(f"Candidates: {len(candidates)}, slate: {len(slate)}, submitted: {sum(c.submitted for c in slate)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
