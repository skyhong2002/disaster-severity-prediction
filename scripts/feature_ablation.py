#!/usr/bin/env python3
"""Generate or run coarse feature-group ablation experiments."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GROUPS = [
    "score_history",
    "climatology",
    "calendar",
    "short_lag",
    "rolling",
    "ewm",
    "long_drought_proxy",
    "domain_indices",
    "region_stats",
]


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def base_command(args, group: str | None) -> list[str]:
    if args.model_family == "lightgbm":
        cmd = ["uv", "run", "python", "src/train.py"]
    elif args.model_family == "xgboost":
        cmd = ["uv", "run", "python", "src/train_xgb.py"]
    elif args.model_family == "catboost":
        cmd = ["uv", "run", "python", "src/train_catboost.py"]
    else:
        raise ValueError(f"Unknown model family: {args.model_family}")

    suffix = "baseline" if group is None else f"minus_{group}"
    cmd.extend(
        [
            "--experiment-name",
            f"{args.experiment_prefix}_{suffix}",
            "--feature-profile",
            args.feature_profile,
            "--validation-mode",
            args.validation_mode,
            "--rolling-folds",
            str(args.rolling_folds),
            "--final-train-mode",
            args.final_train_mode,
        ]
    )
    if args.regularized:
        cmd.append("--regularized")
    if args.recency_half_life_days > 0:
        cmd.extend(["--recency-half-life-days", str(args.recency_half_life_days)])
    if args.no_score_history:
        cmd.append("--no-score-history")
    if args.no_climatology:
        cmd.append("--no-climatology")
    if args.use_global_region_stats:
        cmd.append("--use-global-region-stats")
    if group is not None:
        cmd.extend(["--drop-feature-groups", group])

    if args.model_family == "catboost":
        if args.train_tail_days > 0:
            cmd.extend(["--train-tail-days", str(args.train_tail_days)])
        if args.iterations > 0:
            cmd.extend(["--iterations", str(args.iterations)])
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join("'" + part.replace("'", "'\\''") + "'" if any(ch.isspace() for ch in part) else part for part in cmd)


def main():
    parser = argparse.ArgumentParser(description="Run or print feature-group ablation commands.")
    parser.add_argument("--model-family", choices=["lightgbm", "xgboost", "catboost"], default="lightgbm")
    parser.add_argument("--experiment-prefix", default=f"feature_ablation_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--feature-profile", choices=["micro", "lean", "full"], default="micro")
    parser.add_argument("--validation-mode", choices=["holdout", "rolling_origin"], default="rolling_origin")
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--final-train-mode", choices=["last_fold", "fold_ensemble", "refit_full"], default="refit_full")
    parser.add_argument("--regularized", action="store_true")
    parser.add_argument("--recency-half-life-days", type=float, default=0)
    parser.add_argument("--no-score-history", action="store_true")
    parser.add_argument("--no-climatology", action="store_true")
    parser.add_argument("--use-global-region-stats", action="store_true")
    parser.add_argument("--train-tail-days", type=int, default=0, help="CatBoost only.")
    parser.add_argument("--iterations", type=int, default=0, help="CatBoost only.")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only writing a manifest.")
    parser.add_argument("--out", default=None, help="Markdown manifest path.")
    args = parser.parse_args()

    groups = parse_csv(args.groups)
    commands = [base_command(args, None)] + [base_command(args, group) for group in groups]

    out_path = Path(args.out) if args.out else ROOT / "docs" / f"{args.experiment_prefix}_commands.md"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Feature Ablation Commands: {args.experiment_prefix}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for cmd in commands:
        lines.extend(["```bash", shell_join(cmd), "```", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Manifest saved -> {out_path}")

    if args.execute:
        for cmd in commands:
            print(f"\nRunning: {shell_join(cmd)}")
            subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        print("Dry run only. Add --execute to launch the ablation matrix.")


if __name__ == "__main__":
    main()
