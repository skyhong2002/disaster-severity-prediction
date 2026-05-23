#!/usr/bin/env python3
"""Plot train/test distributions for weather and drought-proxy features."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_train_test_distribution import (  # noqa: E402
    ANALYSIS_COLS,
    BASELINE_LABELS,
    DERIVED_COLS,
    FEATURE_LABELS,
    RAW_REPORT_COLS,
    ROLLING_COLS,
    build_feature_frames,
    load_weather,
    select_baselines,
)

DEFAULT_BASELINES = ["full_train", "tail_1095d", "prev_year_same_md"]
PLOT_BASELINE_LABELS = {
    "full_train": "full train",
    "tail_1095d": "recent 1095d train",
    "tail_365d": "recent 365d train",
    "adjacent_prev_91d": "91d before test",
    "same_md_all_years": "same month-day, all years",
    "same_md_recent_5y": "same month-day, recent 5y",
    "prev_year_same_md": "same month-day, previous year",
}
GROUPS = {
    "Raw weather features": RAW_REPORT_COLS,
    "Daily dryness proxies": DERIVED_COLS,
    "Shifted rolling context": ROLLING_COLS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/test feature distributions.")
    parser.add_argument(
        "--out-dir",
        default="docs/validation/train_test_distribution_plots_20260523",
        help="Directory for PNG plots.",
    )
    parser.add_argument(
        "--index-out",
        default="docs/validation/train_test_distribution_plots_20260523.md",
        help="Markdown gallery path.",
    )
    parser.add_argument(
        "--pdf-out",
        default="docs/validation/train_test_distribution_plots_20260523.pdf",
        help="Optional multi-page PDF path. Use '' to disable.",
    )
    parser.add_argument("--max-samples", type=int, default=80000, help="Max rows sampled per split and baseline.")
    parser.add_argument("--bins", type=int, default=70, help="Histogram bins for continuous features.")
    parser.add_argument(
        "--baselines",
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated train baselines to compare with test.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Optional comma-separated feature subset. Default plots every analysis feature.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG output DPI.")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def slugify(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_").lower()


def clean_sample(series: pd.Series, max_samples: int, seed: int) -> pd.Series:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) > max_samples:
        values = values.sample(max_samples, random_state=seed)
    return values.astype("float64")


def is_binary(train_values: pd.Series, test_values: pd.Series) -> bool:
    values = pd.concat([train_values, test_values], ignore_index=True).dropna().unique()
    if len(values) > 3:
        return False
    rounded = {round(float(value), 6) for value in values}
    return rounded.issubset({0.0, 1.0})


def clipped_limits(train_values: pd.Series, test_values: pd.Series) -> tuple[float, float]:
    values = pd.concat([train_values, test_values], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0, 1.0
    low = float(values.quantile(0.005))
    high = float(values.quantile(0.995))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if math.isclose(low, high):
        pad = abs(low) * 0.05 + 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.03
    return low - pad, high + pad


def plot_hist(ax: plt.Axes, train_values: pd.Series, test_values: pd.Series, bins: int, title: str) -> None:
    if is_binary(train_values, test_values):
        x = np.array([0, 1])
        train_share = [float((train_values == value).mean()) for value in x]
        test_share = [float((test_values == value).mean()) for value in x]
        width = 0.34
        ax.bar(x - width / 2, train_share, width=width, label="train", color="#4C78A8", alpha=0.78)
        ax.bar(x + width / 2, test_share, width=width, label="test", color="#F58518", alpha=0.78)
        ax.set_xticks(x)
        ax.set_ylim(0, max(train_share + test_share + [0.05]) * 1.18)
        ax.set_ylabel("share")
    else:
        low, high = clipped_limits(train_values, test_values)
        train_plot = train_values.clip(low, high)
        test_plot = test_values.clip(low, high)
        ax.hist(train_plot, bins=bins, density=True, alpha=0.36, color="#4C78A8", label="train")
        ax.hist(test_plot, bins=bins, density=True, alpha=0.36, color="#F58518", label="test")
        ax.set_xlim(low, high)
        ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.22)


def plot_ecdf(ax: plt.Axes, train_values: pd.Series, test_values: pd.Series, title: str) -> None:
    low, high = clipped_limits(train_values, test_values)
    for values, label, color in (
        (train_values.clip(low, high), "train", "#4C78A8"),
        (test_values.clip(low, high), "test", "#F58518"),
    ):
        sorted_values = np.sort(values.to_numpy())
        if len(sorted_values) == 0:
            continue
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.plot(sorted_values, y, label=label, color=color, linewidth=1.6)
    ax.set_xlim(low, high)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.22)


def stats_text(train_values: pd.Series, test_values: pd.Series) -> str:
    train_mean = train_values.mean()
    test_mean = test_values.mean()
    train_med = train_values.median()
    test_med = test_values.median()
    return (
        f"mean: {train_mean:.3g} -> {test_mean:.3g}   "
        f"median: {train_med:.3g} -> {test_med:.3g}   "
        f"n: {len(train_values):,} / {len(test_values):,}"
    )


def plot_feature(
    feature: str,
    baselines: dict[str, pd.DataFrame],
    test: pd.DataFrame,
    baseline_names: list[str],
    out_path: Path,
    max_samples: int,
    bins: int,
    dpi: int,
    pdf: PdfPages | None = None,
) -> None:
    label = FEATURE_LABELS.get(feature, feature)
    fig, axes = plt.subplots(2, len(baseline_names), figsize=(5.2 * len(baseline_names), 7.3), squeeze=False)
    test_values = clean_sample(test[feature], max_samples, seed=991)

    for idx, baseline_name in enumerate(baseline_names):
        train_values = clean_sample(baselines[baseline_name][feature], max_samples, seed=100 + idx)
        baseline_label = PLOT_BASELINE_LABELS.get(baseline_name, BASELINE_LABELS.get(baseline_name, baseline_name))
        plot_hist(axes[0, idx], train_values, test_values, bins, baseline_label)
        plot_ecdf(axes[1, idx], train_values, test_values, stats_text(train_values, test_values))
        if idx == 0:
            axes[0, idx].legend(loc="best", fontsize=8)
            axes[1, idx].legend(loc="best", fontsize=8)
        axes[1, idx].set_xlabel(feature)

    fig.suptitle(f"{feature}: {label}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def build_index(index_path: Path, plot_paths: dict[str, Path], features: list[str], out_dir: Path, pdf_path: Path | None) -> None:
    rel_dir = out_dir.relative_to(index_path.parent)
    lines = [
        "# Train/Test Distribution Plot Gallery",
        "",
        "Each PNG has density or share bars on the top row and ECDF curves on the bottom row.",
        "The train panels compare test against full train, recent tail train, and season-matched previous-year train.",
        "",
    ]
    if pdf_path is not None:
        lines.append(f"- [Multi-page PDF]({pdf_path.relative_to(index_path.parent)})")
    lines.append(f"- PNG directory: `{rel_dir}`")
    lines.append("")

    for group_name, group_features in GROUPS.items():
        group_present = [feature for feature in group_features if feature in features]
        if not group_present:
            continue
        lines.extend([f"## {group_name}", "", "| feature | plot |", "|---|---|"])
        for feature in group_present:
            rel_plot = plot_paths[feature].relative_to(index_path.parent)
            label = FEATURE_LABELS.get(feature, feature)
            lines.append(f"| `{feature}` | [{label}]({rel_plot}) |")
        lines.append("")

    leftovers = [feature for feature in features if feature not in {item for values in GROUPS.values() for item in values}]
    if leftovers:
        lines.extend(["## Other Features", "", "| feature | plot |", "|---|---|"])
        for feature in leftovers:
            rel_plot = plot_paths[feature].relative_to(index_path.parent)
            lines.append(f"| `{feature}` | [{FEATURE_LABELS.get(feature, feature)}]({rel_plot}) |")
        lines.append("")

    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    index_path = resolve_path(args.index_out)
    pdf_path = resolve_path(args.pdf_out) if args.pdf_out else None
    baseline_names = parse_csv_list(args.baselines)
    features = parse_csv_list(args.features) or ANALYSIS_COLS

    unknown_features = sorted(set(features) - set(ANALYSIS_COLS))
    if unknown_features:
        raise ValueError(f"Unknown features: {', '.join(unknown_features)}")

    train_raw, test_raw = load_weather()
    train_features, test_features = build_feature_frames(train_raw, test_raw)
    baselines = select_baselines(train_features, test_features)

    unknown_baselines = sorted(set(baseline_names) - set(baselines))
    if unknown_baselines:
        raise ValueError(f"Unknown baselines: {', '.join(unknown_baselines)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    plot_paths: dict[str, Path] = {}
    pdf_context = PdfPages(pdf_path) if pdf_path is not None else None
    try:
        for idx, feature in enumerate(features, start=1):
            out_path = out_dir / f"{idx:02d}_{slugify(feature)}.png"
            print(f"[{idx:02d}/{len(features):02d}] Plotting {feature} -> {out_path}")
            plot_feature(
                feature,
                baselines,
                test_features,
                baseline_names,
                out_path,
                args.max_samples,
                args.bins,
                args.dpi,
                pdf_context,
            )
            plot_paths[feature] = out_path
    finally:
        if pdf_context is not None:
            pdf_context.close()

    build_index(index_path, plot_paths, features, out_dir, pdf_path)
    print(f"Plot gallery saved -> {index_path}")
    if pdf_path is not None:
        print(f"PDF saved -> {pdf_path}")


if __name__ == "__main__":
    main()
