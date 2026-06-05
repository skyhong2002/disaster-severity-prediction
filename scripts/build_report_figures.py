#!/usr/bin/env python3
"""Build reproducible figures used by report/main.tex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_FIG_DIR = ROOT / "report" / "figures"
SHIFT_CSV = (
    ROOT
    / "docs"
    / "validation"
    / "train_test_distribution_comparison_20260523_feature_metrics.csv"
)


ABLATION_ROWS = [
    {
        "run": "minus_climatology",
        "features": 386,
        "avg_mae": 0.3046,
        "delta": -0.0078,
        "weeks": [0.2711, 0.2868, 0.3094, 0.3189, 0.3368],
    },
    {
        "run": "baseline",
        "features": 401,
        "avg_mae": 0.3124,
        "delta": 0.0,
        "weeks": [0.2865, 0.2960, 0.3215, 0.3275, 0.3305],
    },
    {
        "run": "minus_region_stats",
        "features": 401,
        "avg_mae": 0.3124,
        "delta": 0.0,
        "weeks": [0.2865, 0.2960, 0.3215, 0.3275, 0.3305],
    },
    {
        "run": "minus_domain_indices",
        "features": 367,
        "avg_mae": 0.3144,
        "delta": 0.0020,
        "weeks": [0.2794, 0.2958, 0.3211, 0.3217, 0.3541],
    },
    {
        "run": "minus_long_drought_proxy",
        "features": 380,
        "avg_mae": 0.3645,
        "delta": 0.0521,
        "weeks": [0.3625, 0.3683, 0.3568, 0.3601, 0.3748],
    },
    {
        "run": "minus_score_history",
        "features": 379,
        "avg_mae": 0.3645,
        "delta": 0.0521,
        "weeks": [0.3446, 0.3479, 0.3630, 0.3758, 0.3914],
    },
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _score_and_weather_readout() -> dict:
    score_counts = pd.Series(0, index=range(6), dtype="int64")
    monthly_sum = pd.Series(0.0, index=range(1, 13))
    monthly_pos = pd.Series(0, index=range(1, 13), dtype="int64")
    monthly_count = pd.Series(0, index=range(1, 13), dtype="int64")
    train_weather_parts = []

    usecols = ["date", "prec", "tmp", "humidity", "score"]
    for i, chunk in enumerate(
        pd.read_csv(DATA_DIR / "train.csv", usecols=usecols, chunksize=500_000)
    ):
        labels = chunk["score"].dropna().astype("int64")
        if not labels.empty:
            score_counts = score_counts.add(labels.value_counts(), fill_value=0).astype(
                "int64"
            )
            months = (
                chunk.loc[labels.index, "date"].str.split("-").str[-2].astype("int64")
            )
            label_frame = pd.DataFrame(
                {"month": months.to_numpy(), "score": labels.to_numpy()}
            )
            monthly_sum = monthly_sum.add(
                label_frame.groupby("month")["score"].sum(), fill_value=0.0
            )
            monthly_pos = monthly_pos.add(
                label_frame.assign(positive=label_frame["score"].gt(0).astype("int64"))
                .groupby("month")["positive"]
                .sum(),
                fill_value=0,
            ).astype("int64")
            monthly_count = monthly_count.add(
                label_frame.groupby("month").size(), fill_value=0
            )
            monthly_count = monthly_count.astype("int64")

        train_weather_parts.append(
            chunk[["prec", "tmp", "humidity"]].sample(
                frac=0.02, random_state=20260605 + i
            )
        )

    train_weather = pd.concat(train_weather_parts, ignore_index=True)
    test_weather = pd.read_csv(
        DATA_DIR / "test.csv", usecols=["prec", "tmp", "humidity"]
    )

    score_total = int(score_counts.sum())
    monthly_mean = monthly_sum / monthly_count.replace(0, np.nan)
    monthly_positive_rate = monthly_pos / monthly_count.replace(0, np.nan)

    return {
        "score_counts": {str(k): int(v) for k, v in score_counts.items()},
        "score_percent": {
            str(k): float(v / score_total * 100) for k, v in score_counts.items()
        },
        "score_labeled_rows": score_total,
        "score_positive_rate": float((score_total - score_counts.loc[0]) / score_total),
        "monthly_mean_score": {
            str(k): float(v) for k, v in monthly_mean.fillna(0).items()
        },
        "monthly_positive_rate": {
            str(k): float(v) for k, v in monthly_positive_rate.fillna(0).items()
        },
        "monthly_labeled_rows": {str(k): int(v) for k, v in monthly_count.items()},
        "train_weather_sample_rows": int(len(train_weather)),
        "test_weather_rows": int(len(test_weather)),
        "_train_weather": train_weather,
        "_test_weather": test_weather,
    }


def build_eda_figure(readout: dict) -> None:
    train_weather = readout["_train_weather"]
    test_weather = readout["_test_weather"]
    score_percent = pd.Series(readout["score_percent"], dtype="float64")
    months = np.arange(1, 13)
    monthly_mean = pd.Series(readout["monthly_mean_score"], dtype="float64")
    monthly_positive = pd.Series(readout["monthly_positive_rate"], dtype="float64")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2))

    ax = axes[0, 0]
    ax.bar(score_percent.index.astype(int), score_percent.values, color="#4c78a8")
    ax.set_title("Observed weekly score distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Share of labeled rows (%)")
    ax.set_xticks(range(6))
    for score, pct in score_percent.items():
        ax.text(int(score), pct + 0.5, f"{pct:.1f}", ha="center", va="bottom", fontsize=7)

    ax = axes[0, 1]
    ax.plot(months, monthly_mean.reindex([str(m) for m in months]).values, marker="o")
    ax.plot(
        months,
        5 * monthly_positive.reindex([str(m) for m in months]).values,
        marker="s",
        linestyle="--",
        color="#f58518",
        label="Positive-rate x 5",
    )
    ax.set_title("Seasonality in labeled scores")
    ax.set_xlabel("Month")
    ax.set_ylabel("Score scale")
    ax.set_xticks(months)
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    bins = np.linspace(0, 25, 51)
    ax.hist(
        np.clip(train_weather["prec"], 0, 25),
        bins=bins,
        density=True,
        alpha=0.58,
        label="Train sample",
        color="#54a24b",
    )
    ax.hist(
        np.clip(test_weather["prec"], 0, 25),
        bins=bins,
        density=True,
        alpha=0.52,
        label="Test",
        color="#e45756",
    )
    ax.set_title("Precipitation distribution")
    ax.set_xlabel("Daily precipitation, clipped at 25")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[1, 1]
    bins = np.linspace(-15, 45, 61)
    ax.hist(
        train_weather["tmp"],
        bins=bins,
        density=True,
        alpha=0.58,
        label="Train sample",
        color="#54a24b",
    )
    ax.hist(
        test_weather["tmp"],
        bins=bins,
        density=True,
        alpha=0.52,
        label="Test",
        color="#e45756",
    )
    ax.set_title("Temperature distribution")
    ax.set_xlabel("Daily mean temperature")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "eda_overview.png")
    plt.close(fig)


def build_shift_figure() -> dict:
    df = pd.read_csv(SHIFT_CSV)
    selected = [
        ("prec", "Daily precip."),
        ("dry_day", "Dry-day share"),
        ("rain_day_ge_1", "Rain-day share"),
        ("prec_sum_90d", "90d precip."),
        ("dry_days_90d", "90d dry days"),
        ("rain_days_90d", "90d rain days"),
        ("dew_point_depression", "Dew-point spread"),
        ("tmp_range", "Temp. range"),
    ]
    lookup = (
        df[df["baseline"].eq("prev_year_same_md")]
        .set_index("feature")
        .loc[[name for name, _ in selected]]
        .assign(short_label=[label for _, label in selected])
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    order = np.arange(len(lookup))
    colors = np.where(lookup["std_mean_delta"] >= 0, "#f58518", "#4c78a8")

    axes[0].barh(order, lookup["std_mean_delta"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(order)
    axes[0].set_yticklabels(lookup["short_label"])
    axes[0].invert_yaxis()
    axes[0].set_title("Test minus same-season previous year")
    axes[0].set_xlabel("Standardized mean delta")

    axes[1].barh(order, lookup["psi"], color="#72b7b2")
    axes[1].set_title("Population stability index")
    axes[1].set_xlabel("PSI")
    axes[1].set_yticks(order)
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "train_test_shift.png")
    plt.close(fig)

    return {
        feature: {
            "label": row["feature_label"],
            "train_mean": float(row["train_mean"]),
            "test_mean": float(row["test_mean"]),
            "mean_delta": float(row["mean_delta"]),
            "std_mean_delta": float(row["std_mean_delta"]),
            "psi": float(row["psi"]),
            "ks": float(row["ks"]),
        }
        for feature, row in lookup.iterrows()
    }


def build_ablation_figure() -> dict:
    rows = pd.DataFrame(ABLATION_ROWS)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    ordered = rows.sort_values("avg_mae", ascending=True)
    axes[0].barh(ordered["run"], ordered["avg_mae"], color="#4c78a8")
    axes[0].axvline(0.3124, color="#e45756", linestyle="--", linewidth=1.1)
    axes[0].invert_yaxis()
    axes[0].set_title("Rolling-origin ablation MAE")
    axes[0].set_xlabel("Average MAE")
    for y, (_, row) in enumerate(ordered.iterrows()):
        axes[0].text(row["avg_mae"] + 0.003, y, f"{row['delta']:+.4f}", va="center")

    week_axis = np.arange(1, 6)
    for run in [
        "baseline",
        "minus_climatology",
        "minus_long_drought_proxy",
        "minus_score_history",
    ]:
        row = rows[rows["run"].eq(run)].iloc[0]
        axes[1].plot(week_axis, row["weeks"], marker="o", label=run)
    axes[1].set_title("Horizon-specific MAE")
    axes[1].set_xlabel("Forecast week")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(week_axis)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "feature_ablation.png")
    plt.close(fig)

    return {
        row["run"]: {
            "features": int(row["features"]),
            "avg_mae": float(row["avg_mae"]),
            "delta_vs_baseline": float(row["delta"]),
            "week_mae": [float(v) for v in row["weeks"]],
        }
        for _, row in rows.iterrows()
    }


def main() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    readout = _score_and_weather_readout()
    build_eda_figure(readout)
    shift_readout = build_shift_figure()
    ablation_readout = build_ablation_figure()

    serializable = {
        key: value
        for key, value in readout.items()
        if not key.startswith("_")
    }
    serializable["train_test_shift_prev_year_same_md"] = shift_readout
    serializable["feature_ablation_lgbm_lean_tail1095"] = ablation_readout
    serializable["figure_paths"] = {
        "eda_overview": "report/figures/eda_overview.png",
        "train_test_shift": "report/figures/train_test_shift.png",
        "feature_ablation": "report/figures/feature_ablation.png",
    }

    with (REPORT_FIG_DIR / "report_figure_readout.json").open("w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
