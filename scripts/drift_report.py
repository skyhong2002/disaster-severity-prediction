#!/usr/bin/env python3
"""Compare train-tail candidates against the Kaggle test-window distribution."""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import save_json  # noqa: E402
from features import METEO_COLS  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a train/test distribution drift report.")
    parser.add_argument(
        "--tail-days",
        default="1095,1825,2737,3650,0",
        help="Comma-separated train-tail daily row counts per region. 0 means full train.",
    )
    parser.add_argument("--max-samples", type=int, default=100000, help="Max rows per side for expensive metrics.")
    parser.add_argument("--bins", type=int, default=10, help="PSI quantile bins.")
    parser.add_argument("--out", default=None, help="Markdown report path.")
    parser.add_argument("--json-out", default=None, help="Optional JSON metrics path.")
    return parser.parse_args()


def parse_tail_days(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one tail-days value is required.")
    return values


def sample_frame(df: pd.DataFrame, max_samples: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    return df.sample(max_samples, random_state=random_state)


def psi(expected: pd.Series, actual: pd.Series, bins: int) -> float:
    expected = expected.dropna().to_numpy()
    actual = actual.dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    a = np.sort(a.dropna().to_numpy())
    b = np.sort(b.dropna().to_numpy())
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def quantile_wasserstein(a: pd.Series, b: pd.Series, n_quantiles: int = 101) -> float:
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    qs = np.linspace(0, 1, n_quantiles)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["month"] = work["date"].astype(str).str.slice(5, 7).fillna("00")
    work["region_id"] = work["region_id"].astype(str)
    return work


def adversarial_auc(train_tail: pd.DataFrame, test: pd.DataFrame, max_samples: int, include_context: bool = False) -> float:
    left = sample_frame(train_tail[METEO_COLS], max_samples // 2, random_state=11)
    right = sample_frame(test[METEO_COLS], max_samples // 2, random_state=12)
    if include_context:
        left = sample_frame(add_context_columns(train_tail)[["region_id", "month"] + METEO_COLS], max_samples // 2, random_state=11)
        right = sample_frame(add_context_columns(test)[["region_id", "month"] + METEO_COLS], max_samples // 2, random_state=12)
    X = pd.concat([left, right], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y = np.array([0] * len(left) + [1] * len(right))
    if len(np.unique(y)) < 2:
        return float("nan")
    if include_context:
        transformer = ColumnTransformer(
            transformers=[
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), METEO_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["region_id", "month"]),
            ]
        )
        model = make_pipeline(
            transformer,
            LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", n_jobs=-1),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def select_tail(train: pd.DataFrame, tail_days: int) -> pd.DataFrame:
    if tail_days <= 0:
        return train
    return (
        train.sort_values(["region_id", "date"])
        .groupby("region_id", group_keys=False)
        .tail(tail_days)
        .reset_index(drop=True)
    )


def summarize_tail(train_tail: pd.DataFrame, test: pd.DataFrame, bins: int, max_samples: int) -> dict:
    train_sample = sample_frame(train_tail, max_samples, random_state=21)
    test_sample = sample_frame(test, max_samples, random_state=22)
    feature_metrics = {}
    for col in METEO_COLS:
        feature_metrics[col] = {
            "train_mean": float(train_tail[col].mean()),
            "test_mean": float(test[col].mean()),
            "mean_delta": float(test[col].mean() - train_tail[col].mean()),
            "psi": psi(train_sample[col], test_sample[col], bins),
            "ks": ks_stat(train_sample[col], test_sample[col]),
            "q_wasserstein": quantile_wasserstein(train_sample[col], test_sample[col]),
        }
    return {
        "rows": int(len(train_tail)),
        "adversarial_auc_weather": adversarial_auc(train_tail, test, max_samples, include_context=False),
        "adversarial_auc_with_region_month": adversarial_auc(train_tail, test, max_samples, include_context=True),
        "feature_metrics": feature_metrics,
        "avg_psi": float(np.nanmean([m["psi"] for m in feature_metrics.values()])),
        "avg_ks": float(np.nanmean([m["ks"] for m in feature_metrics.values()])),
        "avg_q_wasserstein": float(np.nanmean([m["q_wasserstein"] for m in feature_metrics.values()])),
    }


def format_report(metrics: dict, tail_values: list[int]) -> str:
    lines = [
        "# Drift Report",
        "",
        f"Generated: {metrics['created_at']}",
        "",
        "Lower PSI/KS/Wasserstein and lower adversarial AUC mean the train-tail candidate looks more like the Kaggle test window. AUC near 0.5 means the classifier struggles to separate train from test.",
        "",
        "## Tail Candidate Summary",
        "",
        "| tail_days | rows | auc_weather | auc_region_month | avg_psi | avg_ks | avg_q_wasserstein |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tail in tail_values:
        item = metrics["tail_candidates"][str(tail)]
        label = "full" if tail == 0 else str(tail)
        lines.append(
            f"| {label} | {item['rows']} | {item['adversarial_auc_weather']:.4f} | "
            f"{item['adversarial_auc_with_region_month']:.4f} | "
            f"{item['avg_psi']:.4f} | {item['avg_ks']:.4f} | {item['avg_q_wasserstein']:.4f} |"
        )

    lines.extend(["", "## Most Drifted Features", ""])
    for tail in tail_values:
        item = metrics["tail_candidates"][str(tail)]
        label = "full" if tail == 0 else str(tail)
        ranked = sorted(item["feature_metrics"].items(), key=lambda pair: pair[1]["psi"], reverse=True)[:8]
        lines.extend([f"### tail_days={label}", "", "| feature | psi | ks | mean_delta |", "|---|---:|---:|---:|"])
        for feature, vals in ranked:
            lines.append(f"| {feature} | {vals['psi']:.4f} | {vals['ks']:.4f} | {vals['mean_delta']:.4f} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    tail_values = parse_tail_days(args.tail_days)
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "docs" / "validation" / f"drift_report_{datetime.now().strftime('%Y%m%d')}.md"
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    json_path = Path(args.json_out) if args.json_out else out_path.with_suffix(".json")
    if not json_path.is_absolute():
        json_path = ROOT / json_path

    usecols = ["region_id", "date"] + METEO_COLS
    print("Loading train/test meteorological data ...")
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    test = pd.read_csv(ROOT / "data" / "test.csv", usecols=usecols)
    print(f"  train={train.shape}, test={test.shape}")

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_samples": args.max_samples,
        "bins": args.bins,
        "tail_candidates": {},
    }
    for tail in tail_values:
        label = "full" if tail == 0 else str(tail)
        print(f"\nEvaluating tail_days={label} ...")
        train_tail = select_tail(train, tail)
        metrics["tail_candidates"][str(tail)] = summarize_tail(train_tail, test, args.bins, args.max_samples)
        item = metrics["tail_candidates"][str(tail)]
        print(
            f"  auc_weather={item['adversarial_auc_weather']:.4f}, "
            f"auc_region_month={item['adversarial_auc_with_region_month']:.4f}, "
            f"avg_psi={item['avg_psi']:.4f}, "
            f"avg_ks={item['avg_ks']:.4f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_report(metrics, tail_values), encoding="utf-8")
    save_json(json_path, metrics)
    print(f"\nMarkdown report saved -> {out_path}")
    print(f"JSON metrics saved -> {json_path}")


if __name__ == "__main__":
    main()
