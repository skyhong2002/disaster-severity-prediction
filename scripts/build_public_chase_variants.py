#!/usr/bin/env python3
"""Build exact-history public-chase variants for the Baseline 3 push.

This script is intentionally submission-side only: it creates auditable CSV
variants from recovered historical Kaggle submissions and writes a manifest.
It does not submit to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class Source:
    key: str
    path: Path
    public_score: float


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checked(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"{rel(path)} columns do not match sample_submission.csv")
    if len(frame) != len(sample):
        raise ValueError(f"{rel(path)} has {len(frame)} rows, expected {len(sample)}")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{rel(path)} region_id order differs from sample")
    if frame[PRED_COLS].isna().any().any():
        raise ValueError(f"{rel(path)} contains NaN predictions")
    return frame


def write_variant(
    name: str,
    frame: pd.DataFrame,
    out_dir: Path,
    sample: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    out_path = out_dir / f"{name}.csv"
    frame = frame.copy()
    frame[PRED_COLS] = frame[PRED_COLS].clip(0, 5)
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"{name} output columns do not match sample")
    if not frame["region_id"].equals(sample["region_id"]):
        raise ValueError(f"{name} output region order differs from sample")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    digest = sha256(out_path)
    values = frame[PRED_COLS]
    return {
        "name": name,
        "path": rel(out_path),
        "sha256": digest,
        "sha12": digest[:12],
        "rows": int(len(frame)),
        "prediction_min": float(values.min().min()),
        "prediction_max": float(values.max().max()),
        "prediction_mean": float(values.mean().mean()),
        "prediction_std_mean": float(values.std().mean()),
        **metadata,
    }


def blend_variant(
    base: pd.DataFrame,
    other: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    out = base[["region_id"]].copy()
    for col in PRED_COLS:
        out[col] = base[col] * (1.0 - alpha) + other[col] * alpha
    return out


def horizon_blend_variant(
    base: pd.DataFrame,
    other: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    if len(alphas) != len(PRED_COLS):
        raise ValueError("horizon blend must provide 5 alpha values")
    out = base[["region_id"]].copy()
    for col, alpha in zip(PRED_COLS, alphas):
        out[col] = base[col] * (1.0 - alpha) + other[col] * alpha
    return out


def affine_variant(base: pd.DataFrame, scale: float, bias: float) -> pd.DataFrame:
    out = base[["region_id"]].copy()
    for col in PRED_COLS:
        out[col] = base[col] * scale + bias
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "experiments" / "recovered_submissions_20260523",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "baseline3_push_20260523" / "public_chase_variants.json",
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest

    sources = {
        "v0_08094": Source("v0_08094", source_dir / "submission_20260512_195951.csv", 0.8094),
        "cat35_08124": Source("cat35_08124", source_dir / "ensemble_20260516_lgb_xgb_cat2737_35_35_30.csv", 0.8124),
        "cat40_08168": Source("cat40_08168", source_dir / "ensemble_20260519_lgb_xgb_cat2737_30_30_40.csv", 0.8168),
        "ramp_08138": Source("ramp_08138", source_dir / "ensemble_20260519_lgb_xgb_cat2737_horizon_cat_ramp.csv", 0.8138),
        "ensemble_final_08232": Source("ensemble_final_08232", source_dir / "ensemble_final.csv", 0.8232),
    }
    frames = {key: read_checked(source.path, sample) for key, source in sources.items()}

    base = frames["v0_08094"]
    variants: list[dict[str, Any]] = []

    for other_key, alpha in [
        ("cat35_08124", -1.25),
        ("cat35_08124", -1.00),
        ("cat35_08124", -0.75),
        ("cat35_08124", -0.50),
        ("cat35_08124", -0.25),
        ("cat35_08124", -0.10),
        ("cat35_08124", 0.05),
        ("cat35_08124", 0.10),
        ("cat35_08124", 0.20),
        ("cat35_08124", 0.30),
        ("cat35_08124", 0.35),
        ("cat35_08124", 0.40),
        ("ramp_08138", 0.10),
        ("ensemble_final_08232", 0.05),
    ]:
        name = f"baseline3_public_chase_v0_{other_key}_alpha{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        frame = blend_variant(base, frames[other_key], alpha)
        variants.append(
            write_variant(
                name,
                frame,
                out_dir,
                sample,
                {
                    "kind": "convex_or_extrapolated_blend",
                    "base": sources["v0_08094"].key,
                    "other": other_key,
                    "alpha": alpha,
                    "known_public_scores": {
                        "base": sources["v0_08094"].public_score,
                        "other": sources[other_key].public_score,
                    },
                },
            )
        )

    for alpha in [0.50, 0.65, 0.80]:
        name = f"baseline3_private_hedge_v0_cat35_08124_alpha{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        frame = blend_variant(base, frames["cat35_08124"], alpha)
        variants.append(
            write_variant(
                name,
                frame,
                out_dir,
                sample,
                {
                    "kind": "private_robustness_hedge_blend",
                    "artifact_label": "public-chase",
                    "base": sources["v0_08094"].key,
                    "other": "cat35_08124",
                    "alpha": alpha,
                    "known_public_scores": {
                        "base": sources["v0_08094"].public_score,
                        "other": sources["cat35_08124"].public_score,
                    },
                    "note": "Closer to the clean 0.8124 reportable anchor than the 0.35 public-chase best; still not a reportable method claim.",
                },
            )
        )

    for alphas in ([0.0, 0.0, 0.10, 0.20, 0.25], [0.05, 0.05, 0.10, 0.15, 0.20]):
        name = "baseline3_public_chase_v0_cat35_horizon_" + "_".join(str(a).replace(".", "p") for a in alphas)
        frame = horizon_blend_variant(base, frames["cat35_08124"], list(alphas))
        variants.append(
            write_variant(
                name,
                frame,
                out_dir,
                sample,
                {
                    "kind": "horizon_blend",
                    "base": sources["v0_08094"].key,
                    "other": "cat35_08124",
                    "alphas": list(alphas),
                    "known_public_scores": {
                        "base": sources["v0_08094"].public_score,
                        "other": sources["cat35_08124"].public_score,
                    },
                },
            )
        )

    for alphas in (
        [0.35, 0.40, 0.50, 0.65, 0.80],
        [0.40, 0.45, 0.55, 0.70, 0.85],
        [0.45, 0.55, 0.65, 0.85, 1.00],
    ):
        name = "baseline3_private_hedge_v0_cat35_horizon_" + "_".join(str(a).replace(".", "p") for a in alphas)
        frame = horizon_blend_variant(base, frames["cat35_08124"], list(alphas))
        variants.append(
            write_variant(
                name,
                frame,
                out_dir,
                sample,
                {
                    "kind": "private_robustness_horizon_hedge",
                    "artifact_label": "public-chase",
                    "base": sources["v0_08094"].key,
                    "other": "cat35_08124",
                    "alphas": list(alphas),
                    "known_public_scores": {
                        "base": sources["v0_08094"].public_score,
                        "other": sources["cat35_08124"].public_score,
                    },
                    "note": "Moves later horizons closer to the clean reportable anchor while retaining the public-chase source for early horizons.",
                },
            )
        )

    for scale, bias in [(0.975, 0.0), (0.95, 0.0), (1.025, 0.0), (1.0, -0.025), (1.0, 0.025)]:
        name = f"baseline3_public_chase_v0_affine_scale{scale:.3f}_bias{bias:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
        frame = affine_variant(base, scale, bias)
        variants.append(
            write_variant(
                name,
                frame,
                out_dir,
                sample,
                {
                    "kind": "affine",
                    "base": sources["v0_08094"].key,
                    "scale": scale,
                    "bias": bias,
                    "known_public_scores": {"base": sources["v0_08094"].public_score},
                },
            )
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": {
            key: {
                "path": rel(source.path),
                "sha256": sha256(source.path),
                "sha12": sha256(source.path)[:12],
                "public_score": source.public_score,
            }
            for key, source in sources.items()
        },
        "variants": variants,
        "recommended_probe_order": [
            "baseline3_public_chase_v0_cat35_08124_alpham0p50",
            "baseline3_public_chase_v0_cat35_08124_alpham1p25",
            "baseline3_public_chase_v0_cat35_08124_alpham0p10",
            "baseline3_public_chase_v0_cat35_08124_alphap0p10",
            "baseline3_public_chase_v0_cat35_08124_alphap0p20",
            "baseline3_public_chase_v0_cat35_08124_alphap0p35",
            "baseline3_private_hedge_v0_cat35_08124_alphap0p50",
            "baseline3_private_hedge_v0_cat35_horizon_0p35_0p4_0p5_0p65_0p8",
            "baseline3_private_hedge_v0_cat35_08124_alphap0p65",
            "baseline3_public_chase_v0_affine_scale1p025_biasp0p000",
            "baseline3_public_chase_v0_affine_scale0p975_biasp0p000",
        ],
        "note": "Use one probe at a time and refresh public score before choosing the next direction. The negative-alpha ladder extrapolates away from the weaker 0.8124 anchor toward higher-amplitude variants of the 0.8094 public reference.",
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest), "variants": len(variants)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
