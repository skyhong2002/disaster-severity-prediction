#!/usr/bin/env python3
"""Check that current-state docs agree on the active submission narrative."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "docs" / "current_state.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_prefix(path: Path, length: int) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:length]


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    state = json.loads(read_text(STATE_PATH))
    failures: list[str] = []

    best = state["current_best"]
    best_path = ROOT / best["submission"]
    require(best_path.exists(), f"Missing current best submission: {best['submission']}", failures)
    if best_path.exists():
        require(
            csv_row_count(best_path) == int(best["rows"]),
            f"Current best row count differs from docs/current_state.json: {best['submission']}",
            failures,
        )
        expected_sha = best["sha256_prefix"]
        require(
            sha256_prefix(best_path, len(expected_sha)) == expected_sha,
            f"Current best SHA-256 prefix differs from docs/current_state.json: {best['submission']}",
            failures,
        )

    anchor = state["previous_anchor"]
    anchor_path = ROOT / anchor["submission"]
    require(anchor_path.exists(), f"Missing previous anchor submission: {anchor['submission']}", failures)
    if anchor_path.exists():
        expected_sha = anchor["sha256_prefix"]
        require(
            sha256_prefix(anchor_path, len(expected_sha)) == expected_sha,
            f"Previous anchor SHA-256 prefix differs from docs/current_state.json: {anchor['submission']}",
            failures,
        )

    required_tokens = [
        best["public_mae"],
        best["submission"],
        "CatBoost",
    ]
    for doc in state["current_docs"]:
        path = ROOT / doc
        require(path.exists(), f"Missing current-state doc: {doc}", failures)
        if not path.exists():
            continue
        text = read_text(path)
        for token in required_tokens:
            require(token in text, f"{doc} is missing current-state token: {token}", failures)
        for phrase in state["banned_current_state_phrases"]:
            require(phrase not in text, f"{doc} contains banned stale phrase: {phrase}", failures)

    for doc in state["archived_docs"]:
        path = ROOT / doc
        require(path.exists(), f"Missing archived doc: {doc}", failures)
        if path.exists():
            text = read_text(path)
            require(
                "archived" in text.lower(),
                f"{doc} should explicitly mark itself as archived",
                failures,
            )

    if failures:
        print("Current-state consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Current-state consistency check passed.")
    print(f"- Current best: {best['submission']} ({best['public_mae']} public MAE)")
    print(f"- Previous anchor: {anchor['submission']} ({anchor['public_mae']} public MAE)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
