"""
Utilities for versioned experiment outputs.

Each training run writes its config, metrics, models, and submissions under
experiments/<run_id>/ so future algorithms can be compared without overwriting
the current implementation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = ROOT / "experiments"
LATEST_FILE = EXPERIMENT_DIR / "latest.txt"


def make_run_id(model_family: str, experiment_name: str | None = None) -> str:
    """Create a readable run id that is stable enough for reports and files."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment_name or "run"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return f"{ts}_{model_family}_{safe_name}"


def create_run_dir(model_family: str, experiment_name: str | None = None) -> Path:
    """Create and return experiments/<run_id> with standard subdirectories."""
    run_dir = EXPERIMENT_DIR / make_run_id(model_family, experiment_name)
    (run_dir / "models").mkdir(parents=True, exist_ok=False)
    (run_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_latest_run(run_dir: Path) -> None:
    """Record the latest completed run for predict.py defaults."""
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    LATEST_FILE.write_text(run_dir.name + "\n", encoding="utf-8")


def get_latest_run_dir() -> Path | None:
    """Return the most recent experiment run directory, if available."""
    if not LATEST_FILE.exists():
        return None
    run_id = LATEST_FILE.read_text(encoding="utf-8").strip()
    if not run_id:
        return None
    run_dir = EXPERIMENT_DIR / run_id
    return run_dir if run_dir.exists() else None


def to_jsonable(value: Any) -> Any:
    """Convert numpy/scalar/path values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
