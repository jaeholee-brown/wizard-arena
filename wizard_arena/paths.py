# wizard_arena/paths.py
from __future__ import annotations

from pathlib import Path

# Central location for all generated results (CSV, text logs, etc.).
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_results_dir() -> Path:
    """Create the results directory if it does not exist and return it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def resolve_results_path(path_like: str | Path) -> Path:
    """
    Resolve a user-specified path into the results directory.

    Absolute paths are returned unchanged. Relative paths are anchored inside
    RESULTS_DIR so runs consistently write outputs under the results folder.
    """
    path = Path(path_like)
    if path.is_absolute():
        return path
    ensure_results_dir()
    return RESULTS_DIR / path
