"""Task library: reads the CSV-backed catalog of recordable tasks."""

from __future__ import annotations

import csv
from pathlib import Path

from egocap.studio.studio_path_names import (
    canonicalize_studio_dir_name,
    sanitize_studio_dir_name,
)

TASK_CSV_DIR = Path(__file__).resolve().parents[1] / "task_library"


def _parse_task_csv(csv_path: Path, library: dict[str, list[tuple[str, str]]]) -> None:
    """Parse a single studio CSV and merge rows into *library* keyed by Studio."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        return
    headers = [h.strip().lower() for h in rows[2]]
    try:
        studio_idx = headers.index("studio")
        task_idx = headers.index("task")
        desc_idx = headers.index("description")
    except ValueError:
        return
    max_idx = max(studio_idx, task_idx, desc_idx)
    for row in rows[3:]:
        if len(row) <= max_idx:
            continue
        studio = sanitize_studio_dir_name(row[studio_idx].strip())
        task = row[task_idx].strip()
        description = row[desc_idx].strip()
        if studio and task:
            library.setdefault(studio, []).append((task, description))


def load_task_library() -> dict[str, list[tuple[str, str]]]:
    """Load local studio CSVs, returning an empty library when none are usable."""
    if not TASK_CSV_DIR.is_dir():
        return {}
    csv_files = sorted(TASK_CSV_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if "Summary" not in f.name]
    if not csv_files:
        return {}
    try:
        library: dict[str, list[tuple[str, str]]] = {}
        for csv_path in csv_files:
            _parse_task_csv(csv_path, library)
        for k in library:
            library[k].sort(key=lambda x: x[0])
        return library
    except Exception:
        return {}


TASK_LIBRARY = load_task_library()


def lookup_task_description(
    task_library: dict[str, list[tuple[str, str]]],
    studio: str,
    task_name: str,
) -> str | None:
    """Return the CSV description for *task_name* under *studio*, or None."""
    if not studio or not task_name:
        return None
    known = frozenset(task_library.keys())
    studio_key = canonicalize_studio_dir_name(studio.strip(), known)
    for t, desc in task_library.get(studio_key, []):
        if t == task_name.strip():
            return desc
    return None
