"""Mid Train task library (CSV-backed) for the egocap "Mid Train" instruction mode.

Unlike the diverse-language ``TASK_LIBRARY`` (keyed by *Studio*), Mid Train tasks
are grouped by their manipulation **primitive** so the operator can pick a
primitive first and then the specific task. The backing CSV lives in a
``mid_train/`` subfolder of ``task_library/`` so it is *not* picked up by the
top-level ``*.csv`` studio glob in :mod:`studio.task_library_data` /
:mod:`studio.studio_lookup`.

CSV layout (header on row 0)::

    Task #, Task Name, Task Description, Primitive(s), ... , Props, ...

* **Task Name** is already ``lower_snake``; we normalize defensively anyway.
* **Primitive(s)** may be compound (``"unscrew + shake"``). The task is listed
  under *each* component primitive so selecting either shows it.
* **Props** seeds the operator's editable props box.

All Mid Train recordings route to a single ``mid_train`` studio folder (uploads
stay sorted by glove upstream of the studio segment).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Single studio bucket for every Mid Train recording (folder + zarr ``studio`` attr).
MID_TRAIN_STUDIO = "mid_train"

# Human-facing instruction-mode value persisted to ``meta.instruction_mode``.
MID_TRAIN_MODE = "mid_train"

MID_TRAIN_CSV = Path(__file__).resolve().parents[1] / "task_library" / "mid_train" / "grasp_tasks.csv"


def to_lower_snake(name: str) -> str:
    """Normalize a task name to ``lower_snake_case``.

    Lowercases, converts any run of non-alphanumeric characters to a single
    underscore, and strips leading/trailing underscores. Already-snake names
    (the common case for the Mid Train CSV) pass through unchanged.
    """
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _split_primitives(raw: str) -> list[str]:
    """Split a (possibly compound) ``Primitive(s)`` cell into component primitives.

    Compound cells use ``+`` as the separator (e.g. ``"scoop + pour"``). Hyphens
    inside a single primitive (``"pick-place"``) are preserved.
    """
    parts = [p.strip() for p in (raw or "").split("+")]
    return [p for p in parts if p]


# ``{primitive: [(task_name_snake, description, props), ...]}``
MidTrainLibrary = dict[str, list[tuple[str, str, str]]]


def load_mid_train_library(csv_path: Path | None = None) -> MidTrainLibrary:
    """Parse the Mid Train CSV into ``{primitive: [(task, description, props)]}``.

    Returns an empty dict if the CSV is missing or malformed (the UI then shows
    no primitives rather than crashing).
    """
    path = csv_path or MID_TRAIN_CSV
    library: MidTrainLibrary = {}
    if not path.is_file():
        return library
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return library
    if len(rows) < 2:
        return library

    headers = [h.strip().lower() for h in rows[0]]
    try:
        name_idx = headers.index("task name")
        desc_idx = headers.index("task description")
        prim_idx = headers.index("primitive(s)")
    except ValueError:
        return library
    try:
        props_idx = headers.index("props")
    except ValueError:
        props_idx = -1

    max_idx = max(name_idx, desc_idx, prim_idx, props_idx)
    for row in rows[1:]:
        if len(row) <= max_idx:
            continue
        task = to_lower_snake(row[name_idx])
        description = row[desc_idx].strip()
        props = row[props_idx].strip() if props_idx >= 0 else ""
        primitives = _split_primitives(row[prim_idx])
        if not task or not primitives:
            continue
        entry = (task, description, props)
        for primitive in primitives:
            bucket = library.setdefault(primitive, [])
            # Avoid duplicate (task) rows within the same primitive.
            if not any(e[0] == task for e in bucket):
                bucket.append(entry)

    for primitive in library:
        library[primitive].sort(key=lambda e: e[0])
    return library


MID_TRAIN_LIBRARY: MidTrainLibrary = load_mid_train_library()


def lookup_mid_train_task(
    primitive: str,
    task_name: str,
    library: MidTrainLibrary | None = None,
) -> tuple[str, str] | None:
    """Return ``(description, props)`` for *task_name* under *primitive*, or None."""
    lib = library if library is not None else MID_TRAIN_LIBRARY
    if not primitive or not task_name:
        return None
    target = to_lower_snake(task_name)
    for task, description, props in lib.get(primitive, []):
        if task == target:
            return description, props
    return None


def lookup_mid_train_description(
    task_name: str,
    library: MidTrainLibrary | None = None,
) -> str | None:
    """Return a task's description by name across all primitives, or None."""
    lib = library if library is not None else MID_TRAIN_LIBRARY
    if not task_name:
        return None
    target = to_lower_snake(task_name)
    for tasks in lib.values():
        for task, description, _props in tasks:
            if task == target:
                return description
    return None


# ---------------------------------------------------------------------------
# Grasp columns — for the live-collection grasp-reference thumbnails.
#
# Every task row also names a *Left Hand Grasp* and *Right Hand Grasp* from the
# Feix GRASP taxonomy, e.g. ``"Power Disk [10] (cap)"``. The bracket number maps
# 1:1 to a thumbnail in ``grasp_thumbnails/`` (``10`` -> ``10_power_disk.png``)
# and the parenthetical is a short role hint. The live recorder shows these two
# thumbnails under the operator's hands so they can see which grasp to perform.
# These columns are deliberately *not* surfaced by ``load_mid_train_library``
# (which only feeds the picker), so they get their own loader here.
# ---------------------------------------------------------------------------

GRASP_THUMBNAIL_DIR = Path(__file__).resolve().parents[1] / "task_library" / "mid_train" / "grasp_thumbnails"

_GRASP_NAME_RE = re.compile(r"^\s*(.*?)\s*\[(\d+)\]")
_GRASP_ROLE_RE = re.compile(r"\[\d+\]\s*\((.*?)\)")


def parse_grasp_cell(cell: str) -> tuple[str, int, str] | None:
    """Parse a ``Left/Right Hand Grasp`` cell into ``(name, number, role)``.

    A cell looks like ``"Power Disk [10] (cap)"``. The rare compound cell
    (``"Quadpod [27] (peach), then Sphere 4 Finger [26] (apple)"``) yields only
    the *first* (primary) grasp. Returns None if there is no ``[number]`` tag.
    """
    m = _GRASP_NAME_RE.match(cell or "")
    if not m:
        return None
    name = m.group(1).strip()
    number = int(m.group(2))
    role_m = _GRASP_ROLE_RE.search(cell)
    role = role_m.group(1).strip() if role_m else ""
    return name, number, role


def grasp_thumbnail_path(number: int) -> Path | None:
    """Return the thumbnail path for taxonomy grasp *number*, or None if absent."""
    matches = sorted(GRASP_THUMBNAIL_DIR.glob(f"{number:02d}_*.png"))
    return matches[0] if matches else None


def load_mid_train_grasps(csv_path: Path | None = None) -> dict[str, tuple[str, str]]:
    """Parse the Mid Train CSV into ``{task_name_snake: (left_cell, right_cell)}``.

    Returns an empty dict if the CSV is missing or malformed (the live view then
    shows no grasp thumbnails rather than crashing).
    """
    path = csv_path or MID_TRAIN_CSV
    grasps: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return grasps
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return grasps
    if len(rows) < 2:
        return grasps

    headers = [h.strip().lower() for h in rows[0]]
    try:
        name_idx = headers.index("task name")
        left_idx = headers.index("left hand grasp")
        right_idx = headers.index("right hand grasp")
    except ValueError:
        return grasps

    max_idx = max(name_idx, left_idx, right_idx)
    for row in rows[1:]:
        if len(row) <= max_idx:
            continue
        task = to_lower_snake(row[name_idx])
        if not task:
            continue
        # First occurrence wins (mirrors load_mid_train_library dedup by task).
        grasps.setdefault(task, (row[left_idx].strip(), row[right_idx].strip()))
    return grasps


MID_TRAIN_GRASPS: dict[str, tuple[str, str]] = load_mid_train_grasps()


def lookup_mid_train_grasps(
    task_name: str,
    grasps: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Return ``(left_grasp_cell, right_grasp_cell)`` for *task_name*, or None."""
    g = grasps if grasps is not None else MID_TRAIN_GRASPS
    if not task_name:
        return None
    return g.get(to_lower_snake(task_name))
