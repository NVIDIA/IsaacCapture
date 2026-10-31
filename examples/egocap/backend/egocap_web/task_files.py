# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only access to the task CSV catalogs in ``data/taskfiles/``.

Two small functions, ``list_task_files`` and ``read_task_file``, take the catalog root explicitly and touch the
filesystem only when called. Two existing CSV layouts are supported, each by an explicit branch:

* studio catalogs: a title record, a blank record, then ``Studio, Scenario, Task, Description`` headers;
* Mid Train: headers on the first record (``Task Name``, ``Task Description`` and optional ``Primitive(s)``,
  ``Props``, ``Left Hand Grasp``, ``Right Hand Grasp``).

Only task-definition columns are read; tracking or operator columns in a file are never exposed.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 2 * 1024 * 1024

_STUDIO_HEADERS = ("studio", "scenario", "task", "description")
_MID_TRAIN_REQUIRED = ("task name", "task description")
_MID_TRAIN_OPTIONAL = ("primitive(s)", "props", "left hand grasp", "right hand grasp")


class TaskFileError(Exception):
    """Base for every error this module raises on purpose."""


class InvalidTaskFileName(TaskFileError):
    """The requested name is not a plain ``*.csv`` filename."""


class TaskFileNotFound(TaskFileError):
    """The name is well formed but no such catalog is available."""


class TaskFileTooLarge(TaskFileError):
    """The catalog on the server exceeds ``MAX_FILE_BYTES``."""


class MalformedTaskFile(TaskFileError):
    """The catalog on the server is not a supported layout or has an incomplete task record."""


class TaskFilesUnavailable(TaskFileError):
    """The catalog root, or a listed catalog, cannot be read (a deployment problem, not a missing file)."""


@dataclass(frozen=True)
class TaskDefinition:
    """One task as the catalog defines it; fields a layout lacks stay None."""

    name: str
    instruction: str
    studio: str | None = None
    scenario: str | None = None
    primitives: tuple[str, ...] | None = None
    props: str | None = None
    left_grasp: str | None = None
    right_grasp: str | None = None

    def to_json(self) -> dict:
        """Only the fields this layout has; ``name`` and ``instruction`` are always present."""
        out: dict = {"name": self.name, "instruction": self.instruction}
        for key in ("studio", "scenario", "props", "left_grasp", "right_grasp"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.primitives is not None:
            out["primitives"] = list(self.primitives)
        return out


@dataclass(frozen=True)
class TaskFileContents:
    """A parsed catalog: its file name, layout name and task definitions in file order."""

    name: str
    format: str
    tasks: tuple[TaskDefinition, ...]

    def to_json(self) -> dict:
        """The HTTP response body for this catalog."""
        return {"name": self.name, "format": self.format, "tasks": [t.to_json() for t in self.tasks]}


# Files that live beside the catalogs in the native task library but are not catalogs.
_NOT_CATALOGS = frozenset({"Summary.csv"})


def _is_plain_csv_name(name: str) -> bool:
    """A plain ``*.csv`` filename: no separators, NUL or ``..``, and representable as UTF-8 (the JSON listing)."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a file name that is not valid UTF-8 on disk
        return False
    return (
        len(name) > len(".csv")
        and name.endswith(".csv")
        and not any(c in name for c in ("/", "\\", "\0"))
        and ".." not in name
    )


def _is_catalog_file(path: Path) -> bool:
    """The one eligibility rule for listing and reading: a regular, non-symlink catalog file.

    Hidden files, editor leftovers (``._x.csv``, ``~$x.csv``) and the library's summary file are not catalogs.
    """
    name = path.name
    if not _is_plain_csv_name(name) or name.startswith((".", "~")) or name in _NOT_CATALOGS:
        return False
    return path.is_file() and not path.is_symlink()


def list_task_files(root: Path) -> list[str]:
    """Sorted names of the top-level regular ``*.csv`` files in ``root`` (symlinks and subdirectories skipped)."""
    try:
        return sorted(p.name for p in root.iterdir() if _is_catalog_file(p))
    except OSError as exc:
        logger.error("task file root %s is unavailable: %s", root, exc)
        raise TaskFilesUnavailable("task file storage is unavailable") from exc


def read_task_file(root: Path, name: str) -> TaskFileContents:
    """Parse one catalog into task definitions, in file order."""
    # A malformed name is the caller's mistake (400); a well-formed one that is not a listed catalog is absent.
    if not _is_plain_csv_name(name):
        raise InvalidTaskFileName("a task file name is a plain filename ending in .csv")
    if name not in list_task_files(root):  # the same rules as listing decide what exists
        raise TaskFileNotFound(f"no task file named {name}")
    path = root / name
    # Read at most one byte past the limit, so the bound applies to the bytes actually read, not a prior stat.
    try:
        inside_root = path.resolve().parent == root.resolve()  # belt and braces: the file is directly in the root
        if inside_root:
            with path.open("rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError as exc:  # deleted between listing and reading
        raise TaskFileNotFound(f"no task file named {name}") from exc
    except OSError as exc:
        logger.error("task file %s is unreadable: %s", path, exc)
        raise TaskFilesUnavailable("task file storage is unavailable") from exc
    if not inside_root:
        raise TaskFileNotFound(f"no task file named {name}")
    if len(raw) > MAX_FILE_BYTES:
        raise TaskFileTooLarge(f"{name} is larger than {MAX_FILE_BYTES} bytes")
    # Decode (a leading BOM is allowed), then parse with strict CSV: quoted newlines work, bad quoting is refused.
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MalformedTaskFile(f"{name} is not valid UTF-8") from exc
    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise MalformedTaskFile(f"{name} is not valid CSV: {exc}") from exc
    return _parse(name, records)


def _headers(record: list[str]) -> list[str]:
    """Header cells trimmed and lowercased for case-insensitive matching."""
    return [cell.strip().lower() for cell in record]


def _column_indexes(name: str, headers: list[str], wanted: tuple[str, ...], *, required: bool) -> dict[str, int]:
    """Column positions for ``wanted`` headers; a duplicate is an error, a missing one is if ``required``."""
    found: dict[str, int] = {}
    for column in wanted:
        count = headers.count(column)
        if count > 1:
            raise MalformedTaskFile(f"{name} has more than one {column!r} column")
        if count == 1:
            found[column] = headers.index(column)
        elif required:
            raise MalformedTaskFile(f"{name} has no {column!r} column")
    return found


def _parse(name: str, records: list[list[str]]) -> TaskFileContents:
    """Pick the layout from the required header names (never from the filename) and parse it."""
    if len(records) >= 3 and set(_STUDIO_HEADERS) <= set(_headers(records[2])):
        return _parse_studio(name, records[3:], _headers(records[2]), first_record=4)
    if records and set(_MID_TRAIN_REQUIRED) <= set(_headers(records[0])):
        return _parse_mid_train(name, records[1:], _headers(records[0]), first_record=2)
    raise MalformedTaskFile(f"{name} is neither a studio catalog nor a Mid Train file")


def _cell(name: str, record: list[str], index: int, number: int) -> str:
    """The trimmed cell at ``index``; a record too short to have it is incomplete."""
    if index >= len(record):
        raise MalformedTaskFile(f"{name} record {number} is incomplete")
    return record[index].strip()


def _required_cell(name: str, record: list[str], index: int, number: int, what: str) -> str:
    """A cell that must be non-empty."""
    value = _cell(name, record, index, number)
    if not value:
        raise MalformedTaskFile(f"{name} record {number} has an empty {what}")
    return value


def _parse_studio(name: str, records: list[list[str]], headers: list[str], first_record: int) -> TaskFileContents:
    """Studio catalog records after the header: every non-blank record must be a complete task."""
    at = _column_indexes(name, headers, _STUDIO_HEADERS, required=True)
    tasks = []
    for number, record in enumerate(records, start=first_record):
        if not any(cell.strip() for cell in record):  # blank records are ignored
            continue
        # Every other record must be a complete task; a partial one refuses the whole catalog.
        tasks.append(
            TaskDefinition(
                name=_required_cell(name, record, at["task"], number, "task name"),
                instruction=_required_cell(name, record, at["description"], number, "description"),
                studio=_cell(name, record, at["studio"], number),
                scenario=_cell(name, record, at["scenario"], number),
            )
        )
    return TaskFileContents(name, "studio_csv", tuple(tasks))


def _lower_snake(text: str) -> str:
    """The native Mid Train task-name convention: lowercase, runs of non-alphanumerics become one underscore."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _parse_mid_train(
    name: str, records: list[list[str]], headers: list[str], first_record: int
) -> TaskFileContents:
    """Mid Train records after the header, with the native task-name and primitive conventions."""
    at = _column_indexes(name, headers, _MID_TRAIN_REQUIRED, required=True)
    at.update(_column_indexes(name, headers, _MID_TRAIN_OPTIONAL, required=False))
    tasks = []
    for number, record in enumerate(records, start=first_record):
        if not any(cell.strip() for cell in record):  # blank records are ignored
            continue
        # The task name follows the native lower_snake convention (pinned to the recorder's by a test).
        task_name = _lower_snake(_required_cell(name, record, at["task name"], number, "task name"))
        if not task_name:
            raise MalformedTaskFile(f"{name} record {number} has an empty task name")

        def optional(column: str) -> str | None:
            """The trimmed cell of an optional column, or None when the file has no such column."""
            return _cell(name, record, at[column], number) if column in at else None

        primitives = optional("primitive(s)")
        tasks.append(
            TaskDefinition(
                name=task_name,
                instruction=_required_cell(name, record, at["task description"], number, "description"),
                primitives=None
                if primitives is None
                else tuple(p.strip() for p in primitives.split("+") if p.strip()),
                props=optional("props"),
                left_grasp=optional("left hand grasp"),
                right_grasp=optional("right hand grasp"),
            )
        )
    return TaskFileContents(name, "mid_train_csv", tuple(tasks))
