"""
Canonical studio folder names for local disk.

Use :func:`sanitize_studio_dir_name` for a single directory segment (theme / studio
bucket).

Rules: no ``&``, ``#``, ``?``, ``%``, slashes, or shell-hostile whitespace; collapse
runs of underscores.

The reserved routing bucket ``_unmatched`` is passed through unchanged.
"""

from __future__ import annotations

import re
from typing import AbstractSet, Optional

_UNMATCHED_BUCKET = "_unmatched"


def sanitize_studio_dir_name(name: str) -> str:
    """Return a single path segment safe for POSIX paths and ``s3://`` tools.

    Empty input returns ``\"\"``. A segment that sanitizes to nothing becomes
    ``\"unknown\"``.
    """
    s = (name or "").strip()
    if not s:
        return ""
    if s == _UNMATCHED_BUCKET:
        return s
    s = re.sub(r"[&#?%]", "_", s)
    s = re.sub(r"[/\\\x00]+", "_", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s if s else "unknown"


def canonicalize_studio_dir_name(
    name: str,
    known_studios: Optional[AbstractSet[str]],
) -> str:
    """Map *name* to a known studio spelling (case-insensitive), else sanitized *name*.

    After :func:`sanitize_studio_dir_name`, if ``known_studios`` contains a string
    whose :meth:`str.casefold` matches *name*, return that exact known string so
    ``Dining_room`` and ``Dining_Room`` both resolve to the same folder as the
    task library key (e.g. ``Dining_Room``).

    ``_unmatched`` and empty strings are returned unchanged (no lookup).
    """
    s = sanitize_studio_dir_name(name)
    if not s or s == _UNMATCHED_BUCKET or not known_studios:
        return s
    for k in known_studios:
        if k.casefold() == s.casefold():
            return k
    return s
