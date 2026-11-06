# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live status panel, redrawn in place.

A scrolling log makes you reconstruct the current state from the last few
lines; a panel just shows it. Everything here is a snapshot of *now* --
timings, per-camera settings, and the last thing a button did.

Falls back to one line per period when stderr is not a terminal, because
redrawing in a systemd journal or a piped log produces nothing but escape
codes.
"""

from __future__ import annotations

import shutil
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

_ACCENT = "\033[38;5;148m"  # NVIDIA-ish green
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Units live in the header so the rows stay numbers.
_COLUMNS = (
    ("camera", 12),
    ("shape", 15),  # equirect carries its heading here
    ("lock", 8),
    ("eyes", 7),
    ("size m", 8),
    ("height m", 10),
    ("planes cm", 12),
    ("submit/s", 9),
)


@dataclass
class CameraRow:
    """One camera's line in the panel."""

    name: str
    shape: str
    lock_mode: str
    stereo: bool
    size_m: Optional[float]
    offset_y_m: Optional[float]
    plane_distance_cm: Optional[float]
    suggested_cm: Optional[float]
    submit_fps: float

    def cells(self) -> List[str]:
        def opt(value, fmt):
            return format(value, fmt) if value is not None else "-"

        planes = opt(self.plane_distance_cm, ".1f")
        if self.plane_distance_cm is not None and self.suggested_cm is not None:
            planes = f"{planes}/{self.suggested_cm:.1f}"
        return [
            self.name,
            self.shape,
            self.lock_mode,
            "stereo" if self.stereo else "mono",
            opt(self.size_m, ".2f"),
            opt(self.offset_y_m, "+.2f"),
            planes,
            f"{self.submit_fps:.1f}",
        ]


@dataclass
class Snapshot:
    """Everything the panel draws."""

    header: str
    render: str
    rows: Sequence[CameraRow] = ()
    ipd_mm: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    last_event: str = ""


class Dashboard:
    """Redraws a fixed block of lines in place.

    The render thread drives :meth:`show`, but a capture thread can call
    :meth:`note` at any moment, and two interleaved writes would tear an
    escape sequence in half -- so the stream is held under a lock.
    """

    def __init__(self, stream=None, colour: Optional[bool] = None) -> None:
        self._out = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()
        self._painted = 0
        self._columns = 0
        self._closed = False
        self._live = self._out.isatty() if hasattr(self._out, "isatty") else False
        self._colour = self._live if colour is None else colour

    @property
    def live(self) -> bool:
        """True when redrawing in place; False means one line per period."""
        return self._live

    def show(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._show(snapshot)

    def _show(self, snapshot: Snapshot) -> None:
        if not self._live:
            self._out.write(self._one_line(snapshot) + "\n")
            self._out.flush()
            return
        size = shutil.get_terminal_size((100, 24))
        if size.columns != self._columns:
            # Rewrapped underneath us: the old panel's height no longer says
            # how far up its top is. Start a fresh one below the wreckage
            # rather than erase whatever is there now.
            self._columns, self._painted = size.columns, 0
        lines = self._compose(snapshot)
        # One row per line is the whole premise of the cursor arithmetic, so
        # a panel taller than the terminal cannot be redrawn in place: the
        # top scrolls off and every repaint leaves a copy behind.
        if len(lines) >= size.lines:
            self._painted = 0
            self._out.write(self._one_line(snapshot) + "\n")
            self._out.flush()
            return
        if self._painted:
            self._out.write(f"\033[{self._painted}A")
        for line in lines:
            self._out.write("\033[2K" + line + "\n")
        # A shorter panel than last time would leave stale rows below.
        for _ in range(max(0, self._painted - len(lines))):
            self._out.write("\033[2K\n")
        self._out.flush()
        self._painted = max(len(lines), self._painted)

    def note(self, text: str) -> None:
        """Print a line above the panel, keeping the panel intact.

        The panel owns the cursor while it is live, so anything else writing
        to the same stream lands inside it and every later repaint is off by
        however many lines that write scrolled -- which is how you end up
        with a column of half-erased headers. Route lifecycle messages here
        and they scroll away above the panel like an ordinary log.
        """
        with self._lock:
            self._note(text)

    def _note(self, text: str) -> None:
        if self._closed:
            # A source thread can outlive runner.stop() and notify into a
            # panel that has already handed the cursor back.
            return
        if self._live and self._painted:
            self._out.write(f"\033[{self._painted}A")
            for _ in range(self._painted):
                self._out.write("\033[2K\n")
            self._out.write(f"\033[{self._painted}A")
            self._painted = 0
        self._out.write(text + "\n")
        self._out.flush()

    def close(self) -> None:
        """Leave the cursor below the panel so a later print doesn't land in
        the middle of it. Idempotent, and silences later notes."""
        with self._lock:
            self._closed = True
            if self._live and self._painted:
                self._out.write("\n")
                self._out.flush()
                self._painted = 0

    # ── composition ──────────────────────────────────────────────────

    def _paint(self, text: str, width: int, style: str = "") -> str:
        """Truncate first, then colour. Doing it the other way round can cut a
        line mid-escape and leave the terminal stuck in that style."""
        text = text[:width]
        return f"{style}{text}{_RESET}" if style and self._colour else text

    def _width(self) -> int:
        """Widest line we will draw.

        One short of the terminal: a line that fills the last column leaves
        the cursor wrap-pending, and the newline after it costs two rows on
        some terminals instead of one -- which slides the panel down a row
        per repaint and strands a copy of the header each time.
        """
        columns = shutil.get_terminal_size((100, 24)).columns
        return max(20, min(columns - 1, 100))

    def _compose(self, snapshot: Snapshot) -> List[str]:
        width = self._width()
        lines = [
            self._paint(f"camera_viz  {snapshot.header}", width, _BOLD),
            self._paint("─" * width, width, _DIM),
            self._paint(f"  {snapshot.render}", width),
            "",
            self._paint("  " + _header_row(), width, _DIM),
        ]
        lines += [
            self._paint("  " + _format_row(r.cells()), width) for r in snapshot.rows
        ]

        if snapshot.ipd_mm is not None:
            lines += ["", self._paint(f"  headset IPD {snapshot.ipd_mm:.0f} mm", width)]
        lines += [self._paint(f"  {note}", width, _DIM) for note in snapshot.notes]
        if snapshot.last_event:
            lines += ["", self._paint(f"  {snapshot.last_event}", width, _ACCENT)]
        return lines

    def _one_line(self, snapshot: Snapshot) -> str:
        """Log-friendly form: the same numbers, one line, no escape codes."""
        cameras = " | ".join(
            f"{r.name} {r.submit_fps:.1f} submit/s" for r in snapshot.rows
        )
        return f"camera_viz: stats: {snapshot.render} | {cameras}"


def _header_row() -> str:
    return "".join(name.ljust(width) for name, width in _COLUMNS)


def _format_row(cells: Sequence[str]) -> str:
    return "".join(
        str(cell)[: width - 1].ljust(width) for cell, (_, width) in zip(cells, _COLUMNS)
    )
