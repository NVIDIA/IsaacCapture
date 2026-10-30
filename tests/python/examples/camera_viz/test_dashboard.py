# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Status panel: layout, redraw bookkeeping, and the non-terminal fallback."""

from __future__ import annotations

import io
import os
import re

from isaaccapture_examples.camera_viz.dashboard import CameraRow, Dashboard, Snapshot

ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _snapshot(rows=None, **kwargs):
    defaults = dict(
        header="xr · local · 1 camera",
        render="render 58.0 fps   missed 0",
        rows=rows
        if rows is not None
        else [CameraRow("zed", "cylinder", "lazy", True, 1.0, 0.0, 5.0, 5.2, 64.0)],
    )
    defaults.update(kwargs)
    return Snapshot(**defaults)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_not_a_terminal_falls_back_to_one_line():
    """Redrawing into a journal or a pipe produces nothing but escape codes."""
    out = io.StringIO()  # StringIO.isatty() is False
    dash = Dashboard(stream=out)
    assert not dash.live
    dash.show(_snapshot())
    text = out.getvalue()
    assert text.count("\n") == 1
    assert "\x1b[" not in text
    assert "zed 64.0 submit/s" in text


def test_live_panel_redraws_in_place():
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    assert dash.live
    dash.show(_snapshot())
    first = out.getvalue()
    assert "\x1b[2K" in first  # clears each line
    assert "\x1b[1A" not in first  # nothing to move up over yet

    out.truncate(0), out.seek(0)
    dash.show(_snapshot())
    # Second paint walks back up over the first.
    assert re.search(r"\x1b\[\d+A", out.getvalue())


def test_a_shorter_panel_clears_the_rows_it_left_behind():
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    two = [
        CameraRow("a", "quad", "lazy", True, 1.0, 0.0, 5.0, 5.2, 60.0),
        CameraRow("b", "quad", "lazy", True, 1.0, 0.0, 5.0, 5.2, 60.0),
    ]
    dash.show(_snapshot(rows=two))
    tall = dash._painted

    out.truncate(0), out.seek(0)
    dash.show(_snapshot(rows=two[:1]))
    # Same number of lines written, so the vacated row is blanked not stale.
    assert out.getvalue().count("\x1b[2K") == tall


def test_lines_fit_the_width_and_never_cut_an_escape():
    """Truncating after colouring would clip the reset and leave the terminal
    stuck in that style."""
    dash = Dashboard(stream=FakeTTY(), colour=True)
    snap = _snapshot(notes=["x" * 300], last_event="y" * 300, ipd_mm=63.0)
    for line in dash._compose(snap):
        assert len(ESCAPE.sub("", line)) <= 100
        if "\x1b[" in line:
            assert line.endswith("\x1b[0m")


def test_row_shows_the_suggestion_beside_the_value():
    dash = Dashboard(stream=FakeTTY(), colour=False)
    body = "\n".join(dash._compose(_snapshot()))
    assert "5.0/5.2" in body


def test_row_renders_missing_values_as_a_dash():
    """equirect has no plane gap and window mode has no lock mode."""
    row = CameraRow("sky", "equirect", "-", True, None, None, None, None, 30.0)
    cells = row.cells()
    assert cells[4] == "-" and cells[6] == "-"


def test_close_leaves_the_cursor_below_the_panel():
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    out.truncate(0), out.seek(0)
    dash.close()
    assert out.getvalue() == "\n"
    assert dash._painted == 0


# ── Staying aligned with the terminal ─────────────────────────────────
#
# Every one of these is the same failure seen from a different angle: one
# logical line stopped costing exactly one terminal row, the cursor
# arithmetic drifted, and each repaint stranded another copy of the header.


def test_lines_stop_one_short_of_the_last_column(monkeypatch):
    """A line that fills the last column leaves the cursor wrap-pending, and
    the newline after it costs two rows on some terminals."""
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((64, 24))
    )
    dash = Dashboard(stream=FakeTTY(), colour=False)
    lines = dash._compose(_snapshot())
    assert max(len(line) for line in lines) == 63


def test_a_panel_taller_than_the_terminal_gives_up_on_redrawing(monkeypatch):
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((100, 5))
    )
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    text = out.getvalue()
    assert text.count("\n") == 1
    assert "\x1b[" not in text
    assert dash._painted == 0


def test_a_resize_starts_a_fresh_panel(monkeypatch):
    """After a rewrap the old panel's height no longer says how far up its
    top is, so erasing that many lines would eat unrelated output."""
    columns = [100]
    monkeypatch.setattr(
        "shutil.get_terminal_size",
        lambda fallback=None: os.terminal_size((columns[0], 40)),
    )
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    columns[0] = 70
    out.truncate(0), out.seek(0)
    dash.show(_snapshot())
    assert "\x1b[" in out.getvalue()  # still painting
    assert not re.search(r"\x1b\[\d+A", out.getvalue())  # but not upwards


def test_a_note_scrolls_above_the_panel_instead_of_into_it():
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    painted = dash._painted
    out.truncate(0), out.seek(0)

    dash.note("[zed] reconnected")
    text = out.getvalue()
    # Panel erased, cursor back at its top, note written there: the next
    # show() paints below it rather than half over it.
    assert text.startswith(f"\x1b[{painted}A")
    assert text.endswith("[zed] reconnected\n")
    assert dash._painted == 0


def test_a_note_without_a_panel_is_just_a_line():
    out = io.StringIO()
    dash = Dashboard(stream=out)
    dash.note("[zed] reconnected")
    assert out.getvalue() == "[zed] reconnected\n"


def test_a_note_after_close_is_dropped():
    """Source threads can outlive runner.stop() and notify into a panel that
    has already handed the cursor back."""
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    dash.close()
    out.truncate(0), out.seek(0)

    dash.note("[zed] stopped")
    assert out.getvalue() == ""


def test_close_is_idempotent():
    out = FakeTTY()
    dash = Dashboard(stream=out, colour=False)
    dash.show(_snapshot())
    dash.close()
    out.truncate(0), out.seek(0)
    dash.close()
    assert out.getvalue() == ""
