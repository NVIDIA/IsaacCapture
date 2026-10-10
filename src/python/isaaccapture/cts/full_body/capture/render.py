# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sidebar's HTML, as strings. Colours assume viser's light theme."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from math import ceil

from steps import Phase, Take

_BLOCK = "margin:0 0 9px"
_FAINT = "font-size:12px;color:#5a6270"


def _key(name: str) -> str:
    return (
        f'<span style="display:inline-block;padding:1px 6px;border:1px solid #b6bcc5;'
        f"border-bottom-width:2px;border-radius:4px;background:#f7f8fa;"
        f'font-family:monospace;font-size:11px">{escape(name)}</span>'
    )


def _row(ok: bool, label: str, value: str) -> str:
    mark, colour = ("\u2713", "#1f7a44") if ok else ("\u2717", "#b0342c")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;font-size:12.5px">'
        f'<span style="width:14px;text-align:center;font-weight:700;color:{colour}">'
        f"{mark}</span>"
        f'<span style="flex:1">{escape(label)}</span>'
        f'<span style="color:#8a929c;font-family:monospace;font-size:11px">'
        f"{escape(value)}</span></div>"
    )


def ready_to_start(attached: bool, runtime: str, destination: str) -> str:
    """The idle sidebar: what is attached, and where the take will land."""
    rows = _row(attached, "CloudXR runtime", runtime) + _row(
        True, "writes to", destination
    )
    return (
        f'<div style="{_BLOCK};font-size:11px;color:#6b727c;letter-spacing:.06em;'
        f'text-transform:uppercase;font-weight:600">idle</div>'
        f'<div style="{_BLOCK}">{rows}</div>'
        f'<div style="{_BLOCK};{_FAINT}">The file is created when you press start, and '
        f"the first cue is spoken once a frame with valid joints has arrived.</div>"
    )


def session_failed(reason: object) -> str:
    """Why the take did not start, on the page the performer is looking at."""
    return (
        f'<div style="{_BLOCK};background:#fcebeb;border-radius:5px;padding:9px 11px;'
        f'color:#501313;font-size:12.5px">the session did not open: '
        f"{escape(str(reason))}</div>"
    )


def recording(name: str, elapsed_s: float) -> str:
    """The red light, the wall clock, and which file is growing."""
    minutes, seconds = divmod(int(elapsed_s), 60)
    return (
        f'<div style="{_BLOCK};font-size:12px;display:flex;align-items:center;gap:7px">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:#c0322b">'
        f"</span>"
        f'<span style="color:#b0342c;font-weight:600">recording</span>'
        f'<span style="color:#5a6270;font-family:monospace">'
        f"{minutes:02d}:{seconds:02d}</span></div>"
        f'<div style="{_BLOCK};font-size:11px;color:#8a929c;font-family:monospace;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
        f"{escape(name)}</div>"
    )


def step_block(take: Take, now_s: float) -> str:
    """The block whose wording follows the phase.

    ``HOLDING`` and ``CLOSING`` render the same; a swallowed press must change nothing here.
    """
    if take.phase is Phase.IDLE:
        return _no_frames_yet()
    if take.phase is Phase.DONE:
        return _finished()
    if take.phase is Phase.CUEING:
        return _listen(take)
    if take.phase is Phase.WAITING:
        return _ready(take)
    return _holding(take, now_s)


def _heading(take: Take, tail: str) -> str:
    return (
        f"step {take.index + 1} of {take.count} &middot; "
        f"{escape(take.label)} &middot; {tail}"
    )


def _cue_line(text: str, colour: str) -> str:
    return (
        f'<div style="font-size:19px;font-weight:600;color:{colour};line-height:1.2;'
        f'margin-top:3px">{escape(text)}</div>'
    )


def _listen(take: Take) -> str:
    return (
        f'<div style="{_BLOCK};background:#fff4e0;border-radius:5px;padding:9px 11px">'
        f'<div style="font-size:11px;color:#854f0b;letter-spacing:.04em">'
        f"{_heading(take, 'listen')}</div>"
        f"{_cue_line(take.cue_text, '#412402')}</div>"
    )


def _ready(take: Take) -> str:
    return (
        f'<div style="{_BLOCK};background:#fff4e0;border-radius:5px;padding:9px 11px">'
        f'<div style="font-size:11px;color:#854f0b;letter-spacing:.04em">'
        f"{_heading(take, 'ready when you are')}</div>"
        f"{_cue_line(take.cue_text, '#412402')}"
        f'<div style="font-size:12.5px;color:#633806;margin-top:7px">get into '
        f"position, then press {_key('trigger')} or {_key('space')}</div></div>"
    )


def _holding(take: Take, now_s: float) -> str:
    """Seconds left in this pose."""
    remaining = take.remaining_s(now_s) or 0.0
    filled = 100.0 * take.elapsed_fraction(now_s)
    return (
        f'<div style="{_BLOCK};background:#e6f1fb;border-radius:5px;padding:9px 11px">'
        f'<div style="font-size:11px;color:#185fa5;letter-spacing:.04em">'
        f"{_heading(take, 'hold')}</div>"
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:3px">'
        f'<div style="font-size:19px;font-weight:600;color:#042c53;line-height:1.2;'
        f'flex:1">{escape(take.cue_text)}</div>'
        f'<div style="font-size:46px;font-weight:600;color:#042c53;line-height:1">'
        f"{ceil(remaining):d}</div></div>"
        f'<div style="height:4px;background:#b5d4f4;border-radius:2px;margin-top:9px;'
        f'overflow:hidden"><div style="height:100%;background:#185fa5;'
        f'width:{filled:.0f}%"></div></div></div>'
    )


def _no_frames_yet() -> str:
    return (
        f'<div style="{_BLOCK};background:#f0f3f7;border-radius:5px;padding:11px;'
        f'text-align:center;color:#5a6270;font-size:13px">'
        f"waiting for a frame with valid joints</div>"
    )


def _finished() -> str:
    return (
        f'<div style="{_BLOCK};background:#f0f3f7;border-radius:5px;padding:11px;'
        f'text-align:center;color:#5a6270;font-size:13px">'
        f"script done &mdash; closing the file and writing the labels</div>"
    )


def next_up(take: Take) -> str:
    text = take.next_cue_text
    if text is None or take.phase in (Phase.IDLE, Phase.DONE):
        return ""
    return (
        f'<div style="{_BLOCK};font-size:12.5px;color:#5a6270">next '
        f'<b style="color:#1c1f24;font-weight:600">{escape(text)}</b></div>'
    )


def joints(valid: int, total: int, held: Sequence[str], ever_valid: bool) -> str:
    """The valid-joint count, and the names of whatever is missing."""
    short = valid < total
    background = "#fcebeb" if short else "#f0f3f7"
    colour = "#501313" if short else "#2b2b2b"
    if not ever_valid:
        detail = "no joint has been valid yet &mdash; the take starts at the first valid frame"
    else:
        detail = escape(", ".join(held))
    names = (
        f'<div style="font-size:11px;color:#791f1f;font-family:monospace">'
        f"{detail}</div>"
        if detail
        else ""
    )
    return (
        f'<div style="{_BLOCK};background:{background};border-radius:5px;'
        f'padding:7px 11px">'
        f'<div style="font-size:15px;font-weight:600;color:{colour}">'
        f"{valid} / {total} joints valid</div>{names}</div>"
    )


def no_data(seconds: float, vendor: str | None) -> str:
    """Causes to work through when no joint has ever been valid; the list depends on ``vendor``."""
    if vendor is None:
        causes = (
            "The headset is not streaming, or the session dropped.",
            "The browser has no WebXR body tracking. Use the headset's own browser; "
            "on a consumer PICO 4 Ultra it needs no enterprise activation.",
            "The performer is out of the tracked area.",
        )
    else:
        causes = (
            f"The {vendor} plugin is not running, or it exited. Check its terminal.",
            "The plugin cannot reach the device's data server.",
            "The plugin is pushing to a different collection_id than --vendor-param "
            "asked for. A mismatch looks exactly like this.",
            "The suit or gloves are off, unpaired, or uncalibrated.",
        )
    items = "".join(
        f'<li style="margin:0 0 4px">{escape(cause)}</li>' for cause in causes
    )
    return (
        f'<div style="background:#fcebeb;border:1px solid #e0a9a4;border-radius:5px;'
        f'padding:9px 12px;color:#501313">'
        f'<div style="font-size:14px;font-weight:700;margin:0 0 6px">'
        f"No body data after {seconds:.0f} s</div>"
        f'<div style="font-size:12px;margin:0 0 6px">The take is still running and '
        f"nothing has been lost. It starts at the first valid frame.</div>"
        f'<ul style="font-size:12px;margin:0;padding-left:16px">{items}</ul></div>'
    )


def _names(labels: Sequence[str]) -> str:
    # Joined after escaping, so the separator's own entity survives.
    return " &middot; ".join(escape(label) for label in labels)


def step_list(take: Take) -> str:
    """Every step at once: what is behind, what is open, what is left."""
    done, todo = take.done_labels, take.todo_labels
    lines = []
    if done:
        lines.append(f'<div style="color:#9aa2ac">&#10003; {_names(done)}</div>')
    if take.phase is not Phase.DONE:
        lines.append(
            f'<div style="color:#185fa5;font-weight:600">&#9656; '
            f"{escape(take.label)}</div>"
        )
    if todo:
        lines.append(f'<div style="color:#6b727c">&hellip; {_names(todo)}</div>')
    return (
        f'<div style="{_BLOCK};border-top:1px solid #e3e6ea;padding-top:7px;'
        f'font-size:11px;font-family:monospace;line-height:1.5">'
        f"{''.join(lines)}</div>"
    )


def wrote_labels(name: str, checks: Sequence[tuple[str, bool, str]]) -> str:
    """What ``make_labels`` produced, shown once the session has closed."""
    rows = "".join(
        f'<div style="display:flex;gap:8px;font-size:11px;font-family:monospace">'
        f'<span style="color:{"#1f7a44" if ok else "#b0342c"};font-weight:700">'
        f"{'ok' if ok else 'BAD'}</span>"
        f'<span style="flex:1">{escape(label)}</span>'
        f'<span style="color:#8a929c">{escape(detail)}</span></div>'
        for label, ok, detail in checks
    )
    return (
        f'<div style="{_BLOCK};font-size:11px;color:#6b727c;letter-spacing:.06em;'
        f'text-transform:uppercase;font-weight:600">labels</div>'
        f'<div style="{_BLOCK};font-size:11px;font-family:monospace">'
        f"{escape(name)}</div>"
        f'<div style="{_BLOCK}">{rows}</div>'
    )
