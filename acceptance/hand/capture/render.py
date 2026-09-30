# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sidebar's HTML, as strings. No viser here.

Nothing in this file renders a pass or a fail mid-take: thresholds are not settled, so
the only status coloured is the per-hand valid-joint count.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from math import ceil

from steps import Phase, Take

_BLOCK = "margin:0 0 9px"
_FAINT = "font-size:12px;color:#5a6270"


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


def ready_to_start(attached: bool, runtime: str, destination: str, source: str) -> str:
    rows = (
        _row(attached, "CloudXR runtime", runtime)
        + _row(True, "hand source", source)
        + _row(True, "writes to", destination)
    )
    return (
        f'<div style="{_BLOCK};font-size:11px;color:#6b727c;letter-spacing:.06em;'
        f'text-transform:uppercase;font-weight:600">idle</div>'
        f'<div style="{_BLOCK}">{rows}</div>'
        f'<div style="{_BLOCK};{_FAINT}">The file is created when you press start. The '
        f"script starts once both hands have valid joints, and runs on its own: every "
        f"step is spoken, counted down, held until the second tone.</div>"
    )


def session_failed(reason: object) -> str:
    return (
        f'<div style="{_BLOCK};background:#fcebeb;border-radius:5px;padding:9px 11px;'
        f'color:#501313;font-size:12.5px">the session did not open: '
        f"{escape(str(reason))}</div>"
    )


def recording(name: str, elapsed_s: float) -> str:
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


def step_block(take: Take, now_ns: int) -> str:
    if take.phase is Phase.IDLE:
        return _plain("waiting for both hands to have valid joints")
    if take.phase is Phase.DONE:
        return _plain("script done &mdash; closing the file and writing the labels")
    if take.phase is Phase.CUEING:
        return _amber(take, "listen", "")
    if take.phase is Phase.WAITING:
        return _amber(take, "get into position", f"{take.beats_left + 1}")
    return _holding(take, now_ns)


def _heading(take: Take, tail: str) -> str:
    return (
        f"step {take.index + 1} of {take.count} &middot; "
        f"{escape(take.label)} &middot; {tail}"
    )


def _amber(take: Take, tail: str, big: str) -> str:
    return (
        f'<div style="{_BLOCK};background:#fff4e0;border-radius:5px;padding:9px 11px">'
        f'<div style="font-size:11px;color:#854f0b;letter-spacing:.04em">'
        f"{_heading(take, tail)}</div>"
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:3px">'
        f'<div style="font-size:19px;font-weight:600;color:#412402;line-height:1.2;'
        f'flex:1">{escape(take.cue_text)}</div>'
        f'<div style="font-size:46px;font-weight:600;color:#854f0b;line-height:1">'
        f"{escape(big)}</div></div></div>"
    )


def _holding(take: Take, now_ns: int) -> str:
    remaining = take.remaining_s(now_ns) or 0.0
    filled = 100.0 * (1.0 - remaining / take.duration_s) if take.duration_s else 100.0
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


def _plain(text: str) -> str:
    return (
        f'<div style="{_BLOCK};background:#f0f3f7;border-radius:5px;padding:11px;'
        f'text-align:center;color:#5a6270;font-size:13px">{text}</div>'
    )


def next_up(take: Take) -> str:
    text = take.next_cue_text
    if text is None or take.phase in (Phase.IDLE, Phase.DONE):
        return ""
    return (
        f'<div style="{_BLOCK};font-size:12.5px;color:#5a6270">next '
        f'<b style="color:#1c1f24;font-weight:600">{escape(text)}</b></div>'
    )


def hands(valid: dict[str, int | None], total: int) -> str:
    """Valid joints per hand; ``None`` is a hand the runtime reports inactive."""
    rows = []
    for side in ("left", "right"):
        count = valid.get(side)
        short = count is None or count < total
        text = "inactive" if count is None else f"{count} / {total} valid"
        colour = "#501313" if short else "#2b2b2b"
        rows.append(
            f'<div style="flex:1;background:{"#fcebeb" if short else "#f0f3f7"};'
            f'border-radius:5px;padding:7px 11px">'
            f'<div style="font-size:11px;color:#6b727c">{side}</div>'
            f'<div style="font-size:15px;font-weight:600;color:{colour}">{text}</div>'
            f"</div>"
        )
    return f'<div style="{_BLOCK};display:flex;gap:8px">{"".join(rows)}</div>'


def no_data(seconds: float, plugins: Sequence[str]) -> str:
    """Causes to work through when a hand has never had a valid joint."""
    if plugins:
        causes = (
            f"The {', '.join(plugins)} plugin is not running, or it exited.",
            "The glove is off, unpaired, uncalibrated, or its vendor service is down.",
            "The glove plugin needs a wrist from the headset or a controller and "
            "neither is tracked.",
        )
    else:
        causes = (
            "The headset is not streaming, or the session dropped.",
            "Hand tracking is off in the headset, or the hands are out of view.",
        )
    items = "".join(
        f'<li style="margin:0 0 4px">{escape(cause)}</li>' for cause in causes
    )
    return (
        f'<div style="background:#fcebeb;border:1px solid #e0a9a4;border-radius:5px;'
        f'padding:9px 12px;color:#501313">'
        f'<div style="font-size:14px;font-weight:700;margin:0 0 6px">'
        f"No hand data after {seconds:.0f} s</div>"
        f'<div style="font-size:12px;margin:0 0 6px">The take is still running. The '
        f"script starts once both hands have valid joints.</div>"
        f'<ul style="font-size:12px;margin:0;padding-left:16px">{items}</ul></div>'
    )


def step_list(take: Take) -> str:
    labels = [take_label for take_label, _, _ in take.steps]
    lines = []
    for index, label in enumerate(labels):
        if take.phase is Phase.DONE or index < take.index:
            lines.append(f'<div style="color:#9aa2ac">&#10003; {escape(label)}</div>')
        elif index == take.index:
            lines.append(
                f'<div style="color:#185fa5;font-weight:600">&#9656; '
                f"{escape(label)}</div>"
            )
        else:
            lines.append(f'<div style="color:#6b727c">&middot; {escape(label)}</div>')
    return (
        f'<div style="{_BLOCK};border-top:1px solid #e3e6ea;padding-top:7px;'
        f'font-size:11px;font-family:monospace;line-height:1.5">'
        f"{''.join(lines)}</div>"
    )


def wrote_labels(name: str, checks: Sequence[tuple[str, bool, str]]) -> str:
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
