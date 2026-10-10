# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The viser panel: both hands replayed on one playhead, beside the final results.

The only module in this package that imports viser. The result list is the final
result of every check, never a partial one.
"""

from __future__ import annotations

import time
from html import escape

import numpy as np
import viser
from viser import uplot

from cts_common.panel import render
from cts_common.panel.skeleton import HELD_JOINT_COLOUR, Skeleton
from cts_common.panel.track import Track
from cts_common.report import Mark

from ..profile import CHAINS, HAND, JOINT_NAMES, METACARPALS
from ..report import Report
from . import status
from .track import SIDES, centre

NUM_JOINTS = len(JOINT_NAMES)
RATE_WINDOW_S = 8.0
PLOT_PERIOD_S = 0.2
TICK_S = 1.0 / 60.0
SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
# Per-side colours, greys.
COLOURS = {"left": "#1f2937", "right": "#9ca3af"}
POINT_SIZE = 0.008
# One colour per digit, none of them red: red is a held (invalid) joint.
DIGIT_COLOURS: dict[str, tuple[int, int, int]] = {
    "thumb": (230, 159, 0),
    "index": (0, 114, 178),
    "middle": (0, 158, 115),
    "ring": (148, 103, 189),
    "little": (188, 189, 34),
}
PALM_COLOUR = (120, 128, 140)


def _joint_colours() -> list[tuple[int, int, int]]:
    colours = [PALM_COLOUR] * NUM_JOINTS
    for digit, chain in CHAINS.items():
        metacarpal = (METACARPALS[digit],) if digit in METACARPALS else ()
        for index in (*chain, *metacarpal):
            colours[index] = DIGIT_COLOURS[digit]
    return colours


def _legend() -> str:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'margin-right:10px"><span style="width:10px;height:10px;border-radius:50%;'
        f'background:rgb{colour}"></span>{digit}</span>'
        for digit, colour in DIGIT_COLOURS.items()
    )
    return (
        f'<div style="font-size:11px">{swatches}'
        f'<span style="display:inline-flex;align-items:center;gap:4px">'
        f'<span style="width:10px;height:10px;border-radius:50%;'
        f'background:rgb{HELD_JOINT_COLOUR}"></span>invalid</span></div>'
    )


class Panel:
    def __init__(
        self, server: viser.ViserServer, report: Report, tracks: dict[str, Track]
    ) -> None:
        self._server = server
        self._report = report
        self._tracks = tracks
        self._duration_s = max((t.duration_s for t in tracks.values()), default=0.0)
        self._playhead_s = 0.0

        server.gui.configure_theme(control_width="large", dark_mode=False)
        server.scene.set_up_direction("+y")
        look = centre(tracks) or (0.0, 1.1, -0.3)
        self._table_m = status.table_height(report)
        server.scene.add_grid(
            "/table",
            width=1.2,
            height=1.2,
            plane="xz",
            cell_size=0.05,
            position=(look[0], self._table_m or 0.0, look[2]),
        )
        server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.002)
        server.initial_camera.position = (look[0], look[1] + 0.25, look[2] + 0.55)
        server.initial_camera.look_at = look
        self._skeletons = {
            side: Skeleton(
                server,
                HAND,
                root=f"/hands/{side}",
                point_size=POINT_SIZE,
                joint_colours=_joint_colours(),
            )
            for side in tracks
        }
        self._build_gui()
        self._draw()

    def _build_gui(self) -> None:
        gui = self._server.gui
        report = self._report

        gui.add_html(render.verdict_banner(report))
        facts = [
            ("recording", report.source.rsplit("/", 1)[-1]),
            ("frames", f"{report.frames}"),
            ("duration", f"{self._duration_s:.1f} s"),
            (
                "grid",
                "stage floor: no flat window was held"
                if self._table_m is None
                else f"table at {self._table_m:.3f} m, from the flat windows",
            ),
        ]
        for side in SIDES:
            track = self._tracks.get(side)
            facts.append(
                (
                    f"{side} valid",
                    "no channel"
                    if track is None
                    else f"at worst {track.min_valid_count}/{NUM_JOINTS}",
                )
            )
        gui.add_html(render.facts(facts))
        for note in report.notes:
            gui.add_html(
                f'<div style="font-size:11px;opacity:0.7">note: {escape(note)}</div>'
            )

        with gui.add_folder("Playback"):
            self._scrub = gui.add_slider(
                "time s",
                min=0.0,
                max=max(self._duration_s, 0.01),
                step=0.01,
                initial_value=0.0,
            )
            self._playing = gui.add_checkbox("play", initial_value=False)
            self._speed = gui.add_dropdown("speed", tuple(SPEEDS), initial_value="1x")
            self._name_held = gui.add_checkbox("name held joints", initial_value=False)
            gui.add_html(_legend())
            self._readout = gui.add_html("")

            @self._scrub.on_update
            def _(_: viser.GuiEvent) -> None:
                self._playhead_s = float(self._scrub.value)
                self._draw()

        with gui.add_folder("Decides the take"):
            gui.add_html(render.result_rows(status.decisive(report)))
            self._rate_plot = self._add_rate_plot(gui)
            self._add_validity_plot(gui)

        for group in status.groups(report):
            with gui.add_folder(
                render.group_heading(group),
                expand_by_default=group.mark is not Mark.PASS,
            ):
                gui.add_html(render.group_body(group))

    def _series(self, label: str) -> tuple[uplot.Series, ...]:
        return (uplot.Series(label="s"),) + tuple(
            uplot.Series(label=f"{side} {label}", stroke=COLOURS[side], width=1.5)
            for side in self._tracks
        )

    def _add_rate_plot(self, gui: viser.GuiApi) -> viser.GuiUplotHandle:
        extents = [t.rate_extent() for t in self._tracks.values()]
        extents = [e for e in extents if e is not None] or [(0.0, 1.0)]
        low = min(e[0] for e in extents)
        high = max(e[1] for e in extents)
        pad = max(0.5, (high - low) * 0.1)
        empty = (np.zeros(1),) * (1 + len(self._tracks))
        return gui.add_uplot(
            data=empty,
            series=self._series("Hz"),
            scales={
                "x": uplot.Scale(time=False),
                "y": uplot.Scale(min=low - pad, max=high + pad),
            },
            title=f"frame rate, last {RATE_WINDOW_S:.0f} s",
            height=110,
        )

    def _add_validity_plot(self, gui: viser.GuiApi) -> None:
        times, columns = self._aligned(lambda track: track.validity_series())
        gui.add_uplot(
            data=(np.asarray(times or [0.0]),)
            + tuple(np.asarray(c or [0.0]) for c in columns),
            series=self._series("valid"),
            scales={
                "x": uplot.Scale(time=False),
                "y": uplot.Scale(min=0.0, max=float(NUM_JOINTS)),
            },
            title="valid joints, whole take",
            height=110,
        )

    def _aligned(self, series):
        """Each hand's series resampled onto the first hand's times."""
        sides = list(self._tracks)
        if not sides:
            return [], []
        base_times, _ = series(self._tracks[sides[0]])
        columns = []
        for side in sides:
            times, values = series(self._tracks[side])
            if not times:
                columns.append([0.0] * len(base_times))
                continue
            positions = np.searchsorted(times, base_times, side="right") - 1
            positions = np.clip(positions, 0, len(values) - 1)
            columns.append([values[i] for i in positions])
        return base_times, columns

    def _draw(self) -> None:
        rows = []
        step = None
        with self._server.atomic():
            for side, track in self._tracks.items():
                if not track.samples:
                    continue
                sample = track.samples[track.index_at(self._playhead_s)]
                step = step or sample.step
                self._skeletons[side].draw(sample, self._name_held.value)
                colour = "#b0342c" if sample.valid_count < NUM_JOINTS else "#12502c"
                rows.append(
                    f'<span style="color:{COLOURS[side]};font-weight:700">{side}</span> '
                    f'<span style="color:{colour};font-weight:700">'
                    f"{sample.valid_count}/{NUM_JOINTS}</span>"
                )
            self._readout.content = (
                f'<div style="font-family:monospace;font-size:12px">'
                f"{self._playhead_s:7.2f} s / {self._duration_s:.2f} s</div>"
                f'<div style="font-size:12px">joints valid &nbsp;'
                f"{' &nbsp; '.join(rows)}</div>"
                f'<div style="font-size:12px">window: '
                f"<b>{escape(step) if step else 'unlabelled'}</b></div>"
            )

    def draw_rate(self) -> None:
        times, columns = self._aligned(
            lambda track: track.rate_window(
                track.index_at(self._playhead_s), RATE_WINDOW_S
            )
        )
        if not times:
            return
        self._rate_plot.data = (np.asarray(times),) + tuple(
            np.asarray(c) for c in columns
        )

    def advance(self, elapsed_s: float) -> None:
        if not self._playing.value or self._duration_s <= 0:
            return
        self._playhead_s += elapsed_s * SPEEDS[self._speed.value]
        if self._playhead_s > self._duration_s:
            self._playhead_s = 0.0
        self._scrub.value = round(self._playhead_s, 2)
        self._draw()

    def run(self) -> None:
        last = time.monotonic()
        plotted = 0.0
        while True:
            now = time.monotonic()
            elapsed, last = now - last, now
            self.advance(elapsed)
            if now - plotted > PLOT_PERIOD_S:
                plotted = now
                self.draw_rate()
            time.sleep(TICK_S)


def serve(
    report: Report,
    tracks: dict[str, Track],
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    server = viser.ViserServer(host=host, port=port)
    panel = Panel(server, report, tracks)
    print(f"[panel] http://localhost:{server.get_port()} — Ctrl+C to stop")
    try:
        panel.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
