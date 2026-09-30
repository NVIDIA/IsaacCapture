# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records one take of the G4 motion script, paced by the performer.

viser comes up first and idles: nothing is open against the device and no file exists.
The ``TeleopSession`` -- which both writes the MCAP and feeds the skeleton -- is
constructed when the operator presses start, so the button means what it says and
neither end of the file carries anything useless. One take per process.

Usage:
    python capture_panel.py OUTPUT.mcap [--port 8081] [--accept-eula]

Run it through ``record.sh``, which names the take and records what produced it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import viser

from isaacteleop.cloudxr import CloudXRLauncher, NoopContext
from isaacteleop.deviceio import McapRecordingConfig, TrackerVendor
from isaacteleop.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

from acceptance_common.panel.sample import Sample, Vec3
from acceptance_common.panel.skeleton import Skeleton
from full_body_acceptance.frames import NUM_JOINTS, Frame
from full_body_acceptance.profile import FULL_BODY

import cues
import make_labels
import render
from live import LiveFrameSource, build_pipeline
from session import STEPS
from steps import KEYBOARD, TRIGGER, Phase, Press, Sound, Take

# The scene follows the device so motion looks right; every GUI element is a websocket
# message per change, so the text blocks run slower. A countdown updated per frame would
# be six times the traffic for no readable difference.
STEP_PERIOD_S = 1.0 / 60.0
GUI_PERIOD_S = 0.1

# How long to keep recording after the tenth window closes. The take ends on the last
# event rather than on a timer, so this is the whole of the file's tail.
TAIL_S = 2.0

# 8080 is the replay panel's, so the previous take's report can stay open while the
# next one records.
DEFAULT_PORT = 8081

# Docking insets the canvas rather than covering it, so the skeleton keeps whatever the
# panel does not take. Pixels, not a fraction of the window: tune it for the operator's
# display, or drag the panel's inner edge, which overrides this.
PANEL_WIDTH_PX = 480

# How long the take may run with no valid joint before the panel names the causes. Long
# enough to cover a plugin's cold start and its connection to the device's data server,
# which happen after the button is pressed. Nothing is stopped when it elapses.
NO_DATA_S = 10.0


class HeldJoints:
    """Last valid position per joint, which is where an invalid one is drawn.

    The checker's ``TrackBuilder`` also does this, but it stamps every sample from
    ``frame.log_time_ns`` -- a container time the recorder writes and a live frame does
    not have. Holding the positions here is smaller than handing it a clock to misread.
    """

    def __init__(self) -> None:
        self._held: list[Vec3 | None] = [None] * NUM_JOINTS
        self.ever_valid = False

    def sample(self, frame: Frame) -> Sample:
        """One frame as the ``Sample`` the skeleton draws.

        ``t_s`` and ``interval_ms`` describe a recording's timing and a live frame has
        none, so they carry nothing: the skeleton reads ``positions`` and ``valid``.
        """
        valid = [False] * NUM_JOINTS
        if frame.joints is not None:
            for index, joint in enumerate(frame.joints):
                if joint.is_valid:
                    valid[index] = True
                    self._held[index] = joint.position
        self.ever_valid = self.ever_valid or any(valid)
        return Sample(
            sequence=frame.sequence,
            t_s=0.0,
            positions=tuple(self._held),
            valid=tuple(valid),
            interval_ms=None,
            step=None,
        )


class CapturePanel:
    def __init__(
        self,
        server: viser.ViserServer,
        launcher: CloudXRLauncher | NoopContext,
        recording: Path,
        vendor: TrackerVendor | None = None,
        plugins: tuple[PluginConfig, ...] = (),
    ) -> None:
        self._server = server
        self._launcher = launcher
        self._recording = recording
        self._vendor = vendor
        self._plugins = plugins
        self._spoken = {label: cues.wav(label, text) for label, _, text in STEPS}
        self._cue_process: subprocess.Popen | None = None
        self._armed = threading.Event()
        self._key = threading.Event()
        self._sample: Sample | None = None
        self._listed: tuple[int, bool] | None = None

        # The mark colours in `render` are chosen against a light background.
        server.gui.configure_theme(control_width="large", dark_mode=False)
        # `set_width` is the docked region's width, so it has to follow the dock; the
        # theme's `control_width` is what a panel dragged back out to float reverts to.
        server.gui.main_panel.dock_right()
        server.gui.main_panel.set_width(PANEL_WIDTH_PX)
        server.scene.set_up_direction("+y")
        server.scene.add_grid(
            "/floor", width=3.0, height=3.0, plane="xz", cell_size=0.25
        )
        server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.004)
        server.initial_camera.position = (2.2, 1.6, 2.6)
        server.initial_camera.look_at = (0.0, 0.9, 0.0)
        self._skeleton = Skeleton(server, FULL_BODY)
        self._build_gui()
        self._build_warning()

    def _build_warning(self) -> None:
        """The no-data causes, parked at the canvas's bottom-left until they apply.

        Floated rather than docked so it costs the skeleton no width, and floating
        costs no visibility either: it is only ever shown when no skeleton is being
        drawn. ``float`` measures from the canvas, so it cannot land under the
        right-docked control panel.
        """
        panel = self._server.gui.add_panel(visible=False)
        panel.float(x=20, y=-20, width=440)
        with panel.add_tab("No data"):
            self._warning_text = self._server.gui.add_html("")
        self._warning = panel
        self._warned = False

    def _build_gui(self) -> None:
        gui = self._server.gui
        attached, detail = self._runtime()
        self._idle = gui.add_html(
            render.ready_to_start(attached, detail, self._destination())
        )
        self._start = gui.add_button("Start recording")
        self._status = gui.add_html("")
        self._steps = gui.add_html("")
        self._report = gui.add_html("")

        @self._start.on_click
        def _(_: viser.GuiEvent) -> None:
            # viser dispatches callbacks on a thread pool. The take runs on the thread
            # that owns the session, so this only raises a flag.
            self._armed.set()

        # The keyboard is not a fallback: two people is the common case and the one at
        # the screen does the pressing, and a mocap suit has no controllers at all, so
        # for a whole class of device this is the only input. viser 1.1 exposes no key
        # events, so the command palette's hotkey is the binding.
        #
        # Deliberately never disabled. `Take.press` is the single place a press is
        # accepted or swallowed, and a control that greys itself out is a visible
        # reaction to a press that did not count.
        self._command = gui.add_command(
            "Step ready",
            description="open the current window -- same as the controller trigger",
            hotkey="space",
        )

        @self._command.on_trigger
        def _(_: viser.GuiEvent) -> None:
            self._key.set()

    def _runtime(self) -> tuple[bool, str]:
        """Whether a CloudXR runtime is still there. It cannot promise more than that.

        A failure to open the OpenXR session shows up after the click, with the
        performer already standing there; this catches the common case of no runtime.
        """
        try:
            self._launcher.health_check()
        except RuntimeError as gone:
            return (False, str(gone))
        return (True, "attached")

    def _destination(self) -> str:
        return f"{self._recording.parent.name}/{self._recording.name}"

    def run(self) -> None:
        print(
            f"[capture] idle -- press start in the browser; take is {self._recording}"
        )
        while not self._armed.wait(0.2):
            pass
        self._start.visible = False
        self._idle.visible = False
        try:
            self._record()
        except RuntimeError as failed:
            # The performer is standing there and the browser is what they are looking
            # at, so the reason has to land on the page and not only on the terminal.
            self._status.content = render.session_failed(failed)
            print(f"[capture] session failed: {failed}", file=sys.stderr)
        print("[capture] serving the report -- Ctrl+C to stop")
        while True:
            time.sleep(0.5)

    def _record(self) -> None:
        pipeline, body = build_pipeline(self._vendor)
        config = TeleopSessionConfig(
            app_name="FullBodyAcceptanceCapture",
            pipeline=pipeline,
            plugins=list(self._plugins),
            mcap_config=McapRecordingConfig(str(self._recording)),
        )
        take = Take(
            cue_seconds={
                label: cues.duration_of(path) for label, path in self._spoken.items()
            },
            end_tone_s=cues.duration_of(cues.END_TONE),
        )
        with TeleopSession(config) as session:
            self._loop(LiveFrameSource(session, body), take)
        # Spoken after the file is closed, so "you can stop now" is literally true.
        cues.play(cues.wav("closing", cues.CLOSING))
        self._label(take)

    def _loop(self, source: LiveFrameSource, take: Take) -> None:
        held = HeldJoints()
        started = time.monotonic()
        drawn = 0.0
        finished_at: float | None = None

        while True:
            live = source.step()
            if live.frame is not None:
                self._sample = held.sample(live.frame)
                self._skeleton.draw(self._sample, name_held=True)

            now = time.monotonic()
            if take.phase is Phase.IDLE and held.ever_valid:
                self._play(take.start(now), take)
            # Time first, then the press: a hold that ran out on this tick has closed
            # before a press arriving with it can be considered, which is what keeps a
            # late press out of the window it would otherwise reopen.
            self._play(take.advance(now, source.records), take)

            # Two record numbers, two meanings. A press belongs to the record this step
            # observed, and becomes the window's first; a close needs the first record
            # *after* the window, which is the count written so far.
            if live.pressed:
                self._offer(take, now, source.records - 1, TRIGGER)
            elif self._key.is_set():
                self._key.clear()
                self._offer(take, now, source.records - 1, KEYBOARD)

            if now - drawn > GUI_PERIOD_S:
                drawn = now
                self._draw_sidebar(take, now, now - started, held.ever_valid)
                self._warn(held.ever_valid, now - started)

            if take.phase is Phase.DONE:
                finished_at = finished_at or now
                if now - finished_at > TAIL_S:
                    return
            time.sleep(STEP_PERIOD_S)

    def _warn(self, ever_valid: bool, elapsed_s: float) -> None:
        """Show the causes once the take has run this long with nothing valid.

        Advisory only: the take keeps running, the file keeps being written, and the
        first valid frame both starts the script and takes this back down. Nothing
        here decides anything -- the operator does.
        """
        show = not ever_valid and elapsed_s > NO_DATA_S
        if show and not self._warned:
            self._warned = True
            vendor = None if self._vendor is None else self._vendor.id
            self._warning_text.content = render.no_data(NO_DATA_S, vendor)
            print(f"[capture] no body data after {NO_DATA_S:.0f}s", file=sys.stderr)
        self._warning.visible = show

    def _play(self, sound: Sound | None, take: Take) -> None:
        if sound is Sound.CUE:
            self._cue_process = cues.play(self._spoken[take.label])
        elif sound is Sound.START:
            cues.play(cues.START_TONE)
        elif sound is Sound.END:
            cues.play(cues.END_TONE)

    def _offer(self, take: Take, now_s: float, frame: int, source: str) -> None:
        outcome = take.press(now_s, frame, source)
        if outcome is Press.ACCEPTED:
            cues.play(cues.START_TONE)
        elif outcome is Press.CUT_CUE:
            self._silence()
        # IGNORED does nothing at all, and nothing here may make it visible.

    def _silence(self) -> None:
        process = self._cue_process
        self._cue_process = None
        if process is not None:
            process.terminate()

    def _draw_sidebar(
        self, take: Take, now_s: float, elapsed_s: float, ever_valid: bool
    ) -> None:
        sample = self._sample
        valid = sample.valid_count if sample is not None else 0
        held = self._skeleton.held_names(sample) if sample is not None else []
        with self._server.atomic():
            self._status.content = (
                render.recording(self._recording.name, elapsed_s)
                + render.step_block(take, now_s)
                + render.next_up(take)
                + render.joints(valid, NUM_JOINTS, held, ever_valid)
            )
            listed = (take.index, take.phase is Phase.DONE)
            if listed != self._listed:
                self._listed = listed
                self._steps.content = render.step_list(take)

    def _label(self, take: Take) -> None:
        """Turn the recorded windows into the sidecar, now that the file is closed."""
        sidecar, checks = make_labels.write(self._recording, take.windows)
        name = sidecar.name if sidecar is not None else "no sidecar written"
        self._report.content = render.wrote_labels(name, checks)
        print(f"[capture] {name}")
        for label, ok, detail in checks:
            print(f"    {'ok  ' if ok else 'BAD '} {label:<24} {detail}")


def _vendor(
    parser: argparse.ArgumentParser, vendor_id: str | None, params: list[str]
) -> TrackerVendor | None:
    pairs: dict[str, str] = {}
    for item in params:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error(f"--vendor-param wants KEY=VALUE, got {item!r}")
        pairs[key] = value
    if vendor_id is None:
        if pairs:
            parser.error("--vendor-param needs --vendor")
        return None
    print(f"[capture] full body from {vendor_id} {pairs or ''}".rstrip())
    return TrackerVendor(vendor_id, pairs)


def _plugins(
    parser: argparse.ArgumentParser, names: list[str]
) -> tuple[PluginConfig, ...]:
    """Plugin processes the session launches, searched for the way the examples do."""
    if not names:
        return ()
    repo = Path(__file__).resolve().parents[3]
    roots = [
        path
        for path in (repo / "plugins", repo / "install" / "plugins")
        if path.is_dir()
    ]
    if not roots:
        parser.error(
            f"--plugin given but neither {repo}/plugins nor {repo}/install/plugins "
            "exists; run `cmake --install build`"
        )
    for name in names:
        print(f"[capture] plugin {name}")
    # required, because a plugin that quietly fails to load records a take with no body
    # data in it -- which is the one outcome this panel exists to prevent.
    return tuple(
        PluginConfig(
            plugin_name=name,
            plugin_root_id=name,
            search_paths=roots,
            required=True,
        )
        for name in names
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capture-panel", description=__doc__)
    parser.add_argument("output", type=Path, help="where to write the take")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="bind address (default: all interfaces, so the headset can reach it)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--vendor",
        help="full-body backend id, e.g. body.noitom (default: the tracker's own "
        "body.pico-xr)",
    )
    parser.add_argument(
        "--vendor-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="parameter for --vendor, repeatable, e.g. collection_id=noitom_mocap",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="NAME",
        help="plugin to launch for the session, repeatable, e.g. noitom_mocap",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv)
    vendor = _vendor(parser, args.vendor, args.vendor_param)
    plugins = _plugins(parser, args.plugin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(
                f"[capture] CloudXR runtime started (WSS log: {launcher.wss_log_path})"
            )
        server = viser.ViserServer(host=args.host, port=args.port)
        # viser moves to the next free port rather than failing, so args.port is not
        # necessarily where it landed.
        print(f"[capture] http://localhost:{server.get_port()}")
        try:
            CapturePanel(server, launcher, args.output, vendor, plugins).run()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
