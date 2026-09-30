# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records one take of the hand script, paced by a timer.

viser comes up idle with nothing open against the device. Start constructs the
``TeleopSession``, which writes the MCAP and feeds the two skeletons; the script begins
once both hands have had a valid joint, and every step is cue, 3-2-1, hold, end tone.
One take per process. Run it through ``record.sh``, which names the take.

    python capture_panel.py OUTPUT.mcap --device NAME [--plugin NAME] [--accept-eula]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import viser

from isaacteleop.cloudxr import CloudXRLauncher, NoopContext
from isaacteleop.deviceio import McapRecordingConfig
from isaacteleop.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

from acceptance_common.panel.sample import Sample, Vec3
from acceptance_common.panel.skeleton import Skeleton
from hand_acceptance.frames import Frame
from hand_acceptance.labels import SETTLE_FRACTION
from hand_acceptance.profile import HAND, JOINT_NAMES

import cues
import make_labels
import render
from live import LiveHands, build_pipeline
from session import BEAT_S, COUNTDOWN_BEATS, STEPS
from steps import TIMER, Phase, Sound, Take

STEP_PERIOD_S = 1.0 / 90.0
GUI_PERIOD_S = 0.1
TAIL_S = 2.0
DEFAULT_PORT = 8081
PANEL_WIDTH_PX = 480
NO_DATA_S = 10.0
NUM_JOINTS = len(JOINT_NAMES)
SIDES = ("left", "right")
WRIST_ENV = "WUJI_GLOVE_WRIST_SOURCE"


class HeldJoints:
    """Last valid position per joint, which is where an invalid one is drawn."""

    def __init__(self) -> None:
        self._held: list[Vec3 | None] = [None] * NUM_JOINTS
        self.ever_valid = False

    def sample(self, frame: Frame) -> Sample:
        valid = [False] * NUM_JOINTS
        for index, joint in enumerate(frame.joints or ()):
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
        plugins: tuple[PluginConfig, ...] = (),
    ) -> None:
        self._server = server
        self._launcher = launcher
        self._recording = recording
        self._plugins = plugins
        self._spoken = {label: cues.wav(label, text) for label, _, text in STEPS}
        self._armed = threading.Event()
        self._valid: dict[str, int | None] = dict.fromkeys(SIDES)
        self._listed: tuple[int, bool] | None = None

        server.gui.configure_theme(control_width="large", dark_mode=False)
        server.gui.main_panel.dock_right()
        server.gui.main_panel.set_width(PANEL_WIDTH_PX)
        server.scene.set_up_direction("+y")
        server.scene.add_grid(
            "/floor", width=2.0, height=2.0, plane="xz", cell_size=0.1
        )
        server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.002)
        server.initial_camera.position = (0.0, 1.6, 0.6)
        server.initial_camera.look_at = (0.0, 1.1, -0.3)
        self._skeletons = {
            side: Skeleton(server, HAND, root=f"/hands/{side}", point_size=0.008)
            for side in SIDES
        }
        self._build_gui()
        self._build_warning()

    def _build_warning(self) -> None:
        panel = self._server.gui.add_panel(visible=False)
        panel.float(x=20, y=-20, width=440)
        with panel.add_tab("No data"):
            self._warning_text = self._server.gui.add_html("")
        self._warning = panel
        self._warned = False

    def _build_gui(self) -> None:
        gui = self._server.gui
        attached, detail = self._runtime()
        source = ", ".join(p.plugin_name for p in self._plugins) or "headset optical"
        self._idle = gui.add_html(
            render.ready_to_start(attached, detail, self._destination(), source)
        )
        self._start = gui.add_button("Start recording")
        self._status = gui.add_html("")
        self._steps = gui.add_html("")
        self._report = gui.add_html("")

        @self._start.on_click
        def _(_: viser.GuiEvent) -> None:
            self._armed.set()

    def _runtime(self) -> tuple[bool, str]:
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
            self._status.content = render.session_failed(failed)
            print(f"[capture] session failed: {failed}", file=sys.stderr)
        print("[capture] serving the report -- Ctrl+C to stop")
        while True:
            time.sleep(0.5)

    def _record(self) -> None:
        pipeline, hands = build_pipeline()
        config = TeleopSessionConfig(
            app_name="HandAcceptanceCapture",
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
            self._loop(LiveHands(session, hands), take)
        cues.play(cues.wav("closing", cues.CLOSING))
        self._label(take)

    def _loop(self, live: LiveHands, take: Take) -> None:
        held = {side: HeldJoints() for side in SIDES}
        started = time.monotonic_ns()
        drawn = 0
        finished_at: int | None = None

        while True:
            frames = live.step()
            for side, frame in frames.items():
                if frame is None:
                    self._valid[side] = None
                    continue
                sample = held[side].sample(frame)
                self._valid[side] = sample.valid_count
                self._skeletons[side].draw(sample, name_held=False)

            # The session clock: sample_time_local_common_clock is CLOCK_MONOTONIC.
            now = time.monotonic_ns()
            both = all(h.ever_valid for h in held.values())
            if take.phase is Phase.IDLE and both:
                self._play(take.start(now), take)
            self._play(take.advance(now), take)

            if now - drawn > GUI_PERIOD_S * 1e9:
                drawn = now
                self._draw_sidebar(take, now, (now - started) / 1e9)
                self._warn(both, (now - started) / 1e9)

            if take.phase is Phase.DONE:
                finished_at = finished_at or now
                if now - finished_at > TAIL_S * 1e9:
                    return
            time.sleep(STEP_PERIOD_S)

    def _warn(self, both_valid: bool, elapsed_s: float) -> None:
        show = not both_valid and elapsed_s > NO_DATA_S
        if show and not self._warned:
            self._warned = True
            names = [p.plugin_name for p in self._plugins]
            self._warning_text.content = render.no_data(NO_DATA_S, names)
            print(
                f"[capture] a hand has no data after {NO_DATA_S:.0f}s", file=sys.stderr
            )
        self._warning.visible = show

    def _play(self, sound: Sound | None, take: Take) -> None:
        if sound is Sound.CUE:
            cues.play(self._spoken[take.label])
        elif sound is Sound.COUNT:
            cues.play(cues.COUNT_TONE)
        elif sound is Sound.START:
            cues.play(cues.START_TONE)
        elif sound is Sound.END:
            cues.play(cues.END_TONE)

    def _draw_sidebar(self, take: Take, now_ns: int, elapsed_s: float) -> None:
        with self._server.atomic():
            self._status.content = (
                render.recording(self._recording.name, elapsed_s)
                + render.step_block(take, now_ns)
                + render.next_up(take)
                + render.hands(self._valid, NUM_JOINTS)
            )
            listed = (take.index, take.phase is Phase.DONE)
            if listed != self._listed:
                self._listed = listed
                self._steps.content = render.step_list(take)

    def _label(self, take: Take) -> None:
        sidecar, checks = make_labels.write(self._recording, take.windows)
        name = sidecar.name if sidecar is not None else "no sidecar written"
        self._report.content = render.wrote_labels(name, checks)
        print(f"[capture] {name}")
        for label, ok, detail in checks:
            print(f"    {'ok  ' if ok else 'BAD '} {label:<48} {detail}")


def _roots(repo: Path) -> list[Path]:
    return [p for p in (repo / "plugins", repo / "install" / "plugins") if p.is_dir()]


def _plugin_version(roots: list[Path], name: str) -> str | None:
    for root in roots:
        for manifest in root.glob("*/plugin.yaml"):
            text = manifest.read_text()
            if re.search(rf"^name:\s*{re.escape(name)}\s*$", text, re.M):
                found = re.search(r'^version:\s*"?([^"\n]+)"?', text, re.M)
                return found.group(1).strip() if found else None
    return None


def _plugins(
    parser: argparse.ArgumentParser, names: list[str], roots: list[Path]
) -> tuple[PluginConfig, ...]:
    if names and not roots:
        parser.error("--plugin given but no plugins/ or install/plugins/ in the repo")
    return tuple(
        PluginConfig(
            plugin_name=name, plugin_root_id=name, search_paths=roots, required=True
        )
        for name in names
    )


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(args: argparse.Namespace, repo: Path, roots: list[Path]) -> dict:
    import isaacteleop

    versions = dict(args.plugin_version)
    for name in args.plugin:
        versions.setdefault(name, _plugin_version(roots, name))
    return {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "device": args.device,
        "hand_source": list(args.plugin) or ["headset_optical"],
        "plugin_versions": versions,
        "wrist_source": args.wrist_source,
        "env": {WRIST_ENV: os.environ.get(WRIST_ENV)},
        "pacing": {
            "mode": TIMER,
            "countdown_beats": COUNTDOWN_BEATS,
            "beat_s": BEAT_S,
            "settle_fraction": SETTLE_FRACTION,
            "hold_s": {label: hold for label, hold, _ in STEPS},
        },
        "notes": args.note,
        "isaacteleop_version": isaacteleop.__version__,
        "repo_commit": _git(repo, "rev-parse", "HEAD"),
        "repo_dirty": bool(_git(repo, "status", "--porcelain")),
        "host": socket.gethostname(),
    }


def _pair(parser: argparse.ArgumentParser, item: str) -> tuple[str, str]:
    key, separator, value = item.partition("=")
    if not separator or not key:
        parser.error(f"--plugin-version wants NAME=VERSION, got {item!r}")
    return key, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hand-capture-panel", description=__doc__)
    parser.add_argument("output", type=Path, help="where to write the take")
    parser.add_argument("--device", required=True, help="e.g. quest3, pico4u, manus")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="NAME",
        help="hand plugin to launch, repeatable, e.g. manus_hand_plugin; none means "
        "the headset's own hand tracking",
    )
    parser.add_argument(
        "--plugin-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="overrides the version read from the plugin's plugin.yaml",
    )
    parser.add_argument(
        "--wrist-source",
        help="where the glove's wrist came from, e.g. hand_tracking or controller; "
        f"defaults to ${WRIST_ENV} when set",
    )
    parser.add_argument("--note", default="", help="free text kept with the take")
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv)
    args.plugin_version = [_pair(parser, item) for item in args.plugin_version]
    args.wrist_source = args.wrist_source or os.environ.get(WRIST_ENV)

    repo = Path(__file__).resolve().parents[3]
    roots = _roots(repo)
    plugins = _plugins(parser, args.plugin, roots)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = args.output.with_suffix(".json")
    provenance.write_text(json.dumps(_provenance(args, repo, roots), indent=2) + "\n")
    print(f"[capture] provenance {provenance}")

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(
                f"[capture] CloudXR runtime started (WSS log: {launcher.wss_log_path})"
            )
        server = viser.ViserServer(host=args.host, port=args.port)
        print(f"[capture] http://localhost:{server.get_port()}")
        try:
            CapturePanel(server, launcher, args.output, plugins).run()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
