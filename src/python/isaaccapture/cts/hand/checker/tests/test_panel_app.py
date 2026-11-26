# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand renderer, driven without a browser."""

from __future__ import annotations

import io
import socket
import zipfile
from pathlib import Path
from typing import Iterator

import pytest
from synth import StubSource, both_hands, take

from hand_cts.panel.track import HandTee
from hand_cts.report import run

viser = pytest.importorskip("viser")

from hand_cts.panel import app as panel_app  # noqa: E402  needs viser
from hand_cts.panel.app import Panel  # noqa: E402  needs viser


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def server() -> Iterator["viser.ViserServer"]:
    running = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
    yield running
    running.stop()


def panel_over(frames: list, server, recording: Path | None = None) -> Panel:
    source = HandTee(StubSource(frames))
    report = run(source)
    return Panel(server, report, source.tracks(), recording)


class FakeClient:
    """Stands in for the browser: records the transfer instead of performing it."""

    def __init__(self, panel: Panel | None = None) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.disabled_mid_transfer: bool | None = None
        self._panel = panel

    def send_file_download(self, filename: str, content: bytes) -> None:
        self.sent.append((filename, content))
        if self._panel is not None:
            self.disabled_mid_transfer = self._panel._package_button.disabled


def test_packaging_sends_one_archive_and_locks_the_button_until_it_is_done(
    server, tmp_path, monkeypatch
):
    monkeypatch.setattr(panel_app, "REENABLE_DELAY_S", 0.0)
    recording = tmp_path / "take-hand.mcap"
    recording.write_bytes(b"the bundle moves bytes and does not parse them")
    panel = panel_over(both_hands(30), server, recording)

    client = FakeClient(panel)
    panel.package(client)

    [(name, data)] = client.sent
    assert name == f"take-hand.{panel._report.verdict}.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert f"{name[:-4]}/report.json" in archive.namelist()
        assert archive.read(f"{name[:-4]}/take-hand.mcap") == recording.read_bytes()
    assert client.disabled_mid_transfer is True
    assert panel._package_button.disabled is False
    assert panel._package_bar.visible is False
    assert name in panel._package_note.content


def test_a_take_that_went_away_reports_why_and_unlocks_the_button(
    server, tmp_path, monkeypatch
):
    monkeypatch.setattr(panel_app, "REENABLE_DELAY_S", 0.0)
    recording = tmp_path / "take-hand.mcap"
    recording.write_bytes(b"moved before the button was pressed")
    panel = panel_over(both_hands(30), server, recording)
    recording.unlink()

    client = FakeClient(panel)
    panel.package(client)

    assert client.sent == []
    assert "could not package" in panel._package_note.content
    assert panel._package_button.disabled is False
    assert panel._package_bar.visible is False


def test_a_panel_with_no_recording_has_no_package_button(server):
    panel = panel_over(both_hands(30), server)
    assert not hasattr(panel, "_package_button")
    panel.package(FakeClient())  # and asking anyway is a no-op


def test_separate_hands_moves_left_and_right_apart_and_back(server):
    panel = panel_over(both_hands(30), server)
    panel._separate.value = True
    panel._place_hands()
    left = tuple(panel._roots["left"].position)
    right = tuple(panel._roots["right"].position)
    assert left[0] < 0.0 < right[0]
    assert left[1:] == right[1:] == (0.0, 0.0)

    panel._separate.value = False
    panel._place_hands()
    assert tuple(panel._roots["left"].position) == (0.0, 0.0, 0.0)
    assert tuple(panel._roots["right"].position) == (0.0, 0.0, 0.0)


def drawn_joint_count(panel: Panel, side: str) -> int:
    return len(panel._skeletons[side]._points.points)


def test_a_hidden_digit_drops_its_joints_and_bones_on_both_hands(server):
    panel = panel_over(both_hands(30), server)
    full = {side: drawn_joint_count(panel, side) for side in ("left", "right")}
    bones = len(panel._skeletons["left"]._lines.points)

    panel._show_digit["thumb"].value = False
    panel._show_digit["index"].value = False
    panel._draw()

    gone = len(panel_app.DIGIT_JOINTS["thumb"]) + len(panel_app.DIGIT_JOINTS["index"])
    for side in ("left", "right"):
        assert drawn_joint_count(panel, side) == full[side] - gone
    assert len(panel._skeletons["left"]._lines.points) < bones

    panel._show_digit["thumb"].value = True
    panel._show_digit["index"].value = True
    panel._draw()
    assert drawn_joint_count(panel, "left") == full["left"]


def test_a_hidden_hand_stays_hidden_when_digits_are_shown(server):
    panel = panel_over(both_hands(30), server)
    panel._show_hand["left"].value = False
    panel._show_hands("left")
    panel._show_digit["ring"].value = False
    panel._draw()
    panel._show_digit["ring"].value = True
    panel._draw()

    assert panel._roots["left"].visible is False
    assert panel._roots["right"].visible is True

    panel._show_hand["left"].value = True
    panel._show_hands("left")
    assert panel._roots["left"].visible is True


def test_filled_joints_estimate_only_what_was_never_reported():
    from hand_cts.profile import METACARPALS, PALM, PROXIMAL, WRIST

    positions = [None] * panel_app.NUM_JOINTS
    positions[WRIST] = (0.0, 0.0, 0.0)
    for digit in METACARPALS:
        positions[PROXIMAL[digit]] = (0.0, 0.0, -0.1)
    positions[METACARPALS["index"]] = (0.0, 0.0, -0.05)  # reported, so kept as is

    joints, bones = panel_app.filled_joints(tuple(positions))

    assert set(joints) == {PALM, *METACARPALS.values()} - {METACARPALS["index"]}
    assert joints[PALM] == (0.0, 0.0, -0.05)
    # Wrist to metacarpal and metacarpal to knuckle for each filled finger.
    assert len(bones) == 2 * 3

    joints, bones = panel_app.filled_joints(
        tuple(positions), frozenset({METACARPALS["ring"], PROXIMAL["ring"]})
    )
    assert METACARPALS["ring"] not in joints
    assert len(bones) == 2 * 2


def test_fill_missing_joints_draws_a_separate_layer_only_when_ticked(server):
    from hand_cts.profile import OPTIONAL, REQUIRED

    left = take(30, "left", valid=REQUIRED)
    right = take(30, "right", valid=REQUIRED)
    panel = panel_over([f for pair in zip(left, right) for f in pair], server)
    reported = drawn_joint_count(panel, "left")
    filled, bones = panel._filled["left"]
    assert len(filled.points) == 0

    panel._fill.value = True
    panel._draw()
    assert drawn_joint_count(panel, "left") == reported
    assert len(filled.points) == len(OPTIONAL)
    assert len(bones.points) == 2 * (len(OPTIONAL) - 1)  # PALM has no bone

    panel._fill.value = False
    panel._draw()
    assert len(filled.points) == 0


def test_joint_axes_skip_hidden_digits_and_hands_keep_their_own_layer(server):
    panel = panel_over(both_hands(30), server)
    axes = panel._skeletons["left"]._axes
    assert len(axes.batched_positions) == 0

    panel._axes.value = True
    panel._draw()
    assert len(axes.batched_positions) == drawn_joint_count(panel, "left")

    panel._show_digit["thumb"].value = False
    panel._draw()
    assert len(axes.batched_positions) == drawn_joint_count(panel, "left")
    assert axes.name.startswith("/hands/left/")
