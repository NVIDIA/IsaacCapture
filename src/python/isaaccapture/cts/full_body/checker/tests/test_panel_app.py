# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The renderer, driven without a browser."""

from __future__ import annotations

import io
import socket
import zipfile
from pathlib import Path
from typing import Iterator

import pytest
import synth
from full_body_cts.frames import NUM_JOINTS
from full_body_cts.panel.track import TeeSource
from full_body_cts.report import run

viser = pytest.importorskip("viser")

from full_body_cts.panel import app as panel_app  # noqa: E402  needs viser
from full_body_cts.panel.app import Panel  # noqa: E402  needs viser


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
    source = TeeSource(synth.StubSource(frames))
    report = run(source)
    return Panel(server, report, source.track(), recording)


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


def test_a_take_plays_from_end_to_end(server):
    """Every sample goes through the draw path, including the joint that drops out."""
    frames = synth.frames(400) + synth.frames(400, valid_joints=NUM_JOINTS - 3)
    panel = panel_over(frames, server)
    panel._playing.value = True

    seen = set()
    for _ in range(2000):
        panel.advance(0.01)
        panel.draw_rate()
        seen.add(panel._index)
    assert len(seen) > 100


def test_a_recording_with_no_frames_still_builds(server):
    panel = panel_over([], server)
    panel._playing.value = True
    panel.advance(1.0)
    panel.draw_rate()


def test_seeking_by_hand_moves_the_playhead(server):
    panel = panel_over(synth.frames(200), server)
    panel._scrub.value = 150
    assert panel._index == 150


def test_packaging_sends_one_archive_and_locks_the_button_until_it_is_done(
    server, tmp_path, monkeypatch
):
    monkeypatch.setattr(panel_app, "REENABLE_DELAY_S", 0.0)
    recording = tmp_path / "take.mcap"
    recording.write_bytes(b"the bundle moves bytes and does not parse them")
    panel = panel_over(synth.frames(60), server, recording)

    client = FakeClient(panel)
    panel.package(client)

    [(name, data)] = client.sent
    assert "take" in name and name.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert f"{name[:-4]}/report.json" in archive.namelist()
    assert client.disabled_mid_transfer is True
    assert panel._package_button.disabled is False


def test_a_panel_with_no_recording_has_no_package_button(server):
    panel = panel_over(synth.frames(60), server)
    assert not hasattr(panel, "_package_button")
    panel.package(FakeClient())  # and asking anyway is a no-op
