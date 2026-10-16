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
from synth import StubSource, both_hands

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
