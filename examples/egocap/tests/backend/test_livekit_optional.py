# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The backend works without the optional ``livekit`` package; only receiving LiveKit video needs it."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

import egocap_web
from capture_server import protocol as p
from egocap_web import livekit_receiver as original_receiver
from egocap_web import livekit_recording as lrec


def _stream() -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=16, name="cam", track="t", url="ws://livekit.invalid:7880", room="r", publisher="orin"
    )


@pytest.fixture
def without_livekit(monkeypatch: pytest.MonkeyPatch):
    """A fresh import of the receiver module as if the package were not installed; undone afterwards."""
    monkeypatch.setitem(sys.modules, "livekit", None)  # makes ``from livekit import rtc`` raise ImportError
    monkeypatch.delitem(sys.modules, "egocap_web.livekit_receiver")
    # The import below rebinds this package attribute; restore it so later tests see the original module.
    monkeypatch.setattr(egocap_web, "livekit_receiver", original_receiver)
    return importlib.import_module("egocap_web.livekit_receiver")


def test_the_receiver_module_imports_without_livekit(without_livekit):
    assert without_livekit.livekit_available() is False


def test_receiving_without_livekit_names_the_fix(without_livekit):
    with pytest.raises(RuntimeError, match="'livekit' extra"):
        without_livekit.LiveKitFrameReceiver([_stream()], "token")


def test_a_livekit_stream_is_reported_unrecordable_without_livekit(monkeypatch):
    monkeypatch.setattr(lrec, "livekit_available", lambda: False)
    reason = lrec.unrecordable_reason(SimpleNamespace(streams=[_stream()]))
    assert reason is not None and "'livekit' extra" in reason


def test_a_session_without_livekit_streams_is_unaffected_when_it_is_missing(monkeypatch):
    monkeypatch.setattr(lrec, "livekit_available", lambda: False)
    assert lrec.unrecordable_reason(SimpleNamespace(streams=[])) is None
