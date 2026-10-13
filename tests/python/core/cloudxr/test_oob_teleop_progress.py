# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator progress delivered by the shared OOB service."""

from isaaccapture.cloudxr.oob_teleop_progress import OobProgressReporter


def test_absent_headset_wait_is_visible_and_rate_limited():
    now = [0.0]
    reporter = OobProgressReporter(clock=lambda: now[0])
    waiting = {
        "health": "degraded",
        "state": "WAITING_FOR_ADB",
        "reason": "Waiting for selected headset",
        "adbReady": False,
    }
    assert reporter.observe({**waiting, "health": "starting"}) == []
    assert reporter.observe(waiting) == [
        "No USB-attached HMD detected; connect a headset to continue."
    ]
    now[0] = 5
    assert reporter.observe(waiting) == []
    now[0] = 61
    assert reporter.observe(waiting) == [
        "No USB-attached HMD detected; connect a headset to continue."
    ]


def test_selected_headset_and_adb_diagnostics_are_actionable():
    reporter = OobProgressReporter(clock=lambda: 0)
    waiting = {
        "health": "degraded",
        "state": "WAITING_FOR_ADB",
        "reason": "Selected headset HMD-1 offline; reconnect USB",
        "selectedSerial": "HMD-1",
        "adbReady": False,
    }
    assert reporter.observe(waiting) == [
        "Waiting for USB-attached HMD: Selected headset HMD-1 offline; reconnect USB"
    ]
    diagnostic = {
        **waiting,
        "reason": "adb: failed to connect to daemon at tcp:5037",
        "adbDiagnostic": "adb: failed to connect to daemon at tcp:5037",
    }
    assert reporter.observe(diagnostic) == [
        "ADB diagnostic: adb: failed to connect to daemon at tcp:5037"
    ]


def test_replug_reports_reentered_phases_without_steady_state_spam():
    reporter = OobProgressReporter(clock=lambda: 0)
    base = {
        "health": "degraded",
        "selectedSerial": "HMD-1",
        "adbReady": True,
    }
    phases = (
        ("PREPARING_DEVICE", "Preparing headset HMD-1"),
        ("REBUILDING_USB", "Configuring USB reverse rules"),
        ("AUTOMATING_BROWSER", "Opening headset browser"),
        ("VERIFYING_BROWSER", "Waiting for headset browser"),
    )
    for state, expected in phases:
        messages = reporter.observe({**base, "state": state})
        assert any(expected in message for message in messages)
        assert reporter.observe({**base, "state": state}) == []

    lost = {
        **base,
        "state": "WAITING_FOR_ADB",
        "reason": "Selected headset disconnected",
        "adbReady": False,
    }
    assert any("Waiting for USB-attached HMD" in m for m in reporter.observe(lost))
    for state, expected in phases:
        messages = reporter.observe({**base, "state": state})
        assert any(expected in message for message in messages)
        if state == "PREPARING_DEVICE":
            assert any("HMD reconnected" in message for message in messages)


def test_new_episode_reannounces_same_phase():
    reporter = OobProgressReporter(clock=lambda: 0)
    phase = {
        "health": "degraded",
        "state": "REBUILDING_USB",
        "reason": "Configuring USB reverse rules",
        "episodeStartedAt": 10,
    }
    assert reporter.observe(phase) == ["Configuring USB reverse rules and TURN."]
    assert reporter.observe(phase) == []
    assert reporter.observe({**phase, "episodeStartedAt": 20}) == [
        "Configuring USB reverse rules and TURN."
    ]


def test_browser_ready_is_not_claimed_to_be_streaming():
    reporter = OobProgressReporter(clock=lambda: 0)
    base = {"selectedSerial": "HMD-1", "adbReady": True, "state": "ACTIVE"}
    messages = reporter.observe({**base, "health": "browser_ready"})
    assert "OOB browser ready; waiting for OpenXR stream." in messages
    assert reporter.observe({**base, "health": "browser_ready"}) == []
    assert reporter.observe({**base, "health": "active"}) == []


def test_recovery_delay_is_rate_limited():
    now = [0.0]
    reporter = OobProgressReporter(clock=lambda: now[0])
    delayed = {
        "health": "degraded",
        "state": "DEGRADED",
        "reason": "coturn is not listening",
    }
    assert reporter.observe(delayed) == [
        "OOB recovery delayed: coturn is not listening"
    ]
    assert reporter.observe(delayed) == []
    now[0] = 31
    assert reporter.observe({**delayed, "generation": 2}) == [
        "OOB recovery delayed: coturn is not listening"
    ]
