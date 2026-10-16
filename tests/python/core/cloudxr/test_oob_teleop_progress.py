# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator progress delivered by the shared OOB service."""

import logging

from isaaccapture.cloudxr.oob_teleop_progress import (
    OobProgressEvent,
    OobProgressReporter,
)


def test_absent_headset_wait_is_visible_once():
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
        OobProgressEvent(
            logging.INFO, "No USB-attached HMD detected; connect a headset to continue."
        )
    ]
    now[0] = 61
    assert reporter.observe(waiting) == []
    assert reporter.observe({**waiting, "generation": 2}) == []


def test_adb_authorization_overrides_diagnostic_and_remains_visible():
    reporter = OobProgressReporter(clock=lambda: 0)
    waiting = {
        "health": "degraded",
        "state": "WAITING_FOR_ADB",
        "reason": "Selected headset HMD-1 offline; reconnect USB",
        "selectedSerial": "HMD-1",
        "adbReady": False,
    }
    assert reporter.observe(waiting) == [
        OobProgressEvent(
            logging.INFO,
            "Waiting for USB-attached HMD: Selected headset HMD-1 offline; reconnect USB",
        )
    ]
    unauthorized = {
        **waiting,
        "reason": "Selected headset unauthorized; accept the USB debugging prompt",
        "adbDiagnostic": "adb transport unavailable",
    }
    assert reporter.observe(unauthorized) == [
        OobProgressEvent(
            logging.WARNING,
            "ADB authorization required: unlock the headset and accept "
            "the USB debugging prompt.",
        )
    ]
    assert (
        reporter.observe({**unauthorized, "adbDiagnostic": "another adb detail"}) == []
    )
    assert reporter.observe(
        {
            **waiting,
            "reason": "adb: failed to connect to daemon at tcp:5037",
            "adbDiagnostic": "adb: failed to connect to daemon at tcp:5037",
        }
    ) == [
        OobProgressEvent(
            logging.WARNING,
            "ADB diagnostic: adb: failed to connect to daemon at tcp:5037",
        )
    ]


def test_replug_reports_status_and_debug_phases_without_steady_state_spam():
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
        events = reporter.observe({**base, "state": state})
        assert any(
            expected in event.message and event.level == logging.DEBUG
            for event in events
        )
        assert reporter.observe({**base, "state": state}) == []

    lost = {
        **base,
        "state": "WAITING_FOR_ADB",
        "reason": "Selected headset disconnected",
        "adbReady": False,
    }
    assert any(
        "Waiting for USB-attached HMD" in event.message
        for event in reporter.observe(lost)
    )
    events = reporter.observe({**base, "state": "PREPARING_DEVICE"})
    assert any(
        "HMD reconnected" in event.message and event.level == logging.INFO
        for event in events
    )


def test_new_episode_reannounces_debug_phase_only():
    reporter = OobProgressReporter(clock=lambda: 0)
    phase = {
        "health": "degraded",
        "state": "REBUILDING_USB",
        "reason": "Configuring USB reverse rules",
        "episodeStartedAt": 10,
    }
    expected = OobProgressEvent(
        logging.DEBUG, "Configuring USB reverse rules and TURN."
    )
    assert reporter.observe(phase) == [expected]
    assert reporter.observe(phase) == []
    assert reporter.observe({**phase, "episodeStartedAt": 20}) == [expected]


def test_normal_startup_keeps_info_progress_sparse():
    reporter = OobProgressReporter(clock=lambda: 0)
    base = {
        "health": "degraded",
        "selectedSerial": "HMD-1",
        "adbReady": True,
    }
    events = []
    for state in (
        "PREPARING_DEVICE",
        "REBUILDING_USB",
        "AUTOMATING_BROWSER",
        "VERIFYING_BROWSER",
    ):
        events.extend(reporter.observe({**base, "state": state}))
    events.extend(
        reporter.observe(
            {**base, "state": "VERIFYING_BROWSER", "health": "browser_ready"}
        )
    )
    assert [event.message for event in events if event.level >= logging.INFO] == [
        "USB-attached HMD detected (serial HMD-1); continuing OOB setup.",
        "OOB browser ready; waiting for OpenXR stream.",
    ]


def test_browser_ready_is_not_claimed_to_be_streaming():
    reporter = OobProgressReporter(clock=lambda: 0)
    base = {"selectedSerial": "HMD-1", "adbReady": True, "state": "ACTIVE"}
    events = reporter.observe({**base, "health": "browser_ready"})
    assert (
        OobProgressEvent(logging.INFO, "OOB browser ready; waiting for OpenXR stream.")
        in events
    )
    assert reporter.observe({**base, "health": "browser_ready"}) == []
    assert reporter.observe({**base, "health": "active"}) == []


def test_network_and_health_blocks_are_operator_visible_once_per_entry():
    reporter = OobProgressReporter(clock=lambda: 0)
    network = {
        "health": "degraded",
        "state": "WAITING_FOR_NETWORK",
        "reason": "No non-loopback network",
    }
    assert reporter.observe(network) == [
        OobProgressEvent(
            logging.WARNING,
            "Headset network unavailable; connect it to Wi-Fi to continue.",
        )
    ]
    assert reporter.observe({**network, "reason": "Still no network"}) == []
    blocked = {**network, "state": "USB_HEALTH_BLOCKED", "reason": "USB too slow"}
    assert reporter.observe(blocked) == [
        OobProgressEvent(
            logging.WARNING, "USB health policy blocked browser launch: USB too slow"
        )
    ]
    assert reporter.observe({**blocked, "reason": "USB still too slow"}) == []


def test_recovery_delay_is_rate_limited():
    now = [0.0]
    reporter = OobProgressReporter(clock=lambda: now[0])
    delayed = {
        "health": "degraded",
        "state": "DEGRADED",
        "reason": "coturn is not listening",
    }
    expected = OobProgressEvent(
        logging.WARNING, "OOB recovery delayed: coturn is not listening"
    )
    assert reporter.observe(delayed) == [expected]
    assert reporter.observe(delayed) == []
    now[0] = 31
    assert reporter.observe({**delayed, "generation": 2}) == [expected]
