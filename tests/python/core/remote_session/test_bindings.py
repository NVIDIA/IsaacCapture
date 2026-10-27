# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bindings a remote session relies on, exercised without a network."""

import pytest

from isaaccapture import deviceio, deviceio_session, deviceio_trackers, oxr
from isaaccapture.deviceio_trackers import (
    HapticCommandPushTracker,
    JointSe3PoseTracker,
    KeyboardTracker,
    MessageChannelTracker,
    Se3Tracker,
)
from isaaccapture.deviceio_trackers._generated_tracker_exports import (
    __all__ as GENERATED_TRACKERS,
)


def test_pull_tracker_exposes_its_constructor_arguments():
    tracker = JointSe3PoseTracker("manus_sensors_left", 2048)
    assert tracker.collection_id == "manus_sensors_left"
    assert tracker.max_flatbuffer_size == 2048
    assert (
        Se3Tracker("se3").max_flatbuffer_size == Se3Tracker.DEFAULT_MAX_FLATBUFFER_SIZE
    )
    with pytest.raises(AttributeError):
        tracker.collection_id = "other"


def test_push_tracker_exposes_its_constructor_arguments():
    tracker = HapticCommandPushTracker("haptics", "command", 512)
    assert tracker.collection_id == "haptics"
    assert tracker.tensor_identifier == "command"
    assert tracker.max_payload_size == 512
    default = HapticCommandPushTracker("haptics", "command")
    assert default.max_payload_size == HapticCommandPushTracker.DEFAULT_MAX_PAYLOAD_SIZE
    with pytest.raises(AttributeError):
        tracker.max_payload_size = 1


@pytest.mark.parametrize("name", GENERATED_TRACKERS)
def test_every_generated_tracker_has_descriptor_properties(name):
    cls = getattr(deviceio_trackers, name)
    pushes = hasattr(cls, "push")
    expected = (
        ("collection_id", "tensor_identifier", "max_payload_size")
        if pushes
        else ("collection_id", "max_flatbuffer_size")
    )
    for attribute in expected:
        assert isinstance(getattr(cls, attribute), property), attribute


def test_openxr_session_takes_wait_for_system():
    assert "wait_for_system: bool = True" in oxr.OpenXRSession.__init__.__doc__


def test_remote_classes_are_exported_from_both_packages():
    for name in (
        "RemotePeerClosedError",
        "RemoteSession",
        "RemoteSessionHost",
        "RemoteSessionListener",
    ):
        assert name in deviceio_session.__all__
        assert name in deviceio.__all__
        assert getattr(deviceio, name) is getattr(deviceio_session, name)


def test_remote_peer_closed_error_is_a_runtime_error():
    assert issubclass(deviceio_session.RemotePeerClosedError, RuntimeError)
    assert (
        "closed or reset the connection"
        in deviceio_session.RemotePeerClosedError.__doc__
    )


def test_remote_session_create_needs_no_network():
    keyboard = KeyboardTracker()
    with deviceio_session.RemoteSession.create([(keyboard, "0")]) as session:
        assert session.subscribed_topics() == ["0/keyboard"]
        assert session.latest_status() is None
        assert session.hello_ack() == ""
        assert session.is_fresh is False
        assert session.tick_age_ns is None
        with pytest.raises(RuntimeError, match="not connected"):
            session.update()
        with pytest.raises(RuntimeError, match="not connected"):
            session.wait_ready(0)
    with pytest.raises(RuntimeError, match="closed"):
        session.subscribed_topics()
    with pytest.raises(RuntimeError, match="closed"):
        keyboard.get_keyboard_data(session)
    session.close()  # idempotent


def test_remote_session_create_rejects_bad_arguments():
    with pytest.raises(ValueError, match="MessageChannelTracker"):
        deviceio_session.RemoteSession.create([(MessageChannelTracker(bytes(16)), "0")])
    with pytest.raises(ValueError, match="max_tick_age_ns"):
        deviceio_session.RemoteSession.create([(KeyboardTracker(), "0")], 0)


def test_remote_session_connect_rejects_a_malformed_endpoint():
    with deviceio_session.RemoteSession.create([(KeyboardTracker(), "0")]) as session:
        with pytest.raises(
            ValueError,
            match=r"^wss:// is not supported yet; use ws:// through an encrypted tunnel "
            r"\(VPN, SSH or stunnel\) between the two machines$",
        ):
            session.connect("wss://127.0.0.1:47601", "{}", 1000)


def test_listener_rejects_a_bad_port():
    with pytest.raises(ValueError, match="port"):
        deviceio_session.RemoteSessionListener("127.0.0.1", 70000)
