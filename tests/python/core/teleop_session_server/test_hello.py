# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server's HELLO handling, STATUS JSON and command line, in process."""

import json
import re
import sys
from types import SimpleNamespace

import pytest

import isaaccapture
from isaaccapture import deviceio_trackers, oxr
from isaaccapture.deviceio_session import TrackerVendor
from isaaccapture.deviceio_trackers import (
    ControllerTracker,
    FullBodyTracker,
    HandTracker,
    HeadTracker,
    JointSe3PoseTracker,
    KeyboardTracker,
    Se3Tracker,
)
from isaaccapture.deviceio_trackers._generated_tracker_exports import (
    __all__ as GENERATED_TRACKERS,
)
from isaaccapture.teleop_session_manager import _remote
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server._session import (
    MAX_HELLO_BYTES,
    MAX_TEXT_CHARS,
    MAX_TOPICS,
    MAX_TRACKERS,
    HelloError,
    headset_status,
    parse_hello,
    served_extensions,
    status_json,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

SERVED_EXTENSIONS = served_extensions()


def _hello(**fields):
    hello = {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "Robot",
        "tick_rate_hz": 90.0,
        "trackers": [],
        "topics": [],
    }
    hello.update(fields)
    return json.dumps(hello)


def _tracker(type_name, name="0", params=None, vendor=None):
    return {"name": name, "type": type_name, "params": params or {}, "vendor": vendor}


def _parse(text, placeholder=False, extensions=SERVED_EXTENSIONS):
    return parse_hello(
        text, placeholder=placeholder, extensions=None if placeholder else extensions
    )


def _refused(text, match, **kwargs):
    with pytest.raises(HelloError, match=match):
        _parse(text, **kwargs)


# ============================================================================
# Trackers
# ============================================================================


def test_builds_the_trackers_a_robot_app_describes():
    """The robot side's descriptors (teleop_session_manager._remote) round-trip into trackers."""
    body, noitom_body = FullBodyTracker(), FullBodyTracker()
    sensors = JointSe3PoseTracker("manus_sensors_left", 2048)
    robot_trackers = [
        HeadTracker(),
        HandTracker(),
        ControllerTracker(),
        body,
        noitom_body,
        sensors,
    ]
    vendors = {
        id(body): TrackerVendor("body.pico-xr"),
        id(noitom_body): TrackerVendor("body.noitom"),
    }
    descriptors = [
        _remote.describe_tracker(tracker, str(i), vendors.get(id(tracker)), "test")
        for i, tracker in enumerate(robot_trackers)
    ]
    hello = _hello(
        app_name="Robot",
        tick_rate_hz=120.0,
        trackers=descriptors,
        topics=["0/head", "5/joint_se3_pose_tracked"],
    )

    request = _parse(hello)

    assert request.app_name == "Robot"
    assert request.isaaccapture_version == isaaccapture.__version__
    assert request.tick_rate_hz == 120.0
    assert request.topics == ["0/head", "5/joint_se3_pose_tracked"]
    assert [name for _, name in request.trackers] == ["0", "1", "2", "3", "4", "5"]
    assert [type(t) for t, _ in request.trackers] == [type(t) for t in robot_trackers]
    server_sensors = request.trackers[5][0]
    assert server_sensors.collection_id == "manus_sensors_left"
    assert server_sensors.max_flatbuffer_size == 2048
    assert [
        (tracker, vendor.id, dict(vendor.params)) for tracker, vendor in request.vendors
    ] == [
        (request.trackers[3][0], "body.pico-xr", {}),
        (request.trackers[4][0], "body.noitom", {}),
    ]


def _pull_tracker_names():
    """Generated pull trackers a remote session carries: not plugin status, no metadata."""
    return [
        name
        for name in GENERATED_TRACKERS
        if not hasattr(getattr(deviceio_trackers, name), "push")
        and not _remote.has_metadata(getattr(deviceio_trackers, name))
        and name != "PluginDeviceStatusTracker"
    ]


@pytest.mark.parametrize("type_name", _pull_tracker_names())
def test_serves_every_generated_pull_tracker(type_name):
    request = _parse(
        _hello(trackers=[_tracker(type_name, params={"collection_id": "data"})])
    )
    [(tracker, name)] = request.trackers
    cls = getattr(deviceio_trackers, type_name)
    assert type(tracker) is cls
    assert name == "0"
    assert tracker.collection_id == "data"
    # Omitted max_flatbuffer_size keeps the class default.
    assert tracker.max_flatbuffer_size == cls.DEFAULT_MAX_FLATBUFFER_SIZE


@pytest.mark.parametrize(
    "type_name",
    [
        "KeyboardTracker",
        "MessageChannelTracker",
        "TensorPushTracker",
        "HapticCommandPushTracker",
        "HapticCommandReaderTracker",
        "PluginDeviceStatusTracker",
        "TactileTracker",
    ],
)
def test_refuses_trackers_a_remote_session_cannot_serve(type_name):
    _refused(
        _hello(trackers=[_tracker(type_name)]),
        rf"^{type_name} \(tracker '0'\) is not supported over a remote session: ",
    )


def test_refuses_an_unknown_tracker():
    _refused(
        _hello(trackers=[_tracker("TelepathyTracker", name="7")]),
        r"^TelepathyTracker \(tracker '7'\) is unknown to this Teleop Session Server",
    )


def test_placeholder_mode_serves_only_the_keyboard():
    request = _parse(_hello(trackers=[_tracker("KeyboardTracker")]), placeholder=True)
    assert isinstance(request.trackers[0][0], KeyboardTracker)
    _refused(
        _hello(trackers=[_tracker("HeadTracker")]),
        r"--placeholder-openxr and serves only KeyboardTracker, not HeadTracker",
        placeholder=True,
    )


@pytest.mark.parametrize(
    ("descriptor", "match"),
    [
        (_tracker("HeadTracker", params={"x": 1}), r"takes no params, got \['x'\]"),
        (_tracker("Se3Tracker"), "needs a collection_id"),
        (_tracker("Se3Tracker", params={"collection_id": ""}), "needs a collection_id"),
        (
            _tracker("Se3Tracker", params={"collection_id": "a", "extra": 1}),
            r"unknown params \['extra'\]",
        ),
        (
            _tracker(
                "Se3Tracker", params={"collection_id": "a", "max_flatbuffer_size": 0}
            ),
            "max_flatbuffer_size must be a positive integer",
        ),
        (
            _tracker(
                "Se3Tracker", params={"collection_id": "a", "max_flatbuffer_size": True}
            ),
            "max_flatbuffer_size must be a positive integer",
        ),
        (_tracker("HeadTracker", name=""), r"trackers\[0\] has no name"),
        ({"name": "0"}, "needs a type string"),
        ("HeadTracker", r"trackers\[0\] is not a JSON object"),
        (_tracker("HeadTracker", vendor={"params": {}}), "vendor needs an id string"),
        (
            _tracker("HeadTracker", vendor={"id": "v", "params": {"a": 1}}),
            "vendor params must map strings to strings",
        ),
    ],
)
def test_refuses_malformed_tracker_descriptors(descriptor, match):
    _refused(_hello(trackers=[descriptor]), match)


def test_refuses_a_tracker_name_used_twice():
    _refused(
        _hello(trackers=[_tracker("HeadTracker"), _tracker("HandTracker")]),
        "Tracker name '0' is used twice",
    )


def _pedals(count, id_chars=255):
    """``count`` pedal trackers, each reading its own collection with an id of ``id_chars`` characters."""
    return [
        _tracker(
            "Generic3AxisPedalTracker",
            name=str(index),
            params={"collection_id": f"{index:04d}".ljust(id_chars, "x")},
        )
        for index in range(count)
    ]


def test_serves_a_hello_at_every_limit():
    # The longest legitimate collection id is 255 characters (XR_MAX_TENSOR_IDENTIFIER_SIZE).
    text = _hello(
        app_name="a" * MAX_TEXT_CHARS,
        trackers=_pedals(MAX_TRACKERS),
        topics=[f"{index}/pedals_tracked" for index in range(MAX_TOPICS)],
    )
    assert len(text.encode()) <= MAX_HELLO_BYTES
    request = _parse(text)
    assert len(request.trackers) == MAX_TRACKERS
    assert len(request.topics) == MAX_TOPICS


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            _hello(padding="x" * MAX_HELLO_BYTES),
            f"at most {MAX_HELLO_BYTES} are accepted",
        ),
        (
            _hello(trackers=_pedals(MAX_TRACKERS + 1, 8)),
            f"at most {MAX_TRACKERS} are accepted",
        ),
        (
            _hello(topics=["0/head"] * (MAX_TOPICS + 1)),
            f"at most {MAX_TOPICS} are accepted",
        ),
        (
            _hello(app_name="a" * (MAX_TEXT_CHARS + 1)),
            f"at most {MAX_TEXT_CHARS} are accepted",
        ),
        (
            _hello(trackers=_pedals(1, MAX_TEXT_CHARS + 1)),
            f"at most {MAX_TEXT_CHARS} are accepted",
        ),
        ("[" * 100_000 + "]" * 100_000, "HELLO is not valid JSON"),
    ],
    ids=["size", "trackers", "topics", "app_name", "collection_id", "nesting"],
)
def test_refuses_a_hello_past_a_limit(text, match):
    _refused(text, match)


def test_generated_tracker_size_is_passed_on():
    request = _parse(
        _hello(
            trackers=[
                _tracker(
                    "Se3Tracker",
                    params={"collection_id": "se3", "max_flatbuffer_size": 99},
                )
            ]
        )
    )
    assert request.trackers[0][0].max_flatbuffer_size == 99
    assert isinstance(request.trackers[0][0], Se3Tracker)


# ============================================================================
# OpenXR extensions
# ============================================================================


def test_the_served_extensions_cover_every_tracker_and_body_vendor():
    assert {
        "XR_BD_body_tracking",
        "XR_EXT_hand_tracking",
        "XR_NVX1_action_context",
        "XR_NVX1_tensor_data",
    } <= set(SERVED_EXTENSIONS)
    trackers = [
        _tracker(name, str(i))
        for i, name in enumerate(["HeadTracker", "HandTracker", "ControllerTracker"])
    ]
    trackers += [
        _tracker(
            "FullBodyTracker", f"body {vendor}", vendor={"id": vendor, "params": {}}
        )
        for vendor in ("body.pico-xr", "body.noitom", "body.xsens")
    ]
    trackers += [
        _tracker(name, f"pull {name}", params={"collection_id": "data"})
        for name in _pull_tracker_names()
    ]
    assert len(_parse(_hello(trackers=trackers)).trackers) == len(trackers)


def test_refuses_a_body_vendor_the_server_does_not_have():
    _refused(
        _hello(
            trackers=[
                _tracker(
                    "FullBodyTracker", vendor={"id": "body.telepathy", "params": {}}
                )
            ]
        ),
        r"^The Teleop Session Server could not resolve the trackers: .*body\.telepathy",
    )


def test_refuses_vendor_params_the_trackers_cannot_take():
    vendor = {"id": "body.pico-xr", "params": {"mode": "full"}}
    _refused(
        _hello(trackers=[_tracker("FullBodyTracker", vendor=vendor)]),
        r"^The Teleop Session Server could not resolve the trackers: .*vendor params",
    )


def test_refuses_trackers_that_need_extensions_the_session_lacks():
    _refused(
        _hello(trackers=[_tracker("HandTracker")]),
        r"^The trackers need OpenXR extensions \['XR_EXT_hand_tracking'\] that the "
        r"Teleop Session Server's OpenXR session does not have$",
        extensions=[e for e in SERVED_EXTENSIONS if e != "XR_EXT_hand_tracking"],
    )


# ============================================================================
# Protocol and session fields
# ============================================================================


@pytest.mark.parametrize("protocol", [2, 0, True, "1", None])
def test_refuses_another_protocol(protocol):
    _refused(
        _hello(protocol=protocol),
        rf"^Remote session protocol {re.escape(repr(protocol))} is not supported: "
        r"this Teleop Session Server \(Isaac Capture .*\) speaks protocol 1$",
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{", "HELLO is not valid JSON"),
        ("[1]", "HELLO is not a JSON object"),
        (_hello(app_name=3), "app_name must be a string"),
        (_hello(trackers={}), "trackers must be a list"),
        (_hello(topics=["0/head", 1]), "topics must be a list of strings"),
        (_hello(plugins="manus"), "plugins must be a list"),
    ],
)
def test_refuses_a_malformed_hello(text, match):
    _refused(text, match)


@pytest.mark.parametrize(
    ("rate", "served"), [(0.5, 1.0), (1, 1.0), (90, 90.0), (240.0, 240.0), (1e6, 240.0)]
)
def test_tick_rate_is_clamped(rate, served):
    assert _parse(_hello(tick_rate_hz=rate)).tick_rate_hz == served


@pytest.mark.parametrize("rate", [0, -1.0, float("nan"), float("inf"), "90", True])
def test_refuses_a_tick_rate_that_is_not_positive(rate):
    _refused(_hello(tick_rate_hz=rate), "tick_rate_hz must be a number above zero")


def test_missing_fields_take_defaults():
    request = _parse(json.dumps({"protocol": 1}))
    assert request.app_name == ""
    assert request.tick_rate_hz == 90.0
    assert (request.trackers, request.vendors, request.topics) == ([], [], [])


def test_refuses_a_hello_that_requests_plugins():
    plugin = {
        "plugin_name": "manus_hand_plugin",
        "plugin_root_id": "manus",
        "plugin_args": [],
        "required": True,
    }
    with pytest.raises(HelloError) as raised:
        _parse(_hello(plugins=[plugin]))
    assert str(raised.value) == (
        "The robot app requested plugins ['manus_hand_plugin'], but plugins are started on "
        "the Teleop Session Server machine (by hand or a script), not requested by the "
        "robot app"
    )


def test_an_empty_plugin_list_is_served():
    assert _parse(_hello(plugins=[])).trackers == []


# ============================================================================
# STATUS
# ============================================================================


def test_status_json_shape():
    headset = headset_status(
        SimpleNamespace(
            state=oxr.OpenXRProviderState.AVAILABLE,
            headset_state=oxr.OpenXRHeadsetState.CONNECTED,
            reason=oxr.OpenXRProviderReason.NONE,
        )
    )
    assert json.loads(status_json("waiting_for_headset", "m", headset)) == {
        "stage": "waiting_for_headset",
        "message": "m",
        "headset": {
            "state": "AVAILABLE",
            "headset_state": "CONNECTED",
            "reason": "NONE",
        },
    }
    assert json.loads(status_json("connecting_openxr")) == {
        "stage": "connecting_openxr",
        "message": "",
        "headset": {"state": "", "headset_state": "", "reason": ""},
    }


# ============================================================================
# Command line
# ============================================================================


def test_command_line_defaults():
    parser = server_main._build_parser()
    args = parser.parse_args([])
    assert parser.prog == "python -m isaaccapture.teleop_session_server"
    assert args.listen == ("0.0.0.0", 47600)
    assert args.ws_listen is None
    assert args.verbose is False
    assert args.placeholder_openxr is False
    assert args.launch_cloudxr_runtime is True
    assert args.accept_eula is False
    assert args.host_client is True
    help_text = parser.format_help()
    for option in (
        "--accept-eula",
        "--no-launch-cloudxr-runtime",
        "--cloudxr-install-dir",
        "--cloudxr-device-profile",
        "--cloudxr-env-config",
        "--no-host-client",
    ):
        assert option in help_text
    for option in ("--placeholder-openxr", "--launch-wss-proxy", "--plugin-dir"):
        assert option not in help_text


def test_plugin_directories_are_not_an_option(capsys):
    with pytest.raises(SystemExit) as raised:
        server_main._build_parser().parse_args(["--plugin-dir", "/opt/plugins"])
    assert raised.value.code == 2
    assert "--plugin-dir" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--listen", "tcp://127.0.0.1:0"], ("listen", ("127.0.0.1", 0))),
        (["--listen", "127.0.0.1"], ("listen", ("127.0.0.1", 47600))),
        (["--listen", "[::1]:5000"], ("listen", ("::1", 5000))),
        (["--listen", "tcp://Robot-Server:1/"], ("listen", ("robot-server", 1))),
        (["--ws-listen", "ws://0.0.0.0"], ("ws_listen", ("0.0.0.0", 47601))),
        (["--ws-listen", "localhost:9"], ("ws_listen", ("localhost", 9))),
    ],
)
def test_listen_addresses(argv, expected):
    args = server_main._build_parser().parse_args(argv)
    assert getattr(args, expected[0]) == expected[1]


@pytest.mark.parametrize(
    "argv",
    [
        ["--listen", "ws://0.0.0.0:1"],
        ["--listen", "tcp://0.0.0.0:70000"],
        ["--listen", "tcp://0.0.0.0:1/path"],
        ["--listen", "tcp://:1"],
        ["--listen", "tcp://user@host:1"],
        ["--ws-listen", "wss://0.0.0.0:1"],
        ["--ws-listen", "tcp://0.0.0.0:1"],
    ],
)
def test_refuses_bad_listen_addresses(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        server_main._build_parser().parse_args(argv)
    assert raised.value.code == 2
    assert argv[1] in capsys.readouterr().err
