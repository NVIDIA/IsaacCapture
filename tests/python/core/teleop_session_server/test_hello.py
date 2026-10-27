# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server's HELLO handling, STATUS JSON and command line, in process."""

import json
import os
import re
import sys
from pathlib import Path
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
from isaaccapture.plugin_manager import (
    PluginCrashException,
    PluginManager,
    ProcessReason,
    ProcessState,
)
from isaaccapture.teleop_session_manager import BUNDLED_PLUGINS_DIR, _remote
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server._session import (
    HelloError,
    PluginRequest,
    headset_status,
    parse_hello,
    plugin_status,
    status_json,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

PLUGIN_DIRS = [Path("/srv/plugins")]


class _Plugins:
    """PluginManager stand-in that discovers the given names."""

    def __init__(self, *names):
        self._names = list(names)

    def get_plugin_names(self):
        return list(self._names)


def _hello(**fields):
    hello = {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "Robot",
        "tick_rate_hz": 90.0,
        "trackers": [],
        "topics": [],
        "plugins": [],
    }
    hello.update(fields)
    return json.dumps(hello)


def _tracker(type_name, name="0", params=None, vendor=None):
    return {"name": name, "type": type_name, "params": params or {}, "vendor": vendor}


def _parse(text, placeholder=False, discovered=()):
    return parse_hello(
        text,
        placeholder=placeholder,
        plugin_manager=_Plugins(*discovered),
        plugin_dirs=PLUGIN_DIRS,
    )


def _refused(text, match, **kwargs):
    with pytest.raises(HelloError, match=match):
        _parse(text, **kwargs)


# ============================================================================
# Trackers
# ============================================================================


def test_builds_the_trackers_a_robot_app_describes():
    """The robot side's HELLO (teleop_session_manager._remote) round-trips into matching trackers."""
    body = FullBodyTracker()
    sensors = JointSe3PoseTracker("manus_sensors_left", 2048)
    robot_trackers = [HeadTracker(), HandTracker(), ControllerTracker(), body, sensors]
    vendor = TrackerVendor("body.pico-xr", {"mode": "full"})
    descriptors = [
        _remote.describe_tracker(
            tracker, str(i), vendor if tracker is body else None, "test"
        )
        for i, tracker in enumerate(robot_trackers)
    ]
    hello = _remote.build_hello(
        app_name="Robot",
        tick_rate_hz=120.0,
        trackers=descriptors,
        topics=["0/head", "4/joint_se3_pose_tracked"],
        plugins=[],
    )

    request = _parse(hello)

    assert request.app_name == "Robot"
    assert request.isaaccapture_version == isaaccapture.__version__
    assert request.tick_rate_hz == 120.0
    assert request.topics == ["0/head", "4/joint_se3_pose_tracked"]
    assert [name for _, name in request.trackers] == ["0", "1", "2", "3", "4"]
    assert [type(t) for t, _ in request.trackers] == [type(t) for t in robot_trackers]
    server_sensors = request.trackers[4][0]
    assert server_sensors.collection_id == "manus_sensors_left"
    assert server_sensors.max_flatbuffer_size == 2048
    [(tracker, server_vendor)] = request.vendors
    assert tracker is request.trackers[3][0]
    assert (server_vendor.id, dict(server_vendor.params)) == (
        "body.pico-xr",
        {"mode": "full"},
    )
    assert request.plugins == []


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
    assert request.app_name == "TeleopSessionServer"
    assert request.tick_rate_hz == 90.0
    assert (request.trackers, request.topics, request.plugins) == ([], [], [])


# ============================================================================
# Plugins
# ============================================================================


def _plugin(name="manus_hand_plugin", root_id="manus", args=(), required=True):
    return {
        "plugin_name": name,
        "plugin_root_id": root_id,
        "plugin_args": list(args),
        "required": required,
    }


def test_requested_plugins_the_server_has_are_kept_in_order():
    request = _parse(
        _hello(
            plugins=[
                _plugin("b", "rb", ["--x=1"]),
                _plugin("a", "ra", required=False),
            ]
        ),
        discovered=("a", "b"),
    )
    assert request.plugins == [
        PluginRequest("b", "rb", ["--x=1"], True),
        PluginRequest("a", "ra", [], False),
    ]


def test_a_missing_required_plugin_is_refused_like_teleop_session_does():
    with pytest.raises(HelloError) as raised:
        _parse(_hello(plugins=[_plugin("missing")]), discovered=("other",))
    text = str(raised.value)
    assert text == (
        "Required plugin 'missing' was not discovered on the Teleop Session Server in "
        "['/srv/plugins']. Discovered plugins: ['other']. Install the plugin there, or "
        "start the server with --plugin-dir."
    )
    # The pattern .github/workflows/build-ubuntu.yml fails a bring-up on.
    assert re.search(r"Required plugin .* was not discovered", text)


def test_a_missing_optional_plugin_is_skipped():
    request = _parse(_hello(plugins=[_plugin("missing", required=False)]))
    assert request.plugins == []


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("manus", r"plugins\[0\] is not a JSON object"),
        (_plugin(name=""), "needs a plugin_name and plugin_root_id"),
        (_plugin(root_id=None), "needs a plugin_name and plugin_root_id"),
        (_plugin(args=[1]), "plugin_args must be a list of strings"),
        (_plugin(required="yes"), "required must be true or false"),
    ],
)
def test_refuses_malformed_plugin_requests(entry, match):
    _refused(_hello(plugins=[entry]), match, discovered=("manus_hand_plugin",))


def test_a_later_plugin_directory_wins_a_plugin_name(tmp_path):
    for directory in ("first", "second"):
        plugin_dir = tmp_path / directory / "probe"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: probe\ndescription: {directory}\ncommand: ./probe\n"
        )
    manager = PluginManager([str(tmp_path / "first"), str(tmp_path / "second")])
    assert manager.get_plugin_info("probe").description == "second"


def test_plugin_directories_are_bundled_then_environment_then_command_line(
    tmp_path, monkeypatch
):
    env_a, env_b, extra = (tmp_path / name for name in ("a", "b", "c"))
    monkeypatch.setenv(
        "ISAAC_TELEOP_PLUGIN_PATH", os.pathsep.join([str(env_a), "", str(env_b)])
    )
    # A directory listed again keeps only its last position.
    assert server_main.plugin_dirs([extra, env_a]) == [
        path.resolve() for path in (BUNDLED_PLUGINS_DIR, env_b, extra, env_a)
    ]


# ============================================================================
# STATUS
# ============================================================================


def _snapshot(state, reason, pid=42, exit_code=None, term_signal=None, error=""):
    return SimpleNamespace(
        state=state,
        reason=reason,
        pid=pid,
        exit_code=exit_code,
        term_signal=term_signal,
        error_code=None,
        error=error,
    )


class _Context:
    def __init__(self, snapshot=None, error=None):
        self._snapshot = snapshot
        self._error = error

    def get_process_snapshot(self):
        if self._error is not None:
            raise self._error
        return self._snapshot


class _Session:
    """RemoteSession stand-in for teleop_session_manager._remote.RemoteStatus."""

    def __init__(self, text):
        self.text = text

    def latest_status(self):
        return self.text


PROBE = PluginRequest("probe_plugin", "probe", [], True)


def _robot_view(entries):
    """How a robot app's TeleopSession sees these plugin entries."""
    status = _remote.RemoteStatus(_Session(status_json("running", "", entries)))
    return _remote.RemotePluginContext(status, "probe_plugin", "probe")


def test_running_plugin_status_passes_the_robot_health_check():
    entry = plugin_status(
        PROBE, _Context(_snapshot(ProcessState.RUNNING, ProcessReason.NONE))
    )
    assert entry == {
        "plugin_name": "probe_plugin",
        "plugin_root_id": "probe",
        "state": "RUNNING",
        "reason": "NONE",
        "pid": 42,
        "exit_code": None,
        "term_signal": None,
        "error_code": None,
        "error": "",
    }
    view = _robot_view([entry])
    view.check_health()
    assert view.get_process_snapshot().pid == 42


@pytest.mark.parametrize(
    ("snapshot", "match"),
    [
        (
            _snapshot(
                ProcessState.EXITED,
                ProcessReason.NONZERO_EXIT,
                exit_code=3,
                error="Plugin process unexpectedly exited with code 3",
            ),
            "exited with code 3",
        ),
        (
            _snapshot(ProcessState.EXITED, ProcessReason.CLEAN_EXIT, exit_code=0),
            r"EXITED \(CLEAN_EXIT\)",
        ),
        (
            _snapshot(
                ProcessState.SIGNALED,
                ProcessReason.SIGNAL,
                term_signal=9,
                error="Plugin process crashed with signal 9 (Killed)",
            ),
            "crashed with signal 9",
        ),
    ],
)
def test_stopped_plugin_status_fails_the_robot_health_check(snapshot, match):
    entry = plugin_status(PROBE, _Context(snapshot))
    with pytest.raises(PluginCrashException, match=match):
        _robot_view([entry]).check_health()


def test_a_snapshot_that_cannot_be_read_is_a_failure():
    entry = plugin_status(PROBE, _Context(error=RuntimeError("waitpid failed")))
    assert (entry["state"], entry["error"]) == ("ERROR", "waitpid failed")
    with pytest.raises(PluginCrashException, match="waitpid failed"):
        _robot_view([entry]).check_health()


def test_status_json_shape():
    headset = headset_status(
        SimpleNamespace(
            state=oxr.OpenXRProviderState.AVAILABLE,
            headset_state=oxr.OpenXRHeadsetState.CONNECTED,
            reason=oxr.OpenXRProviderReason.NONE,
        )
    )
    assert json.loads(status_json("waiting_for_headset", "m", [], headset)) == {
        "stage": "waiting_for_headset",
        "message": "m",
        "plugins": [],
        "headset": {
            "state": "AVAILABLE",
            "headset_state": "CONNECTED",
            "reason": "NONE",
        },
    }
    assert json.loads(status_json("connecting_openxr"))["headset"] == {
        "state": "",
        "headset_state": "",
        "reason": "",
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
    assert args.plugin_dir == []
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
    assert "--placeholder-openxr" not in help_text
    assert "--launch-wss-proxy" not in help_text


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
