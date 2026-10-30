"""The Viser 3D view: descriptor binding, the glove/controller/head transforms on asymmetric fixtures, freshness
and visibility, scene ownership against a recording fake server, and real Viser server start/stop.

Samples are encoded with the CaptureServer's own ``encode_tracking_frame`` and descriptors come from its
``tracking_stream_descriptor``, so the layout under test is the real ic_tracking_v1 contract.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from capture_server import protocol as p
from egocap_web import viser_scene as vs
from egocap_web.capture_preview import TrackingSlot
from egocap_web.config import (
    DEFAULT_DEVICE_PROFILE,
    DEVICE_MODEL_PROFILES,
    VisualizationConfig,
    VisualizationConfigError,
    visualization_from_env,
)

STREAM = p.tracking_stream_descriptor((p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS), 60.0)
IDENT = [0.0, 0.0, 0.0, 1.0]


def _blank() -> dict:
    return {f.name: np.zeros(f.shape, dtype=f.numpy_dtype) for f in p.IC_TRACKING_V1_FIELDS}


def _payload(**overrides) -> bytes:
    values = _blank()
    values.update(overrides)
    return p.encode_tracking_frame(values)


def _controller(side: str, *, aim=((0.0, 0.0, 0.0), IDENT), grip=((0.0, 0.0, 0.0), IDENT), valid=True) -> dict:
    pre = f"controller_{side}"
    return {
        f"{pre}.present": 1,
        f"{pre}.aim_position": aim[0],
        f"{pre}.aim_orientation": aim[1],
        f"{pre}.aim_is_valid": int(valid),
        f"{pre}.grip_position": grip[0],
        f"{pre}.grip_orientation": grip[1],
        f"{pre}.grip_is_valid": int(valid),
    }


def _hand(side: str, positions: np.ndarray, *, wrist_quat=IDENT, valid=None, present=True) -> dict:
    pre = f"hand_{side}"
    orientations = np.tile(IDENT, (26, 1)).astype(np.float32)
    orientations[1] = wrist_quat
    return {
        f"{pre}.present": int(present),
        f"{pre}.joint_positions": positions,
        f"{pre}.joint_orientations": orientations,
        f"{pre}.joint_valid": np.ones(26) if valid is None else valid,
    }


def _joints() -> np.ndarray:
    """An asymmetric 26-joint hand: every joint at a distinct, non-axis-aligned place."""
    rng = np.random.default_rng(7)
    return (rng.normal(size=(26, 3)) * 0.05 + [0.3, 1.1, -0.4]).astype(np.float32)


def _frame(**overrides) -> vs.SceneFrame:
    return vs.build_frame(vs.decode(_payload(**overrides), vs.bind_layout(STREAM)))


def _matrix(pose: vs.NodePose) -> np.ndarray:
    w, x, y, z = pose.wxyz
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    out[:3, 3] = pose.position
    return out


# --- descriptor binding and decoding ---------------------------------------------------------------------------


def test_the_real_descriptor_binds_and_every_needed_field_is_found():
    layout = vs.bind_layout(STREAM)
    assert layout.payload_size == p.IC_TRACKING_V1_PAYLOAD_SIZE
    assert layout.fields["hand_left.joint_positions"][2] == (26, 3)


@pytest.mark.parametrize(
    "change",
    [
        {"quaternion_order": 99},
        {"position_unit": 99},
        {"axes": 99},
        {"hand_joint_order": 99},
        {"reference_space": ""},
        {"fields": ()},
        {"payload_size": 100},
    ],
)
def test_a_descriptor_this_view_cannot_render_faithfully_is_refused(change):
    with pytest.raises(vs.UnsupportedTracking):
        vs.bind_layout(dataclasses.replace(STREAM, **change))


def test_a_field_with_the_wrong_dtype_or_shape_is_refused():
    bad = tuple(
        dataclasses.replace(f, shape=(25, 3)) if f.name == "hand_left.joint_positions" else f
        for f in STREAM.fields
    )
    with pytest.raises(vs.UnsupportedTracking, match="hand_left.joint_positions"):
        vs.bind_layout(dataclasses.replace(STREAM, fields=bad))


def test_a_payload_of_the_wrong_length_is_refused():
    with pytest.raises(vs.UnsupportedTracking, match="bytes"):
        vs.decode(b"\0" * 10, vs.bind_layout(STREAM))


# --- head, controllers and gloves -------------------------------------------------------------------------------


def test_head_follows_translation_and_orientation_with_one_xyzw_to_wxyz_reorder():
    quat = Rotation.from_euler("xyz", [20, 50, -70], degrees=True).as_quat()  # xyzw
    frame = _frame(
        **{"head.present": 1, "head.is_valid": 1, "head.position": [0.1, 1.6, -0.3], "head.orientation": quat}
    )
    assert frame.head.position == pytest.approx((0.1, 1.6, -0.3))
    assert frame.head.wxyz == pytest.approx((quat[3], quat[0], quat[1], quat[2]))


def test_an_invalid_nonfinite_or_zero_quaternion_head_is_hidden_and_a_head_at_the_origin_is_not():
    head = {"head.present": 1, "head.is_valid": 1, "head.position": [0, 0, 0], "head.orientation": IDENT}
    assert _frame(**head).head is not None  # a valid pose at the origin is legitimate
    assert _frame(**{**head, "head.is_valid": 0}).head is None
    assert _frame(**{**head, "head.present": 0}).head is None
    assert _frame(**{**head, "head.position": [np.nan, 0, 0]}).head is None
    assert _frame(**{**head, "head.orientation": [0, 0, 0, 0]}).head is None


def test_the_controller_mesh_follows_the_grip_pose_not_the_aim_pose():
    grip = ([0.2, 1.0, -0.1], Rotation.from_euler("z", 90, degrees=True).as_quat())
    frame = _frame(**_controller("left", aim=([5.0, 5.0, 5.0], IDENT), grip=grip))
    assert frame.controllers["left"].position == pytest.approx(grip[0])
    assert frame.controllers["right"] is None


def test_the_glove_is_the_reported_tracking_space_joints_unchanged_at_the_origin_frame():
    joints = _joints()
    glove = _frame(**_hand("left", joints)).gloves["left"]
    assert glove.points == pytest.approx(joints.astype(np.float64))  # exactly as reported


def test_a_glove_does_not_depend_on_the_controller_pose_or_its_validity():
    joints = _joints()
    plain = _frame(**_hand("right", joints)).gloves["right"]
    with_controller = _frame(
        **_controller("right", grip=([5.0, 5.0, 5.0], Rotation.from_euler("y", 90, degrees=True).as_quat())),
        **_hand("right", joints),
    ).gloves["right"]
    invalid_controller = _frame(**_controller("right", valid=False), **_hand("right", joints)).gloves["right"]
    assert with_controller.points == pytest.approx(plain.points)
    assert invalid_controller.points == pytest.approx(plain.points)


def test_a_valid_wrist_at_the_origin_is_drawn_and_an_invalid_wrist_still_draws_the_other_joints():
    joints = _joints()
    joints[1] = 0.0
    assert _frame(**_hand("left", joints)).gloves["left"] is not None  # an origin joint is legitimate
    valid = np.ones(26)
    valid[1] = 0
    glove = _frame(**_hand("left", _joints(), valid=valid)).gloves["left"]
    assert len(glove.points) == 25
    assert len(glove.segments) == len(vs.HAND_BONES) - 5  # the five bones that start at the invalid wrist


def test_an_absent_hand_or_a_hand_with_no_valid_joint_is_hidden_not_pinned_at_the_origin():
    assert _frame(**_hand("left", _joints(), present=False)).gloves["left"] is None
    assert _frame(**_hand("left", _joints(), valid=np.zeros(26))).gloves["left"] is None
    assert _frame().gloves["left"] is None  # nothing reported at all


def test_non_finite_joints_are_not_drawn():
    joints = _joints()
    joints[10] = np.nan
    glove = _frame(**_hand("left", joints)).gloves["left"]
    assert len(glove.points) == 25 and np.isfinite(glove.points).all()


def test_the_mock_hands_point_away_from_the_body_in_the_scene():
    """The mock operator faces -Z with its fingers pointing forward; the drawn hand must keep that direction."""
    from capture_server.mock import mock_isaac_capture as mock

    _, left, right = mock._Operator("demo", 0).core(3.0)
    for hand, side in ((left, "left"), (right, "right")):
        values = _blank()
        values.update(_hand(side, hand.joints[:, :3].astype(np.float32)))
        glove = _frame(**values).gloves[side]
        wrist, middle_tip = glove.points[1], glove.points[15]
        assert middle_tip[2] < wrist[2] - 0.05  # forward is -Z


def test_invalid_joints_are_not_drawn_and_bones_need_both_endpoints_valid():
    valid = np.ones(26)
    valid[[10, 12]] = 0  # index tip, middle proximal
    glove = _frame(**_hand("left", _joints(), valid=valid)).gloves["left"]
    assert len(glove.points) == 24
    assert (
        len(glove.segments) == len(vs.HAND_BONES) - 3
    )  # index distal-tip, wrist-middle proximal, middle prox-inter


def test_the_hand_bone_table_is_the_five_openxr_chains_from_the_wrist():
    assert len(vs.HAND_BONES) == 24 and all(0 <= a < 26 and 0 <= b < 26 for a, b in vs.HAND_BONES)
    assert (1, 2) in vs.HAND_BONES and (1, 6) in vs.HAND_BONES and (9, 10) in vs.HAND_BONES
    assert not any(0 in bone for bone in vs.HAND_BONES)  # the palm is a point, not a bone endpoint


def test_each_object_is_judged_alone():
    frame = _frame(
        **_controller("left"),
        **_hand("left", _joints()),
        **_controller("right", valid=False),
        **{"head.present": 1, "head.is_valid": 1, "head.position": [0, 1.5, 0], "head.orientation": IDENT},
    )
    assert frame.head and frame.gloves["left"] and frame.controllers["left"]
    assert frame.gloves["right"] is None and frame.controllers["right"] is None


# --- the sampler against a recording fake server ----------------------------------------------------------------


class FakeHandle:
    def __init__(self, name):
        self.name, self.visible, self.position, self.wxyz, self.removed = name, True, None, None, False

    def remove(self):
        self.removed = True


class FakeScene:
    def __init__(self):
        self.added = []

    def set_up_direction(self, direction):
        self.up = direction

    def _add(self, kind, name, **kwargs):
        handle = FakeHandle(name)
        handle.visible = kwargs.get("visible", True)
        self.added.append((kind, name))
        return handle

    def add_frame(self, name, **kwargs):
        return self._add("frame", name, **kwargs)

    def add_glb(self, name, data, **kwargs):
        return self._add("glb", name, **kwargs)

    def add_point_cloud(self, name, points, **kwargs):
        return self._add("points", name, **kwargs)

    def add_line_segments(self, name, points, **kwargs):
        return self._add("lines", name, **kwargs)


class FakeServer:
    def __init__(self, port=4321):
        self.scene, self.port, self.stopped, self.depth = FakeScene(), port, False, 0

    @contextmanager
    def atomic(self):
        self.depth += 1
        yield
        self.depth -= 1

    def get_port(self):
        return self.port

    def stop(self):
        self.stopped = True


def _scene(monitor, *, config=None):
    scene = vs.VisualizationScene(monitor, config or VisualizationConfig(enabled=True))
    server = FakeServer()
    scene._server = server
    scene._build_nodes(server, {"head": b"glb", "left": b"glb", "right": b"glb"})
    return scene, server


async def _until(condition, limit_s: float = 2.0) -> None:
    """Yield to the loop until ``condition()`` holds, failing the test rather than sleeping a fixed time."""
    import asyncio

    deadline = asyncio.get_running_loop().time() + limit_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


def _monitor(*, payload=None, sequence=1, received=100.0, terminal=False, stream=STREAM, started=None):
    slot = None if payload is None else TrackingSlot(sequence, 0, 0, payload, received)
    return SimpleNamespace(
        snapshot=lambda: SimpleNamespace(terminal=terminal),
        tracking_descriptor=stream,
        preview=SimpleNamespace(tracking=lambda: slot),
        native=SimpleNamespace(started=started if started is not None else object()),
    )


def _visible(scene) -> dict[str, bool]:
    return {key: handle.visible for key, handle in scene._handles.items()}


def _good_payload() -> bytes:
    return _payload(
        **_controller("left"),
        **_hand("left", _joints()),
        **{"head.present": 1, "head.is_valid": 1, "head.position": [0, 1.5, 0], "head.orientation": IDENT},
    )


def test_every_node_starts_hidden_and_the_scene_is_y_up():
    scene, server = _scene(_monitor())
    assert not any(_visible(scene).values())
    assert server.scene.up == "+y"
    assert ("glb", "/tracking/head/model") in server.scene.added


def test_a_fresh_valid_sample_shows_exactly_the_valid_objects_inside_one_atomic_update():
    scene, server = _scene(_monitor(payload=_good_payload()))
    scene._tick(100.5)
    assert _visible(scene) == {
        "head": True,
        "controller_left": True,
        "controller_right": False,
        "glove_left": True,
        "glove_right": False,
    }
    assert server.depth == 0 and ("points", "/tracking/left_glove/joints") in server.scene.added


def test_a_stale_sample_hides_everything_even_though_its_sequence_did_not_change():
    monitor = _monitor(payload=_good_payload())
    scene, _ = _scene(monitor)
    scene._tick(100.5)
    scene._tick(103.0)  # 3 s after receipt
    assert not any(_visible(scene).values()) and "stale" in scene.projection("localhost")["diagnostic"]


def test_an_unchanged_fresh_sample_is_not_recomputed_but_a_new_sequence_or_attempt_is():
    started = object()
    monitor = _monitor(payload=_good_payload(), started=started)
    scene, server = _scene(monitor)
    scene._tick(100.1)
    count = len(server.scene.added)
    scene._tick(100.2)
    assert len(server.scene.added) == count  # same attempt, same sequence
    monitor.preview.tracking = lambda: TrackingSlot(
        1, 0, 0, _good_payload(), 100.2
    )  # a restarted stream, seq reused
    monitor.native = SimpleNamespace(started=object())
    scene._tick(100.3)
    assert len(server.scene.added) > count


def test_capture_ending_clears_the_scene_immediately():
    monitor = _monitor(payload=_good_payload())
    scene, _ = _scene(monitor)
    scene._tick(100.1)
    monitor.snapshot = lambda: SimpleNamespace(terminal=True)
    scene._tick(100.2)
    assert not any(_visible(scene).values()) and scene.projection("localhost")["diagnostic"] == "capture ended"


def test_a_replaced_descriptor_hides_the_scene_and_an_unsupported_one_says_why():
    monitor = _monitor(payload=_good_payload())
    scene, _ = _scene(monitor)
    scene._tick(100.1)
    monitor.tracking_descriptor = dataclasses.replace(STREAM, quaternion_order=99)
    scene._tick(100.2)
    assert not any(_visible(scene).values()) and "quaternion order" in scene.projection("localhost")["diagnostic"]
    assert (
        scene.projection("localhost")["status"] == "ready"
    )  # the server is fine; tracking is what is unsupported


def test_before_any_tracking_the_server_is_ready_and_says_it_is_waiting():
    scene, _ = _scene(_monitor())
    scene._tick(100.0)
    assert (
        scene.projection("localhost")["status"] == "ready"
        and scene.projection("localhost")["diagnostic"] == "waiting for tracking"
    )


def test_a_geometry_without_bones_leaves_no_old_bones_behind():
    scene, server = _scene(_monitor(payload=_good_payload()))
    scene._tick(100.1)
    first = list(scene._geometry["left"])
    valid = np.zeros(26)
    valid[1] = 1  # only the wrist is valid: a point, no bones
    only_wrist = _payload(**_controller("left"), **_hand("left", _joints(), valid=valid))
    scene._monitor.preview.tracking = lambda: TrackingSlot(2, 0, 0, only_wrist, 100.2)
    scene._tick(100.2)
    assert all(h.removed for h in first) and len(scene._geometry["left"]) == 1


def test_a_failed_redraw_is_retried_for_the_same_sample_rather_than_marked_done(monkeypatch):
    scene, _ = _scene(_monitor(payload=_good_payload()))
    calls = []
    real = scene._apply

    def flaky(frame):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("apply failed")
        real(frame)

    monkeypatch.setattr(scene, "_apply", flaky)
    with pytest.raises(RuntimeError):
        scene._tick(100.1)
    scene._tick(100.2)  # same sequence, same attempt: it must draw, not skip as "already processed"
    assert len(calls) == 2 and scene._handles["head"].visible


async def test_an_update_error_hides_the_scene_reports_unavailable_once_and_ends_the_sampler(caplog):
    monitor = _monitor(payload=_good_payload())
    scene, _ = _scene(monitor)
    scene._tick(100.1)
    monitor.preview.tracking = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    await scene._run()  # returns: no retry loop over a permanent error
    info = scene.projection("localhost")
    assert info["status"] == "unavailable" and "boom" in info["diagnostic"]
    assert not any(_visible(scene).values())
    assert sum("3D view update failed" in r.getMessage() for r in caplog.records) == 1


async def test_stop_settles_a_startup_whose_caller_was_cancelled_and_stops_the_server_it_built():
    import asyncio
    import threading
    import time

    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    server, built, entered = FakeServer(), threading.Event(), threading.Event()

    def slow_build(assets):
        entered.set()
        time.sleep(0.2)
        scene._server = server
        built.set()

    scene._read_assets = lambda: {"head": None, "left": None, "right": None}
    scene._build_server = slow_build
    caller = asyncio.create_task(scene.start())
    await _until(entered.is_set)
    caller.cancel()  # the caller gives up while the server is still being created
    await asyncio.gather(caller, return_exceptions=True)
    await scene.stop()
    assert built.is_set() and server.stopped and scene._server is None


async def test_a_stop_that_outlasts_its_limit_is_reported_and_stays_owned_until_it_finishes(monkeypatch):
    import time

    monkeypatch.setattr(vs, "STOP_TIMEOUT_S", 0.05)
    scene, server = _scene(_monitor())
    server.stop = lambda: (time.sleep(0.3), setattr(server, "stopped", True))
    await scene.stop()
    assert scene._server is server and "did not stop" in scene.projection("localhost")["diagnostic"]
    monkeypatch.setattr(vs, "STOP_TIMEOUT_S", 2.0)
    await scene.stop()  # awaits the same retained stop
    assert server.stopped and scene._server is None


# --- projection and URL ----------------------------------------------------------------------------------------


def test_a_disabled_feature_advertises_disabled_and_never_starts():
    import asyncio

    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=False))
    asyncio.run(scene.start())
    assert scene.projection("localhost") == {"status": "disabled", "url": None, "diagnostic": None}
    assert scene._server is None


@pytest.mark.parametrize(
    ("config", "host", "expected"),
    [
        (VisualizationConfig(host="0.0.0.0"), "lab-pc.local", "http://lab-pc.local:4321/"),
        (VisualizationConfig(host="0.0.0.0"), "2001:db8::5", "http://[2001:db8::5]:4321/"),
        (VisualizationConfig(host="127.0.0.1"), "localhost", "http://localhost:4321/"),
        (
            VisualizationConfig(host="127.0.0.1", public_url="https://x.example/viser/"),
            "other",
            "https://x.example/viser/",
        ),
    ],
)
def test_the_advertised_url_uses_the_bound_port_and_a_browser_reachable_host(config, host, expected):
    assert vs.browser_url(config, 4321, host) == (expected, None)


@pytest.mark.parametrize("host", [None, "lab-pc.local"])
def test_a_loopback_only_server_or_unknown_host_is_not_advertised_to_a_remote_browser(host):
    url, why = vs.browser_url(VisualizationConfig(host="127.0.0.1"), 4321, host)
    assert url is None and why


# --- one real Viser server ---------------------------------------------------------------------------------------


async def test_a_real_server_starts_on_its_port_loads_the_shipped_models_and_stops(unused_tcp_port):
    pytest.importorskip("viser")
    config = VisualizationConfig(enabled=True, port=unused_tcp_port)
    scene = vs.VisualizationScene(_monitor(payload=_good_payload(), received=0.0), config)
    await scene.start()
    try:
        info = scene.projection("localhost")
        assert info["status"] == "ready" and info["url"] == f"http://localhost:{unused_tcp_port}/"
        assert "model" not in (info["diagnostic"] or "")  # the shipped head and controller models all loaded
        scene._tick(0.1)  # a real scene accepts the same updates the fake recorded
    finally:
        await scene.stop()
    assert scene.projection("localhost")["status"] == "unavailable"


async def test_a_missing_model_leaves_a_coordinate_frame_and_a_note_not_a_failure(unused_tcp_port, tmp_path):
    pytest.importorskip("viser")
    config = VisualizationConfig(enabled=True, port=unused_tcp_port, head_model=tmp_path / "absent.glb")
    scene = vs.VisualizationScene(_monitor(), config)
    await scene.start()
    try:
        info = scene.projection("localhost")
        assert (
            info["status"] == "ready" and "head" in info["diagnostic"] and "coordinate frame" in info["diagnostic"]
        )
    finally:
        await scene.stop()


async def test_a_server_that_cannot_start_is_unavailable_and_the_app_keeps_running(monkeypatch):
    viser = pytest.importorskip("viser")

    def failing(**kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(viser, "ViserServer", failing)
    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    await scene.start()  # must not raise
    info = scene.projection("localhost")
    assert info["status"] == "unavailable" and "address already in use" in info["diagnostic"]
    await scene.stop()


def test_the_operator_settings_come_from_the_environment_with_safe_defaults(monkeypatch):
    from egocap_web.config import visualization_from_env

    for name in ("ENABLED", "HOST", "PORT", "PUBLIC_URL", "HEAD_MODEL", "HEAD_MODEL_SCALE"):
        monkeypatch.delenv(f"EGOCAP_VISER_{name}", raising=False)
    default = visualization_from_env()
    assert default.enabled and default.host == "127.0.0.1" and default.head_model is None
    shipped = default.models_for(default.device)  # no explicit files: the default profile's shipped models
    assert default.device == "pico-4-ultra" and shipped.head.is_file() and shipped.left_controller.is_file()
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "0")
    assert not visualization_from_env().enabled  # an operator can still turn it off
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_HOST", "0.0.0.0")
    monkeypatch.setenv("EGOCAP_VISER_PORT", "9000")
    monkeypatch.setenv("EGOCAP_VISER_HEAD_MODEL", "/models/head.glb")
    monkeypatch.setenv("EGOCAP_VISER_HEAD_MODEL_SCALE", "0.5")
    configured = visualization_from_env()
    assert (configured.enabled, configured.host, configured.port) == (True, "0.0.0.0", 9000)
    assert str(configured.head_model) == "/models/head.glb" and configured.head_model_scale == 0.5


@pytest.mark.parametrize("value", ["0", " 0 ", "False", "NO", "off", " Off "])
def test_a_recognized_off_value_is_accepted_even_with_surrounding_whitespace_or_case(monkeypatch, value):
    from egocap_web.config import visualization_from_env

    monkeypatch.setenv("EGOCAP_VISER_ENABLED", value)
    assert not visualization_from_env().enabled


@pytest.mark.parametrize("value", ["", " ", "maybe", "2"])
def test_an_unrecognized_enabled_value_is_rejected_rather_than_silently_treated_as_on(monkeypatch, value):
    from egocap_web.config import visualization_from_env

    monkeypatch.setenv("EGOCAP_VISER_ENABLED", value)
    with pytest.raises(ValueError, match="EGOCAP_VISER_ENABLED"):
        visualization_from_env()


def test_the_real_app_advertises_the_3d_view_in_its_snapshot_and_keeps_serving_without_a_capture_server(
    monkeypatch, unused_tcp_port
):
    from fastapi.testclient import TestClient

    from egocap_web.app import app

    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_PORT", str(unused_tcp_port))
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/api/session", headers={"host": "localhost", "origin": "http://localhost"}
        ) as ws,
    ):
        snapshot = ws.receive_json()
        assert snapshot["visualization"] == {
            "status": "ready",
            "url": f"http://localhost:{unused_tcp_port}/",
            "device": "Pico 4 Ultra",
            "diagnostic": snapshot["visualization"]["diagnostic"],
        }
        assert client.get("/api/capture-status").status_code == 200


def test_the_real_app_advertises_a_disabled_3d_view_when_opted_out(monkeypatch):
    from fastapi.testclient import TestClient

    from egocap_web.app import app

    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "0")
    with (
        TestClient(app) as client,
        client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws,
    ):
        assert ws.receive_json()["visualization"] == {"status": "disabled", "url": None, "diagnostic": None}


async def test_a_stop_that_raises_keeps_the_server_owned_reports_it_and_can_be_retried():
    scene, server = _scene(_monitor())
    attempts = []

    def failing_then_ok():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("socket busy")
        server.stopped = True

    server.stop = failing_then_ok
    await scene.stop()
    assert scene._server is server and "failed while stopping" in scene.projection("localhost")["diagnostic"]
    await scene.stop()  # a later stop tries again instead of treating the first as a release
    assert server.stopped and scene._server is None and len(attempts) == 2


async def test_a_pending_startup_survives_a_cancelled_stop_and_a_later_stop_settles_it():
    import asyncio
    import threading

    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    server, release, entered = FakeServer(), threading.Event(), threading.Event()

    def blocked_build(assets):
        entered.set()
        release.wait(2)
        scene._server = server

    scene._read_assets = lambda: {"head": None, "left": None, "right": None}
    scene._build_server = blocked_build
    starter = asyncio.create_task(scene.start())
    await _until(entered.is_set)
    stopper = asyncio.create_task(scene.stop())
    await _until(lambda: scene._stopping)
    stopper.cancel()  # cancelling the stop must not cancel the shared startup
    await asyncio.gather(stopper, return_exceptions=True)
    assert not scene._startup.cancelled()
    release.set()
    await starter
    await scene.stop()
    assert server.stopped and scene._server is None


class _FakeMonitor:
    def __init__(self, *args, **kwargs):
        self.started = self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def set_recording(self, coordinator):
        """These tests exercise the real lifespan() end-to-end for Viser's
        own start/stop behavior; the real recording coordinator is also
        constructed for real along that path, but these tests have nothing
        to assert about it -- just accept the attachment like the real
        monitor does."""


async def test_the_lifespan_stops_capture_even_when_the_3d_view_fails_to_stop_or_start_is_cancelled(monkeypatch):
    import asyncio

    events = []

    class ExplodingScene:
        def __init__(self, monitor, config):
            self.monitor = monitor

        async def start(self):
            raise asyncio.CancelledError  # the app is being shut down while the view is still starting

        async def stop(self):
            events.append("scene stop")
            raise RuntimeError("scene refused to stop")

    import egocap_web.app as app_module

    created = []
    monkeypatch.setattr(
        app_module, "CaptureStatusMonitor", lambda *a, **k: created.append(_FakeMonitor()) or created[-1]
    )
    monkeypatch.setattr(app_module, "VisualizationScene", ExplodingScene)
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        async with app_module.lifespan(app_module.FastAPI()):
            pass
    assert events == ["scene stop"] and created[0].stopped  # capture is still stopped


async def test_invalid_3d_view_settings_leave_it_unavailable_and_capture_running(monkeypatch):
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_PORT", "not-a-number")
    import egocap_web.app as app_module

    monkeypatch.setattr(app_module, "CaptureStatusMonitor", _FakeMonitor)
    app = app_module.FastAPI()
    async with app_module.lifespan(app):
        info = app.state.visualization.projection("localhost")
        assert info["status"] == "unavailable" and "invalid" in info["diagnostic"]
        assert app.state.monitor.started
    assert app.state.monitor.stopped


def test_a_connecting_browser_is_framed_on_the_head_and_hands():
    client = SimpleNamespace(camera=SimpleNamespace(position=None, look_at=None))
    vs._frame_scene(client)
    assert (
        client.camera.position == vs.INITIAL_CAMERA_POSITION and client.camera.look_at == vs.INITIAL_CAMERA_LOOK_AT
    )


def _blocked_start(scene, server, release, entered):
    scene._read_assets = lambda: {"head": None, "left": None, "right": None}

    def blocked_build(assets):
        entered.set()
        release.wait(5)
        scene._server = server

    scene._build_server = blocked_build


async def test_a_startup_that_finishes_after_stop_gave_up_releases_its_server_and_never_samples(monkeypatch):
    import asyncio
    import threading

    monkeypatch.setattr(vs, "STOP_TIMEOUT_S", 0.05)
    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    server, release, entered = FakeServer(), threading.Event(), threading.Event()
    _blocked_start(scene, server, release, entered)
    starter = asyncio.create_task(scene.start())
    await _until(entered.is_set)
    await scene.stop()  # gives up after its bound; this is the app's one shutdown call
    assert "did not settle" in scene.projection("localhost")["diagnostic"]
    release.set()  # the late startup now finishes on its own
    await asyncio.gather(starter, scene._startup)
    assert server.stopped and scene._server is None and scene._sampler is None


async def test_the_app_does_not_wait_forever_for_a_slow_optional_startup(monkeypatch):
    import asyncio
    import threading

    monkeypatch.setattr(vs, "STARTUP_TIMEOUT_S", 0.05)
    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    server, release = FakeServer(), threading.Event()
    _blocked_start(scene, server, release, threading.Event())
    await asyncio.wait_for(scene.start(), 2)  # returns after its bound although the build is still running
    info = scene.projection("localhost")
    assert info["status"] == "unavailable" and info["diagnostic"] == "3D view is starting"
    release.set()
    await scene._startup  # the startup finishes in the background and the view comes up
    assert scene.projection("localhost")["status"] == "ready"
    await scene.stop()
    assert server.stopped


async def test_malformed_settings_are_ignored_while_the_3d_view_is_off(monkeypatch):
    """A stray bad port or scale must not make the page show an error panel for a view that's turned off."""
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "0")
    monkeypatch.setenv("EGOCAP_VISER_PORT", "")
    monkeypatch.setenv("EGOCAP_VISER_HEAD_MODEL_SCALE", "wide")
    import egocap_web.app as app_module

    monkeypatch.setattr(app_module, "CaptureStatusMonitor", _FakeMonitor)
    app = app_module.FastAPI()
    async with app_module.lifespan(app):
        assert app.state.visualization.projection("localhost") == {
            "status": "disabled",
            "url": None,
            "diagnostic": None,
        }


def test_a_model_file_that_is_not_a_glb_such_as_an_lfs_pointer_is_treated_as_missing(tmp_path):
    pointer = tmp_path / "head.glb"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    shipped = DEVICE_MODEL_PROFILES["pico-4"].left_controller
    # The generic profile has no controller models, so only the explicitly named left file is read.
    config = VisualizationConfig(enabled=True, device="generic", head_model=pointer, left_controller_model=shipped)
    scene = vs.VisualizationScene(_monitor(), config)
    assets = scene._read_assets()
    assert assets["head"] is None and assets["right"] is None and assets["left"][:4] == b"glTF"
    assert "head" in scene._model_note and "coordinate frames" in scene._model_note


async def test_a_missing_viser_package_names_how_to_install_it(monkeypatch):
    scene = vs.VisualizationScene(_monitor(), VisualizationConfig(enabled=True))
    monkeypatch.setitem(__import__("sys").modules, "viser", None)  # makes `import viser` raise ModuleNotFoundError
    scene._read_assets = lambda: {"head": None, "left": None, "right": None}
    await scene.start()
    info = scene.projection("localhost")
    assert info["status"] == "unavailable" and "`uv sync`" in info["diagnostic"]
    # The sampler must never start against a server that was never built (self._server is None),
    # or it would hit None.atomic() once tracking flows.
    assert scene._sampler is None


# --- device models: which hardware is drawn ---------------------------------------------------------------------


def _devices(*descriptions):
    return [SimpleNamespace(type="headset", description=d) for d in descriptions]


@pytest.mark.parametrize(
    ("description", "profile"),
    [
        ("Pico 4 Ultra headset", "pico-4-ultra"),
        ("PICO-4U", "pico-4-ultra"),
        ("Pico 4 headset", "pico-4"),
        ("Pico headset", "pico-4"),
        ("Meta Quest 3", "generic"),
        ("OpenXR headset", None),
        ("", None),
    ],
)
def test_the_headset_description_selects_the_device_profile_most_specific_first(description, profile):
    assert vs.detect_device_profile(_devices(description)) == profile


def test_only_the_headset_entry_names_the_device():
    controllers = [SimpleNamespace(type="controller", description="Pico 4 Ultra left controller")]
    assert vs.detect_device_profile(controllers) is None
    assert vs.detect_device_profile(None) is None


def test_the_default_profile_is_the_station_headset_and_its_models_ship():
    assert DEFAULT_DEVICE_PROFILE == "pico-4-ultra"
    for profile, models in DEVICE_MODEL_PROFILES.items():
        for path in (models.head, models.left_controller, models.right_controller):
            if path is not None:
                assert path.read_bytes()[:4] == b"glTF", f"{profile}: {path} is not a binary glTF"


def test_an_explicit_head_model_overrides_every_profile_but_controllers_follow_the_profile(tmp_path):
    config = VisualizationConfig(enabled=True, head_model=tmp_path / "own.glb")
    for profile in DEVICE_MODEL_PROFILES:
        models = config.models_for(profile)
        assert models.head == tmp_path / "own.glb"
        assert models.left_controller == DEVICE_MODEL_PROFILES[profile].left_controller


@pytest.mark.parametrize("raw", ["pico-4-ultra", " Pico-4-Ultra ", "generic", "pico-4"])
def test_the_device_profile_env_is_case_and_whitespace_tolerant(monkeypatch, raw):
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_DEVICE", raw)
    assert visualization_from_env().device == raw.strip().lower()


def test_an_unknown_device_profile_stops_startup_naming_the_valid_ones(monkeypatch):
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_DEVICE", "quest-pro")
    with pytest.raises(VisualizationConfigError, match="quest-pro.*pico-4-ultra"):
        visualization_from_env()


def test_the_scene_swaps_to_the_device_the_status_names_and_reports_it():
    monitor = _monitor(payload=_good_payload())
    scene, server = _scene(monitor)
    scene._assets["generic"] = {"head": b"glb-generic", "left": None, "right": None}
    first_models = dict(scene._model_handles)
    assert scene.projection("localhost")["device"] == "Pico 4 Ultra"
    monitor.snapshot = lambda: SimpleNamespace(
        terminal=False, status=SimpleNamespace(devices=_devices("Meta Quest 3 headset"))
    )
    scene._tick(100.1)
    # The old meshes are removed, the generic headset is added, and controllers fall back to their frame axes.
    assert all(handle.removed for handle in first_models.values())
    assert set(scene._model_handles) == {"head"} and first_models["head"] is not scene._model_handles["head"]
    assert scene.projection("localhost")["device"] == "generic headset"
    # Tracking still shows on the same frames in the same tick.
    assert _visible(scene)["head"] and _visible(scene)["controller_left"]


def test_a_status_that_names_no_device_keeps_the_configured_profile():
    monitor = _monitor(payload=_good_payload())
    scene, _ = _scene(monitor)
    monitor.snapshot = lambda: SimpleNamespace(
        terminal=False, status=SimpleNamespace(devices=_devices("OpenXR headset"))
    )
    before = dict(scene._model_handles)
    scene._tick(100.1)
    assert scene._model_handles == before and scene.projection("localhost")["device"] == "Pico 4 Ultra"


def test_reading_assets_loads_every_profile_and_notes_only_the_configured_ones_missing(tmp_path):
    config = VisualizationConfig(enabled=True, device="generic", head_model=tmp_path / "missing.glb")
    scene = vs.VisualizationScene(_monitor(), config)
    current = scene._read_assets()
    assert set(scene._assets) == set(DEVICE_MODEL_PROFILES)
    assert current == {"head": None, "left": None, "right": None}
    assert scene.projection  # the note is recorded for the configured profile
    assert "head" in (scene._model_note or "")
    # The other profiles' controller files were still read, ready for a runtime switch.
    assert scene._assets["pico-4-ultra"]["left"] is not None
