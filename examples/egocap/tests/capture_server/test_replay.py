# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay acquisition: a recorded episode played back through the production adapters.

The fixture builds a tiny synthetic episode in the recorder's on-disk layout (Zarr v3 groups plus an mp4v
MP4), so these tests need no real recording and no hardware.
"""

from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import zarr

from capture_server import protocol as p
from capture_server.__main__ import main
from capture_server.acquisition.tensor_indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)
from capture_server.client import CaptureClient
from capture_server.launch import build_replay_source_factory, replay_station_cameras
from capture_server.replay import (
    EpisodeFormatError,
    PlaybackClock,
    load_episode,
    replay_depthai,
    replay_isaac_capture,
    tracking_map,
)
from capture_server.server import ServerConfig
from tests.support.capture_server import start_capture_server, stop_capture_server

T0 = 1000.0
STEP = 0.05
CONTROLLER_ROWS = 6
HAND_ROWS = 5
FRAMES = 4
FRAME_COLORS_BGR = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255))
WIDTH, HEIGHT = 64, 48
MISSING_HAND_NODE = 7


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t


def _rot_y(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _pose(x: float, y: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = _rot_y(yaw_deg)
    pose[:3, 3] = (x, y, z)
    return pose


def head_pose(i: int) -> np.ndarray:
    return _pose(0.1 * i, 1.6, -0.2, yaw_deg=5.0 * i)


def left_pose(i: int) -> np.ndarray:
    return _pose(-0.3, 1.0 + 0.01 * i, -0.4, yaw_deg=30.0)


def right_pose(i: int) -> np.ndarray:
    # The last row is the recorder's untracked sentinel.
    return np.eye(4) if i == CONTROLLER_ROWS - 1 else _pose(0.3, 1.0, -0.4, yaw_deg=-30.0)


def recorded_left_hand(i: int) -> np.ndarray:
    nodes = np.tile(np.eye(4), (tracking_map.MANUS_NODE_COUNT, 1, 1))
    for node in range(1, tracking_map.MANUS_NODE_COUNT):
        nodes[node, :3, 3] = (0.01 * node, 0.002 * i, 0.05)
    nodes[MISSING_HAND_NODE] = 0.0
    return nodes


def _write(group, name: str, data: np.ndarray) -> None:
    array = group.create_array(name, shape=data.shape, dtype=data.dtype)
    array[...] = data


def build_episode(root: Path) -> Path:
    root.mkdir()
    group = zarr.open_group(str(root), mode="w")
    controllers = group.create_group("pico_controller")
    rows = np.arange(CONTROLLER_ROWS)
    _write(controllers, "capture_time", T0 + rows * STEP)
    _write(controllers, "receive_time", T0 + rows * STEP)
    _write(controllers, "head_pose", np.stack([head_pose(i) for i in rows]))
    _write(controllers, "left_pose", np.stack([left_pose(i) for i in rows]))
    _write(controllers, "right_pose", np.stack([right_pose(i) for i in rows]))
    _write(controllers, "left_trigger", np.linspace(0.0, 1.0, CONTROLLER_ROWS))
    _write(controllers, "right_trigger", np.zeros(CONTROLLER_ROWS))
    _write(controllers, "a_button", rows == 2)
    _write(controllers, "b_button", np.zeros(CONTROLLER_ROWS, dtype=bool))
    _write(controllers, "x_button", rows == 1)
    _write(controllers, "y_button", np.zeros(CONTROLLER_ROWS, dtype=bool))

    hands = group.create_group("manus_glove")
    hand_rows = np.arange(HAND_ROWS)
    _write(hands, "capture_time", T0 + 0.01 + hand_rows * STEP)
    _write(hands, "left_hand_pose", np.stack([recorded_left_hand(i) for i in hand_rows]))
    _write(hands, "right_hand_pose", np.tile(np.eye(4), (HAND_ROWS, tracking_map.MANUS_NODE_COUNT, 1, 1)))

    writer = cv2.VideoWriter(str(root / "cam_a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV cannot write mp4v here"
    for color in FRAME_COLORS_BGR:
        writer.write(np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8))
    writer.release()
    camera = group.create_group("cam_a")
    frame_rows = np.arange(FRAMES)
    _write(camera, "capture_time", T0 + 0.02 + frame_rows * STEP)
    _write(camera, "receive_time", T0 + 0.02 + frame_rows * STEP)
    intrinsics = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
    _write(camera, "intrinsics", np.tile(intrinsics, (FRAMES, 1, 1)))
    return root


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    return build_episode(tmp_path / "episode_0")


def test_load_episode_aligns_every_stream_on_one_timeline(episode_dir):
    episode = load_episode(episode_dir)
    assert episode.start_time == pytest.approx(T0)
    assert episode.controllers.times[0] == pytest.approx(0.0)
    assert episode.hands.times[0] == pytest.approx(0.01)
    camera = episode.cameras["cam_a"]
    assert (camera.width, camera.height, camera.frame_count) == (WIDTH, HEIGHT, FRAMES)
    assert camera.fps == pytest.approx(20.0)
    assert camera.times[0] == pytest.approx(0.02)
    assert camera.intrinsics.shape == (FRAMES, 3, 3)
    # The controller table is the longest stream: its last row plus one of its own intervals.
    assert episode.period == pytest.approx((CONTROLLER_ROWS - 1) * STEP + STEP)
    assert episode.controllers.latest_row(-0.001) == -1
    assert episode.controllers.latest_row(0.0) == 0
    assert episode.controllers.latest_row(0.26) == CONTROLLER_ROWS - 1
    assert camera.latest_frame(10.0) == FRAMES - 1


def test_load_episode_rejects_a_camera_without_its_frame_times(episode_dir):
    shutil.rmtree(episode_dir / "cam_a")
    with pytest.raises(EpisodeFormatError, match="cam_a.mp4"):
        load_episode(episode_dir)


def test_load_episode_requires_the_directory_to_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episode(tmp_path / "missing")


def test_head_and_controller_groups_carry_the_recorded_pose():
    pose = _pose(1.0, 2.0, 3.0, yaw_deg=90.0)
    head = tracking_map.head_group(pose)
    np.testing.assert_allclose(head[HeadInputIndex.POSITION], [1.0, 2.0, 3.0])
    half = math.sqrt(0.5)
    np.testing.assert_allclose(head[HeadInputIndex.ORIENTATION], [0.0, half, 0.0, half], atol=1e-6)
    assert head[HeadInputIndex.IS_VALID] and head[HeadInputIndex.IS_TRACKED]

    controller = tracking_map.controller_group(pose, trigger=0.5, primary=True, secondary=False)
    np.testing.assert_allclose(controller[ControllerInputIndex.GRIP_POSITION], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(controller[ControllerInputIndex.AIM_POSITION], [1.0, 2.0, 3.0])
    assert controller[ControllerInputIndex.GRIP_IS_VALID] and controller[ControllerInputIndex.AIM_IS_VALID]
    assert controller[ControllerInputIndex.PRIMARY_CLICK] == 1.0
    assert controller[ControllerInputIndex.SECONDARY_CLICK] == 0.0
    assert controller[ControllerInputIndex.TRIGGER_VALUE] == 0.5
    assert controller[ControllerInputIndex.SQUEEZE_VALUE] == 0.0


def test_identity_pose_is_reported_untracked():
    head = tracking_map.head_group(np.eye(4))
    assert not head[HeadInputIndex.IS_VALID] and not head[HeadInputIndex.IS_TRACKED]
    controller = tracking_map.controller_group(np.eye(4), trigger=0.0, primary=False, secondary=False)
    assert not controller[ControllerInputIndex.GRIP_IS_VALID] and not controller[ControllerInputIndex.AIM_IS_VALID]


def _record_hand(world_joints: np.ndarray, anchor: np.ndarray, side: str, missing: set[int]) -> np.ndarray:
    """The recorder's forward mapping: world OpenXR joints 1..25 -> wrist-relative Manus nodes."""
    align = tracking_map.OXR_TO_MANUS_ALIGN
    root_rot, root_pos = anchor[:3, :3], anchor[:3, 3]
    nodes = np.zeros((tracking_map.MANUS_NODE_COUNT, 4, 4))
    nodes[:, 3, 3] = 1.0
    for node in range(tracking_map.MANUS_NODE_COUNT):
        world = world_joints[node + 1]
        relative_pos = root_rot.T @ (world[:3, 3] - root_pos)
        relative_rot = root_rot.T @ world[:3, :3]
        rot = align @ relative_rot @ align.T
        if node == tracking_map.THUMB_TIP_NODE:
            rot = rot @ tracking_map.thumb_tip_rotation(side)
        nodes[node, :3, :3] = rot
        nodes[node, :3, 3] = align @ relative_pos
    nodes[sorted(missing)] = 0.0
    tilt = np.eye(4)
    tilt[:3, :3] = tracking_map.tilt_correction(side)
    return tilt[None] @ nodes


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_group_recovers_world_joints_from_the_recorded_skeleton(side):
    anchor = _pose(0.2, 1.1, -0.5, yaw_deg=40.0)
    world = np.stack([anchor @ _pose(0.01 * j, -0.002 * j, -0.03 * j, yaw_deg=3.0 * j) for j in range(26)])
    world[HandJointIndex.WRIST] = anchor
    nodes = _record_hand(world, anchor, side, missing={MISSING_HAND_NODE})

    group = tracking_map.hand_group(nodes, anchor, side)
    valid = group[HandInputIndex.JOINT_VALID]
    assert valid[HandJointIndex.PALM] == 0
    assert valid[HandJointIndex.WRIST] == 1
    assert valid[MISSING_HAND_NODE + 1] == 0
    assert valid.sum() == 24
    positions = group[HandInputIndex.JOINT_POSITIONS]
    orientations = group[HandInputIndex.JOINT_ORIENTATIONS]
    for joint in range(1, 26):
        if not valid[joint]:
            continue
        np.testing.assert_allclose(positions[joint], world[joint][:3, 3], atol=1e-5)
        np.testing.assert_allclose(
            orientations[joint], tracking_map.quaternion_xyzw(world[joint][:3, :3]), atol=1e-5
        )


def test_hand_without_an_anchor_is_present_but_invalid():
    nodes = recorded_left_hand(0)
    for anchor in (None, np.eye(4)):
        group = tracking_map.hand_group(nodes, anchor, "left")
        assert group is not None and not group.is_none
        assert group[HandInputIndex.JOINT_VALID].sum() == 0


def test_untracked_hand_is_absent():
    untracked = np.tile(np.eye(4), (tracking_map.MANUS_NODE_COUNT, 1, 1))
    assert tracking_map.hand_group(untracked, _pose(0.0, 1.0, 0.0), "right") is None


def _graph(sdk):
    controllers = sdk.ControllersSource(name="controllers")
    hands = sdk.HandsSource(name="hands")
    return sdk.TeleopSessionConfig(
        app_name="test",
        pipeline=sdk.OutputCombiner(
            {
                "head": sdk.HeadSource(name="head").output("head"),
                "controller_left": controllers.output(sdk.ControllersSource.LEFT),
                "controller_right": controllers.output(sdk.ControllersSource.RIGHT),
                "hand_left": hands.output(sdk.HandsSource.LEFT),
                "hand_right": hands.output(sdk.HandsSource.RIGHT),
            }
        ),
    )


def test_session_follows_the_shared_timeline_and_loops(episode_dir):
    episode = load_episode(episode_dir)
    clock = FakeClock(100.0)
    playback = PlaybackClock(episode.period, clock=clock)
    sdk = replay_isaac_capture
    with sdk.TeleopSession(_graph(sdk), episode=episode, playback=playback, clock=clock) as session:
        first = session.step()
        np.testing.assert_allclose(first["head"][HeadInputIndex.POSITION], head_pose(0)[:3, 3], atol=1e-6)
        # The hand table starts 10 ms after the controller table, so no hand row is current yet.
        assert first["hand_left"].is_none and first["hand_right"].is_none

        clock.t += 0.065
        later = session.step()
        np.testing.assert_allclose(later["head"][HeadInputIndex.POSITION], head_pose(1)[:3, 3], atol=1e-6)
        assert later["controller_left"][ControllerInputIndex.PRIMARY_CLICK] == 1.0
        assert later["controller_left"][ControllerInputIndex.TRIGGER_VALUE] == pytest.approx(0.2)
        assert not later["hand_left"].is_none
        assert later["hand_left"][HandInputIndex.JOINT_VALID][HandJointIndex.WRIST] == 1
        assert later["hand_right"].is_none, "a hand the recorder never tracked stays absent"

        clock.t = 100.0 + episode.period + 0.26
        wrapped = session.step()
        assert wrapped["controller_right"][ControllerInputIndex.GRIP_IS_VALID] is False
        np.testing.assert_allclose(
            wrapped["head"][HeadInputIndex.POSITION], head_pose(CONTROLLER_ROWS - 1)[:3, 3], atol=1e-6
        )

        clock.t = 100.0 + 2 * episode.period + 0.001
        again = session.step()
        np.testing.assert_allclose(again["head"][HeadInputIndex.POSITION], head_pose(0)[:3, 3], atol=1e-6)

        status = session.get_status()
        assert [provider.name for provider in status.providers] == ["Recorded episode replay"]
        devices = {device.path: device.status for device in status.devices}
        assert devices["/head/headset"] is sdk.DeviceState.CONNECTED
        assert devices["/hand/left"] is sdk.DeviceState.CONNECTED


def _start_pipeline(pipeline, quality: int = 90):
    """The same construction sequence DepthAIAdapter._enter_sync performs."""
    sdk = replay_depthai
    camera = pipeline.create(sdk.node.Camera).build(sdk.CameraBoardSocket.CAM_A)
    output = camera.requestOutput((WIDTH, HEIGHT), type=sdk.ImgFrame.Type.NV12, fps=20.0)
    encoder = pipeline.create(sdk.node.VideoEncoder).build(
        output, frameRate=20.0, profile=sdk.VideoEncoderProperties.Profile.MJPEG, quality=quality
    )
    queue = encoder.out.createOutputQueue()
    queue.setMaxSize(1)
    queue.setBlocking(False)
    pipeline.start()
    assert pipeline.isRunning()
    return queue


def _mean_bgr(frame) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(frame.getData(), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image.reshape(-1, 3).mean(axis=0)


def test_pipeline_serves_due_frames_in_order_skips_overdue_ones_and_loops(episode_dir):
    episode = load_episode(episode_dir)
    clock = FakeClock(5.0)
    playback = PlaybackClock(episode.period, clock=clock)
    with replay_depthai.Pipeline(track=episode.cameras["cam_a"], playback=playback, clock=clock) as pipeline:
        queue = _start_pipeline(pipeline)
        assert queue.tryGet() is None, "the first frame is not due until 20 ms in"

        clock.t += 0.025
        frame = queue.tryGet()
        assert frame.getSequenceNum() == 0
        assert (frame.getWidth(), frame.getHeight()) == (WIDTH, HEIGHT)
        assert frame.getData()[:2] == b"\xff\xd8" and frame.getData()[-2:] == b"\xff\xd9"
        np.testing.assert_allclose(_mean_bgr(frame), FRAME_COLORS_BGR[0], atol=40)
        assert frame.getTransformation().getIntrinsicMatrix()[0][0] == 50.0
        assert queue.tryGet() is None, "nothing new until the next frame is due"

        clock.t += 0.10  # position 0.125: frame 2 (0.12) is due, frame 1 (0.07) was never shown
        frame = queue.tryGet()
        assert frame.getSequenceNum() == 2, "frame 1 is overdue and skipped, never replayed late"
        np.testing.assert_allclose(_mean_bgr(frame), FRAME_COLORS_BGR[2], atol=40)

        clock.t = 5.0 + episode.period + 0.025
        frame = queue.tryGet()
        assert frame.getSequenceNum() == 0, "the timeline wrapped, so decoding restarts from the first frame"
        np.testing.assert_allclose(_mean_bgr(frame), FRAME_COLORS_BGR[0], atol=40)


def test_factory_rejects_a_camera_the_episode_lacks_and_is_tagged_mock(episode_dir):
    episode = load_episode(episode_dir)
    factory = build_replay_source_factory(episode)
    assert factory.mock is True
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    unknown = p.CameraStreamConfig(
        stream_id=1, name="cam_b", type=p.CameraType.OAK_D_COLOR, required=True,
        width=WIDTH, height=HEIGHT, fps=20.0, jpeg_quality=80,
    )  # fmt: skip
    with pytest.raises(ValueError, match="not in the recorded episode"):
        factory(begin, (unknown,))

    stations = replay_station_cameras(episode)
    assert list(stations) == ["cam_a"]
    station = stations["cam_a"]
    assert (station.device_id, station.width, station.height) == (
        str(episode.cameras["cam_a"].path),
        WIDTH,
        HEIGHT,
    )
    assert station.fps == pytest.approx(20.0)


def test_replay_and_mock_flags_are_mutually_exclusive(episode_dir, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--mock", "--replay", str(episode_dir)])
    assert excinfo.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_camera_option_is_rejected_with_replay(episode_dir, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--replay", str(episode_dir), "--camera", "front=ABC"])
    assert excinfo.value.code == 2
    assert "only applies to real" in capsys.readouterr().err


def test_missing_episode_is_a_setup_error(tmp_path, capsys):
    assert main(["--replay", str(tmp_path / "missing")]) == 2
    assert "episode directory not found" in capsys.readouterr().err


def _rounded(position) -> tuple[float, ...]:
    return tuple(round(float(v), 4) for v in position)


async def test_replay_streams_recorded_tracking_and_images_through_the_server(episode_dir, unused_tcp_port):
    episode = load_episode(episode_dir)
    running = await start_capture_server(
        unused_tcp_port,
        source_factory=build_replay_source_factory(episode),
        config=ServerConfig(mock=True, cameras=replay_station_cameras(episode)),
    )
    try:
        started, frames, images = [], [], []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS),
            cameras=({"name": "cam_a", "type": p.CameraType.OAK_D_COLOR},),
            on_started=lambda e: started.append(e),
            on_capture_frame=lambda f: frames.append(f),
            on_capture_image=lambda i: images.append(i),
            on_stopped=lambda r: finished.set(),
        )
        await asyncio.sleep(0.7)
        assert started and started[0].mode is p.CaptureMode.MOCK
        # Compared as Python floats: the wire carries float32, the fixture holds float64.
        served = {_rounded(p.decode_tracking_frame(f.payload)["head.position"]) for f in frames}
        recorded = {_rounded(head_pose(i)[:3, 3]) for i in range(CONTROLLER_ROWS)}
        assert served, "no tracking frames arrived"
        assert served <= recorded, f"served head positions {served - recorded} were never recorded"
        assert images, "no camera images arrived"
        assert images[0].name == "cam_a" and images[0].jpeg[:2] == b"\xff\xd8"
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)
