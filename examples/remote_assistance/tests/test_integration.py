# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in tests against a real local SFU, with no camera or robot access."""

import asyncio
import os
import uuid
from pathlib import Path

import pytest
import numpy as np
import yaml
import zmq

from remote_assistance.bridge import run_bridge, sonic_stop_message
from remote_assistance.camera import publish_camera
from remote_assistance.data import DataSession
from remote_assistance.video import VideoReceiver

URL = os.environ.get("LIVEKIT_TEST_URL")
pytestmark = pytest.mark.skipif(
    not URL, reason="Set LIVEKIT_TEST_URL to the local development SFU"
)


def settings():
    return {
        "url": URL,
        "room": "test-" + uuid.uuid4().hex,
        "development": True,
        "max_age_ms": 500,
    }


async def until(predicate, seconds=10):
    async with asyncio.timeout(seconds):
        # SDK peer membership/readiness has no shared Event; keep test-only polling bounded.
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.01)


def test_real_data_packets_round_trip():
    async def scenario():
        cfg, received_edge, received_robot = settings(), [], []
        edge = DataSession(
            cfg,
            "edge",
            "robot",
            {"ra/v1/state"},
            lambda t, p: received_edge.append((t, p)),
        )
        robot = DataSession(
            cfg,
            "robot",
            "edge",
            {"ra/v1/pose", "ra/v1/command"},
            lambda t, p: received_robot.append((t, p)),
        )
        try:
            await edge.start()
            await robot.start()
            await until(lambda: edge.ready and robot.ready)
            payload = bytes(range(256)) * 128
            assert await edge.send("ra/v1/pose", payload)
            assert await edge.send("ra/v1/command", b"start", reliable=True)
            assert await robot.send("ra/v1/state", b"observation")
            await until(lambda: len(received_robot) == 2 and len(received_edge) == 1)
            assert received_robot == [
                ("ra/v1/pose", payload),
                ("ra/v1/command", b"start"),
            ]
            assert received_edge == [("ra/v1/state", b"observation")]
            await edge.stop()
            await until(lambda: not robot.ready)
            assert not await robot.send("ra/v1/state", b"must not queue")
            edge = DataSession(cfg, "edge", "robot", {"ra/v1/state"}, lambda *_: None)
            await edge.start()
            await until(lambda: edge.ready and robot.ready)
            assert await edge.send("ra/v1/command", b"after restart", reliable=True)
            await until(lambda: len(received_robot) == 3)
            assert received_robot[-1] == ("ra/v1/command", b"after restart")
        finally:
            await edge.stop()
            await robot.stop()

    asyncio.run(scenario())


def test_real_video_and_sender_republication():
    async def scenario():
        cfg = {
            **settings(),
            "publisher": "camera",
            "video_encoder": "software",
            "video_decoder": "software",
        }
        spec = {
            "name": "cam",
            "type": "synthetic",
            "width": 320,
            "height": 180,
            "fps": 30,
        }
        receiver = VideoReceiver(
            cfg, identity="viewer", publisher="camera", track="cam"
        )
        await asyncio.to_thread(receiver.start)
        try:
            for _ in range(2):
                stop = asyncio.Event()
                sender = asyncio.create_task(publish_camera(cfg, spec, stop))
                try:
                    frames = []
                    async with asyncio.timeout(15):
                        while len(frames) < 45:
                            frame = receiver.latest()
                            if frame:
                                frames.append(frame)
                            if sender.done():
                                await sender
                                pytest.fail("Sender stopped early")
                            await asyncio.sleep(0.01)
                    assert all(
                        (f.frame.width, f.frame.height) == (320, 180) for f in frames
                    )
                    assert any(f.metadata is not None for f in frames)
                    # A decoder can emit fresh timestamps with frozen pixels. Check
                    # both pulse levels against the sender's frame ID, not just fps.
                    levels = set()
                    for frame in frames:
                        assert frame.metadata is not None
                        expected = 220 if (frame.metadata.frame_id // 30) % 2 else 30
                        levels.add(expected)
                        rgba = np.frombuffer(frame.frame.data, dtype=np.uint8).reshape(
                            180, 320, 4
                        )
                        assert abs(float(rgba[90, :, 0].mean()) - expected) < 5
                    assert levels == {30, 220}
                finally:
                    stop.set()
                    await sender
                await asyncio.sleep(0.2)
        finally:
            await asyncio.to_thread(receiver.stop)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["frozen_input", "disconnect"])
def test_sonic_bridge_suspends_and_recovers(tmp_path, failure):
    async def scenario():
        cfg = yaml.safe_load((Path(__file__).parents[1] / "session.yaml").read_text())
        cfg["livekit"] = settings()
        context = zmq.Context()
        sockets = []
        for role in ("edge", "robot"):
            cfg[role]["subscribe"] = f"ipc://{tmp_path}/{role}-input"
            cfg[role]["publish"] = f"ipc://{tmp_path}/{role}-output"
            pub, sub = context.socket(zmq.PUB), context.socket(zmq.SUB)
            pub.bind(cfg[role]["subscribe"])
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(cfg[role]["publish"])
            sockets.extend((pub, sub))
        edge_input, edge_output, robot_input, robot_output = sockets
        stops = [asyncio.Event(), asyncio.Event()]
        tasks = [
            asyncio.create_task(run_bridge(cfg, role, stop))
            for role, stop in zip(("edge", "robot"), stops)
        ]
        target = b"pose" + bytes(range(256)) * 20
        observation = b"g1_debug" + b"test-observation"
        got_target = got_observation = False
        try:
            # No manager input yet: do not terminate a starting SONIC instance.
            await asyncio.sleep(1.5)
            assert not robot_output.poll(0)
            async with asyncio.timeout(15):
                while not (got_target and got_observation):
                    edge_input.send(target)
                    robot_input.send(observation)
                    for sock, expected in (
                        (robot_output, target),
                        (edge_output, observation),
                    ):
                        while sock.poll(0):
                            data = sock.recv()
                            if expected == target and data == target:
                                got_target = True
                            if expected == observation and data == observation:
                                got_observation = True
                    for task in tasks:
                        if task.done():
                            await task
                            pytest.fail("Bridge stopped early")
                    await asyncio.sleep(0.03)
            # Exercise a discrete start command without sending it to live SONIC.
            command = sonic_stop_message()[:-3] + bytes((1, 0, 1))
            async with asyncio.timeout(5):
                received_command = False
                while not received_command:
                    edge_input.send(command)
                    await asyncio.sleep(0.03)
                    while robot_output.poll(0):
                        received_command |= robot_output.recv() == command
            while robot_output.poll(0):
                robot_output.recv()
            if failure == "disconnect":
                stops[0].set()
                await tasks[0]
            # Drain messages already in flight, then exceed the freshness limit.
            await asyncio.sleep(0.2)
            while robot_output.poll(0):
                assert robot_output.recv() != sonic_stop_message()
            for _ in range(30):
                robot_input.send(observation)
                await asyncio.sleep(0.03)
                assert not robot_output.poll(0)
            if failure == "disconnect":
                stops[0] = asyncio.Event()
                tasks[0] = asyncio.create_task(run_bridge(cfg, "edge", stops[0]))
            # Fresh targets resume without restarting the robot bridge or SONIC.
            async with asyncio.timeout(10):
                resumed = False
                while not resumed:
                    edge_input.send(target)
                    await asyncio.sleep(0.03)
                    while robot_output.poll(0):
                        payload = robot_output.recv()
                        assert payload != sonic_stop_message()
                        resumed |= payload == target
            # Explicit operator stops remain byte-for-byte commands.
            async with asyncio.timeout(5):
                stopped = False
                while not stopped:
                    edge_input.send(sonic_stop_message())
                    await asyncio.sleep(0.03)
                    while robot_output.poll(0):
                        stopped |= robot_output.recv() == sonic_stop_message()
            await asyncio.sleep(0.2)
            while robot_output.poll(0):
                robot_output.recv()
            stops[1].set()
            await tasks[1]
            assert not robot_output.poll(0)
        finally:
            for stop in stops:
                stop.set()
            await asyncio.gather(*tasks)
            for sock in sockets:
                sock.close(0)
            context.term()

    asyncio.run(scenario())
