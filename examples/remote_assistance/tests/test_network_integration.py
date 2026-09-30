# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in check of generated TLS bundles on an isolated SFU and non-loopback IP."""

import asyncio
import os
from pathlib import Path
import shlex
import ssl
from urllib.parse import urlparse

import numpy as np
import pytest
import yaml

from remote_assistance.camera import publish_camera
from remote_assistance.data import DataSession
from remote_assistance.video import VideoReceiver

BUNDLE = os.environ.get("RA_NETWORK_TEST_BUNDLE")
pytestmark = pytest.mark.skipif(
    not BUNDLE, reason="Set RA_NETWORK_TEST_BUNDLE to an isolated generated deployment"
)


def test_generated_tls_data_and_video(monkeypatch):
    root = Path(BUNDLE)
    for role in ("edge", "robot"):
        for line in (root / role / "credentials.env").read_text().splitlines():
            if line.startswith("export SSL_CERT_FILE="):
                continue
            key, value = shlex.split(line)[1].split("=", 1)
            monkeypatch.setenv(role.upper() + "_" + key, value)
    # The caller sets SSL_CERT_FILE before the native SDK initializes.
    assert (
        Path(os.environ["SSL_CERT_FILE"]).resolve() == (root / "edge/ca.crt").resolve()
    )
    cfg = yaml.safe_load((root / "edge/session.yaml").read_text())
    bridge_settings = cfg["livekit"]
    url = urlparse(bridge_settings["url"])
    assert url.scheme == "wss" and url.hostname != "127.0.0.1"

    async def scenario():
        # A foreign trust root must reject our endpoint: never disable TLS checks.
        with pytest.raises(ssl.SSLCertVerificationError):
            async with asyncio.timeout(5):
                await asyncio.open_connection(
                    url.hostname, url.port, ssl=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                )
        edge_messages, robot_messages = [], []
        edge = DataSession(
            {**bridge_settings, "token_env": "EDGE_RA_SESSION_TOKEN"},
            cfg["edge"]["identity"],
            cfg["robot"]["identity"],
            {"ra/v1/check"},
            lambda t, p: edge_messages.append(p),
        )
        robot = DataSession(
            {**bridge_settings, "token_env": "ROBOT_RA_SESSION_TOKEN"},
            cfg["robot"]["identity"],
            cfg["edge"]["identity"],
            {"ra/v1/check"},
            lambda t, p: robot_messages.append(p),
        )
        try:
            await edge.start()
            await robot.start()
            async with asyncio.timeout(15):
                while not (edge_messages and robot_messages):
                    await edge.send("ra/v1/check", b"target" * 1000)
                    await robot.send("ra/v1/check", b"feedback", reliable=True)
                    await asyncio.sleep(0.05)
            assert all(p == b"target" * 1000 for p in robot_messages)
            assert all(p == b"feedback" for p in edge_messages)
        finally:
            await edge.stop()
            await robot.stop()
        camera = yaml.safe_load((root / "robot/camera.yaml").read_text())
        settings = camera["livekit"]
        spec = {
            **camera["cameras"][0],
            "type": "synthetic",
            "width": 320,
            "height": 180,
            "fps": 30,
        }
        receiver = VideoReceiver(
            {
                **settings,
                "token_env": "EDGE_RA_PROBE_TOKEN",
                "video_decoder": "software",
            },
            identity=settings.get("probe_identity", "edge-probe"),
            publisher=settings["publisher"],
            track=spec["name"],
        )
        stop = asyncio.Event()
        sender = None
        await asyncio.to_thread(receiver.start)
        try:
            sender = asyncio.create_task(
                publish_camera(
                    {
                        **settings,
                        "token_env": "ROBOT_RA_CAMERA_TOKEN",
                        "video_encoder": "software",
                    },
                    spec,
                    stop,
                )
            )
            levels = set()
            async with asyncio.timeout(15):
                while len(levels) < 2:
                    if sender.done():
                        await sender
                        pytest.fail("Sender exited early")
                    frame = receiver.latest()
                    if frame and frame.metadata is not None:
                        expected = 220 if (frame.metadata.frame_id // 30) % 2 else 30
                        rgba = np.frombuffer(frame.frame.data, dtype=np.uint8).reshape(
                            180, 320, 4
                        )
                        assert abs(float(rgba[90, :, 0].mean()) - expected) < 5
                        levels.add(expected)
                    await asyncio.sleep(0.01)
        finally:
            stop.set()
            if sender:
                await sender
            await asyncio.to_thread(receiver.stop)

    asyncio.run(scenario())
