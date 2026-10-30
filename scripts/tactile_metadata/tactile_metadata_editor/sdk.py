# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import multiprocessing
import sys
import threading
import time

from .document import JOINTS


class Receiver:
    def __init__(self, config: dict):
        context = multiprocessing.get_context("spawn")
        context.set_executable(sys.executable)
        self.connection, child = context.Pipe(duplex=False)
        self.latest, self.status, self.active = {}, "Connecting", False
        self.process = context.Process(target=live, args=(child, config), daemon=True)
        self.process.start()
        child.close()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self) -> None:
        try:
            while True:
                value = json.loads(self.connection.recv_bytes(16 * 1024 * 1024))
                self.status = value.get("status", "Live")
                if "frame" in value:
                    self.latest = value["frame"]
                    self.active = True
        except (EOFError, OSError, ValueError):
            self.active = False
            if self.status in ("Connecting", "Live"):
                self.status = "Disconnected"

    def close(self) -> None:
        self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()
        self.thread.join(timeout=2)


def _pose_rows(values) -> list[dict]:
    return [
        {
            "position": dict(zip("xyz", row[:3])),
            "orientation": dict(zip("xyzw", row[3:])),
        }
        for row in values.copy().tolist()
    ]


def _metadata(value) -> dict | None:
    if value is None or value.geometry_mapping is None or value.segment_source is None:
        return None
    mapping = value.geometry_mapping
    return {
        "geometry_mapping_type": "LBS",
        "geometry_mapping": {
            "segment_bind_poses": _pose_rows(mapping.segment_bind_poses),
            "taxel_bind_poses": _pose_rows(mapping.taxel_bind_poses),
            "taxel_segment_indices": [
                {"values": row} for row in mapping.taxel_segment_indices.copy().tolist()
            ],
            "taxel_segment_weights": [
                {"values": row} for row in mapping.taxel_segment_weights.copy().tolist()
            ],
        },
        "segment_source_type": "HandSkeletonRef",
        "segment_source": {
            "joint_ids": [
                JOINTS[int(joint)] for joint in value.segment_source.joint_ids
            ]
        },
    }


def _hand(value) -> dict | None:
    if value is None:
        return None
    joints = value.joints
    return {
        name: getattr(joints, name).copy().tolist()
        for name in ("positions", "orientations", "is_valid")
    }


def live(connection, config: dict) -> None:
    try:
        from isaaccapture.deviceio_session import DeviceIOSession
        from isaaccapture.deviceio_trackers import (
            HandTracker,
            JointStateTracker,
            TactileTracker,
        )
        from isaaccapture.oxr import OpenXRSession

        hand = HandTracker()
        tactile = TactileTracker(config["collection"])
        joint = (
            JointStateTracker(config["joint_state"])
            if config.get("joint_state")
            else None
        )
        trackers = [hand, tactile, *([joint] if joint else [])]
        with OpenXRSession(
            "TactileMetadataEditor", DeviceIOSession.get_required_extensions(trackers)
        ) as oxr:
            with DeviceIOSession.run(trackers, oxr.get_handles()) as session:
                while True:
                    session.update()
                    metadata = tactile.get_metadata(session)
                    data = tactile.get_data(session)
                    hand_value = getattr(hand, f"get_{config['side']}_hand")(session)
                    joint_value = joint.get_data(session) if joint else None
                    frame = {
                        "metadata": _metadata(metadata),
                        "forces": data.taxel_forces.copy().tolist()
                        if data is not None
                        else None,
                        "hand": _hand(hand_value),
                        "joints": {
                            item.name: [item.position, item.valid]
                            for item in joint_value.joints
                        }
                        if joint_value is not None
                        else {},
                        "read_time": time.monotonic(),
                    }
                    connection.send_bytes(json.dumps({"frame": frame}).encode())
                    time.sleep(1 / 30)
    except Exception as exc:
        connection.send_bytes(
            json.dumps({"status": f"{type(exc).__name__}: {exc}"}).encode()
        )
    finally:
        connection.close()
