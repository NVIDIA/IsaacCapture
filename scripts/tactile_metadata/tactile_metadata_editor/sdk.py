# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import multiprocessing
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .document import Document, FIELDS, IDENTITY, joint_number
from .storage import load, metadata, write

ROOT = Path(__file__).resolve().parent.parent


def check(document, output=None):
    with tempfile.TemporaryDirectory(prefix="tactile-sdk-") as folder:
        root = Path(folder)
        write(root / "input.json", metadata(document, numerical=True))
        try:
            result = subprocess.run(
                [sys.executable, "-m", __package__ + ".sdk", "check", folder],
                cwd=ROOT,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("SDK construction timed out after 30 seconds") from exc
        report = root / "result.json"
        if result.returncode or not report.exists():
            reason = (
                report.read_text()
                if report.exists()
                else result.stderr.decode(errors="replace")[-2000:]
            )
            raise RuntimeError(
                f"SDK construction failed (exit {result.returncode}): " + reason
            )
        if output:
            Path(output).write_bytes((root / "metadata.bin").read_bytes())
        return json.loads(report.read_text())


def encode(folder):
    import numpy as np
    from isaaccapture import schema
    from isaaccapture.deviceio_trackers import TactileTracker

    root = Path(folder)
    d = load(root / "input.json")
    arrays = [
        np.asarray(
            d.arrays[f], dtype=np.uint32 if f == FIELDS[2] else np.float32
        ).reshape(-1, 7 if f in FIELDS[:2] else 4)
        for f in FIELDS
    ]
    encoded = schema.TactileMetadata(
        schema.LBS(*arrays),
        schema.HandSkeletonRef([schema.HandJoint(joint_number(j)) for j in d.joints]),
    ).to_bytes()
    frame_size = len(
        schema.TactileDataArray(
            np.zeros((len(arrays[1]), 3), dtype=np.float32)
        ).to_bytes()
    )
    if (
        len(encoded) > schema.TRACKER_METADATA_MAX_SIZE
        or frame_size > TactileTracker.DEFAULT_MAX_FLATBUFFER_SIZE
    ):
        raise ValueError(
            "Encoded metadata or force frame exceeds the default tracker capacity"
        )
    (root / "metadata.bin").write_bytes(encoded)
    return dict(
        status="SDK construction and size checks passed",
        metadata_bytes=len(encoded),
        force_bytes=frame_size,
    )


class Receiver:
    def __init__(self, configs):
        context = multiprocessing.get_context("spawn")
        context.set_executable(sys.executable)
        self.connection, child = context.Pipe(duplex=False)
        self.latest, self.status, self.active = {}, "Connecting", False
        self.process = context.Process(target=live, args=(child, configs), daemon=True)
        self.process.start()
        child.close()
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        try:
            while True:
                value = json.loads(self.connection.recv_bytes(16 * 1024 * 1024))
                self.status = value.get("status", "Connected")
                self.active = "sides" in value
                if "sides" in value:
                    self.latest = value["sides"]
        except (EOFError, OSError, ValueError):
            self.active = False
            self.status += "; receiver stopped"

    def close(self):
        self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()
        self.thread.join(timeout=2)


def owned(value, fields):
    return (
        {f: getattr(value, f).copy().tolist() for f in fields}
        if value is not None
        else None
    )


def live(connection, configs):
    try:
        from isaaccapture.deviceio_session import DeviceIOSession
        from isaaccapture.deviceio_trackers import (
            HandTracker,
            JointStateTracker,
            TactileTracker,
        )
        from isaaccapture.oxr import OpenXRSession

        hand = HandTracker()
        tactile = {s: TactileTracker(c["collection"]) for s, c in configs.items()}
        joints = {
            s: JointStateTracker(c["joint_state"])
            for s, c in configs.items()
            if c.get("joint_state")
        }
        trackers = [hand, *tactile.values(), *joints.values()]
        with OpenXRSession(
            "TactileMetadataEditor", DeviceIOSession.get_required_extensions(trackers)
        ) as oxr:
            with DeviceIOSession.run(trackers, oxr.get_handles()) as session:
                while True:
                    session.update()
                    sides = {}
                    for side, tracker in tactile.items():
                        m, data = (
                            tracker.get_metadata(session),
                            tracker.get_data(session),
                        )
                        h = getattr(hand, "get_" + configs[side]["hand"] + "_hand")(
                            session
                        )
                        j = joints[side].get_data(session) if side in joints else None
                        raw = None
                        if m is not None:
                            d = Document(side)
                            d.joints = [int(v) for v in m.segment_source.joint_ids]
                            d.arrays = owned(m.geometry_mapping, FIELDS)
                            d.sync()
                            raw = metadata(d)
                            raw.pop("editor")
                        sides[side] = dict(
                            metadata=raw,
                            forces=data.taxel_forces.copy().tolist()
                            if data is not None
                            else None,
                            hand=owned(h, ("positions", "orientations", "is_valid")),
                            joints={v.name: [v.position, v.valid] for v in j.joints}
                            if j is not None
                            else {},
                            read_time=time.monotonic(),
                            config=configs[side],
                        )
                    connection.send_bytes(json.dumps(dict(sides=sides)).encode())
                    time.sleep(1 / 30)
    except Exception as exc:
        connection.send_bytes(
            json.dumps(dict(status=f"{type(exc).__name__}: {exc}")).encode()
        )
    finally:
        connection.close()


def demo(configs):
    sides, angle = {}, math.sin(time.monotonic()) * 0.4
    for side, config in configs.items():
        d = Document(side)
        d.joints = ["WRIST", "INDEX_DISTAL"]
        d.arrays[FIELDS[0]] = [IDENTITY[:], [0, 0.06, 0, 0, 0, 0, 1]]
        d.add(
            [
                [c * 0.008, 0.06 + r * 0.008, 0, 0, 0, 0, 1]
                for r in range(4)
                for c in range(4)
            ],
            [1],
        )
        positions = [[0, 0, 0] for _ in range(26)]
        orientations = [[0, 0, 0, 1] for _ in range(26)]
        valid = [False] * 26
        positions[9] = [0, 0.06, 0]
        orientations[9] = [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]
        valid[1] = valid[9] = True
        sides[side] = dict(
            metadata=metadata(d),
            forces=[
                [0, 0, 15 + 15 * math.sin(time.monotonic() + i / 4)] for i in range(16)
            ],
            hand=dict(positions=positions, orientations=orientations, is_valid=valid),
            joints={name: [angle, True] for name in config.get("mapping", {})}
            if config.get("joint_state")
            else {},
            read_time=time.monotonic(),
            config=config,
        )
    return sides


if __name__ == "__main__":
    try:
        write(Path(sys.argv[2]) / "result.json", encode(sys.argv[2]))
    except Exception as exc:
        write(
            Path(sys.argv[2]) / "result.json",
            dict(error=f"{type(exc).__name__}: {exc}"),
        )
        sys.exit(1)
