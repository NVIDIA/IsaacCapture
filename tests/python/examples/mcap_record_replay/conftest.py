# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "mcap_record_replay" / "python"))


@pytest.fixture
def command_clock(monkeypatch):
    def patch(module, timestamps):
        ticks = iter(timestamps)
        clock = SimpleNamespace(time=lambda: next(ticks), sleep=MagicMock())
        # Replace the command's clock, not global time used by native log receivers.
        monkeypatch.setattr(module, "time", clock)
        return clock

    return patch


def write_streams(path, streams):
    from mcap.writer import CompressionType, Writer

    with path.open("wb") as stream:
        # The native recorder and reader use uncompressed MCAP.
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start()
        for suffix, collection, record_type, payloads, timestamps in streams:
            bfbs = (
                repo_root() / "src/core/schema/golden" / f"{suffix}.bfbs"
            ).read_bytes()
            schema_id = writer.register_schema(
                f"core.{record_type.__name__}", "flatbuffer", bfbs
            )
            raw = writer.register_channel(
                f"{collection}/{suffix}", "flatbuffer", schema_id
            )
            tracked = writer.register_channel(
                f"{collection}/{suffix}_tracked", "flatbuffer", schema_id
            )
            for payload, timestamp in zip(payloads, timestamps, strict=True):
                log_time = timestamp.available_time_local_common_clock
                # Replay must select tracked records, not these different raw samples.
                writer.add_message(
                    raw, log_time, record_type(None, timestamp).to_bytes(), log_time
                )
                writer.add_message(
                    tracked,
                    log_time,
                    record_type(payload, timestamp).to_bytes(),
                    log_time,
                )
        writer.finish()
    return path


@pytest.fixture
def soma_mcap(tmp_path):
    from isaaccapture import schema

    def write(
        representation, payloads, timestamps, *, collection="soma_body_demo", path=None
    ):
        suffix = "soma_body_" + representation.replace("-", "_")
        record_type = (
            schema.SomaBodyJointRotationsRecord
            if representation == "joint-rotations"
            else schema.SomaBodyJointPosesRecord
        )
        path = path or tmp_path / f"{suffix}.mcap"
        return write_streams(
            path, [(suffix, collection, record_type, payloads, timestamps)]
        )

    return write


@pytest.fixture
def soma_hand_mcap(tmp_path):
    from isaaccapture import schema
    from isaaccapture_examples.mcap_record_replay.soma_hand_common import collections

    def write(representation, payloads, timestamps, *, collection_ids=None, path=None):
        collection_ids = collection_ids or collections(
            "soma_hand_left_demo", "soma_hand_right_demo"
        )
        suffix = "soma_hand_" + representation.replace("-", "_")
        record_type = (
            schema.SomaHandJointRotationsRecord
            if representation == "joint-rotations"
            else schema.SomaHandJointPosesRecord
        )
        return write_streams(
            path or tmp_path / f"{suffix}.mcap",
            [
                (suffix, collection_ids[key], record_type, values, timestamps[key])
                for key, values in payloads.items()
            ],
        )

    return write


@pytest.fixture
def legacy_mcap(tmp_path):
    """Native OpenXR channels do not use SchemaTracker's raw/tracked pair."""
    import numpy as np
    from mcap.writer import CompressionType, Writer
    from isaaccapture import schema

    def write(kind):
        body = kind == "body"
        joints = schema.BodyJoints() if body else schema.HandJoints()
        count = 24 if body else 26
        joints.positions[:] = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        joints.orientations[:, 3] = 1
        joints.is_valid[:] = 1
        data = schema.FullBodyPose(joints) if body else schema.HandPose(joints)
        record_type = schema.FullBodyPoseRecord if body else schema.HandPoseRecord
        suffix = "full_body" if body else "hand"
        topics = (
            ["full_body/full_body"] if body else ["hands/left_hand", "hands/right_hand"]
        )
        path = tmp_path / f"{suffix}.mcap"
        with path.open("wb") as stream:
            writer = Writer(stream, compression=CompressionType.NONE)
            writer.start()
            schema_id = writer.register_schema(
                f"core.{record_type.__name__}",
                "flatbuffer",
                (
                    repo_root() / "src/core/schema/golden" / f"{suffix}.bfbs"
                ).read_bytes(),
            )
            for topic in topics:
                channel = writer.register_channel(topic, "flatbuffer", schema_id)
                for i, payload in enumerate([data, None, data]):
                    stamp = schema.DeviceDataTimestamp(
                        1_000_000_000 + i * 100_000_000, 0, i
                    )
                    writer.add_message(
                        channel,
                        stamp.available_time_local_common_clock,
                        record_type(payload, stamp).to_bytes(),
                        stamp.available_time_local_common_clock,
                    )
            writer.finish()
        return path

    return write
