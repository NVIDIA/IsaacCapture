# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from mcap.writer import CompressionType, IndexType, Writer

from isaaccapture_examples.mcap_record_replay import replay_ego
from isaaccapture.schema import (
    OrbbecEgoRecoveryEvent,
    OrbbecEgoRecoveryStage,
    OrbbecEgoDiagnosticEvent,
)

TOPICS = {
    "orbbec_ego_metadata_colorleft/FrameMetadata",
    "orbbec_ego_metadata_colorright/FrameMetadata",
    "orbbec_ego_imu_accel/Imu",
    "orbbec_ego_imu_gyro/Imu",
    "orbbec_ego_audio/Audio",
    "orbbec_ego_calibration/Calibration",
    "orbbec_ego_device/DeviceState",
}


def write_recording(path, summary, topics=TOPICS):
    with path.open("wb") as stream:
        writer = Writer(
            stream,
            compression=CompressionType.NONE,
            use_chunking=False,
            index_types=IndexType.NONE,
            repeat_schemas=summary,
            repeat_channels=summary,
            use_statistics=summary,
            use_summary_offsets=summary,
        )
        writer.start()
        for topic in topics | {"orbbec_ego_audio/Audio_tracked"}:
            channel = writer.register_channel(topic, "test", 0)
            for sequence in range(1, 4 if topic in topics else 10):
                writer.add_message(channel, sequence, b"sample", sequence)
        writer.finish()


class Replay:
    def __init__(self):
        self.updates = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def update(self):
        self.updates += 1


class Tracker:
    def __init__(self, *_):
        pass

    def get_data(self, session):
        if session.updates == 2:
            return None
        return SimpleNamespace(
            capture_epoch=1,
            sequence_number=session.updates,
            width=1600,
            height=1300,
            fps=30,
            pixel_format="H264",
            encoded_bytes=100,
            sample_rate_hz=48000,
            samples=[1],
            sample_count=480,
            byte_count=960,
            device_uid="ego",
            event_type=OrbbecEgoDiagnosticEvent.RecoveryEvent,
            diagnostic_version=2,
            observed_at_ns=100,
            event=OrbbecEgoRecoveryEvent(
                OrbbecEgoRecoveryStage.Recovered, "recovered", 1
            ),
        )


def mock_trackers(monkeypatch):
    for name in (
        "OrbbecEgoFrameMetadataTracker",
        "OrbbecEgoImuTracker",
        "OrbbecEgoAudioTracker",
        "OrbbecEgoCalibrationTracker",
        "OrbbecEgoDeviceStateTracker",
    ):
        monkeypatch.setattr(replay_ego, name, Tracker)
    session = Replay()
    monkeypatch.setattr(
        replay_ego, "ReplaySession", SimpleNamespace(run=lambda _: session)
    )
    monkeypatch.setattr(replay_ego, "McapReplayConfig", lambda *_: None)
    return session


@pytest.mark.parametrize("summary", [True, False])
def test_gap_does_not_end_replay_and_tracked_topics_do_not_extend_it(
    tmp_path, monkeypatch, capsys, summary
):
    recording = tmp_path / "gap.mcap"
    write_recording(recording, summary)
    session = mock_trackers(monkeypatch)
    counts = replay_ego.replay(recording, 0)
    assert session.updates == 3
    assert session.closed
    assert set(counts.values()) == {2}
    assert "ColorLeft: epoch=1 seq=3" in capsys.readouterr().out


def test_explicit_update_limit_includes_gaps(tmp_path, monkeypatch):
    recording = tmp_path / "gap.mcap"
    write_recording(recording, True)
    session = mock_trackers(monkeypatch)
    counts = replay_ego.replay(recording, 2)
    assert session.updates == 2
    assert set(counts.values()) == {1}


def test_recording_without_raw_topics_finishes_immediately(tmp_path, monkeypatch):
    recording = tmp_path / "other.mcap"
    write_recording(recording, True, {"head/head"})
    session = mock_trackers(monkeypatch)
    counts = replay_ego.replay(recording, 0)
    assert session.updates == 0
    assert set(counts.values()) == {0}


@pytest.mark.parametrize("enabled", [True, False])
def test_recorded_policy_controls_diagnostic_tracker_registration(
    tmp_path, monkeypatch, enabled
):
    path = tmp_path / "policy.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start()
        writer.add_metadata(
            "ego.recording",
            {
                "record_device_diagnostics": "on" if enabled else "off",
                "diagnostic_version": "2",
            },
        )
        channel = writer.register_channel(
            "orbbec_ego_metadata_colorleft/FrameMetadata", "test", 0
        )
        writer.add_message(channel, 1, b"test", 1)
        writer.finish()
    mock_trackers(monkeypatch)
    counts = replay_ego.replay(path, 0)
    assert ("DeviceState" in counts) == enabled


def test_off_rejects_diagnostic_channel_under_a_different_topic(tmp_path):
    path = tmp_path / "conflict.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start()
        writer.add_metadata(
            "ego.recording",
            {"record_device_diagnostics": "off", "diagnostic_version": "2"},
        )
        schema = writer.register_schema(
            "core.OrbbecEgoDeviceStateRecord", "flatbuffer", b"fixture"
        )
        writer.register_channel("custom/topic", "flatbuffer", schema)
        writer.finish()
    with pytest.raises(ValueError, match="channel exists"):
        replay_ego.replay(path, 0)


@pytest.mark.parametrize("version", ["0", "1", "99"])
def test_unsupported_diagnostic_policy_is_rejected(tmp_path, version):
    path = tmp_path / "unsupported.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start()
        writer.add_metadata(
            "ego.recording",
            {"record_device_diagnostics": "on", "diagnostic_version": version},
        )
        writer.finish()
    with pytest.raises(ValueError, match="Unsupported EGO"):
        replay_ego.replay(path, 0)
