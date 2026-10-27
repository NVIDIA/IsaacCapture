# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay all structured EGO trackers from a recorded MCAP file.

Pass the ``metadata.mcap`` produced by ``ego.sh record``, or the
final archive produced by ``isaaccapture.ego.embedded_recording``.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from isaaccapture.deviceio_session import McapReplayConfig, ReplaySession
from isaaccapture.deviceio_trackers import (
    OrbbecEgoFrameMetadataTracker,
    OrbbecEgoAudioTracker,
    OrbbecEgoCalibrationTracker,
    OrbbecEgoDeviceStateTracker,
    OrbbecEgoImuTracker,
)
from isaaccapture.schema import (
    OrbbecEgoSdkReport,
    OrbbecEgoCaptureSummary,
    OrbbecEgoRecoveryEvent,
    OrbbecEgoDeviceSnapshot,
)
from mcap.reader import make_reader

logger = logging.getLogger("isaaccapture.examples.mcap_record_replay.replay_ego")


def _update_count(path: Path, topics: set[str]) -> int:
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        counts: Counter[str] = Counter()
        if (
            summary is not None
            and summary.statistics is not None
            and summary.statistics.channel_message_counts.keys()
            <= summary.channels.keys()
        ):
            for channel_id, count in summary.statistics.channel_message_counts.items():
                topic = summary.channels[channel_id].topic
                if topic in topics:
                    counts[topic] += count
        else:
            for _, channel, _ in reader.iter_messages(
                topics=topics, log_time_order=False
            ):
                counts[channel.topic] += 1
    return max(counts.values(), default=0)


def _diagnostic_policy(path: Path, expected: bool | None) -> bool:
    policy = None
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for metadata in reader.iter_metadata():
            if metadata.name != "ego.recording":
                continue
            value = metadata.metadata.get("record_device_diagnostics")
            if (
                value not in ("on", "off")
                or metadata.metadata.get("diagnostic_version") != "2"
            ):
                raise ValueError("Unsupported EGO recording policy")
            enabled = value == "on"
            if policy is not None and policy != enabled:
                raise ValueError("Conflicting EGO recording policies")
            policy = enabled
        if expected is not None and policy is not None and policy != expected:
            raise ValueError("Requested and recorded diagnostic policies disagree")
        enabled = policy if policy is not None else expected
        if enabled is False:
            summary = reader.get_summary()
            if summary is None:
                raise ValueError(
                    "Diagnostics-off replay requires a complete channel inventory"
                )
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                if (
                    schema is not None
                    and schema.name == "core.OrbbecEgoDeviceStateRecord"
                ):
                    raise ValueError(
                        f"Diagnostics off but channel exists: {channel.topic}"
                    )
        if policy is None:
            logger.warning(
                "No EGO recording policy; current diagnostic schema is still required"
            )
    return enabled is not False


def replay(
    path: Path, max_updates: int, *, record_device_diagnostics: bool | None = None
) -> dict[str, int]:
    diagnostics = _diagnostic_policy(path, record_device_diagnostics)
    # One tracker per stream; each records under orbbec_ego_<kind>_<stream>.
    frame_trackers = {
        name: OrbbecEgoFrameMetadataTracker(f"ego/{name}")
        for name in ("ColorLeft", "ColorRight")
    }
    imu_trackers = {
        name: OrbbecEgoImuTracker(f"ego/{name}") for name in ("Accel", "Gyro")
    }
    audio_tracker = OrbbecEgoAudioTracker("ego_audio/Audio")
    calibration_tracker = OrbbecEgoCalibrationTracker("ego_calibration/Calibration")
    state_tracker = OrbbecEgoDeviceStateTracker("ego_device/DeviceState")
    trackers = [
        *(
            (tracker, f"orbbec_ego_metadata_{name.lower()}")
            for name, tracker in frame_trackers.items()
        ),
        *(
            (tracker, f"orbbec_ego_imu_{name.lower()}")
            for name, tracker in imu_trackers.items()
        ),
        (audio_tracker, "orbbec_ego_audio"),
        (calibration_tracker, "orbbec_ego_calibration"),
    ]
    counts = {
        "ColorLeft": 0,
        "ColorRight": 0,
        "Accel": 0,
        "Gyro": 0,
        "Audio": 0,
        "Calibration": 0,
        "DeviceState": 0,
    }
    topics = {
        "orbbec_ego_metadata_colorleft/FrameMetadata",
        "orbbec_ego_metadata_colorright/FrameMetadata",
        "orbbec_ego_imu_accel/Imu",
        "orbbec_ego_imu_gyro/Imu",
        "orbbec_ego_audio/Audio",
        "orbbec_ego_calibration/Calibration",
        "orbbec_ego_device/DeviceState",
    }
    if diagnostics:
        trackers.append((state_tracker, "orbbec_ego_device"))
    else:
        del counts["DeviceState"]
        topics.remove("orbbec_ego_device/DeviceState")
    # ReplaySession has no EOF accessor; payload-less records also return None.
    updates = _update_count(path, topics)
    if max_updates:
        updates = min(updates, max_updates)

    with ReplaySession.run(McapReplayConfig(str(path), trackers)) as session:
        for _ in range(updates):
            session.update()

            for name, tracker in frame_trackers.items():
                data = tracker.get_data(session)
                if data is not None:
                    counts[name] += 1
                    print(
                        f"{name}: epoch={data.capture_epoch} seq={data.sequence_number} "
                        f"profile={data.width}x{data.height}@{data.fps} "
                        f"format={data.pixel_format} bytes={data.encoded_bytes}"
                    )

            for name, tracker in imu_trackers.items():
                data = tracker.get_data(session)
                if data is not None:
                    counts[name] += 1
                    print(
                        f"{name}: epoch={data.capture_epoch} seq={data.sequence_number} "
                        f"rate_hz={data.sample_rate_hz} samples={len(data.samples)}"
                    )

            audio = audio_tracker.get_data(session)
            if audio is not None:
                counts["Audio"] += 1
                print(
                    f"Audio: epoch={audio.capture_epoch} seq={audio.sequence_number} "
                    f"rate_hz={audio.sample_rate_hz} samples={audio.sample_count} "
                    f"bytes={audio.byte_count}"
                )

            calibration = calibration_tracker.get_data(session)
            if calibration is not None:
                counts["Calibration"] += 1
                print(
                    f"Calibration: epoch={calibration.capture_epoch} "
                    f"device_uid={calibration.device_uid}"
                )

            state = state_tracker.get_data(session) if diagnostics else None
            if state is not None:
                counts["DeviceState"] += 1
                event = state.event
                if isinstance(event, OrbbecEgoSdkReport):
                    detail = f"seq={event.sequence_number} work_mode={event.work_mode} status_flags={event.status_flags} error_flags={event.error_flags} storage_free_bytes={event.storage_free_bytes}"
                elif isinstance(event, OrbbecEgoCaptureSummary):
                    detail = f"health={event.capture_health} queue={event.queue_peak}/{event.queue_capacity} dropped_events={event.dropped_events} dropped_video_frame_sets={event.dropped_video_frame_sets} final={event.is_final} reason={event.reason}"
                elif isinstance(event, OrbbecEgoRecoveryEvent):
                    detail = f"stage={event.stage} attempt={event.reconnect_attempt} reason={event.reason}"
                elif isinstance(event, OrbbecEgoDeviceSnapshot):
                    detail = f"imu_temperature_c={event.imu_temperature_c} properties={len(event.properties)}"
                else:
                    raise ValueError("Unsupported EGO diagnostic event")
                print(
                    f"DeviceState: event={state.event_type} version={state.diagnostic_version} epoch={state.capture_epoch} device_uid={state.device_uid} observed_at_ns={state.observed_at_ns} {detail}"
                )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path, help="Path to an EGO metadata.mcap file")
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many replay updates; 0 reads to end of file",
    )
    parser.add_argument(
        "--record-device-diagnostics",
        choices=("on", "off"),
        help="Expected policy; otherwise use MCAP Metadata (defaults to on when absent)",
    )
    args = parser.parse_args()
    if not args.mcap.is_file():
        parser.error(f"MCAP file does not exist: {args.mcap}")
    if args.max_updates < 0:
        parser.error("--max-updates must be non-negative")

    expected = (
        None
        if args.record_device_diagnostics is None
        else args.record_device_diagnostics == "on"
    )
    counts = replay(args.mcap, args.max_updates, record_device_diagnostics=expected)
    print(
        "Replay counts: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    if not all(counts.values()):
        missing = ", ".join(name for name, count in counts.items() if count == 0)
        logger.error("Replay incomplete; no records for: %s", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
