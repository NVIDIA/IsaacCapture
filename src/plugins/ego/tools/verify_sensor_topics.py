# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys
from mcap.reader import make_reader


def unpack_from(fmt, data, offset, label):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise SystemExit(f"Invalid FlatBuffer while reading {label} at byte {offset}")
    return struct.unpack_from(fmt, data, offset)


def field_position(data, table, field_id):
    (vtable_delta,) = unpack_from("<i", data, table, "vtable offset")
    vtable = table - vtable_delta
    (vtable_size,) = unpack_from("<H", data, vtable, "vtable size")
    entry = vtable + 4 + 2 * field_id
    if entry + 2 > vtable + vtable_size:
        return None
    (field_offset,) = unpack_from("<H", data, entry, f"field {field_id}")
    return None if field_offset == 0 else table + field_offset


def scalar_field(data, table, field_id, fmt, default=0):
    position = field_position(data, table, field_id)
    return (
        default
        if position is None
        else unpack_from(fmt, data, position, f"field {field_id}")[0]
    )


def record_data(data, label):
    (record_table,) = unpack_from("<I", data, 0, f"{label} root table")
    data_position = field_position(data, record_table, 0)
    timestamp_position = field_position(data, record_table, 1)
    if data_position is None or timestamp_position is None:
        raise SystemExit(f"{label} is missing data or DeviceDataTimestamp")
    (data_offset,) = unpack_from("<I", data, data_position, f"{label} data table")
    table = data_position + data_offset
    device_timestamp = unpack_from(
        "<qqq", data, timestamp_position, f"{label} timestamp"
    )[2]
    return table, device_timestamp


def vector_field(data, table, field_id, element_size, label):
    position = field_position(data, table, field_id)
    if position is None:
        return 0, 0
    (offset,) = unpack_from("<I", data, position, label)
    vector = position + offset
    (count,) = unpack_from("<I", data, vector, f"{label} length")
    begin = vector + 4
    if begin + count * element_size > len(data):
        raise SystemExit(f"Invalid FlatBuffer while reading {label} elements")
    return count, begin


def diagnostic_record(data, label):
    table, _ = record_data(data, label)
    version = scalar_field(data, table, 3, "<I")
    kind = scalar_field(data, table, 4, "<B")
    observed = scalar_field(data, table, 2, "<q")
    payload_position = field_position(data, table, 5)
    uid_length, _ = vector_field(data, table, 0, 1, f"{label} device UID")
    if (
        version != 2
        or kind not in (1, 2, 3, 4)
        or observed <= 0
        or not uid_length
        or payload_position is None
    ):
        raise SystemExit(
            f"{label} requires diagnostic version 2, a known event, UID, and observation time"
        )
    offset = unpack_from("<I", data, payload_position, "diagnostic event offset")[0]
    if not offset:
        raise SystemExit(f"{label} has an empty diagnostic event")
    payload = payload_position + offset
    record = unpack_from("<I", data, 0, "record")[0]
    timestamp = field_position(data, record, 1)
    sampled = unpack_from("<qqq", data, timestamp, "timestamp")[1]
    if observed != sampled:
        raise SystemExit(
            f"{label} diagnostic observation differs from sample timestamp"
        )
    if kind == 1:
        for field_id, fmt in enumerate(("<Q", "<i", "<Q", "<Q", "<Q")):
            scalar_field(data, payload, field_id, fmt)
    elif kind == 2:
        health = scalar_field(data, payload, 0, "<b")
        if health not in (0, 1, 2):
            raise SystemExit(f"{label} has invalid capture health")
        if health == 2:
            raise SystemExit(f"{label} reports incomplete capture")
        if scalar_field(data, payload, 4, "<Q") or scalar_field(data, payload, 5, "<Q"):
            raise SystemExit(f"{label} reports dropped capture data")
    elif kind == 3:
        if scalar_field(data, payload, 0, "<b") not in (0, 1, 2):
            raise SystemExit(f"{label} has invalid recovery stage")
    else:
        temperature = scalar_field(data, payload, 0, "<f", None)
        if temperature is not None and not math.isfinite(temperature):
            raise SystemExit(f"{label} temperature must be finite or absent")
        vector_field(data, payload, 1, 16, f"{label} successful properties")
    return kind


path, mode = sys.argv[1:3]
expect_imu = "--expect-imu" in sys.argv
expect_audio = "--expect-audio" in sys.argv
counts = Counter()
last_log_time = {}
non_monotonic = []
video_epochs = {}
imu = {
    "ego_imu/Accel": {"sensor": 0, "batches": 0, "samples": 0, "epochs": {}},
    "ego_imu/Gyro": {"sensor": 1, "batches": 0, "samples": 0, "epochs": {}},
}
audio_topics = ["ego_audio/Audio"]
if mode == "embedded":
    audio_topics.append("ego_media/Audio")
audio = {
    topic: {"chunks": 0, "samples": 0, "timeline": [], "epochs": {}}
    for topic in audio_topics
}

policy = None
expected_policy = None
for argument in sys.argv[3:]:
    if argument.startswith("--requested-profile="):
        manifest = Path(argument.split("=", 1)[1])
        if manifest.is_file():
            expected_policy = json.loads(manifest.read_text()).get(
                "record_device_diagnostics"
            )
    elif argument.startswith("--record-device-diagnostics="):
        expected_policy = argument.split("=", 1)[1]
if expected_policy not in (None, "on", "off"):
    raise SystemExit("Expected diagnostic recording policy must be on or off")
diagnostic_count = 0
with open(path, "rb") as stream:
    reader = make_reader(stream)
    for metadata in reader.iter_metadata():
        if metadata.name != "ego.recording":
            continue
        value = metadata.metadata.get("record_device_diagnostics")
        if (
            value not in ("on", "off")
            or metadata.metadata.get("diagnostic_version") != "2"
        ):
            raise SystemExit("Unsupported EGO recording policy")
        if policy is not None and policy != value:
            raise SystemExit("Conflicting EGO recording policies")
        policy = value
    if expected_policy is not None and policy != expected_policy:
        raise SystemExit(
            f"Expected diagnostic policy {expected_policy}, recorded {policy}"
        )
    summary = reader.get_summary()
    if policy == "off":
        if summary is None:
            raise SystemExit(
                "Diagnostics-off verification requires a complete channel inventory"
            )
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            if schema is not None and schema.name == "core.EgoDeviceStateRecord":
                raise SystemExit(f"Diagnostics off but channel exists: {channel.topic}")
    if policy is None:
        print(
            "  No EGO recording policy: checking diagnostic events without a policy guarantee"
        )

with open(path, "rb") as stream:
    for _schema, channel, message in make_reader(stream).iter_messages(
        log_time_order=False
    ):
        topic = channel.topic
        counts[topic] += 1
        previous = last_log_time.get(topic)
        if previous is not None and message.log_time < previous:
            non_monotonic.append(topic)
        last_log_time[topic] = message.log_time

        if _schema is not None and _schema.name == "core.EgoDeviceStateRecord":
            if policy == "off":
                raise SystemExit(f"Diagnostics off but message exists: {topic}")
            diagnostic_count += 1
            diagnostic_record(message.data, topic)

        if topic == "ego_metadata/ColorLeft":
            table, timestamp = record_data(message.data, topic)
            capture_epoch = scalar_field(message.data, table, 8, "<I")
            epoch = video_epochs.setdefault(
                capture_epoch, {"first": timestamp, "last": timestamp}
            )
            epoch["last"] = timestamp

        if topic in imu:
            table, record_timestamp = record_data(message.data, topic)
            state = imu[topic]
            sensor = scalar_field(message.data, table, 0, "<b")
            sequence = scalar_field(message.data, table, 1, "<Q")
            sample_rate = scalar_field(message.data, table, 2, "<I")
            capture_epoch = scalar_field(message.data, table, 5, "<I")
            epoch = state["epochs"].setdefault(
                capture_epoch,
                {
                    "batches": 0,
                    "samples": 0,
                    "last_sequence": None,
                    "first": None,
                    "last": None,
                },
            )
            sample_count, samples = vector_field(
                message.data, table, 4, 48, f"{topic} samples"
            )
            if sensor != state["sensor"]:
                raise SystemExit(f"{topic} contains sensor id {sensor}")
            if sample_rate <= 0:
                raise SystemExit(f"{topic} has invalid sample_rate_hz {sample_rate}")
            if sample_count == 0:
                raise SystemExit(f"{topic} contains an empty IMU batch")
            if state.get("rate") not in (None, sample_rate):
                raise SystemExit(
                    f"{topic} sample_rate_hz changed from {state['rate']} to {sample_rate}"
                )
            if (
                epoch["last_sequence"] is not None
                and sequence != epoch["last_sequence"] + 1
            ):
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} batch sequence is not contiguous: "
                    f"previous={epoch['last_sequence']} current={sequence}"
                )
            batch_timestamps = [
                unpack_from(
                    "<q",
                    message.data,
                    samples + index * 48 + 40,
                    f"{topic} sample timestamp",
                )[0]
                for index in range(sample_count)
            ]
            previous_sample = epoch["last"]
            for timestamp in batch_timestamps:
                if previous_sample is not None and timestamp <= previous_sample:
                    raise SystemExit(
                        f"{topic} raw device sample timestamp is not increasing: "
                        f"previous={previous_sample} current={timestamp}"
                    )
                previous_sample = timestamp
            if record_timestamp != batch_timestamps[-1]:
                raise SystemExit(
                    f"{topic} batch timestamp {record_timestamp} does not match its last sample "
                    f"{batch_timestamps[-1]}"
                )
            if epoch["first"] is None:
                epoch["first"] = batch_timestamps[0]
            epoch["last"] = batch_timestamps[-1]
            epoch["last_sequence"] = sequence
            epoch["batches"] += 1
            epoch["samples"] += sample_count
            state["rate"] = sample_rate
            state["batches"] += 1
            state["samples"] += sample_count

        if topic in audio:
            table, timestamp = record_data(message.data, topic)
            state = audio[topic]
            sequence = scalar_field(message.data, table, 0, "<Q")
            sample_rate = scalar_field(message.data, table, 1, "<I")
            channels = scalar_field(message.data, table, 2, "<H")
            bits = scalar_field(message.data, table, 3, "<H")
            sample_format = scalar_field(message.data, table, 4, "<b")
            sample_count = scalar_field(message.data, table, 5, "<I")
            capture_epoch = scalar_field(
                message.data, table, 7 if topic == "ego_media/Audio" else 8, "<I"
            )
            epoch = state["epochs"].setdefault(
                capture_epoch,
                {
                    "chunks": 0,
                    "samples": 0,
                    "first": None,
                    "last": None,
                    "last_sequence": None,
                    "first_sample_count": None,
                    "last_sample_count": None,
                },
            )
            profile = (sample_rate, channels, bits, sample_format)
            if profile != (48_000, 1, 16, 0):
                raise SystemExit(
                    f"{topic} has unsupported audio profile {profile}; expected (48000, 1, 16, S16LE)"
                )
            if sample_count == 0:
                raise SystemExit(f"{topic} contains an empty audio chunk")
            expected_bytes = sample_count * channels * (bits // 8)
            if topic == "ego_media/Audio":
                pcm_bytes, _pcm = vector_field(
                    message.data, table, 6, 1, f"{topic} PCM data"
                )
                if pcm_bytes != expected_bytes:
                    raise SystemExit(
                        f"{topic} PCM byte count {pcm_bytes} does not match sample_count/profile {expected_bytes}"
                    )
            else:
                byte_count = scalar_field(message.data, table, 7, "<I")
                if byte_count != expected_bytes:
                    raise SystemExit(
                        f"{topic} byte_count {byte_count} does not match sample_count/profile {expected_bytes}"
                    )
            if state.get("profile") not in (None, profile):
                raise SystemExit(
                    f"{topic} audio profile changed from {state['profile']} to {profile}"
                )
            if (
                epoch["last_sequence"] is not None
                and sequence != epoch["last_sequence"] + 1
            ):
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} chunk sequence is not contiguous: "
                    f"previous={epoch['last_sequence']} current={sequence}"
                )
            if epoch["last"] is not None and timestamp <= epoch["last"]:
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} raw device chunk timestamp is not increasing: "
                    f"previous={epoch['last']} current={timestamp}"
                )
            if epoch["first"] is None:
                epoch["first"] = timestamp
                epoch["first_sample_count"] = sample_count
            epoch["last"] = timestamp
            epoch["last_sample_count"] = sample_count
            epoch["last_sequence"] = sequence
            epoch["chunks"] += 1
            epoch["samples"] += sample_count
            state["profile"] = profile
            state["chunks"] += 1
            state["samples"] += sample_count
            state["timeline"].append(
                (capture_epoch, sequence, timestamp, sample_count, profile)
            )

expect_imu = expect_imu or any(counts[topic] for topic in imu)
expect_audio = expect_audio or any(counts[topic] for topic in audio)

required = [
    "ego_metadata/ColorLeft",
    "ego_metadata/ColorRight",
    "ego_calibration/Calibration",
]
if policy == "on" and diagnostic_count == 0:
    raise SystemExit("Diagnostics enabled but no EgoDeviceState records exist")
if mode == "embedded":
    required += ["ego_media/ColorLeft", "ego_media/ColorRight"]
if expect_imu:
    required += ["ego_imu/Accel", "ego_imu/Gyro"]
if expect_audio:
    required += ["ego_audio/Audio"]
    if mode == "embedded":
        required += ["ego_media/Audio"]
missing = [topic for topic in required if counts[topic] == 0]
for topic in sorted(counts):
    print(f"  {topic}: {counts[topic]}")
if missing:
    raise SystemExit(f"Missing or empty required MCAP topics: {missing}")
if non_monotonic:
    raise SystemExit(
        f"Non-monotonic MCAP log time on topics: {sorted(set(non_monotonic))}"
    )

if expect_imu or expect_audio:
    for capture_epoch, epoch in video_epochs.items():
        if epoch["last"] <= epoch["first"]:
            raise SystemExit(
                f"ColorLeft epoch {capture_epoch} has too few timestamps for auxiliary coverage"
            )

if expect_imu:
    for topic, state in imu.items():
        if set(state["epochs"]) != set(video_epochs):
            raise SystemExit(
                f"{topic} capture epochs differ from video: {sorted(state['epochs'])}"
            )
        for capture_epoch, epoch in sorted(state["epochs"].items()):
            if epoch["samples"] < 2 or epoch["last"] <= epoch["first"]:
                raise SystemExit(f"{topic} epoch {capture_epoch} has too few samples")
            video_span = (
                video_epochs[capture_epoch]["last"]
                - video_epochs[capture_epoch]["first"]
            )
            sample_span = epoch["last"] - epoch["first"]
            observed_rate = (epoch["samples"] - 1) * 1_000_000_000 / sample_span
            tolerance = max(5.0, state["rate"] * 0.05)
            if abs(observed_rate - state["rate"]) > tolerance:
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} raw-device cadence is {observed_rate:.3f} Hz; "
                    f"batches declare {state['rate']} Hz"
                )
            coverage = sample_span + 1_000_000_000 / state["rate"]
            if coverage < video_span * 0.90:
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} covers only {coverage / 1_000_000_000:.3f}s "
                    f"of the {video_span / 1_000_000_000:.3f}s video timeline"
                )
            print(
                f"  {topic}: epoch={capture_epoch} batches={epoch['batches']} samples={epoch['samples']} "
                f"declared_hz={state['rate']} raw_device_hz={observed_rate:.3f}"
            )
    if imu["ego_imu/Accel"]["rate"] != imu["ego_imu/Gyro"]["rate"]:
        raise SystemExit("Accel and Gyro declare different sample rates")

if expect_audio:
    for topic, state in audio.items():
        sample_rate = state["profile"][0]
        if set(state["epochs"]) != set(video_epochs):
            raise SystemExit(
                f"{topic} capture epochs differ from video: {sorted(state['epochs'])}"
            )
        for capture_epoch, epoch in sorted(state["epochs"].items()):
            if epoch["chunks"] < 2 or epoch["last"] <= epoch["first"]:
                raise SystemExit(f"{topic} epoch {capture_epoch} has too few chunks")
            video_span = (
                video_epochs[capture_epoch]["last"]
                - video_epochs[capture_epoch]["first"]
            )
            timestamp_span = epoch["last"] - epoch["first"]
            timestamped_counts = (
                epoch["samples"] - epoch["first_sample_count"],
                epoch["samples"] - epoch["last_sample_count"],
            )
            observed_rates = [
                count * 1_000_000_000 / timestamp_span for count in timestamped_counts
            ]
            observed_rate = min(
                observed_rates, key=lambda rate: abs(rate - sample_rate)
            )
            tolerance = max(5.0, sample_rate * 0.05)
            if abs(observed_rate - sample_rate) > tolerance:
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} raw-device cadence is {observed_rate:.3f} Hz; "
                    f"chunks declare {sample_rate} Hz"
                )
            duration = epoch["samples"] * 1_000_000_000 / sample_rate
            if duration < video_span * 0.90:
                raise SystemExit(
                    f"{topic} epoch {capture_epoch} contains only {duration / 1_000_000_000:.3f}s "
                    f"of PCM for the {video_span / 1_000_000_000:.3f}s video timeline"
                )
            print(
                f"  {topic}: epoch={capture_epoch} chunks={epoch['chunks']} samples={epoch['samples']} "
                f"duration_s={duration / 1_000_000_000:.3f} raw_device_hz={observed_rate:.3f}"
            )
    if mode == "embedded":
        metadata = audio["ego_audio/Audio"]
        media = audio["ego_media/Audio"]
        if metadata["timeline"] != media["timeline"]:
            raise SystemExit("Embedded PCM and audio-metadata timelines differ")
print("MCAP Footer, required topics, and per-topic log-time monotonicity: OK")
