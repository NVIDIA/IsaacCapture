# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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


def scalar_field(data, table, field_id, fmt, default):
    position = field_position(data, table, field_id)
    return (
        default
        if position is None
        else unpack_from(fmt, data, position, f"field {field_id}")[0]
    )


def video_record(data, embedded_media):
    # Metadata and embedded-video roots share field ids for data and timestamp;
    # their data tables share sequence/width/height/fps/format ids 1 through 5.
    (record_table,) = unpack_from("<I", data, 0, "root table")
    data_position = field_position(data, record_table, 0)
    timestamp_position = field_position(data, record_table, 1)
    if data_position is None or timestamp_position is None:
        raise SystemExit("Ego video record is missing data or DeviceDataTimestamp")
    (data_offset,) = unpack_from("<I", data, data_position, "video data table")
    data_table = data_position + data_offset
    timestamp = unpack_from("<qqq", data, timestamp_position, "DeviceDataTimestamp")[2]
    sequence = scalar_field(data, data_table, 1, "<Q", 0)
    width = scalar_field(data, data_table, 2, "<I", 0)
    height = scalar_field(data, data_table, 3, "<I", 0)
    fps = scalar_field(data, data_table, 4, "<I", 0)
    pixel_format = scalar_field(data, data_table, 5, "<B", 0)
    capture_epoch = scalar_field(data, data_table, 7 if embedded_media else 8, "<I", 0)
    return timestamp, sequence, (pixel_format, width, height, fps), capture_epoch


path, mode = sys.argv[1:3]
expected_counts = {
    "ColorLeft": int(sys.argv[3]),
    "ColorRight": int(sys.argv[4]),
}
manifest_path = Path(sys.argv[5])
requested_profiles = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_ids = {"mjpg": 0, "h264": 1, "h265": 2}
        for stream in expected_counts:
            requested = manifest["streams"][stream]
            requested_profiles[stream] = (
                format_ids[requested["format"]],
                requested["width"],
                requested["height"],
                requested["fps"],
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Invalid requested profile manifest {manifest_path}: {error}"
        ) from error

prefixes = ["ego_metadata"]
if mode == "embedded":
    prefixes.append("ego_media")
topics = {
    f"{prefix}/{stream}": {
        "stream": stream,
        "count": 0,
        "first": None,
        "last": None,
        "first_sequence": None,
        "last_sequence": None,
        "profile": None,
        "epochs": {},
        "timeline": [],
    }
    for prefix in prefixes
    for stream in expected_counts
}
with open(path, "rb") as stream:
    for _schema, channel, message in make_reader(stream).iter_messages(
        log_time_order=False
    ):
        state = topics.get(channel.topic)
        if state is None:
            continue
        timestamp, sequence, profile, capture_epoch = video_record(
            message.data, channel.topic.startswith("ego_media/")
        )
        epoch = state["epochs"].setdefault(
            capture_epoch,
            {
                "count": 0,
                "first": None,
                "last": None,
                "first_sequence": None,
                "last_sequence": None,
            },
        )
        if epoch["last"] is not None and timestamp <= epoch["last"]:
            raise SystemExit(
                f"Non-increasing raw device timestamp on {channel.topic} epoch {capture_epoch}: "
                f"previous={epoch['last']} current={timestamp}"
            )
        if epoch["first"] is None:
            epoch["first"] = timestamp
            epoch["first_sequence"] = sequence
        if epoch["last_sequence"] is not None:
            if sequence <= epoch["last_sequence"]:
                raise SystemExit(
                    f"Non-increasing sequence on {channel.topic} epoch {capture_epoch}: "
                    f"previous={epoch['last_sequence']} current={sequence}"
                )
            if sequence != epoch["last_sequence"] + 1:
                raise SystemExit(
                    f"Sequence gap on {channel.topic} epoch {capture_epoch}: "
                    f"previous={epoch['last_sequence']} current={sequence}"
                )
        if state["profile"] is not None and profile != state["profile"]:
            raise SystemExit(
                f"Profile changed on {channel.topic}: {state['profile']} -> {profile}"
            )
        state["last"] = timestamp
        state["last_sequence"] = sequence
        state["profile"] = profile
        state["count"] += 1
        state["timeline"].append((capture_epoch, sequence, timestamp, profile))
        epoch["last"] = timestamp
        epoch["last_sequence"] = sequence
        epoch["count"] += 1

for topic, state in topics.items():
    stream = state["stream"]
    if state["count"] != expected_counts[stream]:
        raise SystemExit(
            f"{topic} count {state['count']} differs from decoded {stream} count {expected_counts[stream]}"
        )
    if stream in requested_profiles and state["profile"] != requested_profiles[stream]:
        raise SystemExit(
            f"{topic} MCAP profile {state['profile']} differs from requested {requested_profiles[stream]}"
        )
    for capture_epoch, epoch in sorted(state["epochs"].items()):
        if epoch["count"] < 2 or epoch["last"] <= epoch["first"]:
            raise SystemExit(
                f"{topic} epoch {capture_epoch} has too few valid raw device timestamps"
            )
        declared_fps = state["profile"][3]
        if declared_fps <= 0:
            raise SystemExit(f"{topic} has invalid declared FPS {declared_fps}")
        observed_fps = (
            (epoch["count"] - 1) * 1_000_000_000 / (epoch["last"] - epoch["first"])
        )
        tolerance = max(0.5, declared_fps * 0.05)
        if abs(observed_fps - declared_fps) > tolerance:
            raise SystemExit(
                f"{topic} epoch {capture_epoch} raw-device cadence is {observed_fps:.3f} FPS; "
                f"the MCAP profile declares {declared_fps} FPS"
            )
        print(
            f"  {topic}: epoch={capture_epoch} count={epoch['count']} profile={state['profile']} "
            f"raw_device_fps={observed_fps:.3f} span_ns={epoch['last'] - epoch['first']}"
        )
if mode == "embedded":
    for stream in expected_counts:
        metadata = topics[f"ego_metadata/{stream}"]
        media = topics[f"ego_media/{stream}"]
        comparable = ("count", "profile", "timeline")
        if any(metadata[field] != media[field] for field in comparable):
            raise SystemExit(f"Embedded and metadata timelines differ for {stream}")
print("MCAP video counts and raw device timestamp cadence: OK")
