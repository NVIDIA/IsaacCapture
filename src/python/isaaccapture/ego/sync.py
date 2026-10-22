# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline EGO sample association, independent of the NV replay lifecycle."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING

from ._records import ByteVector, RecordDecoder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

STREAMS = ("left_rgb", "right_rgb", "accel", "gyro", "audio")
_TOPICS = {
    "orbbec_ego_metadata/ColorLeft": ("left_rgb", "core.OrbbecEgoFrameMetadataRecord"),
    "orbbec_ego_metadata/ColorRight": (
        "right_rgb",
        "core.OrbbecEgoFrameMetadataRecord",
    ),
    "orbbec_ego_imu/Accel": ("accel", "core.OrbbecEgoImuBatchRecord"),
    "orbbec_ego_imu/Gyro": ("gyro", "core.OrbbecEgoImuBatchRecord"),
    "orbbec_ego_audio/Audio": ("audio", "core.OrbbecEgoAudioChunkRecord"),
    "orbbec_ego_media/ColorLeft": ("left_rgb", "core.OrbbecEgoEncodedVideoFrameRecord"),
    "orbbec_ego_media/ColorRight": (
        "right_rgb",
        "core.OrbbecEgoEncodedVideoFrameRecord",
    ),
    "orbbec_ego_media/Audio": ("audio", "core.OrbbecEgoPcmAudioChunkRecord"),
    "orbbec_ego_calibration/Calibration": (
        "calibration",
        "core.OrbbecEgoCalibrationRecord",
    ),
}


@dataclass(frozen=True, slots=True)
class RecordReference:
    topic: str
    record_index: int
    sequence_number: int
    epoch: int
    available_time_ns: int
    sample_time_local_ns: int
    sample_time_raw_ns: int


@dataclass(frozen=True, slots=True)
class MediaReference:
    storage: str
    path: str
    topic: str | None
    record_index: int | None
    byte_offset: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    stream: str
    time_ns: int
    record: RecordReference
    media: MediaReference | None
    width: int
    height: int
    fps: int
    pixel_format: str
    sdk_metadata: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ImuSample:
    stream: str
    time_ns: int
    record: RecordReference
    sample_index: int
    sample_time_raw_ns: int
    sample_time_local_ns: int
    x_si: float
    y_si: float
    z_si: float
    temperature_c: float


@dataclass(frozen=True, slots=True)
class AudioSlice:
    record: RecordReference
    media: MediaReference | None
    segment_id: int
    first_sample_index: int
    sample_offset: int
    sample_count: int
    start_time_ns: int
    end_time_ns: int
    period_ns: float
    sample_rate_hz: int
    channels: int
    bits_per_sample: int
    continuity_verified: bool


@dataclass(frozen=True, slots=True)
class SyncWindow:
    epoch: int
    model_id: str
    model_segment: int
    start_time_ns: int
    end_time_ns: int
    reference_stream: str
    required_streams: tuple[str, ...]
    left_rgb: VideoFrame | None
    right_rgb: VideoFrame | None
    accel: tuple[ImuSample, ...]
    gyro: tuple[ImuSample, ...]
    audio: tuple[AudioSlice, ...]
    complete: bool
    reasons: tuple[str, ...]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(recording):
    path = Path(recording).expanduser().resolve()
    path = path / "metadata.mcap" if path.is_dir() else path
    if not path.is_file():
        raise ValueError(f"Recording does not exist: {path}")
    return path


def _load(recording):
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError(
            "EGO offline synchronization requires isaaccapture[ego]"
        ) from error

    path = _path(recording)
    records = {stream: [] for stream in STREAMS}
    media = {}
    models = []
    identities = []
    decoders = {}
    topic_counts = {}
    ignored = 0
    with path.open("rb") as source:
        reader = make_reader(source, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            if channel.topic not in _TOPICS:
                ignored += 1
                continue
            stream, expected = _TOPICS[channel.topic]
            if (
                schema is None
                or schema.encoding != "flatbuffer"
                or schema.name != expected
            ):
                raise ValueError(f"Unexpected schema for {channel.topic}")
            if schema.id not in decoders:
                decoders[schema.id] = RecordDecoder(schema.data, schema.name)
            record = decoders[schema.id].decode(message.data)
            data, timestamp = record["data"], record["timestamp"]
            index = topic_counts.get(channel.topic, 0)
            topic_counts[channel.topic] = index + 1
            ref = RecordReference(
                channel.topic,
                index,
                data.get("sequence_number", 0),
                data.get("capture_epoch", 0),
                timestamp["available_time_local_common_clock"],
                timestamp["sample_time_local_common_clock"],
                timestamp["sample_time_raw_device_clock"],
            )
            if stream == "calibration":
                timing = data.get("time_alignment")
                if timing and timing.get("json"):
                    model = json.loads(timing["json"])
                    model = {**model, "_capture_epoch": ref.epoch}
                    models.append(model)
                    if model.get("identity"):
                        identities.append(model["identity"])
                continue
            expected_sensor = {
                "left_rgb": 0,
                "right_rgb": 1,
                "accel": 0,
                "gyro": 1,
            }.get(stream)
            if (
                expected_sensor is not None
                and data.get("stream", data.get("sensor")) != expected_sensor
            ):
                raise ValueError(
                    f"Sensor discriminator disagrees with topic {channel.topic}"
                )
            item = {"data": data, "ref": ref, "media": None}
            if channel.topic.startswith("orbbec_ego_media/"):
                vector = data.get("encoded_data", data.get("pcm_data"))
                if not isinstance(vector, ByteVector):
                    raise ValueError(f"Missing embedded media on {channel.topic}")
                key = stream, ref.epoch, ref.sequence_number
                if key in media:
                    raise ValueError(
                        "Duplicate embedded media sequence within an epoch"
                    )
                item["media"] = MediaReference(
                    "mcap", str(path), channel.topic, index, vector.offset, vector.size
                )
                media[key] = item
            else:
                records[stream].append(item)
        for metadata in reader.iter_metadata():
            if metadata.name == "ego.time_alignment":
                encoded = metadata.metadata.get("json")
                if encoded:
                    models.append(json.loads(encoded))
    media_paths = set()
    sidecar_trailing_bytes = dict.fromkeys(("left_rgb", "right_rgb"), 0)
    issues = []
    scoped_issues = []

    def issue(message, streams=()):
        issues.append(message)
        scoped_issues.append((message, streams))

    for stream in STREAMS:
        byte_offset = 0
        retained_samples = {}
        seen = set()
        last_raw = {}
        raw_regressions = 0
        for item in records[stream]:
            data, ref = item["data"], item["ref"]
            if stream != "audio":
                raw_times = (
                    (
                        int(sample["sample_time_raw_device_clock_ns"])
                        for sample in data["samples"]
                    )
                    if stream in ("accel", "gyro")
                    else (ref.sample_time_raw_ns,)
                )
                for raw in raw_times:
                    if ref.epoch in last_raw and raw <= last_raw[ref.epoch]:
                        raw_regressions += 1
                    last_raw[ref.epoch] = raw
            key = stream, ref.epoch, ref.sequence_number
            if key in seen:
                raise ValueError(
                    f"Duplicate {stream} sequence within epoch {ref.epoch}"
                )
            seen.add(key)
            embedded = media.pop(key, None)
            if embedded:
                for field in (
                    "width",
                    "height",
                    "fps",
                    "pixel_format",
                    "sample_count",
                    "sample_rate_hz",
                    "channel_count",
                    "bits_per_sample",
                    "sample_format",
                    "timing",
                ):
                    if field in data and data[field] != embedded["data"].get(field):
                        raise ValueError(
                            f"Embedded {stream} disagrees with its metadata: {field}"
                        )
                if ref.sample_time_raw_ns != embedded["ref"].sample_time_raw_ns:
                    raise ValueError(
                        "Embedded media and metadata have different sample timestamps"
                    )
                item["media"] = embedded["media"]
            elif stream in ("left_rgb", "right_rgb"):
                extension = _format(data["pixel_format"])
                media_path = (
                    path.parent
                    / "raw"
                    / f"{'ColorLeft' if stream == 'left_rgb' else 'ColorRight'}.{extension}"
                )
                size = data["encoded_bytes"]
                if (
                    media_path.is_file()
                    and byte_offset + size <= media_path.stat().st_size
                ):
                    item["media"] = MediaReference(
                        "sidecar", str(media_path), None, None, byte_offset, size
                    )
                    media_paths.add(media_path)
                byte_offset += size
            elif stream == "audio":
                media_path = path.parent / "Audio.wav"
                size, offset = data["byte_count"], data["wav_data_offset"]
                if media_path.is_file() and offset + size <= media_path.stat().st_size:
                    item["media"] = MediaReference(
                        "sidecar", str(media_path), None, None, offset, size
                    )
                    media_paths.add(media_path)
            if stream == "audio":
                if not all(
                    data[field] > 0
                    for field in ("sample_count", "sample_rate_hz", "channel_count")
                ):
                    raise ValueError(
                        "Audio sample count, rate, and channels must be positive"
                    )
                expected_bytes = (
                    data["sample_count"]
                    * data["channel_count"]
                    * data["bits_per_sample"]
                    // 8
                )
                if (
                    data["bits_per_sample"] != 16
                    or data["sample_format"] != 0
                    or data["byte_count"] != expected_bytes
                ):
                    raise ValueError("Invalid S16LE audio chunk size or format")
                if item["media"] and item["media"].byte_count != expected_bytes:
                    raise ValueError(
                        "Embedded audio byte length disagrees with sample count"
                    )
                timing = data.get("timing")
                if not timing:
                    timing = {
                        "segment_id": 0,
                        "first_sample_index": retained_samples.get(ref.epoch, 0),
                        "continuity_known": False,
                        "discontinuity": False,
                    }
                    item["inferred_audio_index"] = True
                item["timing"] = timing
                retained_samples[ref.epoch] = (
                    timing["first_sample_index"] + data["sample_count"]
                )
        if raw_regressions:
            issue(
                f"{stream}: {raw_regressions} nonmonotonic raw sample timestamps within an epoch",
                (stream,),
            )
        if stream in ("left_rgb", "right_rgb") and records[stream]:
            refs = [
                item["media"]
                for item in records[stream]
                if item["media"] and item["media"].storage == "sidecar"
            ]
            if refs:
                size = Path(refs[0].path).stat().st_size
                if size < byte_offset:
                    issue(
                        f"{stream}: sidecar is shorter than its metadata index",
                        (stream,),
                    )
                # Host polling can end before the plugin flushes its final frames.
                # Unindexed suffixes have no associations or verified frame boundaries.
                sidecar_trailing_bytes[stream] = max(0, size - byte_offset)
    if media:
        issue(
            f"{len(media)} embedded media records have no canonical metadata",
            tuple({key[0] for key in media}),
        )
    identity = identities[-1] if identities else {}
    if any(value != identity for value in identities):
        issue("Device identity changes within recording")
    capabilities = path.parent / "capabilities.txt"
    if capabilities.is_file():
        text = capabilities.read_text()
        for key, pattern in {
            "serial": r"\bserial=(\S+)",
            "firmware": r"\bfirmware=(\S+)",
            "sdk_version": r"SDK version=(\S+)",
        }.items():
            match = re.search(pattern, text)
            if match:
                if key in identity and identity[key] != match[1]:
                    issue(f"Conflicting recorded {key}")
                identity = {**identity, key: match[1]}
    return {
        "path": path,
        "records": records,
        "models": models,
        "identity": identity,
        "media_paths": media_paths,
        "sidecar_trailing_bytes": sidecar_trailing_bytes,
        "issues": issues,
        "scoped_issues": scoped_issues,
        "ignored_records": ignored,
    }


def _format(value):
    try:
        return {0: "mjpg", 1: "h264", 2: "h265"}[value]
    except KeyError as error:
        raise ValueError(f"Unknown EGO pixel format: {value}") from error


def recording_fingerprint(recording: str | Path) -> dict[str, str | dict[str, str]]:
    """Hash the MCAP and the media files referenced by its canonical records."""
    return _fingerprint(_load(recording))


def _fingerprint(loaded):
    return {
        "mcap_sha256": _sha256(loaded["path"]),
        "media_sha256": {
            str(path.relative_to(loaded["path"].parent)): _sha256(path)
            for path in sorted(loaded["media_paths"])
        },
    }


def _integer(value, name, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{name} must be an integer"
            + (f" >= {minimum}" if minimum is not None else "")
        )
    return value


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _required(required_streams, reference_stream, default=False):
    if reference_stream not in ("left_rgb", "right_rgb"):
        raise ValueError("reference_stream must be left_rgb or right_rgb")
    if required_streams is None and default:
        return STREAMS
    if isinstance(required_streams, str) or not required_streams:
        raise ValueError("required_streams must be an explicit nonempty collection")
    values = tuple(dict.fromkeys(required_streams))
    if any(value not in STREAMS for value in values) or reference_stream not in values:
        raise ValueError(
            "required_streams must name known streams and include reference_stream"
        )
    return values


def _profile_model(loaded, required, reference):
    snapshots = [
        value
        for value in loaded["models"]
        if value.get("capture_profile", {}).get("status") == "validated"
        and value.get("host_bridge")
    ]
    if not snapshots:
        return None
    profile = snapshots[-1]["capture_profile"]
    if any(value["capture_profile"] != profile for value in snapshots):
        raise ValueError("Conflicting embedded capture profiles")
    if (
        profile.get("version") != 1
        or profile.get("scope") != "capture_profile"
        or profile.get("reference_stream") != "left_rgb"
    ):
        raise ValueError("Invalid embedded capture profile")
    sdk_digest = profile.get("identity", {}).get("sdk_binary_sha256")
    if not isinstance(sdk_digest, str) or not re.fullmatch("[0-9a-f]{64}", sdk_digest):
        raise ValueError("Embedded capture profile requires sdk_binary_sha256")
    duration = _integer(
        profile.get("max_capture_duration_ns"), "max_capture_duration_ns", 1
    )
    validated_duration = _integer(
        profile.get("validation", {}).get("held_out_duration_ns"),
        "held_out_duration_ns",
        900_000_000_000,
    )
    if duration > validated_duration:
        raise ValueError("Capture profile duration exceeds its independent validation")
    for key in ("identity", "profile"):
        if any(value.get(key) != profile.get(key) for value in snapshots):
            raise ValueError(f"Embedded capture profile {key} mismatch")
    if loaded["identity"].get("sdk_binary_sha256") != sdk_digest:
        raise ValueError("Embedded capture profile SDK binary digest mismatch")
    epochs = {}
    for value in snapshots:
        bridge = value["host_bridge"]
        epoch = _integer(bridge["epoch"], "epoch", 0)
        if epoch != value.get("_capture_epoch", epoch):
            raise ValueError("Capture bridge epoch disagrees with its wrapper")
        anchor = _integer(bridge["reference_anchor_ns"], "reference_anchor_ns")
        lo = _integer(bridge.get("valid_from_ns"), "valid_from_ns")
        hi = _integer(bridge.get("valid_to_ns"), "valid_to_ns")
        if lo != anchor or hi != anchor + duration:
            raise ValueError("Capture bridge validity must match its verified duration")
        if bridge.get("status") not in ("active", "expired"):
            raise ValueError("Capture bridge requires an active or expired status")
        expired_at = bridge.get("expired_at_reference_ns")
        if expired_at is not None:
            if (
                bridge["status"] != "expired"
                or _integer(expired_at, "expired_at_reference_ns") < hi
            ):
                raise ValueError(
                    "Capture bridge expiry contradicts its validity interval"
                )
        first = bridge.get("first_mapped_raw_ns")
        if not isinstance(first, dict):
            raise ValueError("Capture bridge requires first_mapped_raw_ns boundaries")
        for stream, raw in first.items():
            if stream not in profile.get("streams", {}):
                raise ValueError("Capture bridge names an uncalibrated stream")
            _integer(raw, "first_mapped_raw_ns")
        if epoch in epochs and epochs[epoch]["anchor"] != anchor:
            raise ValueError("Conflicting capture clock origins within an epoch")
        if (
            epochs.get(epoch, {}).get("status") == "expired"
            and bridge["status"] != "expired"
        ):
            raise ValueError(
                "An expired capture bridge cannot become active within an epoch"
            )
        previous = epochs.get(epoch, {}).get("first", {})
        if any(first.get(stream) != raw for stream, raw in previous.items()):
            raise ValueError(
                "Conflicting first mapped sample boundaries within an epoch"
            )
        epochs[epoch] = {
            "anchor": anchor,
            "lo": lo,
            "hi": hi,
            "first": first,
            "status": bridge["status"],
        }
    result = {
        key: profile[key]
        for key in (
            "version",
            "status",
            "model_id",
            "reference_stream",
            "identity",
            "profile",
            "validation",
        )
    }
    result.update(scope="recording", epochs=[], source="embedded_capture_profile")
    for epoch, value in epochs.items():
        anchor = value["anchor"]
        maps = {}
        for stream, mapping in profile.get("streams", {}).items():
            if stream == "audio" or stream not in STREAMS:
                raise ValueError("Capture profiles may only align RGB/IMU streams")
            offset = _integer(mapping.get("offset_ns"), "offset_ns")
            scale = _positive(mapping.get("scale"), "scale")
            if scale != 1:
                raise ValueError("Reusable capture profiles require unit stream scale")
            if stream not in value["first"]:
                continue
            if not value["lo"] <= value["first"][stream] + offset < value["hi"]:
                raise ValueError("First mapped sample is outside the verified interval")
            maps[stream] = {
                "source_anchor_ns": anchor,
                "reference_anchor_ns": anchor + offset,
                "scale": scale,
                "valid_from_raw_ns": value["first"][stream],
                "validation_start_ns": value["lo"],
                "validation_end_ns": value["hi"],
            }
        times = [
            item["ref"].sample_time_raw_ns
            for item in loaded["records"]["left_rgb"]
            if item["ref"].epoch == epoch
        ]
        if times:
            result["epochs"].append(
                {
                    "epoch": epoch,
                    "valid_from_ns": value["lo"],
                    "valid_to_ns": value["hi"],
                    "streams": maps,
                }
            )
    return result


def _select_model(loaded, alignment, required, reference):
    external = alignment is not None
    if isinstance(alignment, (str, Path)):
        alignment = json.loads(Path(alignment).read_text())
    elif alignment is not None:
        alignment = json.loads(json.dumps(alignment, allow_nan=False))
    if alignment is not None and not isinstance(alignment, dict):
        raise ValueError("Alignment must be a JSON object")
    if alignment is None:
        candidates = [
            {key: value for key, value in m.items() if not key.startswith("_")}
            for m in loaded["models"]
            if m.get("status") == "validated" and m.get("scope") == "recording"
        ]
        if candidates and any(candidate != candidates[-1] for candidate in candidates):
            raise ValueError("Conflicting embedded recording alignment models")
        alignment = (
            candidates[-1]
            if candidates
            else _profile_model(loaded, required, reference)
        )
        if alignment is None and loaded["models"]:
            alignment = loaded["models"][-1]
    return alignment, external


def _validate_model(loaded, model, external, required, reference):
    # The selectable window camera does not change the model's left-camera clock.
    if not isinstance(model, dict) or model.get("status") != "validated":
        raise ValueError("A validated recording-specific alignment model is required")
    if (
        model.get("version") != 1
        or model.get("scope") != "recording"
        or model.get("reference_stream") != "left_rgb"
        or not model.get("model_id")
    ):
        raise ValueError(
            "Invalid alignment version, scope, reference stream, or model_id"
        )
    issues = [
        message
        for message, streams in loaded["scoped_issues"]
        if not streams or set(required).intersection(streams)
    ]
    if issues:
        raise ValueError("; ".join(issues))
    identity = model.get("identity", {})
    for key in ("serial", "firmware", "sdk_version"):
        if not identity.get(key) or identity[key] != loaded["identity"].get(key):
            raise ValueError(
                f"Alignment {key} does not match independently recorded identity"
            )
    if "sdk_binary_sha256" in identity:
        digest = identity["sdk_binary_sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError(
                "Alignment sdk_binary_sha256 must be a lowercase SHA-256 digest"
            )
        if digest != loaded["identity"].get("sdk_binary_sha256"):
            raise ValueError(
                "Alignment SDK binary digest does not match recorded identity"
            )
    if external and model.get("recording_fingerprint") != _fingerprint(loaded):
        raise ValueError("Alignment recording/media fingerprint mismatch")
    profile = model.get("profile", {})
    for stream, items in loaded["records"].items():
        if stream not in required:
            continue
        for item in items:
            data = item["data"]
            if stream.endswith("rgb"):
                actual = {
                    "format": _format(data["pixel_format"]),
                    **{key: data[key] for key in ("width", "height", "fps")},
                }
                if actual != profile.get("video"):
                    raise ValueError(f"Alignment video profile mismatch on {stream}")
            elif stream in ("accel", "gyro") and data["sample_rate_hz"] != profile.get(
                "imu_rate_hz"
            ):
                raise ValueError("Alignment IMU profile mismatch")
            elif stream == "audio":
                actual = {
                    "sample_rate_hz": data["sample_rate_hz"],
                    "channels": data["channel_count"],
                    "bits_per_sample": data["bits_per_sample"],
                }
                if actual != profile.get("audio"):
                    raise ValueError("Alignment audio profile mismatch")
    validation = model.get("validation", {})
    error = _integer(validation.get("max_abs_error_ns"), "max_abs_error_ns", 0)
    uncertainty = _integer(validation.get("uncertainty_ns"), "uncertainty_ns", 0)
    _integer(
        validation.get("held_out_duration_ns"),
        "held_out_duration_ns",
        900_000_000_000,
    )
    fps = _positive(profile.get("video", {}).get("fps"), "fps")
    if error + uncertainty > 500_000_000 / fps:
        raise ValueError("Independent validation exceeds the half-frame error budget")
    evidence = validation.get("evidence")
    if (
        not isinstance(validation.get("method"), str)
        or not validation["method"].strip()
        or not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(value, str) or not value.strip() for value in evidence)
    ):
        raise ValueError("Independent validation method and evidence are required")
    segments = model.get("epochs")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Alignment must define epoch validity intervals")
    ranges = {}
    for segment in segments:
        epoch = _integer(segment.get("epoch"), "epoch", 0)
        lo = _integer(segment.get("valid_from_ns"), "valid_from_ns")
        hi = _integer(segment.get("valid_to_ns"), "valid_to_ns")
        if hi <= lo or any(lo < b and a < hi for a, b in ranges.get(epoch, [])):
            raise ValueError(
                "Alignment validity intervals must be nonempty and nonoverlapping within an epoch"
            )
        ranges.setdefault(epoch, []).append((lo, hi))
        maps = segment.get("streams", {})
        for stream in required:
            if stream not in maps:
                raise ValueError(f"Alignment is missing required stream {stream}")
        for stream, mapping in maps.items():
            if stream not in STREAMS:
                raise ValueError(f"Unknown alignment stream {stream}")
            if stream == "audio":
                audio_ranges = {}
                for audio in mapping.get("segments", []):
                    sid = _integer(audio.get("segment_id"), "segment_id", 0)
                    first = _integer(
                        audio.get("first_sample_index"), "first_sample_index", 0
                    )
                    count = _integer(audio.get("sample_count"), "sample_count", 1)
                    _integer(audio.get("reference_anchor_ns"), "reference_anchor_ns")
                    _positive(audio.get("period_ns"), "period_ns")
                    if stream in required:
                        validation_lo = _integer(
                            audio.get("validation_start_ns"), "validation_start_ns"
                        )
                        validation_hi = _integer(
                            audio.get("validation_end_ns"), "validation_end_ns"
                        )
                        if validation_hi <= validation_lo:
                            raise ValueError(
                                "Audio independent validation bounds are empty"
                            )
                    if type(audio.get("continuity_verified", False)) is not bool:
                        raise ValueError(
                            "continuity_verified must be a boolean backed by evidence"
                        )
                    if any(
                        first < b and a < first + count
                        for a, b in audio_ranges.get(sid, [])
                    ):
                        raise ValueError("Overlapping audio sample ranges")
                    audio_ranges.setdefault(sid, []).append((first, first + count))
                if stream in required and not audio_ranges:
                    raise ValueError("Required audio alignment has no sample segments")
            else:
                _integer(mapping.get("source_anchor_ns"), "source_anchor_ns")
                _integer(mapping.get("reference_anchor_ns"), "reference_anchor_ns")
                _positive(mapping.get("scale"), "scale")
                if stream in required and stream != model["reference_stream"]:
                    validation_lo = _integer(
                        mapping.get("validation_start_ns"), "validation_start_ns"
                    )
                    validation_hi = _integer(
                        mapping.get("validation_end_ns"), "validation_end_ns"
                    )
                    if not validation_lo <= lo < hi <= validation_hi:
                        raise ValueError(
                            "Alignment interval exceeds independent validation bounds"
                        )
                if "valid_from_raw_ns" in mapping:
                    _integer(mapping["valid_from_raw_ns"], "valid_from_raw_ns")
                if stream == model["reference_stream"] and (
                    mapping["source_anchor_ns"] != mapping["reference_anchor_ns"]
                    or mapping["scale"] != 1
                ):
                    raise ValueError(
                        "Reference stream must preserve its raw sample clock"
                    )
    return model


def _map_time(raw, mapping):
    # Subtract integer anchors before scaling; epoch-sized floats lose sample precision.
    return mapping["reference_anchor_ns"] + round(
        (raw - mapping["source_anchor_ns"]) * mapping["scale"]
    )


def _mapped(loaded, model, required):
    mapped = {stream: [] for stream in STREAMS}
    occupied = {stream: {} for stream in STREAMS}
    for segment_index, segment in enumerate(model["epochs"]):
        for stream, mapping in segment["streams"].items():
            if stream == "audio" or stream not in required:
                continue
            for item in loaded["records"][stream]:
                data, ref = item["data"], item["ref"]
                if ref.epoch != segment["epoch"]:
                    continue
                previous = occupied[stream].setdefault(
                    (ref.epoch, ref.record_index), []
                )
                first_index = None
                last_index = None
                samples = data["samples"] if stream in ("accel", "gyro") else [None]
                for index, sample in enumerate(samples):
                    raw = (
                        int(sample["sample_time_raw_device_clock_ns"])
                        if sample is not None
                        else ref.sample_time_raw_ns
                    )
                    if raw < mapping.get("valid_from_raw_ns", raw):
                        continue
                    time = _map_time(raw, mapping)
                    if not segment["valid_from_ns"] <= time < segment["valid_to_ns"]:
                        continue
                    if any(lo <= index < hi for lo, hi in previous):
                        raise ValueError(
                            "A source sample maps into more than one model interval"
                        )
                    first_index = index if first_index is None else first_index
                    last_index = index
                    if sample is not None:
                        value = ImuSample(
                            stream,
                            time,
                            ref,
                            index,
                            raw,
                            int(sample["sample_time_local_common_clock_ns"]),
                            float(sample["x_si"]),
                            float(sample["y_si"]),
                            float(sample["z_si"]),
                            float(sample["temperature_c"]),
                        )
                    else:
                        value = VideoFrame(
                            stream,
                            time,
                            ref,
                            item["media"],
                            data["width"],
                            data["height"],
                            data["fps"],
                            _format(data["pixel_format"]),
                            tuple(
                                (entry["key"], entry["value"])
                                for entry in data.get("sdk_metadata", [])
                            ),
                        )
                    mapped[stream].append((segment_index, value))
                if first_index is not None:
                    previous.append((first_index, last_index + 1))
    return mapped


def _audio_index(loaded, segment):
    spans = []
    for item in loaded["records"]["audio"]:
        ref, data, timing = item["ref"], item["data"], item["timing"]
        if ref.epoch != segment["epoch"]:
            continue
        for mapping in segment["streams"].get("audio", {}).get("segments", []):
            if timing["segment_id"] != mapping["segment_id"]:
                continue
            first, anchor = (
                mapping["first_sample_index"],
                mapping["reference_anchor_ns"],
            )
            period = Fraction(str(mapping["period_ns"]))
            begin = max(timing["first_sample_index"], first)
            end = min(
                timing["first_sample_index"] + data["sample_count"],
                first + mapping["sample_count"],
            )
            # A retained segment can extend beyond its independently measured
            # interval. Its tail has no trusted timing merely because PCM exists.
            begin = max(
                begin,
                first
                + math.ceil(Fraction(mapping["validation_start_ns"] - anchor) / period),
            )
            end = min(
                end,
                first
                + math.ceil(Fraction(mapping["validation_end_ns"] - anchor) / period),
            )
            if end > begin:
                spans.append(
                    (
                        anchor + (begin - first) * period,
                        anchor + (end - first) * period,
                        item,
                        mapping,
                        period,
                        begin,
                        end,
                    )
                )
    spans.sort(key=lambda value: value[0])
    if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
        raise ValueError("Audio chunks overlap within a model interval")
    return spans, [value[0] for value in spans], [value[1] for value in spans]


def _audio_slices(index, lo, hi):
    spans, starts, ends = index
    result = []
    # Index once per model interval; a long recording must not rescan all PCM
    # chunks for every frame window.
    for _, _, item, mapping, period, lower, upper in spans[
        bisect_right(ends, lo) : bisect_left(starts, hi)
    ]:
        ref, data, timing = item["ref"], item["data"], item["timing"]
        first, anchor = mapping["first_sample_index"], mapping["reference_anchor_ns"]
        begin = max(lower, first + math.ceil(Fraction(lo - anchor) / period))
        end = min(upper, first + math.ceil(Fraction(hi - anchor) / period))
        if end <= begin:
            continue
        offset = begin - timing["first_sample_index"]
        frame_bytes = data["channel_count"] * data["bits_per_sample"] // 8
        media = item["media"]
        if media:
            media = replace(
                media,
                byte_offset=media.byte_offset + offset * frame_bytes,
                byte_count=(end - begin) * frame_bytes,
            )
        result.append(
            AudioSlice(
                ref,
                media,
                timing["segment_id"],
                begin,
                offset,
                end - begin,
                anchor + round((begin - first) * period),
                anchor + round((end - first) * period),
                float(period),
                data["sample_rate_hz"],
                data["channel_count"],
                data["bits_per_sample"],
                bool(
                    timing["continuity_known"]
                    or mapping.get("continuity_verified", False)
                ),
            )
        )
    return tuple(sorted(result, key=lambda value: value.start_time_ns))


def _windows(loaded, model, required, reference, epoch, start, end):
    mapped = _mapped(loaded, model, required)
    used_video = set()
    used_audio = {}
    for segment_index, segment in enumerate(model["epochs"]):
        if epoch is not None and epoch != segment["epoch"]:
            continue
        audio_index = (
            _audio_index(loaded, segment) if "audio" in required else ([], [], [])
        )
        values = {
            stream: sorted(
                (value for index, value in mapped[stream] if index == segment_index),
                key=lambda value: value.time_ns,
            )
            for stream in STREAMS
            if stream != "audio"
        }
        times = {
            stream: [value.time_ns for value in items]
            for stream, items in values.items()
        }
        refs = values[reference]
        for current, following in zip(refs, refs[1:]):
            lo, hi = current.time_ns, following.time_ns
            reasons = []
            if hi <= lo:
                continue
            period = 1_000_000_000 / current.fps
            gap = (
                following.record.sequence_number != current.record.sequence_number + 1
                or hi - lo > 1.5 * period
            )
            if gap:
                hi = min(hi, lo + round(period))
                reasons.append("reference_gap")
            if (start is not None and lo < start) or (end is not None and hi > end):
                continue
            frames = {"left_rgb": None, "right_rgb": None, reference: current}
            other = "right_rgb" if reference == "left_rgb" else "left_rgb"
            # SDK eye sequence counters can start one cycle apart; corrected
            # sample times define stereo correspondence, not equal sequence IDs.
            position = bisect_left(times[other], lo)
            candidates = [
                values[other][index]
                for index in (position - 1, position)
                if 0 <= index < len(values[other])
            ]
            candidates = [
                value
                for value in candidates
                if abs(value.time_ns - lo) <= 2_000_000
                and (other, value.record.epoch, value.record.record_index)
                not in used_video
            ]
            if candidates:
                pair = min(candidates, key=lambda value: abs(value.time_ns - lo))
                frames[other] = pair
                used_video.add((other, pair.record.epoch, pair.record.record_index))
            samples = {}
            for stream in ("accel", "gyro"):
                a, b = bisect_left(times[stream], lo), bisect_left(times[stream], hi)
                samples[stream] = tuple(values[stream][a:b])
                if stream in required and samples[stream]:
                    nominal = 1_000_000_000 / model["profile"]["imu_rate_hz"]
                    sample_times = [value.time_ns for value in samples[stream]]
                    if (
                        sample_times[0] - lo > 1.5 * nominal
                        or hi - sample_times[-1] > 1.5 * nominal
                        or any(
                            not 0 < y - x <= 1.5 * nominal
                            for x, y in zip(sample_times, sample_times[1:])
                        )
                    ):
                        reasons.append(f"{stream}_gap")
            audio = _audio_slices(audio_index, lo, hi)
            for value in audio:
                key = value.record.epoch, value.record.record_index
                previous = used_audio.setdefault(key, [])
                begin, stop = (
                    value.sample_offset,
                    value.sample_offset + value.sample_count,
                )
                if any(begin < b and a < stop for a, b in previous):
                    raise ValueError("An audio sample maps into more than one window")
                previous.append((begin, stop))
            for stream in required:
                present = (
                    frames.get(stream)
                    if stream.endswith("rgb")
                    else (audio if stream == "audio" else samples[stream])
                )
                if not present:
                    reasons.append(f"missing_{stream}")
                if stream.endswith("rgb") and present and present.media is None:
                    reasons.append(f"missing_{stream}_media")
            if "audio" in required and audio:
                if any(value.media is None for value in audio):
                    reasons.append("missing_audio_media")
                if not all(value.continuity_verified for value in audio):
                    reasons.append("audio_continuity_unverified")
                if len({value.segment_id for value in audio}) > 1:
                    reasons.append("audio_segment_boundary")
                if any(
                    not any(
                        mapping["segment_id"] == value.segment_id
                        and mapping["first_sample_index"] <= value.first_sample_index
                        and value.first_sample_index + value.sample_count
                        <= mapping["first_sample_index"] + mapping["sample_count"]
                        and mapping["validation_start_ns"] <= lo
                        and hi <= mapping["validation_end_ns"]
                        for mapping in segment["streams"]["audio"]["segments"]
                    )
                    for value in audio
                ):
                    reasons.append("audio_validation_boundary")
                tolerance = max(value.period_ns for value in audio)
                if (
                    audio[0].start_time_ns - lo > tolerance
                    or hi - audio[-1].end_time_ns > tolerance
                    or any(
                        abs(b.start_time_ns - a.end_time_ns) > tolerance
                        for a, b in zip(audio, audio[1:])
                    )
                ):
                    reasons.append("audio_gap")
            yield SyncWindow(
                segment["epoch"],
                segment.get("model_id", model["model_id"]),
                segment_index,
                lo,
                hi,
                reference,
                required,
                frames["left_rgb"],
                frames["right_rgb"],
                samples["accel"],
                samples["gyro"],
                audio,
                not reasons,
                tuple(reasons),
            )


def _arguments(
    required_streams, reference_stream, epoch, start_time_ns, end_time_ns, default=False
):
    required = _required(required_streams, reference_stream, default)
    if epoch is not None:
        _integer(epoch, "epoch", 0)
    for name, value in (("start_time_ns", start_time_ns), ("end_time_ns", end_time_ns)):
        if value is not None:
            _integer(value, name)
    if (
        start_time_ns is not None
        and end_time_ns is not None
        and end_time_ns <= start_time_ns
    ):
        raise ValueError("end_time_ns must exceed start_time_ns")
    return required


def iter_sync_windows(
    recording: str | Path,
    *,
    required_streams: Iterable[str],
    alignment: dict[str, Any] | str | Path | None = None,
    reference_stream: str = "left_rgb",
    epoch: int | None = None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> Iterator[SyncWindow]:
    """Yield immutable half-open reference-frame windows from a validated model.

    Full windows only are returned within a requested range. Epoch/model tails
    and samples outside those windows remain visible in analyze_recording_sync.
    """
    required = _arguments(
        required_streams, reference_stream, epoch, start_time_ns, end_time_ns
    )
    loaded = _load(recording)
    for stream in required:
        if not any(
            epoch is None or item["ref"].epoch == epoch
            for item in loaded["records"][stream]
        ):
            raise ValueError(f"Required stream is absent: {stream}")
    model, external = _select_model(loaded, alignment, required, reference_stream)
    _validate_model(loaded, model, external, required, reference_stream)
    yield from _windows(
        loaded, model, required, reference_stream, epoch, start_time_ns, end_time_ns
    )


def _sample_spans(samples):
    current = None
    for sample in samples:
        if (
            current is not None
            and current[0] == sample.record
            and current[1] + current[2] == sample.sample_index
        ):
            current[2] += 1
            current[4] = sample.time_ns
        else:
            if current is not None:
                yield current
            current = [
                sample.record,
                sample.sample_index,
                1,
                sample.time_ns,
                sample.time_ns,
            ]
    if current is not None:
        yield current


def _sample_ranges(samples):
    return [
        {
            "record": asdict(ref),
            "first_sample_index": first,
            "sample_count": count,
            "first_time_ns": lo,
            "last_time_ns": hi,
        }
        for ref, first, count, lo, hi in _sample_spans(samples)
    ]


def _window_index(window):
    # Reports carry batch/sample ranges; copying every XYZ value into JSON makes
    # a 30-minute audit several gigabytes. Full values stay in the iterator API.
    result = {
        key: getattr(window, key)
        for key in (
            "epoch",
            "model_id",
            "model_segment",
            "start_time_ns",
            "end_time_ns",
            "reference_stream",
            "required_streams",
            "complete",
            "reasons",
        )
    }
    for stream in ("left_rgb", "right_rgb"):
        frame = getattr(window, stream)
        result[stream] = asdict(frame) if frame else None
    result["accel"] = _sample_ranges(window.accel)
    result["gyro"] = _sample_ranges(window.gyro)
    result["audio"] = [asdict(value) for value in window.audio]
    return result


def _merge_ranges(ranges):
    result = []
    for lo, hi in sorted(ranges):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def _unassigned(loaded, assigned, assigned_audio, epoch):
    result = {stream: [] for stream in ("accel", "gyro", "audio")}
    for stream in result:
        for item in loaded["records"][stream]:
            ref, data = item["ref"], item["data"]
            if epoch is not None and ref.epoch != epoch:
                continue
            if stream == "audio":
                count = data["sample_count"]
                ranges = _merge_ranges(
                    assigned_audio.get((ref.epoch, ref.record_index), [])
                )
            else:
                count = len(data["samples"])
                ranges = _merge_ranges(
                    assigned[stream].get((ref.epoch, ref.record_index), [])
                )
            previous = 0
            for lo, hi in [*ranges, (count, count)]:
                if lo > previous:
                    result[stream].append(
                        {
                            "record": asdict(ref),
                            "first_sample_index": previous,
                            "sample_count": lo - previous,
                        }
                    )
                previous = max(previous, hi)
    return result


def analyze_recording_sync(
    recording: str | Path,
    *,
    required_streams: Iterable[str] | None = None,
    alignment: dict[str, Any] | str | Path | None = None,
    reference_stream: str = "left_rgb",
    epoch: int | None = None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> dict[str, Any]:
    """Audit all retained records; associations require independently validated timing."""
    required = _arguments(
        required_streams, reference_stream, epoch, start_time_ns, end_time_ns, True
    )
    loaded = _load(recording)
    inventory = {}
    for stream, items in loaded["records"].items():
        selected = [
            item for item in items if epoch is None or item["ref"].epoch == epoch
        ]
        samples = sum(
            len(item["data"]["samples"])
            if stream in ("accel", "gyro")
            else item["data"]["sample_count"]
            if stream == "audio"
            else 1
            for item in selected
        )
        epochs = sorted({item["ref"].epoch for item in selected})
        sequence_gaps = []
        for a, b in zip(selected, selected[1:]):
            if (
                a["ref"].epoch == b["ref"].epoch
                and b["ref"].sequence_number != a["ref"].sequence_number + 1
            ):
                sequence_gaps.append(
                    {
                        "epoch": a["ref"].epoch,
                        "before": a["ref"].sequence_number,
                        "after": b["ref"].sequence_number,
                    }
                )
        inventory[stream] = {
            "available": bool(selected),
            "records": len(selected),
            "samples": samples,
            "epochs": epochs,
            "sequence_gaps": sequence_gaps,
            "missing_media_records": sum(item["media"] is None for item in selected)
            if stream in ("left_rgb", "right_rgb", "audio")
            else 0,
        }
        if stream in loaded["sidecar_trailing_bytes"]:
            inventory[stream]["sidecar_trailing_bytes"] = loaded[
                "sidecar_trailing_bytes"
            ][stream]
    model = None
    errors = []
    windows = []
    complete_windows = 0
    assigned = {stream: set() for stream in ("left_rgb", "right_rgb")}
    assigned.update({stream: {} for stream in ("accel", "gyro")})
    assigned_audio = {}
    status = "unverified"
    try:
        model, external = _select_model(loaded, alignment, required, reference_stream)
        if model and model.get("status") == "validated":
            _validate_model(loaded, model, external, required, reference_stream)
            missing = [
                stream for stream in required if not inventory[stream]["available"]
            ]
            if missing:
                raise ValueError(f"Required streams are absent: {', '.join(missing)}")
            for window in _windows(
                loaded,
                model,
                required,
                reference_stream,
                epoch,
                start_time_ns,
                end_time_ns,
            ):
                windows.append(_window_index(window))
                complete_windows += window.complete
                for stream in ("left_rgb", "right_rgb"):
                    frame = getattr(window, stream)
                    if frame:
                        assigned[stream].add(
                            (frame.record.epoch, frame.record.record_index)
                        )
                for stream in ("accel", "gyro"):
                    for ref, begin, count, _, _ in _sample_spans(
                        getattr(window, stream)
                    ):
                        assigned[stream].setdefault(
                            (ref.epoch, ref.record_index), []
                        ).append((begin, begin + count))
                for value in window.audio:
                    key = value.record.epoch, value.record.record_index
                    assigned_audio.setdefault(key, []).append(
                        (value.sample_offset, value.sample_offset + value.sample_count)
                    )
            status = "validated"
    except (ValueError, TypeError, KeyError) as error:
        status = "invalid"
        errors.append(str(error))
        windows.clear()
        complete_windows = 0
        for values in assigned.values():
            values.clear()
        assigned_audio.clear()
    audio_count = sum(
        hi - lo
        for ranges in assigned_audio.values()
        for lo, hi in _merge_ranges(ranges)
    )
    assigned_counts = {
        stream: len(assigned[stream]) for stream in ("left_rgb", "right_rgb")
    }
    assigned_counts.update(
        {
            stream: sum(
                hi - lo
                for ranges in assigned[stream].values()
                for lo, hi in _merge_ranges(ranges)
            )
            for stream in ("accel", "gyro")
        }
    )
    assigned_counts["audio"] = audio_count
    coverage = {
        stream: {
            "assigned_samples": assigned_counts[stream],
            "unassigned_samples": inventory[stream]["samples"]
            - assigned_counts[stream],
        }
        for stream in STREAMS
    }
    unpaired = {
        stream: [
            asdict(item["ref"])
            for item in loaded["records"][stream]
            if (epoch is None or item["ref"].epoch == epoch)
            and (item["ref"].epoch, item["ref"].record_index) not in assigned[stream]
        ]
        for stream in ("left_rgb", "right_rgb")
    }
    return {
        "version": 1,
        "recording": str(loaded["path"]),
        "required_streams": list(required),
        "reference_stream": reference_stream,
        "streams": inventory,
        "alignment": {
            "status": status,
            "model_id": model.get("model_id") if isinstance(model, dict) else None,
            "errors": errors,
            "declared_validation": model.get("validation")
            if isinstance(model, dict)
            else None,
        },
        "windows": windows,
        "association_format": "record_and_sample_ranges_v1",
        "five_stream_verified": status == "validated"
        and set(required) == set(STREAMS)
        and bool(windows)
        and complete_windows == len(windows),
        "verification_scope": "returned_windows",
        "coverage": {
            "window_count": len(windows),
            "complete_windows": complete_windows,
            "incomplete_windows": len(windows) - complete_windows,
            "streams": coverage,
            "unpaired_video": unpaired,
            "unassigned_sample_ranges": _unassigned(
                loaded, assigned, assigned_audio, epoch
            ),
        },
        "issues": loaded["issues"],
        "ignored_noncanonical_records": loaded["ignored_records"],
        "physical_error_measurements": None,
    }
