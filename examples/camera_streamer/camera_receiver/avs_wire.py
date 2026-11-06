# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AVS1 wire-format parser.

AVS1 records are framed on TCP as:

    uint32_be_record_length || avs1_record

This module parses only the inner ``avs1_record``. The caller owns the outer
length prefix and any receive timestamps.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

MAGIC = b"AVS1"
VERSION = 1
HEADER = struct.Struct("!4sHHIHHQQQQIIQ")
STREAM = struct.Struct("!HHBBHIIIIQHHI")
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII")

BUNDLE_SENDER_CLOCK_VALID = 1 << 0
BUNDLE_SENDER_CLOCK_TAI = 1 << 1
BUNDLE_SENDER_TSC_VALID = 1 << 2
BUNDLE_KNOWN_FLAGS = (
    BUNDLE_SENDER_CLOCK_VALID | BUNDLE_SENDER_CLOCK_TAI | BUNDLE_SENDER_TSC_VALID
)

# Sensor SOF -> sender capture-ready lag accepted by ``sensor_sof_ns``.
SENSOR_SOF_MAX_LAG_NS = 500_000_000

STREAM_PRESENT = 1 << 0
STREAM_STEREO = 1 << 1
STREAM_KNOWN_FLAGS = STREAM_PRESENT | STREAM_STEREO

SAMPLE_PRESENT = 1 << 0
SAMPLE_SOF_TSC_VALID = 1 << 1
SAMPLE_FRAME_TIMESTAMP_VALID = 1 << 2
SAMPLE_SENDER_TIMING_VALID = 1 << 3
SAMPLE_KNOWN_FLAGS = (
    SAMPLE_PRESENT
    | SAMPLE_SOF_TSC_VALID
    | SAMPLE_FRAME_TIMESTAMP_VALID
    | SAMPLE_SENDER_TIMING_VALID
)

CODEC_RAW_NV12 = 0
CODEC_HEVC = 1
PIXEL_FORMAT_NV12 = 1
ROLE_MONO = 0
ROLE_LEFT = 1
ROLE_RIGHT = 2


@dataclass(frozen=True)
class Avs1Sample:
    stream_index: int
    camera_index: int
    role: int
    flags: int
    device_index: int
    sensor_mode_index: int
    frame_number: int
    sof_tsc_ns: Optional[int]
    frame_timestamp_ns: Optional[int]
    sender_capture_ready_ns: Optional[int]
    sender_process_submit_ns: Optional[int]
    sender_process_done_ns: Optional[int]
    payload_offset: int
    payload_length: int
    payload: memoryview

    @property
    def present(self) -> bool:
        return bool(self.flags & SAMPLE_PRESENT)

    @property
    def role_name(self) -> str:
        if self.role == ROLE_LEFT:
            return "left"
        if self.role == ROLE_RIGHT:
            return "right"
        return "mono"


@dataclass(frozen=True)
class Avs1Stream:
    index: int
    flags: int
    codec: int
    pixel_format: int
    rotation: int
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    sequence: int
    samples: tuple[Avs1Sample, ...]

    @property
    def present(self) -> bool:
        return bool(self.flags & STREAM_PRESENT)

    @property
    def stereo(self) -> bool:
        return bool(self.flags & STREAM_STEREO)

    @property
    def codec_name(self) -> str:
        if self.codec == CODEC_HEVC:
            return "hevc"
        if self.codec == CODEC_RAW_NV12:
            return "raw_nv12"
        return f"unknown:{self.codec}"

    @property
    def tsc_delta_ns(self) -> Optional[int]:
        if not self.stereo or len(self.samples) != 2:
            return None
        left = self.samples[0].sof_tsc_ns
        right = self.samples[1].sof_tsc_ns
        if left is None or right is None:
            return None
        return int(right) - int(left)


@dataclass(frozen=True)
class Avs1Bundle:
    flags: int
    sequence: int
    session_id: int
    sender_capture_ready_ns: Optional[int]
    # Sender TSC (same domain as ``Avs1Sample.sof_tsc_ns``) read at the same
    # instant as ``sender_capture_ready_ns``; None when SENDER_TSC_VALID is clear.
    sender_capture_ready_tsc_ns: Optional[int]
    sender_tcp_enqueue_ns: Optional[int]
    sender_clock_domain: Optional[str]
    streams: tuple[Avs1Stream, ...]


def parse_avs1_record(record: bytes | bytearray | memoryview) -> Avs1Bundle:
    packet = memoryview(record)
    if len(packet) < HEADER.size:
        raise ValueError(f"short AVS1 record: {len(packet)} bytes")
    (
        magic,
        version,
        flags,
        header_bytes,
        stream_count,
        sample_count,
        bundle_sequence,
        session_id,
        capture_ready_ns,
        tcp_enqueue_ns,
        descriptor_bytes,
        payload_bytes,
        capture_ready_tsc_ns,
    ) = HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION or header_bytes != HEADER.size:
        raise ValueError(
            f"unsupported AVS1 header magic={magic!r} version={version} "
            f"header_bytes={header_bytes}"
        )
    if flags & ~BUNDLE_KNOWN_FLAGS:
        raise ValueError(f"unknown AVS1 bundle flags 0x{flags:04x}")
    if flags & BUNDLE_SENDER_CLOCK_TAI and not flags & BUNDLE_SENDER_CLOCK_VALID:
        raise ValueError("AVS1 CLOCK_TAI flag requires valid sender timing")
    if flags & BUNDLE_SENDER_TSC_VALID:
        if not flags & BUNDLE_SENDER_CLOCK_VALID:
            raise ValueError("AVS1 SENDER_TSC_VALID flag requires valid sender timing")
        if capture_ready_tsc_ns == 0:
            raise ValueError(
                "AVS1 SENDER_TSC_VALID flag with zero sender_capture_ready_tsc_ns"
            )
    elif capture_ready_tsc_ns != 0:
        raise ValueError(
            "AVS1 sender_capture_ready_tsc_ns set without SENDER_TSC_VALID flag"
        )
    if stream_count == 0 or bundle_sequence == 0 or session_id == 0:
        raise ValueError("invalid AVS1 bundle identity/count")

    expected_descriptors = stream_count * STREAM.size + sample_count * SAMPLE.size
    if descriptor_bytes != expected_descriptors:
        raise ValueError("AVS1 descriptor byte count mismatch")
    expected_length = header_bytes + descriptor_bytes + payload_bytes
    if expected_length != len(packet):
        raise ValueError(
            f"AVS1 record byte count mismatch: header says {expected_length}, "
            f"record has {len(packet)}"
        )

    offset = header_bytes
    stream_headers: list[tuple] = []
    flat_cursor = 0
    for expected_index in range(stream_count):
        fields = STREAM.unpack_from(packet, offset)
        offset += STREAM.size
        (
            stream_index,
            stream_flags,
            codec,
            pixel_format,
            rotation,
            width,
            height,
            _fps_num,
            _fps_den,
            stream_sequence,
            first_sample,
            stream_sample_count,
            stream_reserved,
        ) = fields
        if stream_index != expected_index or stream_reserved != 0:
            raise ValueError("AVS1 stream descriptors are out of config order")
        if stream_flags & ~STREAM_KNOWN_FLAGS:
            raise ValueError(f"unknown AVS1 stream flags 0x{stream_flags:04x}")
        if codec not in (CODEC_RAW_NV12, CODEC_HEVC):
            raise ValueError(f"unsupported AVS1 codec {codec}")
        if pixel_format != PIXEL_FORMAT_NV12:
            raise ValueError(f"unsupported AVS1 pixel format {pixel_format}")
        if rotation not in (0, 180):
            raise ValueError(f"unsupported AVS1 rotation {rotation}")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("invalid AVS1 stream dimensions")
        expected_sample_count = 2 if stream_flags & STREAM_STEREO else 1
        if stream_sample_count != expected_sample_count:
            raise ValueError("AVS1 stream sample count does not match mono/stereo")
        if first_sample != flat_cursor:
            raise ValueError("AVS1 stream sample range is not contiguous")
        flat_cursor += stream_sample_count
        if flat_cursor > sample_count:
            raise ValueError("AVS1 stream sample range exceeds descriptor table")
        if stream_flags & STREAM_PRESENT:
            if stream_sequence == 0:
                raise ValueError("present AVS1 stream has zero stream_sequence")
        elif stream_sequence != 0:
            raise ValueError("absent AVS1 stream has nonzero stream_sequence")
        stream_headers.append(fields)

    if flat_cursor != sample_count:
        raise ValueError("AVS1 sample descriptors are not fully referenced")

    raw_samples = []
    for _flat_index in range(sample_count):
        fields = SAMPLE.unpack_from(packet, offset)
        offset += SAMPLE.size
        if fields[4] != 0:
            raise ValueError("AVS1 sample reserved field is nonzero")
        raw_samples.append(fields)

    payload_start = header_bytes + descriptor_bytes
    streams: list[Avs1Stream] = []
    consumed_ranges: list[tuple[int, int]] = []
    for stream_index, fields in enumerate(stream_headers):
        (
            _,
            stream_flags,
            codec,
            pixel_format,
            rotation,
            width,
            height,
            fps_num,
            fps_den,
            stream_sequence,
            first_sample,
            stream_sample_count,
            _reserved,
        ) = fields
        if fps_den == 0:
            raise ValueError("AVS1 fps denominator must be nonzero")
        stream_present = bool(stream_flags & STREAM_PRESENT)
        samples: list[Avs1Sample] = []
        for camera_index in range(stream_sample_count):
            sample_fields = raw_samples[first_sample + camera_index]
            (
                sample_stream,
                sample_camera,
                role,
                sample_flags,
                _sample_reserved,
                device_index,
                mode_index,
                frame_number,
                sof_tsc_ns,
                frame_timestamp_ns,
                sample_capture_ready_ns,
                process_submit_ns,
                process_done_ns,
                payload_offset,
                payload_length,
            ) = sample_fields
            if sample_stream != stream_index or sample_camera != camera_index:
                raise ValueError("AVS1 sample descriptors are out of config order")
            expected_role = (
                (ROLE_LEFT if camera_index == 0 else ROLE_RIGHT)
                if stream_flags & STREAM_STEREO
                else ROLE_MONO
            )
            if role != expected_role:
                raise ValueError("invalid AVS1 sample role")
            if sample_flags & ~SAMPLE_KNOWN_FLAGS:
                raise ValueError(f"unknown AVS1 sample flags 0x{sample_flags:04x}")
            sample_present = bool(sample_flags & SAMPLE_PRESENT)
            if sample_present != stream_present:
                raise ValueError("AVS1 stream/sample presence is not atomic")
            if payload_offset + payload_length > payload_bytes:
                raise ValueError("AVS1 sample payload exceeds payload area")
            if sample_present and payload_length == 0:
                raise ValueError("present AVS1 sample has empty payload")
            if not sample_present and payload_length != 0:
                raise ValueError("absent AVS1 sample has payload")
            if codec == CODEC_RAW_NV12 and sample_present:
                expected_raw = width * height * 3 // 2
                if payload_length != expected_raw:
                    raise ValueError(
                        f"raw AVS1 payload has {payload_length} bytes, "
                        f"expected {expected_raw}"
                    )
            begin = payload_start + payload_offset
            end = begin + payload_length
            if payload_length:
                consumed_ranges.append(
                    (payload_offset, payload_offset + payload_length)
                )
            sender_timing_valid = bool(sample_flags & SAMPLE_SENDER_TIMING_VALID)
            samples.append(
                Avs1Sample(
                    stream_index=stream_index,
                    camera_index=camera_index,
                    role=role,
                    flags=sample_flags,
                    device_index=device_index,
                    sensor_mode_index=mode_index,
                    frame_number=frame_number,
                    sof_tsc_ns=(
                        int(sof_tsc_ns) if sample_flags & SAMPLE_SOF_TSC_VALID else None
                    ),
                    frame_timestamp_ns=(
                        int(frame_timestamp_ns)
                        if sample_flags & SAMPLE_FRAME_TIMESTAMP_VALID
                        else None
                    ),
                    sender_capture_ready_ns=(
                        int(sample_capture_ready_ns) if sender_timing_valid else None
                    ),
                    sender_process_submit_ns=(
                        int(process_submit_ns) if sender_timing_valid else None
                    ),
                    sender_process_done_ns=(
                        int(process_done_ns) if sender_timing_valid else None
                    ),
                    payload_offset=int(payload_offset),
                    payload_length=int(payload_length),
                    payload=packet[begin:end],
                )
            )
        streams.append(
            Avs1Stream(
                index=stream_index,
                flags=stream_flags,
                codec=codec,
                pixel_format=pixel_format,
                rotation=rotation,
                width=width,
                height=height,
                fps_numerator=fps_num,
                fps_denominator=fps_den,
                sequence=stream_sequence,
                samples=tuple(samples),
            )
        )

    expected_offset = 0
    for begin, end in sorted(consumed_ranges):
        if begin != expected_offset:
            raise ValueError("AVS1 payload ranges overlap or have gaps")
        expected_offset = end
    if expected_offset != payload_bytes:
        raise ValueError("AVS1 payload descriptors do not cover payload area")

    sender_clock_domain = None
    if flags & BUNDLE_SENDER_CLOCK_VALID:
        sender_clock_domain = "tai" if flags & BUNDLE_SENDER_CLOCK_TAI else "realtime"

    return Avs1Bundle(
        flags=flags,
        sequence=int(bundle_sequence),
        session_id=int(session_id),
        sender_capture_ready_ns=(
            int(capture_ready_ns) if sender_clock_domain is not None else None
        ),
        sender_capture_ready_tsc_ns=(
            int(capture_ready_tsc_ns) if flags & BUNDLE_SENDER_TSC_VALID else None
        ),
        sender_tcp_enqueue_ns=(
            int(tcp_enqueue_ns) if sender_clock_domain is not None else None
        ),
        sender_clock_domain=sender_clock_domain,
        streams=tuple(streams),
    )


def sensor_sof_lag_ns(bundle: Avs1Bundle, sample: Avs1Sample) -> Optional[int]:
    """Sensor SOF -> sender capture-ready lag in the sender TSC domain.

    Returns the raw (possibly out-of-range) lag so callers can report it as a
    diagnostic; ``sensor_sof_ns`` applies the validity window.
    """
    if bundle.sender_capture_ready_tsc_ns is None or sample.sof_tsc_ns is None:
        return None
    return int(bundle.sender_capture_ready_tsc_ns) - int(sample.sof_tsc_ns)


def sensor_sof_ns(bundle: Avs1Bundle, sample: Avs1Sample) -> Optional[int]:
    """Sensor start-of-frame of ``sample`` in the sender wall-clock domain.

        sensor_sof_ns = sender_capture_ready_ns
                        - (sender_capture_ready_tsc_ns - sof_tsc_ns)

    Only defined when the bundle carries both the wall and TSC capture-ready
    stamps, the sample carries its SOF TSC, and the lag lies within
    ``[0, SENSOR_SOF_MAX_LAG_NS]``; otherwise None.
    """
    if bundle.sender_capture_ready_ns is None:
        return None
    lag_ns = sensor_sof_lag_ns(bundle, sample)
    if lag_ns is None or lag_ns < 0 or lag_ns > SENSOR_SOF_MAX_LAG_NS:
        return None
    return int(bundle.sender_capture_ready_ns) - lag_ns
