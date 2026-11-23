#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avs_wire import (  # noqa: E402
    BUNDLE_SENDER_CLOCK_VALID,
    BUNDLE_SENDER_TSC_VALID,
    BUNDLE_TIMING_EXT,
    CODEC_HEVC,
    CODEC_RAW_NV12,
    HEADER,
    SAMPLE,
    SENSOR_SOF_MAX_LAG_NS,
    STREAM,
    TIMING,
    TIMING_DOMAIN_REALTIME,
    TIMING_VALID,
    Avs1Bundle,
    parse_avs1_record,
    sensor_sof_lag_ns,
    sensor_sof_ns,
)
from camera_receiver import (  # noqa: E402
    Avs1TcpReceiver,
    ReceivedRecord,
    ReceiveFpsMonitor,
    ReceiveFpsWarningConfig,
    _OUTPUT_NV12,
    _ipc_layout,
    _parse_optional_gpu_id,
    _parse_stats_log_interval_s,
    _parse_stream_config,
    _resolve_receive_fps_warning_config,
    _resolve_stats_log_interval_s,
)


_CAPTURE_READY_NS = 10_000
_SOF_TSC_NS = (1000, 1210, 3000)  # head.left, head.right, wrist


def _avs1_record(
    *,
    flags: Optional[int] = None,
    bundle_sequence: int = 3,
    session_id: int = 99,
    stream1_present: bool = True,
    sender_capture_ready_tsc_ns: Optional[int] = None,
    timing: Optional[list[tuple[int, int, int, int, int, int]]] = None,
) -> bytes:
    # flags=None derives CLOCK_VALID plus SENDER_TSC_VALID when a TSC is given;
    # pass flags explicitly to build deliberately inconsistent records.
    # timing: one (capture_realtime_ns, uncertainty_ns, anchor_age_ns, flags,
    # clock_domain, reserved) tuple per sample; sets BUNDLE_TIMING_EXT.
    if flags is None:
        flags = BUNDLE_SENDER_CLOCK_VALID
        if sender_capture_ready_tsc_ns is not None:
            flags |= BUNDLE_SENDER_TSC_VALID
    head_left = b"\x00\x00\x00\x01head-left"
    head_right = b"\x00\x00\x00\x01head-right"
    wrist = bytes(640 * 480 * 3 // 2) if stream1_present else b""
    payloads = [head_left, head_right, wrist]
    payload_offsets = []
    offset = 0
    for payload in payloads:
        payload_offsets.append(offset)
        offset += len(payload)
    payload = b"".join(payloads)

    stream0_flags = 0x0001 | 0x0002
    stream1_flags = 0x0001 if stream1_present else 0
    streams = [
        STREAM.pack(
            0,
            stream0_flags,
            CODEC_HEVC,
            1,
            180,
            2560,
            1984,
            60,
            1,
            42,
            0,
            2,
            0,
        ),
        STREAM.pack(
            1,
            stream1_flags,
            CODEC_RAW_NV12,
            1,
            0,
            640,
            480,
            60,
            1,
            7 if stream1_present else 0,
            2,
            1,
            0,
        ),
    ]
    samples = [
        SAMPLE.pack(
            0,
            0,
            1,
            0x000F,
            0,
            0,
            0,
            100,
            _SOF_TSC_NS[0],
            2000,
            10_000,
            10_100,
            10_500,
            payload_offsets[0],
            len(head_left),
        ),
        SAMPLE.pack(
            0,
            1,
            2,
            0x000F,
            0,
            1,
            0,
            101,
            _SOF_TSC_NS[1],
            2210,
            10_000,
            10_100,
            10_550,
            payload_offsets[1],
            len(head_right),
        ),
        SAMPLE.pack(
            1,
            0,
            0,
            0x000F if stream1_present else 0,
            0,
            4,
            0,
            5 if stream1_present else 0,
            _SOF_TSC_NS[2] if stream1_present else 0,
            4000 if stream1_present else 0,
            11_000 if stream1_present else 0,
            11_050 if stream1_present else 0,
            11_100 if stream1_present else 0,
            payload_offsets[2],
            len(wrist),
        ),
    ]
    timing_records: list[bytes] = []
    if timing is not None:
        assert len(timing) == len(samples)
        flags |= BUNDLE_TIMING_EXT
        timing_records = [TIMING.pack(*record) for record in timing]
    descriptor_bytes = (
        len(streams) * STREAM.size
        + len(samples) * SAMPLE.size
        + len(timing_records) * TIMING.size
    )
    header = HEADER.pack(
        b"AVS1",
        1,
        flags,
        HEADER.size,
        len(streams),
        len(samples),
        bundle_sequence,
        session_id,
        _CAPTURE_READY_NS,
        10_800,
        descriptor_bytes,
        len(payload),
        sender_capture_ready_tsc_ns or 0,
    )
    return (
        header
        + b"".join(streams)
        + b"".join(samples)
        + b"".join(timing_records)
        + payload
    )


_REALTIME_NS = 1_762_900_000_123_456_789
_INVALID = (0, 0, 0, 0, 0, 0)


def _valid(
    capture_ns: int = _REALTIME_NS,
    uncertainty_ns: int = 2_500,
    anchor_age_ns: int = 400_000_000,
    domain: int = TIMING_DOMAIN_REALTIME,
) -> tuple[int, int, int, int, int, int]:
    return (capture_ns, uncertainty_ns, anchor_age_ns, TIMING_VALID, domain, 0)


class Avs1ParserTests(unittest.TestCase):
    def test_optional_gpu_id_semantics(self) -> None:
        self.assertIsNone(_parse_optional_gpu_id(None))
        self.assertEqual(_parse_optional_gpu_id(0), 0)
        self.assertEqual(_parse_optional_gpu_id("2"), 2)
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            _parse_optional_gpu_id(-1)

    def test_stats_log_interval_semantics(self) -> None:
        self.assertEqual(_parse_stats_log_interval_s(0), 0.0)
        self.assertEqual(_parse_stats_log_interval_s("2.5"), 2.5)
        with self.assertRaisesRegex(ValueError, "finite number >= 0"):
            _parse_stats_log_interval_s(-1)
        with self.assertRaisesRegex(ValueError, "finite number >= 0"):
            _parse_stats_log_interval_s(float("inf"))

    def test_stats_log_interval_resolution(self) -> None:
        self.assertEqual(_resolve_stats_log_interval_s({}, None), 5.0)
        self.assertEqual(
            _resolve_stats_log_interval_s({"stats_log_interval_s": 0}, None),
            0.0,
        )
        self.assertEqual(
            _resolve_stats_log_interval_s({"stats_log_interval_s": 1}, 3.0),
            3.0,
        )
        self.assertEqual(
            _resolve_stats_log_interval_s({"log_timing": False}, None),
            0.0,
        )

    def test_receive_fps_warning_config(self) -> None:
        disabled = _resolve_receive_fps_warning_config({})
        self.assertFalse(disabled.enabled)

        enabled = _resolve_receive_fps_warning_config(
            {
                "receive_fps_warning_threshold": 55,
                "receive_fps_warning_interval_s": 10,
            }
        )
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.threshold_fps, 55.0)
        self.assertEqual(enabled.interval_s, 10.0)

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            _resolve_receive_fps_warning_config(
                {
                    "receive_fps_warning_threshold": 55,
                    "receive_fps_warning_interval_s": 0,
                }
            )

    def test_receive_fps_monitor_warns_only_for_low_streams(self) -> None:
        now = [0.0]
        monitor = ReceiveFpsMonitor(
            ("head", "wrist1"),
            ReceiveFpsWarningConfig(threshold_fps=55.0, interval_s=2.0),
            clock=lambda: now[0],
        )
        monitor.start()
        for _ in range(120):
            monitor.observe(("head",))
        for _ in range(100):
            monitor.observe(("wrist1",))
        now[0] = 2.0

        with patch("camera_receiver.logger.warning") as warning:
            monitor.poll()

        warning.assert_called_once()
        message_args = warning.call_args.args
        self.assertIn("wrist1=50.0fps", message_args[-1])
        self.assertNotIn("head", message_args[-1])

    def test_receive_fps_monitor_reports_zero_during_connected_stall(self) -> None:
        now = [0.0]
        monitor = ReceiveFpsMonitor(
            ("head",),
            ReceiveFpsWarningConfig(threshold_fps=55.0, interval_s=1.0),
            clock=lambda: now[0],
        )
        monitor.start()
        now[0] = 1.0

        with patch("camera_receiver.logger.warning") as warning:
            monitor.poll()

        warning.assert_called_once()
        self.assertIn("head=0.0fps", warning.call_args.args[-1])

    def test_parses_mixed_stream_bundle(self) -> None:
        bundle = parse_avs1_record(_avs1_record())

        self.assertIsInstance(bundle, Avs1Bundle)
        self.assertEqual(bundle.sequence, 3)
        self.assertEqual(bundle.session_id, 99)
        self.assertEqual(bundle.sender_clock_domain, "realtime")
        self.assertEqual(len(bundle.streams), 2)

        head = bundle.streams[0]
        self.assertTrue(head.present)
        self.assertTrue(head.stereo)
        self.assertEqual(head.codec, CODEC_HEVC)
        self.assertEqual(head.rotation, 180)
        self.assertEqual(head.tsc_delta_ns, 210)
        self.assertEqual(bytes(head.samples[0].payload), b"\x00\x00\x00\x01head-left")
        self.assertEqual(bytes(head.samples[1].payload), b"\x00\x00\x00\x01head-right")

        wrist = bundle.streams[1]
        self.assertTrue(wrist.present)
        self.assertFalse(wrist.stereo)
        self.assertEqual(wrist.codec, CODEC_RAW_NV12)
        self.assertEqual(wrist.samples[0].payload_length, 640 * 480 * 3 // 2)

    def test_accepts_absent_lower_rate_stream(self) -> None:
        bundle = parse_avs1_record(_avs1_record(stream1_present=False))

        self.assertTrue(bundle.streams[0].present)
        self.assertFalse(bundle.streams[1].present)
        self.assertEqual(bundle.streams[1].sequence, 0)
        self.assertEqual(bundle.streams[1].samples[0].payload_length, 0)

    def test_rejects_tai_without_clock_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "CLOCK_TAI"):
            parse_avs1_record(_avs1_record(flags=0x0002))

    def test_legacy_bundle_has_no_sender_tsc(self) -> None:
        bundle = parse_avs1_record(_avs1_record())

        self.assertEqual(bundle.flags, 0x0001)
        self.assertIsNone(bundle.sender_capture_ready_tsc_ns)
        for stream in bundle.streams:
            for sample in stream.samples:
                self.assertIsNone(sensor_sof_lag_ns(bundle, sample))
                self.assertIsNone(sensor_sof_ns(bundle, sample))

    def test_parses_sender_tsc_when_flagged(self) -> None:
        bundle = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=3_500))

        self.assertEqual(bundle.flags, 0x0001 | 0x0004)
        self.assertEqual(bundle.sender_capture_ready_ns, 10_000)
        self.assertEqual(bundle.sender_capture_ready_tsc_ns, 3_500)

    def test_rejects_sender_tsc_flag_without_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero sender_capture_ready_tsc_ns"):
            parse_avs1_record(_avs1_record(flags=0x0001 | 0x0004))

    def test_rejects_sender_tsc_field_without_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "without SENDER_TSC_VALID"):
            parse_avs1_record(
                _avs1_record(flags=0x0001, sender_capture_ready_tsc_ns=3_500)
            )

    def test_rejects_sender_tsc_without_clock_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "SENDER_TSC_VALID flag requires"):
            parse_avs1_record(
                _avs1_record(flags=0x0004, sender_capture_ready_tsc_ns=3_500)
            )

    def test_sensor_sof_math_per_sample(self) -> None:
        # capture_ready wall 10_000 ns, capture_ready TSC 3_500 ns.
        bundle = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=3_500))
        left, right = bundle.streams[0].samples
        (wrist,) = bundle.streams[1].samples

        self.assertEqual(sensor_sof_lag_ns(bundle, left), 2_500)
        self.assertEqual(sensor_sof_lag_ns(bundle, right), 2_290)
        self.assertEqual(sensor_sof_lag_ns(bundle, wrist), 500)
        self.assertEqual(sensor_sof_ns(bundle, left), 10_000 - 2_500)
        # Right eye uses its own SOF: 210 ns later than the left eye.
        self.assertEqual(sensor_sof_ns(bundle, right), 10_000 - 2_290)
        self.assertEqual(
            sensor_sof_ns(bundle, right) - sensor_sof_ns(bundle, left),
            bundle.streams[0].tsc_delta_ns,
        )
        self.assertEqual(sensor_sof_ns(bundle, wrist), 10_000 - 500)

    def test_sensor_sof_lag_window(self) -> None:
        (wrist,) = (
            parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=3_500))
            .streams[1]
            .samples
        )

        # Zero lag (SOF == capture-ready instant) is accepted.
        at_sof = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=3_000))
        self.assertEqual(sensor_sof_lag_ns(at_sof, wrist), 0)
        self.assertEqual(sensor_sof_ns(at_sof, wrist), 10_000)

        # Exactly the maximum lag is accepted.
        at_max = parse_avs1_record(
            _avs1_record(sender_capture_ready_tsc_ns=3_000 + SENSOR_SOF_MAX_LAG_NS)
        )
        self.assertEqual(sensor_sof_lag_ns(at_max, wrist), SENSOR_SOF_MAX_LAG_NS)
        self.assertEqual(sensor_sof_ns(at_max, wrist), 10_000 - SENSOR_SOF_MAX_LAG_NS)

        # One ns past the window: lag is still reported, SOF is not.
        too_late = parse_avs1_record(
            _avs1_record(sender_capture_ready_tsc_ns=3_000 + SENSOR_SOF_MAX_LAG_NS + 1)
        )
        self.assertEqual(sensor_sof_lag_ns(too_late, wrist), SENSOR_SOF_MAX_LAG_NS + 1)
        self.assertIsNone(sensor_sof_ns(too_late, wrist))

        # SOF after the capture-ready instant (negative lag) is rejected.
        before_sof = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=2_999))
        self.assertEqual(sensor_sof_lag_ns(before_sof, wrist), -1)
        self.assertIsNone(sensor_sof_ns(before_sof, wrist))

    def test_sensor_sof_requires_every_input(self) -> None:
        bundle = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=3_500))
        (wrist,) = bundle.streams[1].samples

        no_wall = dataclasses.replace(bundle, sender_capture_ready_ns=None)
        self.assertEqual(sensor_sof_lag_ns(no_wall, wrist), 500)
        self.assertIsNone(sensor_sof_ns(no_wall, wrist))

        no_tsc = dataclasses.replace(bundle, sender_capture_ready_tsc_ns=None)
        self.assertIsNone(sensor_sof_lag_ns(no_tsc, wrist))
        self.assertIsNone(sensor_sof_ns(no_tsc, wrist))

        no_sof = dataclasses.replace(wrist, sof_tsc_ns=None)
        self.assertIsNone(sensor_sof_lag_ns(bundle, no_sof))
        self.assertIsNone(sensor_sof_ns(bundle, no_sof))

    def test_rejects_length_mismatch(self) -> None:
        record = bytearray(_avs1_record())
        fields = list(HEADER.unpack_from(record))
        fields[11] += 1
        HEADER.pack_into(record, 0, *fields)
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            parse_avs1_record(record)

    def test_rejects_raw_payload_size_mismatch(self) -> None:
        record = bytearray(_avs1_record())
        sample2_offset = HEADER.size + 2 * STREAM.size + 2 * SAMPLE.size
        fields = list(SAMPLE.unpack_from(record, sample2_offset))
        fields[-1] -= 1
        SAMPLE.pack_into(record, sample2_offset, *fields)
        with self.assertRaisesRegex(ValueError, "raw AVS1 payload"):
            parse_avs1_record(record)

    def test_stream_config_defaults(self) -> None:
        cfg = _parse_stream_config(
            {
                "index": 0,
                "name": "head",
                "type": "stereo",
                "width": 2560,
                "height": 1984,
            }
        )

        self.assertTrue(cfg.stereo)
        self.assertEqual(cfg.output_format, _OUTPUT_NV12)
        self.assertEqual(cfg.ring_depth, 6)
        self.assertEqual(cfg.ipc_names, ("head.left", "head.right"))

    def test_nv12_ipc_layout(self) -> None:
        layout = _ipc_layout("nv12", 640, 480)

        self.assertEqual(layout["planes"][0]["name"], "y")
        self.assertEqual(layout["planes"][1]["name"], "uv")
        self.assertEqual(layout["planes"][1]["offset"], 640 * 480)


class Avs1TimingExtensionTests(unittest.TestCase):
    def _samples(self, **kwargs):
        bundle = parse_avs1_record(_avs1_record(**kwargs))
        return bundle, [s for stream in bundle.streams for s in stream.samples]

    def test_valid_realtime_record_parses(self) -> None:
        bundle, samples = self._samples(
            timing=[_valid(), _valid(_REALTIME_NS + 10, 3_000, 5), _valid(7, 1, 2)]
        )

        self.assertTrue(bundle.flags & BUNDLE_TIMING_EXT)
        self.assertTrue(samples[0].capture_valid)
        self.assertEqual(samples[0].sensor_capture_realtime_ns, _REALTIME_NS)
        self.assertEqual(samples[0].capture_uncertainty_ns, 2_500)
        self.assertEqual(samples[0].capture_anchor_age_ns, 400_000_000)
        self.assertEqual(samples[1].sensor_capture_realtime_ns, _REALTIME_NS + 10)
        self.assertEqual(samples[1].capture_uncertainty_ns, 3_000)
        self.assertEqual(samples[1].capture_anchor_age_ns, 5)
        self.assertEqual(samples[2].sensor_capture_realtime_ns, 7)
        # Payload slicing is unaffected by the extra descriptor bytes.
        self.assertEqual(bytes(samples[0].payload), b"\x00\x00\x00\x01head-left")

    def test_bundle_without_extension_has_no_capture_time(self) -> None:
        bundle, samples = self._samples()

        self.assertFalse(bundle.flags & BUNDLE_TIMING_EXT)
        for sample in samples:
            self.assertIsNone(sample.sensor_capture_realtime_ns)
            self.assertIsNone(sample.capture_uncertainty_ns)
            self.assertIsNone(sample.capture_anchor_age_ns)
            self.assertFalse(sample.capture_valid)

    def test_all_zero_invalid_records_parse(self) -> None:
        _, samples = self._samples(timing=[_INVALID] * 3)

        for sample in samples:
            self.assertIsNone(sample.sensor_capture_realtime_ns)
            self.assertIsNone(sample.capture_uncertainty_ns)
            self.assertIsNone(sample.capture_anchor_age_ns)
            self.assertFalse(sample.capture_valid)

    def test_mixed_valid_and_invalid_records(self) -> None:
        _, samples = self._samples(timing=[_valid(), _INVALID, _valid(99, 4, 6)])

        self.assertTrue(samples[0].capture_valid)
        self.assertEqual(samples[0].sensor_capture_realtime_ns, _REALTIME_NS)
        self.assertFalse(samples[1].capture_valid)
        self.assertIsNone(samples[1].sensor_capture_realtime_ns)
        self.assertTrue(samples[2].capture_valid)
        self.assertEqual(samples[2].sensor_capture_realtime_ns, 99)
        self.assertEqual(samples[2].capture_uncertainty_ns, 4)
        self.assertEqual(samples[2].capture_anchor_age_ns, 6)

    def test_valid_record_in_other_domains_is_rejected(self) -> None:
        for domain in (0, 1, 3, 0xFFFF):
            with self.subTest(domain=domain):
                with self.assertRaisesRegex(
                    ValueError, "valid timing record has no realtime capture time"
                ):
                    parse_avs1_record(
                        _avs1_record(timing=[_valid(domain=domain), _INVALID, _INVALID])
                    )

    def test_valid_record_with_zero_capture_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "valid timing record has no realtime capture time"
        ):
            parse_avs1_record(_avs1_record(timing=[_valid(0), _INVALID, _INVALID]))

    def test_invalid_record_with_values_is_rejected(self) -> None:
        for record in (
            (1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, TIMING_DOMAIN_REALTIME, 0),
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(
                    ValueError, "invalid timing record carries values"
                ):
                    parse_avs1_record(_avs1_record(timing=[record, _INVALID, _INVALID]))

    def test_valid_record_for_absent_sample_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid for an absent sample"):
            parse_avs1_record(
                _avs1_record(
                    stream1_present=False, timing=[_INVALID, _INVALID, _valid()]
                )
            )

    def test_absent_sample_with_invalid_record_parses(self) -> None:
        _, samples = self._samples(
            stream1_present=False, timing=[_valid(), _valid(), _INVALID]
        )

        self.assertFalse(samples[2].capture_valid)

    def test_reserved_and_extra_flag_bits_are_rejected(self) -> None:
        for record in (
            (_REALTIME_NS, 1, 1, TIMING_VALID, TIMING_DOMAIN_REALTIME, 1),
            (_REALTIME_NS, 1, 1, TIMING_VALID | 0x0002, TIMING_DOMAIN_REALTIME, 0),
            (_REALTIME_NS, 1, 1, TIMING_VALID | 0x8000, TIMING_DOMAIN_REALTIME, 0),
            (0, 0, 0, 0x0002, 0, 0),
            (0, 0, 0, 0, 0, 1),
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "reserved bits set"):
                    parse_avs1_record(_avs1_record(timing=[record, _INVALID, _INVALID]))

    def test_descriptor_count_must_include_timing_records(self) -> None:
        record = bytearray(_avs1_record(timing=[_INVALID] * 3))
        fields = list(HEADER.unpack_from(record))
        fields[10] -= 3 * TIMING.size
        fields[11] += 3 * TIMING.size
        HEADER.pack_into(record, 0, *fields)
        with self.assertRaises(ValueError):
            parse_avs1_record(record)

    def _metadata(self, record: bytes) -> dict[str, object]:
        bundle = parse_avs1_record(record)
        stream = bundle.streams[0]
        runtime = SimpleNamespace(
            config=SimpleNamespace(name="head", output_format="nv12")
        )
        received = ReceivedRecord(b"", 1, 2, 3, 4, "realtime")
        return Avs1TcpReceiver._frame_metadata(
            None,
            bundle,
            stream,
            stream.samples[0],
            runtime,
            received,
            5,
            6,
            7,
            8,
            9,
            10,
            {},
        )

    def test_frame_metadata_for_valid_sample(self) -> None:
        metadata = self._metadata(_avs1_record(timing=[_valid(), _INVALID, _INVALID]))

        self.assertEqual(metadata["sensor_capture_realtime_ns"], _REALTIME_NS)
        self.assertEqual(metadata["sensor_capture_uncertainty_ns"], 2_500)
        self.assertEqual(metadata["sensor_anchor_age_ns"], 400_000_000)
        self.assertIs(metadata["sensor_capture_valid"], True)

    def test_frame_metadata_for_invalid_or_absent_record(self) -> None:
        for record in (
            _avs1_record(timing=[_INVALID, _valid(), _INVALID]),
            _avs1_record(),
        ):
            metadata = self._metadata(record)
            self.assertIsNone(metadata["sensor_capture_realtime_ns"])
            self.assertIsNone(metadata["sensor_capture_uncertainty_ns"])
            self.assertIsNone(metadata["sensor_anchor_age_ns"])
            self.assertIs(metadata["sensor_capture_valid"], False)


if __name__ == "__main__":
    unittest.main()
