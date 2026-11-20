# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest

from avs_wire import parse_avs1_record
from camera_receiver import Avs1TcpReceiver, ReceivedRecord
from test_wire_format import _avs1_record


@pytest.mark.parametrize("usb", [False, True])
def test_usb_metadata_has_no_argus_sensor_clock(usb):
    bundle = parse_avs1_record(_avs1_record(sender_capture_ready_tsc_ns=2000))
    stream = bundle.streams[0]
    sample = stream.samples[0]
    if usb:
        bundle = replace(bundle, flags=1, sender_capture_ready_tsc_ns=None)
        samples = tuple(
            replace(s, flags=9, sof_tsc_ns=None, frame_timestamp_ns=None)
            for s in stream.samples
        )
        stream = replace(stream, samples=samples)
        sample = samples[0]
    receiver = Avs1TcpReceiver.__new__(Avs1TcpReceiver)
    metadata = receiver._frame_metadata(
        bundle,
        stream,
        sample,
        SimpleNamespace(config=SimpleNamespace(name="head", output_format="nv12")),
        ReceivedRecord(b"", 10001, 10002, 10003, 10004, "realtime"),
        10005,
        10006,
        10007,
        10008,
        10009,
        10010,
        {},
    )
    assert metadata["sender_clock"] == (None if usb else "tegra_tsc_ns")
    assert metadata["sof_tsc_valid"] is not usb
    if usb:
        assert metadata["sensor_sof_ns"] is None
        assert metadata["sof_wall_clock_ns"] is None
        assert metadata["tsc_delta_ns"] is None
        assert metadata["sender_capture_ready_ns"] is not None
