# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SensingStreamReader: tails the producer's H.264 file, decodes every unit in order, delivers the newest frame.

CPU-only, against real libx264-encoded units (intra-only and inter-coded GOPs); the native producer's own
output is covered separately.
"""

from __future__ import annotations

import ctypes
import errno
import os
import threading
import time

import numpy as np
import pytest

from capture_server.acquisition import sensing_h264 as h264
from capture_server.acquisition.errors import SourceError
from capture_server.acquisition.sensing_jpeg import encode_rgb_jpeg

from .h264_fixtures import allocated_bytes, encode_units

W, H = 64, 32
RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)


def _close(rgb: np.ndarray, expected: tuple[int, int, int], tol: int = 14) -> bool:
    return bool(np.abs(rgb[H // 2, W // 2].astype(int) - np.array(expected)).max() <= tol)


def _reader(path, **kw) -> h264.SensingStreamReader:
    reader = h264.SensingStreamReader(str(path), width=kw.pop("width", W), height=kw.pop("height", H), **kw)
    reader.connect(1.0)
    return reader


def test_decodes_correct_colours_and_geometry(tmp_path):
    units = encode_units(W, H, [RED, GREEN])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) + units[-1])
    reader = _reader(path)
    rgb, selected_ns = reader.read_latest(1.0)
    assert rgb.shape == (H, W, 3) and selected_ns > 0 and _close(rgb, GREEN)
    reader.close()


def test_newest_complete_unit_wins_and_older_ones_are_skipped(tmp_path):
    units = encode_units(W, H, [RED, GREEN, BLUE])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))  # the last unit is not yet known complete: GREEN is the newest
    reader = _reader(path)
    rgb, _ = reader.read_latest(1.0)
    assert _close(rgb, GREEN)  # RED was decoded first but only the newest decoded frame is delivered
    assert reader.read_latest(0.05) is None  # nothing newer; never invents a frame
    reader.close()


def test_a_half_written_unit_is_not_delivered_until_complete(tmp_path):
    units = encode_units(W, H, [RED, GREEN])
    path = tmp_path / "s.h264"
    reader_path = str(path)
    with open(path, "wb") as f:
        f.write(units[0][: len(units[0]) // 2])
        f.flush()
        reader = h264.SensingStreamReader(reader_path, width=W, height=H)
        reader.connect(1.0)
        assert reader.read_latest(0.05) is None
        f.write(units[0][len(units[0]) // 2 :] + units[1])
        f.flush()
        rgb, _ = reader.read_latest(1.0)
        assert _close(rgb, RED)
    reader.close()


def test_missing_file_times_out_with_oserror(tmp_path):
    reader = h264.SensingStreamReader(str(tmp_path / "never"), width=W, height=H)
    with pytest.raises(FileNotFoundError):
        reader.connect(0.05)


def test_geometry_mismatch_is_terminal(tmp_path):
    units = encode_units(W, H, [RED, GREEN])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) + units[-1])
    reader = _reader(path, width=W * 2)
    with pytest.raises(h264.SensingStreamError, match="expected 128x32"):
        reader.read_latest(1.0)
    with pytest.raises(h264.SensingStreamError, match="failed earlier"):
        reader.read_latest(0.0)


def test_undecodable_bytes_are_a_terminal_stream_error_not_a_frame(tmp_path):
    path = tmp_path / "s.h264"
    # Annex-B shaped but not video, like the native mock encoder's units.
    path.write_bytes((b"\x00\x00\x00\x01\x67MOCK" + b"U" * 200 + b"\x00\x00\x00\x01\x65MOCK" + b"U" * 200) * 3)
    reader = _reader(path)
    with pytest.raises(h264.SensingStreamError):
        for _ in range(5):
            reader.read_latest(0.2)
    with pytest.raises(h264.SensingStreamError, match="failed earlier"):
        reader.read_latest(0.0)


def test_read_before_connect_is_an_error(tmp_path):
    reader = h264.SensingStreamReader(str(tmp_path / "x"), width=W, height=H)
    with pytest.raises(h264.SensingStreamError, match="not connected"):
        reader.read_latest(0.0)


def test_consumed_prefix_is_returned_to_the_filesystem_while_reading_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_PUNCH_CHUNK_BYTES", 4096)
    units = encode_units(W, H, [RED, GREEN, BLUE]) * 600  # each unit is an independent IDR
    path = tmp_path / "s.h264"
    with open(path, "wb") as f:
        reader = None
        for i, unit in enumerate(units):
            f.write(unit)
            f.flush()
            if reader is None:
                reader = h264.SensingStreamReader(str(path), width=W, height=H)
                reader.connect(1.0)
            if i > 2:
                assert reader.read_latest(1.0) is not None
        size = os.stat(path).st_size
        assert allocated_bytes(str(path)) < size  # holes punched behind the read offset (best effort)
    reader.close()


def test_native_size_unit_decodes_independently_and_fits_the_payload_ceiling(tmp_path):
    width, height = 2560, 1984
    units = encode_units(width, height, [RED, GREEN, BLUE])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = h264.SensingStreamReader(str(path), width=width, height=height)
    reader.connect(1.0)
    started = time.monotonic()
    rgb, _ = reader.read_latest(5.0)  # RED then GREEN decoded in order; GREEN delivered
    assert rgb.shape == (height, width, 3)
    assert bool(np.abs(rgb[900, 1200].astype(int) - np.array(GREEN)).max() <= 14)
    assert len(encode_rgb_jpeg(rgb, 80)) < 2 * 1024 * 1024
    assert time.monotonic() - started < 5.0
    reader.close()


def test_oversized_jpeg_names_dimensions_and_quality():
    noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(SourceError, match=r"64x64 JPEG at quality 95.*ceiling"):
        encode_rgb_jpeg(noise, 95, max_bytes=100)


def test_jpeg_rejects_wrong_shape():
    with pytest.raises(ValueError):
        encode_rgb_jpeg(np.zeros((4, 4, 4), np.uint8), 80)


def _failing_fallocate(code: int):
    """A stand-in for the C call that fails the way the real one does: returns -1 and sets errno."""

    def fallocate(_fd, _mode, _offset, _length):
        ctypes.set_errno(code)
        return -1

    return fallocate


def test_a_failing_space_reclamation_ends_the_stream_with_the_reason_and_never_marks_bytes_reclaimed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(h264, "_PUNCH_CHUNK_BYTES", 256)
    units = encode_units(W, H, [RED, GREEN, BLUE]) * 40
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = h264.SensingStreamReader(str(path), width=W, height=H, fallocate=_failing_fallocate(errno.EOPNOTSUPP))
    reader.connect(1.0)
    with pytest.raises(h264.SensingStreamError, match=r"cannot reclaim.*errno 95"):
        reader.read_latest(1.0)
    assert reader._punched == 0  # the reclaimed offset advances only after the call succeeds
    with pytest.raises(h264.SensingStreamError, match="failed earlier"):
        reader.read_latest(0.0)


def test_a_file_we_cannot_reclaim_from_is_an_error_not_a_quiet_read_only_fallback(tmp_path):
    path = tmp_path / "s.h264"
    path.write_bytes(b"x")
    path.chmod(0o444)
    if os.access(path, os.W_OK):
        pytest.skip("running with privileges that ignore file modes")
    reader = h264.SensingStreamReader(str(path), width=W, height=H)
    with pytest.raises(h264.SensingStreamError, match="reclamation"):
        reader.connect(0.5)


def test_one_poll_reads_at_most_its_byte_cap_even_with_a_larger_backlog(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_POLL_BYTES", 2048)
    monkeypatch.setattr(h264, "_READ_BYTES", 512)
    units = encode_units(W, H, [RED, GREEN, BLUE]) * 200
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = _reader(path)
    assert reader.read_latest(1.0) is not None
    assert reader._offset <= 2048 and os.stat(path).st_size > 20 * 2048  # the backlog is read over later polls
    reader.close()


def test_a_producer_that_keeps_appending_cannot_extend_a_poll_past_its_deadline(tmp_path):
    units = encode_units(W, H, [RED, GREEN])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 3)
    stop = threading.Event()

    def append() -> None:
        with open(path, "ab") as f:
            while not stop.is_set():
                f.write(units[0])
                f.flush()

    writer = threading.Thread(target=append, daemon=True)
    reader = _reader(path)
    writer.start()
    started = time.monotonic()
    assert reader.read_latest(0.3) is not None
    assert time.monotonic() - started < 2.0
    stop.set()
    writer.join(timeout=2)
    reader.close()


def test_input_that_never_completes_a_unit_is_rejected_instead_of_retained_without_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_UNIT_BYTES", 8192)
    path = tmp_path / "s.h264"
    path.write_bytes(b"\x01" * 64 * 1024)  # no start codes: the parser never completes a unit
    reader = _reader(path)
    with pytest.raises(h264.SensingStreamError, match="no complete access unit in"):
        reader.read_latest(1.0)


def test_inter_coded_units_are_decoded_in_order_and_the_newest_decoded_frame_is_delivered(tmp_path):
    # gop=30: after the first IDR the units are P-frames that cannot be decoded without their predecessors.
    units = encode_units(W, H, [RED] * 3 + [GREEN] * 3 + [BLUE] * 3, gop=30)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = _reader(path)
    rgb, _ = reader.read_latest(2.0)
    assert _close(rgb, BLUE)  # the last complete unit; the final one is still held back by the parser
    reader.close()


def test_finish_flushes_the_unit_the_parser_was_holding_back_and_only_once(tmp_path):
    units = encode_units(W, H, [RED, GREEN, BLUE])
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = _reader(path)
    rgb, _ = reader.read_latest(1.0)
    assert _close(rgb, GREEN) and reader.read_latest(0.05) is None  # BLUE is incomplete while the file may grow
    last = reader.finish()  # the producer ended: now the end of the file is the end of the stream
    assert last is not None and _close(last[0], BLUE)
    assert reader.finish() is None
    reader.close()


def test_a_backlog_beyond_the_pending_bound_resumes_at_the_newest_idr(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_PENDING_BYTES", 400)
    units = encode_units(W, H, [RED, GREEN, BLUE] * 17)  # 51 independent IDR units
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = _reader(path)
    rgb, _ = reader.read_latest(2.0)
    assert _close(rgb, GREEN)  # unit 49, the newest complete one; everything older was skipped undecoded
    reader.close()


def test_an_inter_coded_backlog_that_cannot_be_caught_up_ends_the_stream_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_MAX_PENDING_BYTES", 100)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED] * 40, gop=1000)))  # one IDR, then only P-frames
    reader = _reader(path)
    with pytest.raises(h264.SensingStreamError, match="decoder cannot keep up"):
        reader.read_latest(2.0)


def test_a_restart_point_needs_an_idr_slice_and_in_band_parameter_sets():
    idr, p_frame = encode_units(W, H, [RED, RED], gop=30)[:2]
    assert h264._is_restart_point(idr) and not h264._is_restart_point(p_frame)
    slice_at = idr.index(b"\x00\x00\x01\x65")
    assert not h264._is_restart_point(idr[slice_at:])  # the IDR slice alone: no SPS/PPS to restart from


def test_finish_straight_away_decodes_every_pending_unit_in_order_before_flushing(tmp_path):
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED] * 3 + [GREEN] * 3 + [BLUE] * 3, gop=30)))
    reader = _reader(path)
    last = reader.finish()  # no read_latest first: the units are still queued/held back, all must be decoded
    assert last is not None and _close(last[0], BLUE)
    reader.close()


def test_finish_is_bounded_by_its_budget_and_fails_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_FINISH_BUDGET_S", 0.0)
    monkeypatch.setattr(h264, "_READ_BYTES", 512)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED, GREEN, BLUE]) * 300))
    reader = _reader(path)
    with pytest.raises(h264.SensingStreamError, match="could not finish decoding"):
        reader.finish()


def test_reading_after_finish_is_an_error_not_a_decode_on_a_flushed_decoder(tmp_path):
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED, GREEN])))
    reader = _reader(path)
    reader.finish()
    with pytest.raises(h264.SensingStreamError, match="already finished"):
        reader.read_latest(0.0)
