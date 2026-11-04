# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware H.264 decoding for the SENSING reader: the decoder process wrapper and the reader's use of it.

The Jetson decoder is a GStreamer subprocess; these tests replace it with stand-ins that follow the same pipe
contract (Annex-B on stdin, tightly packed I420 frames on stdout), so every code path runs on any machine. A
separate check at the end exercises the real pipeline when this machine has NVDEC.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from capture_server.acquisition import sensing_h264 as h264
from capture_server.acquisition import sensing_nvdec as nvdec

from .h264_fixtures import encode_units

W, H = 64, 32
FRAME = W * H * 3 // 2
RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)

# A stand-in that decodes the Annex-B on stdin with PyAV and writes each frame as raw I420: the real decoder's
# contract, so the reader sees correct pixels end to end.
PYAV_DECODER = textwrap.dedent(
    """
    import sys
    import av

    parser = av.CodecContext.create("h264", "r")
    decoder = av.CodecContext.create("h264", "r")
    out = sys.stdout.buffer

    def emit(packets):
        for packet in packets:
            for frame in decoder.decode(packet):
                out.write(frame.to_ndarray(format="yuv420p").tobytes())
                out.flush()

    while chunk := sys.stdin.buffer.read1(65536):
        emit(parser.parse(chunk))
    emit(parser.parse(None))
    emit([None])
    """
)


def _command(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def _counting_decoder(unit_bytes: int = 4) -> list[str]:
    """A stand-in that writes one frame, filled with its sequence number, per ``unit_bytes`` read."""
    return _command(
        textwrap.dedent(
            f"""
            import sys
            n = 0
            while True:
                data = sys.stdin.buffer.read({unit_bytes})
                if len(data) < {unit_bytes}:
                    break
                sys.stdout.buffer.write(bytes([n % 251]) * {FRAME})
                sys.stdout.buffer.flush()
                n += 1
            """
        )
    )


def _wait_for(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the decoder"
        time.sleep(0.01)


@pytest.fixture
def decoders():
    """Decoders created by a test, closed afterwards so a failing test never leaks a process."""
    made = []

    def make(command, width: int = W, height: int = H) -> nvdec.HardwareDecoder:
        decoder = nvdec.HardwareDecoder(width, height, command=command)
        made.append(decoder)
        return decoder

    yield make
    for decoder in made:
        decoder.close()


# --- availability and the pipeline ---


@pytest.fixture
def fresh_probe():
    nvdec.hardware_decoder_unavailable_reason.cache_clear()
    yield
    nvdec.hardware_decoder_unavailable_reason.cache_clear()


def _fake_environment(monkeypatch, *, node=True, tools=("gst-launch-1.0", "gst-inspect-1.0"), probe=0):
    monkeypatch.setattr(nvdec.os.path, "exists", lambda p: node if p == nvdec._DEVICE_NODE else True)
    monkeypatch.setattr(nvdec.shutil, "which", lambda t: f"/usr/bin/{t}" if t in tools else None)
    if isinstance(probe, Exception):

        def run(*a, **k):
            raise probe

    else:

        def run(*a, **k):
            return subprocess.CompletedProcess(a[0], probe)

    monkeypatch.setattr(nvdec.subprocess, "run", run)


def test_available_when_node_tools_and_element_exist(monkeypatch, fresh_probe):
    _fake_environment(monkeypatch)
    assert nvdec.hardware_decoder_unavailable_reason() is None


def test_unavailable_without_the_decoder_node(monkeypatch, fresh_probe):
    _fake_environment(monkeypatch, node=False)
    assert "not a Jetson" in nvdec.hardware_decoder_unavailable_reason()


def test_unavailable_without_gstreamer_tools(monkeypatch, fresh_probe):
    _fake_environment(monkeypatch, tools=("gst-launch-1.0",))
    assert "gst-inspect-1.0 is not installed" in nvdec.hardware_decoder_unavailable_reason()


def test_unavailable_when_the_element_is_missing(monkeypatch, fresh_probe):
    _fake_environment(monkeypatch, probe=1)
    assert "nvv4l2decoder is not installed" in nvdec.hardware_decoder_unavailable_reason()


def test_unavailable_when_the_probe_cannot_run(monkeypatch, fresh_probe):
    _fake_environment(monkeypatch, probe=subprocess.TimeoutExpired("gst-inspect-1.0", 10))
    assert "could not query GStreamer" in nvdec.hardware_decoder_unavailable_reason()


def test_pipeline_is_low_delay_with_a_leaky_queue_by_default():
    command = nvdec.pipeline_command()
    assert "disable-dpb=true" in command and "leaky=downstream" in command
    assert command.index("queue") < command.index("nvvidconv")  # frames are dropped before the costly copy


def test_pipeline_keeps_the_reorder_buffer_when_asked():
    assert "disable-dpb=true" not in nvdec.pipeline_command(low_delay=False)


# --- the decoder process ---


@pytest.mark.parametrize("width, height", [(0, 32), (64, 0), (63, 32), (64, 31)])
def test_geometry_must_be_positive_and_even(width, height):
    with pytest.raises(ValueError, match="even dimensions"):
        nvdec.HardwareDecoder(width, height)


def test_frames_flow_and_the_newest_wins(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    for _ in range(5):
        decoder.write(b"\x00\x00\x00\x00")
    _wait_for(lambda: decoder.frames_received == 5)
    frame = decoder.take_frame()
    assert frame.shape == (H * 3 // 2, W) and frame.dtype == np.uint8
    assert (frame == 4).all()  # frames 0..3 were replaced before being taken
    assert decoder.take_frame() is None and not decoder.has_frame
    decoder.release_frame()


def test_a_taken_frame_is_not_overwritten_while_held(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    decoder.write(b"\x00" * 4)
    _wait_for(lambda: decoder.has_frame)
    held = decoder.take_frame()
    for _ in range(6):
        decoder.write(b"\x00" * 4)
    _wait_for(lambda: decoder.frames_received == 7)
    assert (held == 0).all()  # the output thread filled other buffers
    decoder.release_frame()
    assert (decoder.take_frame() == 6).all()
    decoder.release_frame()


def test_taking_twice_without_releasing_is_a_programming_error(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    decoder.write(b"\x00" * 8)
    _wait_for(lambda: decoder.frames_received == 2)
    decoder.take_frame()
    decoder.write(b"\x00" * 4)
    _wait_for(lambda: decoder.frames_received == 3)
    with pytest.raises(RuntimeError, match="release_frame"):
        decoder.take_frame()


def test_closing_the_input_flushes_and_ends_the_output(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    decoder.write(b"\x00" * 12)
    assert decoder.close_input() is True
    assert decoder.wait_finished(10.0)
    decoder.check()  # a closed input makes the exit expected, not an error
    assert decoder.frames_received == 3


def test_writes_after_the_input_is_closed_are_refused(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    decoder.close_input()
    with pytest.raises(nvdec.HardwareDecodeError, match="already closed"):
        decoder.write(b"x")


def test_close_input_waits_while_input_is_still_queued(decoders):
    decoder = decoders(_command("import time\ntime.sleep(60)"))  # never reads its input
    decoder.start()
    decoder.write(b"\x00" * (8 << 20))
    assert decoder.pending_input_bytes > 0
    assert decoder.close_input() is False


def test_a_stalled_decoder_never_blocks_the_writer(decoders):
    decoder = decoders(_command("import time\ntime.sleep(60)"))
    decoder.start()
    started = time.monotonic()
    for _ in range(8):
        decoder.write(b"\x00" * (1 << 20))
    assert time.monotonic() - started < 2.0
    assert decoder.pending_input_bytes >= 6 << 20  # the pipe took about one MiB; the rest is queued here


def test_an_early_exit_is_reported_with_its_error_output(decoders):
    decoder = decoders(_command("import sys\nsys.stderr.write('no such device')\nsys.exit(3)"))
    decoder.start()
    _wait_for(lambda: decoder._reader_done.is_set())
    time.sleep(0.2)  # let the diagnostics thread collect stderr
    with pytest.raises(nvdec.HardwareDecodeError, match=r"exited unexpectedly \(code 3\).*no such device"):
        decoder.check()


def test_writing_to_a_dead_decoder_is_reported(decoders):
    decoder = decoders(_command("import sys\nsys.exit(1)"))
    decoder.start()
    _wait_for(lambda: decoder._reader_done.is_set())
    with pytest.raises(nvdec.HardwareDecodeError, match="closed its input"):
        for _ in range(50):
            decoder.write(b"\x00" * 4096)
            time.sleep(0.01)


def test_output_that_ends_mid_frame_is_an_error(decoders):
    decoder = decoders(_command(f"import sys\nsys.stdout.buffer.write(b'x' * {FRAME // 2})\nsys.stdout.flush()"))
    decoder.start()
    _wait_for(lambda: decoder._reader_done.is_set())
    with pytest.raises(nvdec.HardwareDecodeError, match="mid-frame"):
        decoder.check()


def test_close_stops_a_stalled_process_quickly_and_is_repeatable(decoders):
    decoder = decoders(_command("import time\ntime.sleep(60)"))
    decoder.start()
    proc = decoder._proc
    started = time.monotonic()
    decoder.close()
    assert time.monotonic() - started < 1.5 and proc.poll() is not None
    decoder.close()  # idempotent


def test_closing_a_decoder_that_never_started_is_harmless():
    nvdec.HardwareDecoder(W, H).close()


def test_pumping_before_the_process_starts_does_nothing():
    decoder = nvdec.HardwareDecoder(W, H)
    decoder.pump()
    assert decoder.pending_input_bytes == 0


def test_close_kills_a_process_that_ignores_termination(decoders):
    ignoring = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)"
    decoder = decoders(_command(ignoring))
    decoder.start()
    proc = decoder._proc
    time.sleep(0.5)  # the child must have installed its handler before the terminate arrives
    started = time.monotonic()
    decoder.close()
    assert time.monotonic() - started < 2.0 and proc.poll() is not None


def test_a_pipe_that_cannot_be_enlarged_is_tolerated(decoders, monkeypatch):
    def refuse(*args):
        raise OSError("not permitted")

    monkeypatch.setattr(nvdec.fcntl, "fcntl", refuse)
    decoder = decoders(_counting_decoder())
    decoder.start()
    decoder.write(b"\x00" * 4)
    _wait_for(lambda: decoder.has_frame)


def test_close_input_twice_is_harmless(decoders):
    decoder = decoders(_counting_decoder())
    decoder.start()
    assert decoder.close_input() is True
    assert decoder.close_input() is True


# --- the reader on the hardware path ---


def _reader(path, **kw) -> h264.SensingStreamReader:
    kw.setdefault("decoder", "hardware")
    kw.setdefault("hardware_command", _command(PYAV_DECODER))
    reader = h264.SensingStreamReader(str(path), width=kw.pop("width", W), height=kw.pop("height", H), **kw)
    reader.connect(1.0)
    return reader


def _close(rgb: np.ndarray, expected, tol: int = 14) -> bool:
    return bool(np.abs(rgb[H // 2, W // 2].astype(int) - np.array(expected)).max() <= tol)


def _read_until(reader, timeout_s: float = 20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = reader.read_latest(0.2)
        if out is not None:
            return out
    raise AssertionError("no frame decoded in time")


def test_hardware_reader_delivers_correct_colours(tmp_path, caplog):
    units = encode_units(W, H, [RED, GREEN, BLUE], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 4)
    with caplog.at_level(logging.INFO, logger="isaaccapture.capture_server.sensing_h264"):
        reader = _reader(path)
    assert "NVDEC hardware decoder" in caplog.text
    rgb, selected_ns = _read_until(reader)
    assert rgb.shape == (H, W, 3) and selected_ns > 0
    assert any(_close(rgb, c) for c in (RED, GREEN, BLUE))
    reader.close()


def test_hardware_reader_finish_returns_the_last_frame(tmp_path):
    units = encode_units(W, H, [RED, GREEN, BLUE], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units))
    reader = _reader(path)
    reader.read_latest(0.05)
    rgb, _ = reader.finish()
    assert _close(rgb, BLUE)  # the final unit is only known complete once the stream ends
    assert reader.finish() is None  # terminal
    reader.close()


def test_hardware_reader_full_range_flag_changes_the_conversion(tmp_path):
    units = encode_units(W, H, [GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 3)
    limited = _reader(path)
    full = _reader(path, full_range=True)
    a, _ = _read_until(limited)
    b, _ = _read_until(full)
    assert not np.array_equal(a, b)
    limited.close()
    full.close()


def test_hardware_reader_rejects_a_stream_of_the_wrong_size(tmp_path):
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 2)
    reader = _reader(path, width=W * 2, height=H * 2)
    with pytest.raises(h264.SensingStreamError, match=r"decoded 64x32, expected 128x64"):
        _read_until(reader)
    reader.close()


def test_hardware_reader_reports_a_dead_decoder(tmp_path):
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 2)
    reader = _reader(path, hardware_command=_command("import sys\nsys.stderr.write('boom')\nsys.exit(2)"))
    with pytest.raises(h264.SensingStreamError, match="decoder"):
        for _ in range(100):
            reader.read_latest(0.1)
    with pytest.raises(h264.SensingStreamError, match="failed earlier"):
        reader.read_latest(0.0)
    reader.close()


def test_hardware_reader_close_ends_the_process(tmp_path):
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 2)
    reader = _reader(path)
    _read_until(reader)
    proc = reader._hw._proc
    reader.close()
    assert proc.poll() is not None and reader._hw is None


def test_input_beyond_the_decoders_capacity_stays_in_the_readers_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "_HW_MAX_INPUT_BYTES", 1)
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    # More than the decoder's pipe can hold, so the input really is full and later units must wait here.
    path.write_bytes(b"".join(units) * (4 * nvdec._STDIN_PIPE_BYTES // sum(map(len, units))))
    reader = _reader(path, hardware_command=_command("import time\ntime.sleep(60)"))
    assert reader.read_latest(0.3) is None
    assert reader._pending  # not handed over while the decoder's input is full
    reader.close()


# --- choosing the decoder ---


def test_unknown_decoder_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="decoder must be one of"):
        h264.SensingStreamReader(str(tmp_path / "s"), width=W, height=H, decoder="gpu")


def test_hardware_requested_but_unavailable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "hardware_decoder_unavailable_reason", lambda _backend: "no NVDEC here")
    (tmp_path / "s.h264").write_bytes(b"")
    reader = h264.SensingStreamReader(str(tmp_path / "s.h264"), width=W, height=H, decoder="hardware")
    with pytest.raises(h264.SensingStreamError, match="requested but is unavailable: no NVDEC here"):
        reader.connect(1.0)
    reader.close()


def test_auto_falls_back_to_software_and_says_why(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(h264, "hardware_decoder_unavailable_reason", lambda _backend: "no NVDEC here")
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 2)
    with caplog.at_level(logging.INFO, logger="isaaccapture.capture_server.sensing_h264"):
        reader = h264.SensingStreamReader(str(path), width=W, height=H, decoder="auto")
        reader.connect(1.0)
    assert "falling back to CPU" in caplog.text and "no NVDEC here" in caplog.text
    rgb, _ = reader.read_latest(1.0)
    assert rgb.shape == (H, W, 3) and reader._hw is None
    reader.close()


def test_auto_uses_hardware_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(h264, "hardware_decoder_unavailable_reason", lambda _backend: None)
    units = encode_units(W, H, [RED, GREEN], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 4)
    reader = _reader(path, decoder="auto")
    _read_until(reader)
    assert reader._hw is not None
    reader.close()


def test_software_choice_is_logged(tmp_path, caplog):
    path = tmp_path / "s.h264"
    path.write_bytes(b"")
    with caplog.at_level(logging.INFO, logger="isaaccapture.capture_server.sensing_h264"):
        reader = h264.SensingStreamReader(str(path), width=W, height=H, decoder="software")
        reader.connect(1.0)
    assert "CPU (PyAV software), as configured" in caplog.text
    reader.close()


# --- the rig configuration ---


def test_rig_decoder_defaults_to_auto_and_rejects_unknown_values():
    from capture_server.acquisition import sensing as sn

    base = dict(
        executable="/x", expected_backend="MOCK", platform_config="/p", config_name="C", link_masks=("0x0000",)
    )
    assert sn.SensingRigConfig(**base).decoder == "auto"
    with pytest.raises(ValueError, match="decoder must be one of"):
        sn.SensingRigConfig(**base, decoder="gpu")


def test_station_file_accepts_a_decoder_key_and_rejects_a_bad_one(tmp_path):
    from capture_server.acquisition import sensing as sn

    def write(decoder: str) -> str:
        path = tmp_path / "station.yaml"
        path.write_text(
            "sensing_rig:\n"
            "  executable: /x\n  expected_backend: MOCK\n  platform_config: /p\n  config_name: C\n"
            f"  link_masks: ['0x0000']\n  decoder: {decoder}\n"
            "cameras:\n  left:\n    type: SENSING_COLOR\n    module_name: M\n"
        )
        return str(path)

    rig, _ = sn.load_sensing_station(write("software"))
    assert rig.decoder == "software"
    with pytest.raises(ValueError, match="decoder"):
        sn.load_sensing_station(write("gpu"))
    with pytest.raises(ValueError, match="decoder"):
        sn.load_sensing_station(write("1"))


# --- the real pipeline, when this machine has NVDEC ---


@pytest.mark.skipif(
    nvdec.hardware_decoder_unavailable_reason() is not None, reason="no Jetson NVDEC on this machine"
)
def test_real_nvdec_pipeline_decodes_correct_colours(tmp_path):
    # Hardware decoders have a minimum surface size; a 640x480 stream is well above it.
    width, height = 640, 480
    units = encode_units(width, height, [RED, GREEN, BLUE], gop=1)
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(units) * 8)
    reader = h264.SensingStreamReader(str(path), width=width, height=height, decoder="hardware")
    reader.connect(1.0)
    try:
        rgb, _ = _read_until(reader, timeout_s=30.0)
    finally:
        reader.close()
    centre = rgb[height // 2, width // 2].astype(int)
    assert min(np.abs(centre - np.array(c)).max() for c in (RED, GREEN, BLUE)) <= 16
