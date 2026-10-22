# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EGO producer failures reach NV's runner without changing SDK retry policy."""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from pipeline import VizRunner
from sources.ego import EgoSource
from sources.stereo_sbs import StereoSideBySideSource


@pytest.fixture
def preview(monkeypatch):
    cp = MagicMock()
    cp.uint8 = np.uint8
    cp.empty.side_effect = np.empty
    cp.cuda.runtime.getDevice.return_value = 0
    capture = MagicMock()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "ego_preview", SimpleNamespace(Capture=capture))
    source = EgoSource("ego-test", width=2, height=1)
    yield source, cp, capture
    source.stop()


def join_producer(source):
    source._thread.join(timeout=2)
    assert not source._thread.is_alive(), "EGO producer did not exit"


def nal_unit(codec, nal_type, *, short_start=False):
    start = b"\x00\x00\x01" if short_start else b"\x00\x00\x00\x01"
    if codec == "h265":
        header = bytes((nal_type << 1, 1))
    else:
        header = bytes((nal_type | (0x60 if nal_type in (1, 5, 7, 8) else 0),))
    return start + header + b"\x80"


def parameter_sets(codec):
    types = (32, 33, 34) if codec == "h265" else (7, 8)
    return b"".join(nal_unit(codec, nal_type) for nal_type in types)


def encoded_pair(codec, sequence_left=0, sequence_right=None, *, left=None, right=None):
    keyframe = parameter_sets(codec) + nal_unit(codec, 19 if codec == "h265" else 5)
    return {
        "left": keyframe if left is None else left,
        "right": keyframe if right is None else right,
        "sequence_left": sequence_left,
        "sequence_right": sequence_left if sequence_right is None else sequence_right,
        "timestamp_ns": 123,
    }


def run_encoded_pairs(preview, monkeypatch, codec, pairs):
    source, _, capture = preview
    source._format = codec
    decoders = [MagicMock(), MagicMock()]
    for decoder in decoders:
        decoder.decode.return_value = True
    monkeypatch.setattr("sources.ego.NvVideoDecoder", MagicMock(side_effect=decoders))
    pending = iter(pairs)

    def next_pair(timeout):
        try:
            pair = next(pending)
        except StopIteration:
            source._stop.set()
            return None
        if isinstance(pair, Exception):
            raise pair
        return pair

    capture.return_value.next_pair.side_effect = next_pair
    source.start()
    join_producer(source)
    source.latest()  # Surface any unexpected producer failure.
    return source, decoders


@pytest.mark.parametrize("failure_type", [RuntimeError, SystemExit])
@pytest.mark.parametrize("sbs", [False, True])
def test_producer_failure_reaches_consumer_before_sdk_close(
    preview, monkeypatch, failure_type, sbs
):
    source, _, capture = preview
    failure = failure_type("injected CUDA failure")
    closing = threading.Event()
    release = threading.Event()

    def close():
        closing.set()
        assert release.wait(timeout=3)

    capture.return_value.close.side_effect = close

    def produce():
        source._capture = capture.return_value
        raise failure

    monkeypatch.setattr(source, "_capture_loop", produce)
    consumer = StereoSideBySideSource(source) if sbs else source
    try:
        consumer.start()
        assert closing.wait(timeout=2)
        # Match source and cause text without fixing the wrapper punctuation.
        with pytest.raises(
            RuntimeError, match=r"ego-test.*injected CUDA failure"
        ) as error:
            consumer.latest()
        assert error.value.__cause__ is failure
        assert source._thread.is_alive()
    finally:
        release.set()
        consumer.stop()
    capture.return_value.close.assert_called_once()


@pytest.mark.parametrize("producer_fails", [False, True])
def test_cleanup_failure_cannot_overwrite_producer_error(
    preview, monkeypatch, producer_fails
):
    source, _, capture = preview
    failure = RuntimeError("decode failed")
    cleanup = RuntimeError("SDK close failed")
    capture.return_value.close.side_effect = cleanup

    def produce():
        source._capture = capture.return_value
        if producer_fails:
            raise failure

    monkeypatch.setattr(source, "_capture_loop", produce)
    source.start()
    join_producer(source)
    with pytest.raises(RuntimeError) as error:
        source.latest()
    assert error.value.__cause__ is (failure if producer_fails else cleanup)
    capture.return_value.close.assert_called_once()


def test_fatal_error_takes_precedence_over_unconsumed_frame(preview, monkeypatch):
    source, _, _ = preview

    def produce():
        with source._lock:
            source._publish_idx = 0
            source._timestamp_ns = 123
        raise RuntimeError("upload failed")

    monkeypatch.setattr(source, "_capture_loop", produce)
    source.start()
    join_producer(source)
    with pytest.raises(RuntimeError, match="upload failed"):
        source.latest()


def test_restart_clears_failure_and_mailbox_and_resolves_auto_gpu_again(
    preview, monkeypatch
):
    source, cp, _ = preview

    def fail():
        source._publish_idx = 0
        source._timestamp_ns = 123
        source._write_idx = 2
        source._frames = 8
        source._incomplete_pairs = 3
        raise RuntimeError("decode failed")

    monkeypatch.setattr(source, "_capture_loop", fail)
    source.start()
    join_producer(source)
    source.stop()
    cp.cuda.runtime.getDevice.return_value = 2
    monkeypatch.setattr(source, "_capture_loop", lambda: None)
    source.start()
    join_producer(source)
    assert source.latest() is None
    assert source._gpu_id == 2
    assert (
        source._timestamp_ns,
        source._write_idx,
        source._frames,
        source._incomplete_pairs,
    ) == (0, 0, 0, 0)


def test_failed_thread_start_allows_retry(preview, monkeypatch):
    source, _, capture = preview
    failure = RuntimeError("cannot start thread")
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", MagicMock(side_effect=failure))
        with pytest.raises(RuntimeError, match="cannot start thread"):
            source.start()
    assert source._thread is None
    assert not source._buffers_left and not source._buffers_right
    capture.assert_not_called()
    monkeypatch.setattr(source, "_capture_loop", lambda: None)
    source.start()
    join_producer(source)


def test_interrupted_start_retains_thread_for_shutdown(preview, monkeypatch):
    source, _, _ = preview
    thread = MagicMock()
    thread.start.side_effect = KeyboardInterrupt
    thread.is_alive.return_value = False
    monkeypatch.setattr("sources.ego.threading.Thread", lambda **kwargs: thread)
    with pytest.raises(KeyboardInterrupt):
        source.start()
    assert source._thread is thread
    assert source._buffers_left and source._buffers_right
    source.stop()
    thread.join.assert_called_once_with(timeout=5.0)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_reconnect_keeps_both_decoders_on_renderer_gpu(preview, monkeypatch, codec):
    source, cp, capture = preview
    source._format = codec
    source._renderer_cuda_device_id = 2
    decoders = [MagicMock(), MagicMock()]
    make_decoder = MagicMock(side_effect=decoders)
    monkeypatch.setattr("sources.ego.NvVideoDecoder", make_decoder)
    calls = 0

    def next_pair(timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient SDK read failure")
        source._stop.set()
        return encoded_pair(codec)

    capture.return_value.next_pair.side_effect = next_pair
    source.start()
    join_producer(source)
    assert capture.call_count == 2
    assert capture.return_value.close.call_count == 2
    assert make_decoder.call_count == 2
    assert all(
        c.kwargs == {"gpu_id": 2, "codec": codec} for c in make_decoder.call_args_list
    )
    assert all(c.args == (2,) for c in cp.cuda.Device.call_args_list)
    for decoder in decoders:
        decoder.reset.assert_called_once()
        decoder.decode.assert_called_once()
    assert source.latest().timestamp_ns == 123


def test_open_retry_is_cancellable_and_does_not_report_fatal_error(preview):
    source, _, capture = preview
    attempted = threading.Event()

    def open_device(*args):
        attempted.set()
        raise RuntimeError("EGO not found")

    capture.side_effect = open_device
    source.start()
    assert attempted.wait(timeout=2)
    source.stop()
    assert source.latest() is None
    assert source._thread is None


@pytest.mark.parametrize("fails", [False, True])
def test_decode_failure_is_fatal_but_waiting_for_output_is_not(
    preview, monkeypatch, fails
):
    source, _, capture = preview
    decoder = MagicMock()
    failure = RuntimeError("NVDEC failed")
    if fails:
        decoder.decode.side_effect = failure
    else:
        decoder.decode.return_value = False
    monkeypatch.setattr("sources.ego.NvVideoDecoder", lambda *args, **kwargs: decoder)

    def next_pair(timeout):
        source._stop.set()
        return encoded_pair("h264")

    capture.return_value.next_pair.side_effect = next_pair
    source.start()
    join_producer(source)
    if fails:
        with pytest.raises(RuntimeError, match="NVDEC failed") as error:
            source.latest()
        assert error.value.__cause__ is failure
    else:
        assert source.latest() is None
    capture.return_value.close.assert_called_once()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_startup_waits_for_parameters_and_keyframe(preview, monkeypatch, codec):
    predicted = nal_unit(codec, 1)
    idr = nal_unit(codec, 19 if codec == "h265" else 5, short_start=True)
    params = parameter_sets(codec)
    source, decoders = run_encoded_pairs(
        preview,
        monkeypatch,
        codec,
        [
            encoded_pair(codec, 0, left=predicted, right=predicted),
            encoded_pair(codec, 1, left=idr, right=idr),
            encoded_pair(codec, 2, left=params, right=params),
            encoded_pair(codec, 3, left=predicted, right=predicted),
            encoded_pair(codec, 4, left=idr, right=idr),
        ],
    )
    assert source._frames == 1
    for decoder in decoders:
        decoder.reset.assert_not_called()
        decoder.decode.assert_called_once()
        assert decoder.decode.call_args.args[0] == params + idr


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_recovery_replays_partial_parameters_in_dependency_order(
    preview, monkeypatch, codec
):
    types = (32, 33, 34) if codec == "h265" else (7, 8)
    pps = nal_unit(codec, types[-1])
    partial = b"".join(nal_unit(codec, nal_type) for nal_type in types[:-1])
    partial += nal_unit(codec, 19 if codec == "h265" else 5)
    source, decoders = run_encoded_pairs(
        preview,
        monkeypatch,
        codec,
        [
            encoded_pair(codec, 0, left=pps, right=pps),
            encoded_pair(codec, 1, left=partial, right=partial),
        ],
    )
    assert source._frames == 1
    for decoder in decoders:
        decoder.decode.assert_called_once()
        assert decoder.decode.call_args.args[0] == parameter_sets(codec) + partial


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_gap_resets_once_and_recovers_each_eye_independently(
    preview, monkeypatch, codec
):
    predicted = nal_unit(codec, 1)
    source, decoders = run_encoded_pairs(
        preview,
        monkeypatch,
        codec,
        [
            encoded_pair(codec, 0),
            None,
            encoded_pair(codec, 2, 1, left=predicted, right=predicted),
            encoded_pair(codec, 3, 2, left=predicted, right=predicted),
            encoded_pair(codec, 4, 3, right=predicted),
            encoded_pair(codec, 5, 5, left=predicted, right=predicted),
            encoded_pair(codec, 6, 6, left=predicted, right=predicted),
            encoded_pair(codec, 7, 7, left=predicted),
        ],
    )
    assert source._frames == 3
    left_packets = [call.args[0] for call in decoders[0].decode.call_args_list]
    right_packets = [call.args[0] for call in decoders[1].decode.call_args_list]
    keyframe = encoded_pair(codec)["left"]
    assert left_packets == [keyframe, keyframe, predicted, predicted, predicted]
    assert right_packets == [keyframe, predicted, predicted, predicted, keyframe]
    for decoder in decoders:
        decoder.reset.assert_called_once()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_timeouts_sei_and_no_output_keep_sequence_continuity(
    preview, monkeypatch, codec
):
    source, _, capture = preview
    source._format = codec
    predicted = nal_unit(codec, 1)
    sei = nal_unit(codec, 39 if codec == "h265" else 6)
    decoders = [MagicMock(), MagicMock()]
    for decoder in decoders:
        decoder.decode.side_effect = [True, False, False, True]
    monkeypatch.setattr("sources.ego.NvVideoDecoder", MagicMock(side_effect=decoders))
    pairs = iter(
        [
            encoded_pair(codec, 0),
            None,
            encoded_pair(codec, 1, left=sei, right=sei),
            encoded_pair(codec, 2, left=predicted, right=predicted),
            encoded_pair(codec, 3, left=predicted, right=predicted),
        ]
    )

    def next_pair(timeout):
        pair = next(pairs)
        if pair is not None and pair["sequence_left"] == 3:
            source._stop.set()
        return pair

    capture.return_value.next_pair.side_effect = next_pair
    source.start()
    join_producer(source)
    assert source.latest() is not None and source._frames == 2
    for decoder in decoders:
        decoder.reset.assert_not_called()
        assert decoder.decode.call_count == 4


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_reconnect_clears_parameter_cache_and_waits_again(
    preview, monkeypatch, codec
):
    params = parameter_sets(codec)
    idr = nal_unit(codec, 19 if codec == "h265" else 5)
    source, decoders = run_encoded_pairs(
        preview,
        monkeypatch,
        codec,
        [
            encoded_pair(codec, 100, left=params, right=params),
            RuntimeError("transient SDK read failure"),
            encoded_pair(codec, 0, left=idr, right=idr),
            encoded_pair(codec, 1),
        ],
    )
    assert source._frames == 1
    for decoder in decoders:
        decoder.reset.assert_called_once()
        decoder.decode.assert_called_once()
        assert decoder.decode.call_args.args[0] == params + idr


def test_hevc_cra_recovery_drops_rasl_until_trailing_picture(preview, monkeypatch):
    codec = "h265"
    cra = parameter_sets(codec) + nal_unit(codec, 21)
    rasl = nal_unit(codec, 8)
    radl = nal_unit(codec, 6)
    predicted = nal_unit(codec, 1)
    source, decoders = run_encoded_pairs(
        preview,
        monkeypatch,
        codec,
        [
            encoded_pair(codec, 0, left=cra, right=cra),
            encoded_pair(codec, 1, left=rasl, right=rasl),
            encoded_pair(codec, 2, left=radl, right=radl),
            encoded_pair(codec, 3, left=rasl, right=rasl),
            encoded_pair(codec, 4, left=predicted, right=predicted),
        ],
    )
    assert source._frames == 3
    for decoder in decoders:
        assert [call.args[0] for call in decoder.decode.call_args_list] == [
            cra,
            radl,
            predicted,
        ]
        decoder.reset.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["gpu-conflict", "producer"])
def test_viewer_cleans_up_after_ego_startup_or_producer_failure(
    preview, monkeypatch, tmp_path, failure_kind
):
    import camera_viz

    _, _, capture = preview
    events = []
    session = MagicMock()
    session.is_xr_mode.return_value = False
    session.should_close.return_value = False
    session.cuda_device_id = 2
    session.destroy.side_effect = lambda: events.append("destroy")
    capture.return_value.close.side_effect = lambda: events.append("close")
    path = tmp_path / "ego.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": "local",
                "display": {"mode": "window"},
                "cameras": [
                    {
                        "name": "ego",
                        "type": "ego",
                        "stereo": True,
                        "width": 2,
                        "height": 1,
                        "gpu_id": 1 if failure_kind == "gpu-conflict" else 0,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(camera_viz.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        camera_viz.display, "make_session", lambda *args, **kwargs: session
    )
    monkeypatch.setattr(
        camera_viz,
        "_build_display",
        lambda session, entries, shapes: (
            [entry.source for entry in entries],
            [MagicMock()],
            [None],
            [{}],
        ),
    )
    dashboard = MagicMock()
    dashboard.live = False
    monkeypatch.setattr(camera_viz, "Dashboard", lambda: dashboard)

    def produce(source):
        assert source._gpu_id == 2
        source._capture = capture()
        raise RuntimeError("injected producer failure")

    monkeypatch.setattr(EgoSource, "_capture_loop", produce)
    original_wait = VizRunner.wait

    def bounded_wait(runner, **kwargs):
        runner._submit_thread.join(timeout=2)
        runner._render_thread.join(timeout=2)
        assert not runner._submit_thread.is_alive()
        assert not runner._render_thread.is_alive()
        return original_wait(runner, **kwargs)

    monkeypatch.setattr(VizRunner, "wait", bounded_wait)
    expected = (
        "renderer uses CUDA GPU 2"
        if failure_kind == "gpu-conflict"
        else "injected producer failure"
    )
    with pytest.raises((ValueError, RuntimeError), match=expected):
        camera_viz.main([str(path)])
    assert events == (
        ["destroy"] if failure_kind == "gpu-conflict" else ["close", "destroy"]
    )
    dashboard.close.assert_called_once()
    if failure_kind == "gpu-conflict":
        capture.assert_not_called()


def test_nv_runner_wait_reports_ego_producer_failure(preview, monkeypatch):
    source, _, capture = preview
    failure = RuntimeError("GPU upload failed")

    def produce():
        source._capture = capture.return_value
        raise failure

    monkeypatch.setattr(source, "_capture_loop", produce)
    session = MagicMock()
    session.is_xr_mode.return_value = False
    session.should_close.return_value = False
    runner = VizRunner(session, [source], [MagicMock()])
    try:
        runner.start()
        runner._submit_thread.join(timeout=2)
        runner._render_thread.join(timeout=2)
        assert not runner._submit_thread.is_alive()
        assert not runner._render_thread.is_alive()
        with pytest.raises(RuntimeError, match="GPU upload failed") as error:
            runner.wait()
        assert error.value.__cause__ is failure
    finally:
        assert runner.stop()
    assert source._thread is None
    capture.return_value.close.assert_called_once()
