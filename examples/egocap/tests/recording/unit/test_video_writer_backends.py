# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for video_writer_backends.py's public contract: the config-validation helpers, a real
CPU (ffmpeg_h264/libx264) encode-write-release-readback roundtrip -- this is the existing default
encoder `VideoSourceSpec` already uses, not a new test dependency -- and a few behavior-oriented
option/error/lifecycle checks callers actually rely on. NVENC (GPU) is an explicitly optional
hardware check, honestly skipped by default rather than probed at collection time; set
EGOCAP_TEST_NVENC=1 to opt in on a real GPU.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import egocap.recording.video_writer_backends as vwb
from egocap.recording.video_writer_backends import (
    FFmpegH264Backend,
    FFmpegH264NvencBackend,
    FFmpegPipeBackend,
    OpenCVBackend,
    VideoEncodingFailed,
    create_video_writer,
    resolve_video_encoder_options,
    validate_ffmpeg_threads,
)

_NVENC_OPT_IN = os.environ.get("EGOCAP_TEST_NVENC") == "1"


def _frame(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01, what: str = "condition") -> None:
    """Polls *predicate* until it's true, rather than sleeping a guessed duration -- bounded, and
    raises a diagnostic TimeoutError (not a silent pass) if *timeout* is exceeded."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for: {what}")


# A real executable standing in for ffmpeg, placed on PATH -- same real-subprocess approach the
# existing round-trip tests above already use (not a mocked Popen), so the thread/process
# lifecycle code under test (stdin pipe, SIGTERM/SIGKILL, wait/poll, stderr drain) runs for real.
# Controlled per-test via env vars (set through `monkeypatch`, so each test gets its own
# configuration without touching the script itself):
#   FAKE_FFMPEG_ARGV_FILE  -- if set, the received argv is JSON-dumped here on start.
#   FAKE_FFMPEG_READY_FILE -- if set, touched immediately on start, before anything else --
#                             the one synchronization point tests poll on instead of sleeping.
#   FAKE_FFMPEG_MODE       -- "drain" (default): read stdin to EOF, then exit 0.
#                             "ignore_sigterm": install a SIGTERM no-op handler first, then drain
#                             normally (SIGKILL still works -- a real process can't block that).
#                             "never_read_stdin": never touch stdin at all; block forever until
#                             killed, so the OS pipe fills and the writer thread's write() blocks.
#                             "exit_immediately": exit 0 without reading stdin at all.
#                             "spew_stderr": write FAKE_FFMPEG_STDERR_BYTES bytes to stderr, then
#                             drain stdin normally -- deterministic content for stderr-cap tests.
#                             "drain_then_hang": drain stdin to EOF (so the writer thread finishes
#                             and joins normally), then ignore SIGTERM and block forever -- models
#                             a process that's done consuming input but won't actually exit.
_FAKE_FFMPEG_SRC = """#!{python}
import os, signal, sys

def main():
    # argv is published FIRST, atomically (write to a temp file, then rename -- a reader that
    # sees the final filename exist always sees the complete, valid content, never a
    # partially-written one). The ready marker is published LAST, only once argv is fully visible
    # AND any mode-specific signal handler is installed -- a test that waits for "ready" is then
    # guaranteed both are true, not racing either one.
    argv_file = os.environ.get("FAKE_FFMPEG_ARGV_FILE")
    if argv_file:
        import json
        # PID-qualified, not a fixed ".tmp" suffix: two fake-ffmpeg processes sharing the same
        # FAKE_FFMPEG_ARGV_FILE (a test running more than one writer at once) must never collide
        # renaming the SAME temp path -- the final argv_file may still end up written by whichever
        # process finishes last, which is fine for a test that doesn't read it in that situation.
        tmp = f"{{argv_file}}.tmp.{{os.getpid()}}"
        with open(tmp, "w") as f:
            json.dump(sys.argv[1:], f)
        os.rename(tmp, argv_file)
    mode = os.environ.get("FAKE_FFMPEG_MODE", "drain")
    if mode in ("ignore_sigterm", "drain_then_hang"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "spew_stderr":
        n = int(os.environ.get("FAKE_FFMPEG_STDERR_BYTES", "0"))
        sys.stderr.buffer.write(b"E" * n)
        sys.stderr.buffer.flush()
    ready = os.environ.get("FAKE_FFMPEG_READY_FILE")
    if ready:
        open(ready, "w").close()
    if mode == "exit_immediately":
        sys.exit(0)
    if mode == "never_read_stdin":
        signal.pause()
        return
    data = sys.stdin.buffer
    while True:
        chunk = data.read(1 << 20)
        if not chunk:
            break
    if mode == "drain_then_hang":
        signal.pause()
        return
    sys.exit(0)

if __name__ == "__main__":
    main()
"""


class _FakeFfmpeg:
    def __init__(self, argv_file: Path, ready_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._argv_file = argv_file
        self._ready_file = ready_file
        self._monkeypatch = monkeypatch

    def set_mode(self, mode: str) -> None:
        self._monkeypatch.setenv("FAKE_FFMPEG_MODE", mode)

    def set_stderr_bytes(self, n: int) -> None:
        self._monkeypatch.setenv("FAKE_FFMPEG_STDERR_BYTES", str(n))

    def argv(self) -> list[str]:
        # Wait on the ready marker, not just the argv file's own existence -- the fake script
        # only touches "ready" after argv is fully (atomically) published, so this can't observe
        # a partially-written file even if the rename below somehow weren't already atomic.
        self.wait_until_ready()
        return json.loads(self._argv_file.read_text())

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        _wait_until(self._ready_file.exists, timeout=timeout, what="fake ffmpeg ready marker")


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeFfmpeg:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "ffmpeg"
    script.write_text(_FAKE_FFMPEG_SRC.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    argv_file = tmp_path / "fake_ffmpeg_argv.json"
    ready_file = tmp_path / "fake_ffmpeg_ready"
    monkeypatch.setenv("FAKE_FFMPEG_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_FFMPEG_READY_FILE", str(ready_file))
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "drain")
    return _FakeFfmpeg(argv_file, ready_file, monkeypatch)


def _decode_all_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# --- validate_ffmpeg_threads / resolve_video_encoder_options (pure, no I/O) -------------------


def test_validate_ffmpeg_threads_accepts_none_and_positive_ints():
    assert validate_ffmpeg_threads(None) is None
    assert validate_ffmpeg_threads(8) == 8


@pytest.mark.parametrize("bad", [0, -1, 1.5, "4", True])
def test_validate_ffmpeg_threads_rejects_non_positive_or_non_int(bad):
    with pytest.raises(ValueError, match="positive integer"):
        validate_ffmpeg_threads(bad)


def test_resolve_video_encoder_options_opencv_rejects_any_options():
    assert resolve_video_encoder_options("opencv_mp4v", None) == {}
    with pytest.raises(ValueError, match="only supported with"):
        resolve_video_encoder_options("opencv_mp4v", {"crf": 10})


def test_resolve_video_encoder_options_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_video_encoder_options("ffmpeg_h264", ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_resolve_video_encoder_options_h264_defaults_and_override():
    defaults = resolve_video_encoder_options("ffmpeg_h264", None)
    assert defaults == {"crf": 12, "preset": "medium", "pix_fmt": "yuv420p"}
    assert resolve_video_encoder_options("ffmpeg_h264", {"crf": 20})["crf"] == 20


def test_resolve_video_encoder_options_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown video_encoder_options keys"):
        resolve_video_encoder_options("ffmpeg_h264", {"bogus": 1})


@pytest.mark.parametrize("crf", [-1, 52, 12.5])
def test_resolve_video_encoder_options_rejects_bad_crf(crf):
    with pytest.raises(ValueError, match="crf must be an int"):
        resolve_video_encoder_options("ffmpeg_h264", {"crf": crf})


def test_resolve_video_encoder_options_rejects_bad_preset():
    with pytest.raises(ValueError, match="preset must be one of"):
        resolve_video_encoder_options("ffmpeg_h264", {"preset": "not-a-real-preset"})


def test_resolve_video_encoder_options_rejects_bad_pix_fmt():
    with pytest.raises(ValueError, match="pix_fmt must be one of"):
        resolve_video_encoder_options("ffmpeg_h264", {"pix_fmt": "not-a-real-format"})


def test_resolve_video_encoder_options_nvenc_translates_x264_preset_names():
    # Pure option resolution -- no ffmpeg/GPU involved, just the preset-name mapping.
    assert resolve_video_encoder_options("ffmpeg_h264_nvenc", {"preset": "medium"})["preset"] == "p4"
    assert resolve_video_encoder_options("ffmpeg_h264_nvenc", {"preset": "p2"})["preset"] == "p2"


# --- OpenCVBackend (real encoding, no external process) ----------------------------------------


def test_opencv_backend_writes_real_frames_and_produces_a_playable_file(tmp_path: Path):
    path = str(tmp_path / "out.mp4")
    writer = OpenCVBackend(path, width=64, height=48, fps=10.0)
    for _ in range(5):
        writer.write(_frame(64, 48))
    writer.release()
    frames = _decode_all_frames(path)
    assert len(frames) == 5
    assert all(f.shape == (48, 64, 3) for f in frames)


def test_opencv_backend_rejects_wrong_dtype_or_shape(tmp_path: Path):
    writer = OpenCVBackend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0)
    try:
        with pytest.raises(ValueError, match="uint8 BGR frames"):
            writer.write(_frame(32, 32))
    finally:
        writer.release()


def test_opencv_backend_raises_when_the_writer_fails_to_open(tmp_path: Path):
    # A real failure (no mocking needed): OpenCV can't open an output path under a directory
    # that doesn't exist -- "missing" is never created, confined to this test's own tmp_path.
    with pytest.raises(RuntimeError, match="failed to open"):
        OpenCVBackend(str(tmp_path / "missing" / "out.mp4"), width=64, height=48, fps=10.0)


# --- FFmpegH264Backend: the real, existing default encoder -------------------------------------


def test_ffmpeg_h264_backend_writes_real_frames_and_produces_a_playable_file(tmp_path: Path):
    path = str(tmp_path / "out.mp4")
    writer = FFmpegH264Backend(path, width=64, height=48, fps=10.0, preset="ultrafast")
    for i in range(5):
        writer.write(_frame(64, 48, value=i * 10))
    writer.release()
    frames = _decode_all_frames(path)
    assert len(frames) == 5
    assert all(f.shape == (48, 64, 3) for f in frames)


def test_ffmpeg_h264_backend_rejects_wrong_dtype_or_shape(tmp_path: Path):
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    try:
        with pytest.raises(ValueError, match="H.264 requires uint8 BGR frames"):
            writer.write(_frame(32, 32))
    finally:
        writer.release()


def test_ffmpeg_h264_backend_write_after_release_raises(tmp_path: Path):
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    writer.release()
    with pytest.raises(VideoEncodingFailed, match="write\\(\\) after release\\(\\)"):
        writer.write(_frame(64, 48))


def test_ffmpeg_h264_backend_release_is_idempotent(tmp_path: Path):
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    writer.release()
    writer.release()


def test_ffmpeg_h264_backend_a_real_encoder_failure_surfaces_at_release_not_falsely_succeeds(tmp_path: Path):
    # A real failure, not a mock: ffmpeg itself cannot open an output path under a directory that
    # doesn't exist ("missing" is never created), so it exits nonzero -- release() must raise
    # rather than silently reporting a clean finalize, and the error must carry real diagnostics.
    path = str(tmp_path / "missing" / "out.mp4")
    writer = FFmpegH264Backend(path, width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    with pytest.raises(VideoEncodingFailed) as exc_info:
        writer.release()
    assert "ffmpeg" in str(exc_info.value).lower()


def test_ffmpeg_h264_backend_accepts_an_explicit_thread_limit(tmp_path: Path):
    path = str(tmp_path / "out.mp4")
    writer = FFmpegH264Backend(path, width=64, height=48, fps=10.0, preset="ultrafast", thread_limit=1)
    writer.write(_frame(64, 48))
    writer.release()
    assert Path(path).stat().st_size > 0


# --- create_video_writer factory ----------------------------------------------------------------


def test_create_video_writer_dispatches_by_encoder(tmp_path: Path):
    opencv = create_video_writer("opencv_mp4v", str(tmp_path / "a.mp4"), 64, 48, 10.0)
    assert isinstance(opencv, OpenCVBackend)
    opencv.release()

    h264 = create_video_writer(
        "ffmpeg_h264",
        str(tmp_path / "b.mp4"),
        64,
        48,
        10.0,
        source_config={"video_encoder_options": {"preset": "ultrafast"}},
    )
    assert isinstance(h264, FFmpegH264Backend)
    h264.release()


def test_create_video_writer_rejects_unknown_encoder(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown video encoder"):
        create_video_writer("not_a_real_encoder", str(tmp_path / "x.mp4"), 64, 48, 10.0)  # type: ignore[arg-type]


# --- NVENC (GPU): explicitly optional hardware check, never required for the normal suite -------


@pytest.mark.skipif(not _NVENC_OPT_IN, reason="NVENC hardware check is opt-in: set EGOCAP_TEST_NVENC=1")
def test_ffmpeg_h264_nvenc_backend_writes_real_frames_and_produces_a_playable_file(tmp_path: Path):
    path = str(tmp_path / "out.mp4")
    writer = FFmpegH264NvencBackend(path, width=256, height=256, fps=10.0, preset="p1")
    writer.write(_frame(256, 256))
    writer.release()
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    cap.release()


def test_fake_ffmpeg_drains_stdin_and_exits_clean(tmp_path: Path, fake_ffmpeg):
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    writer.release()
    argv = fake_ffmpeg.argv()
    assert argv[0] == "-y"
    assert "libx264" in argv


# --- OpenCVBackend: output_path, and write()/release() after an already-released writer ---------


def test_opencv_backend_output_path_and_post_release_calls_are_quiet_no_ops(tmp_path: Path):
    path = str(tmp_path / "out.mp4")
    writer = OpenCVBackend(path, width=64, height=48, fps=10.0)
    assert writer.output_path == path
    writer.release()
    writer.release()  # idempotent
    writer.write(_frame(64, 48))  # silently dropped, unlike the ffmpeg backends' hard raise


# --- FFmpegPipeBackend: output_path, and the non-contiguous-frame path in the writer thread ------


def test_ffmpeg_pipe_backend_output_path(tmp_path: Path, fake_ffmpeg):
    path = str(tmp_path / "out.mp4")
    writer = FFmpegPipeBackend(
        path, 64, 48, 10.0, input_pixel_format="bgr24", encoder_args=["-c:v", "libx264"], writer_name="test"
    )
    try:
        assert writer.output_path == path
    finally:
        writer.release()


def test_ffmpeg_pipe_backend_makes_a_non_contiguous_frame_contiguous_before_writing(tmp_path: Path, fake_ffmpeg):
    # FFmpegH264Backend's own write() always calls np.ascontiguousarray first, so this base-class
    # branch is only reachable by driving FFmpegPipeBackend directly with a frame that is already
    # non-contiguous (a reversed view, via a negative step slice).
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    try:
        non_contiguous = _frame(4, 4)[:, ::-1, :]
        assert not non_contiguous.flags["C_CONTIGUOUS"]
        writer.write(non_contiguous)
    finally:
        writer.release()


# --- FFmpegPipeBackend: stderr capture edge cases -------------------------------------------------


def test_ffmpeg_pipe_backend_handles_a_missing_stderr_pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # subprocess.Popen(..., stderr=subprocess.PIPE) always gives a real pipe in production; the
    # defensive "stderr_r is None" branch in _start() only exists for a Popen that doesn't behave
    # that way, which real ffmpeg never does -- so this needs a stand-in Popen, not the fake binary.
    import io

    class _FakeCompletedProc:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = None
            self.stderr = None
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(vwb.subprocess, "Popen", lambda *a, **k: _FakeCompletedProc())
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    writer.write(_frame(4, 4))
    writer.release()  # must not raise despite no stderr thread ever having started


def test_ffmpeg_pipe_backend_stderr_overflow_is_capped_not_grown_without_bound(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vwb, "_MAX_FFMPEG_STDERR_BYTES", 8)
    fake_ffmpeg.set_mode("spew_stderr")
    # Bigger than one read(8192) chunk, so the drain loop sees at least two chunks: the first
    # fills the 8-byte cap (room > 0, taken), the second finds room already exhausted (room <= 0,
    # the skip branch) -- a single chunk under 8192 bytes could only ever exercise the first.
    fake_ffmpeg.set_stderr_bytes(20_000)
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    try:
        fake_ffmpeg.wait_until_ready()  # the spew already happened by the time "ready" is touched
        stderr_thread = writer._stderr_thread
        assert stderr_thread is not None
        writer.write(_frame(4, 4))
        writer.release()  # closes stdin -> the fake process (already past its stderr write) exits;
        # release() itself nulls self._stderr_thread at the end, hence the reference captured above
        stderr_thread.join(timeout=5.0)
        assert not stderr_thread.is_alive()
    finally:
        writer.release()
    with writer._stderr_lock:
        assert 0 < len(writer._stderr_buf) <= 8


def test_ffmpeg_pipe_backend_stderr_drain_failure_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import io

    class _RaisingStderr(io.RawIOBase):
        def read(self, size=-1):
            raise OSError("stderr read failed")

    class _FakeCompletedProc:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = None
            self.stderr = _RaisingStderr()
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(vwb.subprocess, "Popen", lambda *a, **k: _FakeCompletedProc())
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    stderr_thread = writer._stderr_thread
    assert stderr_thread is not None
    stderr_thread.join(timeout=5.0)
    assert not stderr_thread.is_alive()
    assert "ffmpeg stderr drain failed" in caplog.text
    writer.write(_frame(4, 4))
    writer.release()


# --- FFmpegPipeBackend: _failure_detail with and without a recorded writer error -----------------


def test_failure_detail_includes_the_writer_error_when_one_was_recorded(tmp_path: Path, fake_ffmpeg):
    writer_with_error = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    writer_without_error = FFmpegPipeBackend(
        str(tmp_path / "out2.mp4"),
        4,
        4,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test2",
    )
    try:
        writer_with_error._writer_error = RuntimeError("boom")
        detail = writer_with_error._failure_detail()
        assert "writer_thread_error" in detail and "boom" in detail
        detail_without = writer_without_error._failure_detail()
        assert "writer_thread_error" not in detail_without
    finally:
        # Nested try/finally, not two sequential statements: if the pytest.raises expectation
        # itself fails, writer_without_error must still get released rather than leaking its
        # process/thread.
        try:
            # The injected error above surfaces here -- this writer's release() is expected to raise.
            with pytest.raises(VideoEncodingFailed):
                writer_with_error.release()
        finally:
            writer_without_error.release()


# --- FFmpegPipeBackend: writer-thread failure and stdin-close failure ----------------------------


def _force_cleanup(writer: FFmpegPipeBackend) -> None:
    """Best-effort teardown for a writer a test deliberately drove into a broken/stuck state --
    safe to call from a `finally` even after an earlier wait or assertion already failed. Wakes a
    writer thread blocked on EITHER self._queue.get() (nothing was ever written -- e.g. a
    readiness wait failed before any frame/sentinel was queued, so the child dying doesn't help)
    OR self._proc.stdin.write() (kills the child), then joins it -- and asserts it actually
    exited, rather than silently reporting success for a leaked non-daemon thread."""
    q = writer._queue
    if q is not None:
        try:
            q.put_nowait(None)
        except queue.Full:
            pass  # something is already queued/draining; no sentinel needed
    proc = writer._proc
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            pass
    th = writer._thread
    if th is not None:
        th.join(timeout=5.0)
        assert not th.is_alive(), "test cleanup failed to stop the writer thread"


def _write_until_error_recorded(writer: FFmpegPipeBackend) -> None:
    """Writes frames one at a time until the writer thread records a broken-pipe error, catching
    (and stopping on) the VideoEncodingFailed that write() itself raises once that happens -- a
    tight unconditional loop risks raising outside an intended `pytest.raises` block."""
    try:
        for _ in range(50):
            if writer._writer_error is not None:
                break
            writer.write(_frame(64, 48))
    except VideoEncodingFailed:
        pass
    _wait_until(lambda: writer._writer_error is not None, what="writer thread to record its error")


def test_writer_thread_records_the_error_when_the_pipe_breaks(tmp_path: Path, fake_ffmpeg):
    fake_ffmpeg.set_mode("exit_immediately")
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        64,
        48,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    try:
        fake_ffmpeg.wait_until_ready()
        _wait_until(lambda: writer._proc.poll() is not None, what="fake process to exit immediately")
        _write_until_error_recorded(writer)
        with pytest.raises(VideoEncodingFailed):
            writer.release()
    finally:
        _force_cleanup(writer)


def test_write_after_a_writer_thread_failure_raises_with_the_recorded_cause(tmp_path: Path, fake_ffmpeg):
    fake_ffmpeg.set_mode("exit_immediately")
    writer = FFmpegPipeBackend(
        str(tmp_path / "out.mp4"),
        64,
        48,
        10.0,
        input_pixel_format="bgr24",
        encoder_args=["-c:v", "libx264"],
        writer_name="test",
    )
    try:
        fake_ffmpeg.wait_until_ready()
        _wait_until(lambda: writer._proc.poll() is not None, what="fake process to exit immediately")
        _write_until_error_recorded(writer)
        with pytest.raises(VideoEncodingFailed, match="already failed"):
            writer.write(_frame(64, 48))
    finally:
        with pytest.raises(VideoEncodingFailed):
            writer.release()


def test_release_logs_but_does_not_raise_when_closing_stdin_fails(
    tmp_path: Path, fake_ffmpeg, caplog: pytest.LogCaptureFixture
):
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    real_close = writer._proc.stdin.close

    def _raise_close():
        real_close()
        raise OSError("close failed")

    writer._proc.stdin.close = _raise_close
    writer.release()  # must not raise despite the close() failure -- it's a logged warning only
    assert "closing ffmpeg stdin failed" in caplog.text


# --- FFmpegPipeBackend: write() backpressure (queue.Full) and release()'s sentinel-enqueue failure ---


def test_write_raises_when_the_backlog_queue_is_full(tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vwb, "_QUEUE_PUT_TIMEOUT_S", 0.2)
    fake_ffmpeg.set_mode("never_read_stdin")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=1920, height=1440, fps=10.0, preset="ultrafast")
    try:
        fake_ffmpeg.wait_until_ready()
        big_frame = _frame(1920, 1440)
        with pytest.raises(VideoEncodingFailed, match="queue full"):
            for _ in range(80):  # comfortably exceeds the 60-slot queue plus the kernel pipe buffer
                writer.write(big_frame)
    finally:
        _force_cleanup(writer)


def test_release_force_stops_when_the_shutdown_sentinel_cannot_be_enqueued(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vwb, "_QUEUE_PUT_TIMEOUT_S", 0.2)
    fake_ffmpeg.set_mode("never_read_stdin")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=1920, height=1440, fps=10.0, preset="ultrafast")
    try:
        fake_ffmpeg.wait_until_ready()
        big_frame = _frame(1920, 1440)
        for _ in range(65):  # fill the queue (maxsize=60) so release()'s own sentinel put() fails
            try:
                writer.write(big_frame)
            except VideoEncodingFailed:
                break
        with pytest.raises(VideoEncodingFailed):
            writer.release()
    finally:
        _force_cleanup(writer)


# --- FFmpegPipeBackend._force_stop_ffmpeg: terminate, then escalate to kill ----------------------


def test_force_stop_escalates_from_terminate_to_kill_when_the_process_ignores_sigterm(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vwb, "_FORCE_STOP_TERMINATE_TIMEOUT_S", 0.2)
    fake_ffmpeg.set_mode("ignore_sigterm")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    try:
        # wait_until_ready() is published only AFTER the SIGTERM handler is installed (see the
        # fake script's ordering comment), so terminate() below is guaranteed to hit the real
        # ignore-SIGTERM state, not race a default handler that hasn't been installed yet.
        fake_ffmpeg.wait_until_ready()
        writer._force_stop_ffmpeg()  # white-box: exercises terminate -> timeout -> kill directly
        # Assert the actual escalation outcome, not just "died somehow" -- poll()!=None alone
        # would also pass if SIGTERM had (wrongly) killed it before SIG_IGN took effect.
        assert writer._proc.returncode == -signal.SIGKILL
    finally:
        with pytest.raises(VideoEncodingFailed):
            try:
                writer.write(_frame(64, 48))
            finally:
                writer.release()


def test_force_stop_is_a_no_op_once_the_process_reference_is_already_gone(tmp_path: Path, fake_ffmpeg):
    # No readiness wait needed: writer._proc is set synchronously by __init__ regardless of
    # whether the child has reached its own internal "ready" state.
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    real_proc = writer._proc  # saved so a real, normal shutdown still happens in `finally`
    try:
        writer._proc = None  # white-box: the state release() leaves things in after it completes
        writer._force_stop_ffmpeg()  # must return immediately, not raise
    finally:
        writer._proc = real_proc
        writer.write(_frame(64, 48))
        writer.release()


def test_force_stop_is_a_no_op_once_the_process_has_already_exited(tmp_path: Path, fake_ffmpeg):
    fake_ffmpeg.set_mode("exit_immediately")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    try:
        fake_ffmpeg.wait_until_ready()
        _wait_until(lambda: writer._proc.poll() is not None, what="fake process to exit immediately")
        writer._force_stop_ffmpeg()  # poll() already shows it exited -- must skip terminate/kill
    finally:
        _force_cleanup(writer)


def test_force_stop_logs_but_does_not_raise_when_terminate_itself_fails(
    tmp_path: Path, fake_ffmpeg, caplog: pytest.LogCaptureFixture
):
    # No readiness wait needed here either: terminate() is injected and never actually reaches
    # the child, so the child's own internal startup timing is irrelevant.
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    real_terminate = writer._proc.terminate

    def _raise(*a, **k):
        raise OSError("terminate failed")

    try:
        writer._proc.terminate = _raise
        writer._force_stop_ffmpeg()  # must not raise despite the injected failure
        assert "failed to stop ffmpeg" in caplog.text
    finally:
        # Restore the real terminate -- a normal write+release never calls it in "drain" mode (the
        # fake process just reads stdin to EOF and exits), so this is purely a safety net, not
        # something the rest of this test's cleanup path relies on.
        writer._proc.terminate = real_terminate
        writer.write(_frame(64, 48))
        writer.release()


# --- FFmpegPipeBackend.release(): the writer thread hangs even after a force stop ----------------


def test_release_raises_when_the_writer_thread_is_still_stuck_after_a_force_stop(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    # A real deadlock: _force_stop_ffmpeg is neutered to a no-op (simulating it failing to free a
    # stuck writer thread), and the fake process never reads stdin, so a frame big enough to
    # exceed the kernel pipe buffer leaves the writer thread permanently blocked inside its one
    # os.write() call. release()'s own two join() timeouts are shrunk so the test stays fast.
    monkeypatch.setattr(vwb, "_WRITER_JOIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(vwb, "_WRITER_JOIN_AFTER_FORCE_STOP_TIMEOUT_S", 0.05)
    fake_ffmpeg.set_mode("never_read_stdin")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=1920, height=1440, fps=10.0, preset="ultrafast")
    try:
        fake_ffmpeg.wait_until_ready()
        monkeypatch.setattr(writer, "_force_stop_ffmpeg", lambda: None)
        writer.write(_frame(1920, 1440))  # one frame is already bigger than any kernel pipe buffer
        with pytest.raises(VideoEncodingFailed, match="hung"):
            writer.release()
    finally:
        # The monkeypatched no-op above is why release() didn't actually free anything -- do it
        # for real here so the test doesn't leak a live process or a stuck non-daemon thread.
        _force_cleanup(writer)


def test_release_recovers_when_a_real_force_stop_frees_the_stuck_writer_thread_in_time(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    # Same stuck-writer-thread setup as above, but force_stop is REAL this time (not neutered): it
    # kills the process, which frees the thread's blocked write() before the second (post-force-
    # stop) join timeout elapses. release() must then proceed past the "still stuck" check rather
    # than raising from it -- it still raises overall, from the already-covered nonzero-returncode
    # path below (a kill() doesn't leave a clean exit), but that's a different branch; what this
    # test actually exercises is the is_alive() check after the recovery join coming back False.
    monkeypatch.setattr(vwb, "_WRITER_JOIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(vwb, "_WRITER_JOIN_AFTER_FORCE_STOP_TIMEOUT_S", 2.0)
    fake_ffmpeg.set_mode("never_read_stdin")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=1920, height=1440, fps=10.0, preset="ultrafast")
    try:
        fake_ffmpeg.wait_until_ready()
        writer.write(_frame(1920, 1440))  # one frame is already bigger than any kernel pipe buffer
        with pytest.raises(VideoEncodingFailed, match="ffmpeg failed"):
            writer.release()
        assert not writer._thread.is_alive()  # recovered, not still stuck
    finally:
        _force_cleanup(writer)


# --- FFmpegPipeBackend.release(): proc.wait() itself times out after stdin closes ----------------


def test_release_raises_when_ffmpeg_does_not_exit_after_stdin_closes(
    tmp_path: Path, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vwb, "_PROC_WAIT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(vwb, "_FORCE_STOP_TERMINATE_TIMEOUT_S", 0.2)  # real force-stop still runs
    # "drain_then_hang": the writer thread finishes and joins normally once stdin reaches EOF, but
    # the process itself then ignores SIGTERM and blocks forever -- exactly the scenario where the
    # writer thread is done while proc.wait() itself times out.
    fake_ffmpeg.set_mode("drain_then_hang")
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    try:
        writer.write(_frame(64, 48))
        with pytest.raises(VideoEncodingFailed, match="did not exit"):
            writer.release()
        assert writer._proc.returncode == -signal.SIGKILL  # real force_stop (not neutered) killed it
    finally:
        _force_cleanup(writer)


# --- FFmpegPipeBackend.release(): the writer_error check at the very end -------------------------


def test_release_raises_the_writer_error_even_when_the_process_itself_exits_cleanly(tmp_path: Path, fake_ffmpeg):
    # A writer_error can be recorded (e.g. a transient write failure) even though the fake process
    # still exits 0 afterward -- release() must still surface it rather than reporting success.
    writer = FFmpegH264Backend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0, preset="ultrafast")
    writer.write(_frame(64, 48))
    writer._writer_error = RuntimeError("synthetic transient failure")
    with pytest.raises(VideoEncodingFailed, match="before finalize"):
        writer.release()


# --- NVENC: argv/lifecycle only, via the fake binary -- no GPU needed, matches the reviewer's ----
# --- "patch Popen"/boundary-test suggestion, just via a real subprocess instead of a mock --------


def test_nvenc_backend_builds_the_expected_argv_and_completes_via_the_factory(tmp_path: Path, fake_ffmpeg):
    writer = create_video_writer(
        "ffmpeg_h264_nvenc",
        str(tmp_path / "out.mp4"),
        64,
        48,
        10.0,
        source_config={"video_encoder_options": {"crf": 19, "preset": "medium"}},
    )
    assert isinstance(writer, FFmpegH264NvencBackend)
    writer.write(_frame(64, 48))
    writer.release()
    argv = fake_ffmpeg.argv()
    assert "h264_nvenc" in argv and "-cq" in argv and "19" in argv and "p4" in argv


def test_nvenc_backend_rejects_wrong_dtype_or_shape(tmp_path: Path, fake_ffmpeg):
    writer = FFmpegH264NvencBackend(str(tmp_path / "out.mp4"), width=64, height=48, fps=10.0)
    try:
        with pytest.raises(ValueError, match="NVENC H.264 requires uint8 BGR frames"):
            writer.write(_frame(32, 32))
    finally:
        writer.release()
