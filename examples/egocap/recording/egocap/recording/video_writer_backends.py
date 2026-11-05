# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: BLE001
"""RGB video writer backends and their factory."""

import logging
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import cv2
import numpy as np

log = logging.getLogger(__name__)

VideoEncoderKind = Literal["opencv_mp4v", "ffmpeg_h264", "ffmpeg_h264_nvenc"]

_H264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)
# yuv420p requires even width/height; the 4:4:4 formats support odd dimensions
# and preserve chroma (yuvj444p is additionally full-range).
_H264_PIX_FMTS = ("yuv420p", "yuv444p", "yuvj444p")
_H264_DEFAULT_ENCODER_OPTIONS: dict[str, Any] = {
    "crf": 12,
    "preset": "medium",
    "pix_fmt": "yuv420p",
}

# --- NVENC (GPU) ------------------------------------------------------------
# Hardware H.264 on the NVIDIA encoder block. Offloads encoding off the CPU
# entirely, which is what keeps a 60 Hz control loop scheduling on time while
# several 1920x1440 streams record: software libx264 at crf 12/medium measures
# ~0.45x realtime for two such streams and saturates every core, which stalls
# frame ingestion and eventually blocks on the bounded frame queue below.
#
# The option KEYS are deliberately identical to the libx264 ones (crf, preset,
# pix_fmt) so a config switches encoders by changing only `video_encoder`.
# `crf` maps to NVENC's -cq (same 0..51 scale, same "lower = better" sense) and
# x264 preset NAMES are accepted and translated to p1..p7.
_NVENC_PRESETS = ("p1", "p2", "p3", "p4", "p5", "p6", "p7")
_X264_TO_NVENC_PRESET = {
    "ultrafast": "p1",
    "superfast": "p1",
    "veryfast": "p2",
    "faster": "p2",
    "fast": "p3",
    "medium": "p4",
    "slow": "p5",
    "slower": "p6",
    "veryslow": "p7",
    "placebo": "p7",
}
# NVENC advertises yuv420p/nv12/p010le/yuv444p/... but NOT yuvj444p, so an RGB source
# configured for that full-range 4:4:4 format must stay on libx264 instead of NVENC.
_NVENC_PIX_FMTS = ("yuv420p", "yuv444p")
_NVENC_DEFAULT_ENCODER_OPTIONS: dict[str, Any] = {
    "crf": 12,
    "preset": "p4",  # NVENC's own default; equivalent to x264 "medium"
    "pix_fmt": "yuv420p",
}

# Cap captured ffmpeg stderr so a chatty process cannot exhaust memory.
_MAX_FFMPEG_STDERR_BYTES = 256 * 1024

# Lifecycle timeouts (seconds), module-level so tests can shrink them instead of taking the real
# wall-clock hit -- the values themselves are the real, intended production behavior.
_QUEUE_PUT_TIMEOUT_S = 5.0
_FORCE_STOP_TERMINATE_TIMEOUT_S = 5.0
_WRITER_JOIN_TIMEOUT_S = 30.0
_WRITER_JOIN_AFTER_FORCE_STOP_TIMEOUT_S = 10.0
_PROC_WAIT_TIMEOUT_S = 30.0


class VideoEncodingFailed(RuntimeError):
    """Raised when encoded video cannot be completed or ffmpeg exits with an error."""


def validate_ffmpeg_threads(value: int | None) -> int | None:
    """None retains FFmpeg defaults; explicit limits must be positive integers."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("recording.ffmpeg_threads must be a positive integer or null")
    return value


def resolve_video_encoder_options(
    encoder: VideoEncoderKind | str,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate ``video_encoder_options`` and merge them over built-in defaults.

    Returns the effective ``{crf, preset, pix_fmt}`` dict for ``ffmpeg_h264``
    and ``ffmpeg_h264_nvenc``, and an empty dict for encoders without tunable
    options. Raises ValueError on unknown keys or out-of-range values so config
    typos fail at startup instead of silently recording with defaults.

    Both ffmpeg encoders take the same keys. For NVENC, x264 preset names are
    accepted and translated to p1..p7, so flipping ``video_encoder`` alone is
    enough to move a source onto the GPU.
    """
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError(f"video_encoder_options must be a mapping, got {type(options).__name__}")
    if encoder == "ffmpeg_h264":
        defaults, valid_presets, valid_pix_fmts = (
            _H264_DEFAULT_ENCODER_OPTIONS,
            _H264_PRESETS,
            _H264_PIX_FMTS,
        )
    elif encoder == "ffmpeg_h264_nvenc":
        defaults, valid_presets, valid_pix_fmts = (
            _NVENC_DEFAULT_ENCODER_OPTIONS,
            _NVENC_PRESETS,
            _NVENC_PIX_FMTS,
        )
    else:
        if options:
            raise ValueError(
                "video_encoder_options is only supported with "
                f"video_encoder='ffmpeg_h264' or 'ffmpeg_h264_nvenc'; got {dict(options)!r} "
                f"with encoder {encoder!r}"
            )
        return {}
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown video_encoder_options keys {sorted(unknown)}; allowed keys are {sorted(defaults)}"
        )
    resolved = {**defaults, **options}
    crf = resolved["crf"]
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError(f"video_encoder_options crf must be an int in [0, 51], got {crf!r}")
    if encoder == "ffmpeg_h264_nvenc":
        # Accept x264 preset names so a config can switch encoders in one word.
        resolved["preset"] = _X264_TO_NVENC_PRESET.get(resolved["preset"], resolved["preset"])
    if resolved["preset"] not in valid_presets:
        allowed = (
            f"{valid_presets} (or an x264 preset name {tuple(_X264_TO_NVENC_PRESET)})"
            if encoder == "ffmpeg_h264_nvenc"
            else str(valid_presets)
        )
        raise ValueError(f"video_encoder_options preset must be one of {allowed}, got {resolved['preset']!r}")
    if resolved["pix_fmt"] not in valid_pix_fmts:
        raise ValueError(
            f"video_encoder_options pix_fmt must be one of {valid_pix_fmts}, got {resolved['pix_fmt']!r}"
        )
    return resolved


class VideoWriterBackend(ABC):
    """Interface for writing video frames to a file."""

    @property
    @abstractmethod
    def output_path(self) -> str:
        """Path of the video file produced by this writer."""

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        """Queue one frame for encoding."""

    @abstractmethod
    def release(self) -> None:
        """Finish and close the file. Idempotent."""


class OpenCVBackend(VideoWriterBackend):
    """Write video using OpenCV VideoWriter (MPEG-4 Part 2 / mp4v)."""

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._expected_shape = (height, width, 3)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer: cv2.VideoWriter | None = None
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {path}")

    @property
    def output_path(self) -> str:
        return self._path

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != self._expected_shape:
            raise ValueError(
                "RGB video requires uint8 BGR frames with shape "
                f"{self._expected_shape}; got dtype={frame.dtype}, "
                f"shape={frame.shape}"
            )
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class FFmpegPipeBackend(VideoWriterBackend):
    """Bounded asynchronous raw-frame pipe into an FFmpeg encoder.

    Subclasses own pixel validation and supply only their codec arguments. This
    class owns subprocess lifecycle, backpressure, stderr capture, and failure
    propagation, shared equally by the RGB CPU (:class:`FFmpegH264Backend`) and
    GPU (:class:`FFmpegH264NvencBackend`) subclasses below.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        input_pixel_format: str,
        encoder_args: list[str],
        writer_name: str,
        thread_limit: int | None = None,
    ) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._input_pixel_format = input_pixel_format
        self._encoder_args = encoder_args
        self._writer_name = writer_name
        self._thread_limit = validate_ffmpeg_threads(thread_limit)
        self._queue: queue.Queue[np.ndarray | None] | None = queue.Queue(maxsize=60)
        self._proc: subprocess.Popen[Any] | None = None
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buf = bytearray()
        self._stderr_lock = threading.Lock()
        self._writer_error: BaseException | None = None
        self._released = False
        self._start()

    @property
    def output_path(self) -> str:
        return self._path

    def _start(self) -> None:
        # Bound both filter pools and codec pools. Input/output -threads are
        # separate file options; output placement also covers lossless tactile
        # libx264, not just the NVENC camera encoders. None keeps the old argv.
        thread_args = [] if self._thread_limit is None else ["-threads", str(self._thread_limit)]
        filter_args = (
            []
            if self._thread_limit is None
            else [
                "-filter_threads",
                str(self._thread_limit),
                "-filter_complex_threads",
                str(self._thread_limit),
            ]
        )
        cmd = [
            "ffmpeg",
            "-y",
            *filter_args,
            *thread_args,
            "-f",
            "rawvideo",
            "-pix_fmt",
            self._input_pixel_format,
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            *self._encoder_args,
            *thread_args,
            self._path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        stderr_r = self._proc.stderr
        if stderr_r is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr_r,),
                daemon=True,
                name=f"{self._writer_name}-stderr",
            )
            self._stderr_thread.start()
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=False,
            name=f"{self._writer_name}-writer",
        )
        self._thread.start()

    def _drain_stderr(self, stderr_r: Any) -> None:
        try:
            while True:
                chunk = stderr_r.read(8192)
                if not chunk:
                    break
                with self._stderr_lock:
                    room = _MAX_FFMPEG_STDERR_BYTES - len(self._stderr_buf)
                    if room > 0:
                        self._stderr_buf.extend(chunk[:room])
        except Exception:
            log.exception("ffmpeg stderr drain failed (path=%s)", self._path)

    def _failure_detail(self) -> str:
        with self._stderr_lock:
            tail = self._stderr_buf.decode("utf-8", errors="replace")
        lines = [
            f"writer={self._writer_name} path={self._path}",
            f"width={self._width} height={self._height} fps={self._fps}",
            f"input_pixel_format={self._input_pixel_format}",
            f"encoder_args={self._encoder_args!r}",
        ]
        if self._writer_error is not None:
            lines.append(f"writer_thread_error={self._writer_error!r}")
        if tail.strip():
            lines.append("--- ffmpeg stderr (tail, may be truncated) ---")
            lines.append(tail[-8000:] if len(tail) > 8000 else tail)
        return "\n".join(lines)

    def _writer_loop(self) -> None:
        # self._proc/stdin/self._queue are always set by _start() before this thread is started,
        # and are only ever nulled at the very end of release() -- after this thread has already
        # exited and been joined -- so there is no window where any of them is None here.
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                # Write the frame's own buffer rather than frame.tobytes():
                # tobytes() allocates AND copies the whole frame every time --
                # 8.3 MB per 1920x1440 BGR head-cam frame, ~500 MB/s across two
                # head cams at 30 fps -- with the GIL held, which stalls the
                # 60 Hz control loop. A memoryview writes straight from the
                # existing buffer (no allocation, no copy) and the underlying
                # os.write releases the GIL.
                #
                # cast("B") flattens to bytes and requires a C-contiguous
                # source; every concrete backend already guarantees that via
                # np.ascontiguousarray in its own write(). The guard keeps this
                # correct for any future backend that forgets.
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                self._proc.stdin.write(memoryview(frame).cast("B"))
        except Exception as e:
            self._writer_error = e
            log.exception(
                "ffmpeg writer thread failed (writer=%s, path=%s). Video file may be incomplete or corrupt.",
                self._writer_name,
                self._path,
            )
        finally:
            try:
                self._proc.stdin.close()
            except Exception as e:
                log.warning(
                    "closing ffmpeg stdin failed (path=%s): %s",
                    self._path,
                    e,
                    exc_info=True,
                )

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise VideoEncodingFailed(f"write() after release() is invalid (path={self._path})")
        if self._writer_error is not None:
            err = VideoEncodingFailed(
                "Video encoding already failed; cannot write more frames.\n" + self._failure_detail()
            )
            raise err from self._writer_error
        try:
            self._queue.put(frame, block=True, timeout=_QUEUE_PUT_TIMEOUT_S)
        except queue.Full:
            msg = (
                f"ffmpeg encode backlog: queue full after {_QUEUE_PUT_TIMEOUT_S:g}s (path={self._path}). "
                "Encoder cannot keep up; recording cannot continue safely."
            )
            log.error(msg)  # noqa: TRY400
            exc = VideoEncodingFailed(msg)
            self._writer_error = exc
            raise exc from None

    def _force_stop_ffmpeg(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                log.error("terminating stuck ffmpeg process (path=%s)", self._path)
                proc.terminate()
                try:
                    proc.wait(timeout=_FORCE_STOP_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    log.error("killing ffmpeg process (path=%s)", self._path)  # noqa: TRY400
                    proc.kill()
                    proc.wait(timeout=_FORCE_STOP_TERMINATE_TIMEOUT_S)
        except Exception:
            log.exception("failed to stop ffmpeg (path=%s)", self._path)

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        try:
            self._queue.put(None, block=True, timeout=_QUEUE_PUT_TIMEOUT_S)
        except queue.Full:
            log.error(  # noqa: TRY400
                "cannot enqueue ffmpeg shutdown sentinel — queue full (path=%s)",
                self._path,
            )
            self._force_stop_ffmpeg()

        th = self._thread
        th.join(timeout=_WRITER_JOIN_TIMEOUT_S)
        if th.is_alive():
            log.error(
                "ffmpeg writer thread did not finish in %gs (path=%s)",
                _WRITER_JOIN_TIMEOUT_S,
                self._path,
            )
            self._force_stop_ffmpeg()
            th.join(timeout=_WRITER_JOIN_AFTER_FORCE_STOP_TIMEOUT_S)
            if th.is_alive():
                detail = self._failure_detail()
                log.error("writer thread still alive after force stop:\n%s", detail)
                raise VideoEncodingFailed("ffmpeg writer thread hung; video output is unreliable.\n" + detail)

        proc = self._proc
        try:
            proc.wait(timeout=_PROC_WAIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.error(  # noqa: TRY400
                "ffmpeg did not exit after stdin close (path=%s)", self._path
            )
            self._force_stop_ffmpeg()
            detail = self._failure_detail()
            raise VideoEncodingFailed("ffmpeg did not exit in time; output may be corrupt.\n" + detail) from None
        rc = proc.returncode
        if rc != 0:
            detail = self._failure_detail()
            log.error(
                "ffmpeg exited with code %s (path=%s)\n%s",
                rc,
                self._path,
                detail,
            )
            err = VideoEncodingFailed(f"ffmpeg failed (returncode={rc}) for {self._path}.\n{detail}")
            raise err from self._writer_error

        if self._writer_error is not None:
            detail = self._failure_detail()
            log.error(
                "video encoding failed before finalize (path=%s)\n%s",
                self._path,
                detail,
            )
            raise VideoEncodingFailed(
                f"Video encoding failed before finalize: {self._path}\n{detail}"
            ) from self._writer_error

        self._thread = None
        self._proc = None
        self._queue = None
        self._stderr_thread = None


class FFmpegH264Backend(FFmpegPipeBackend):
    """Write uint8 BGR RGB frames using H.264 (defaults: CRF 12, medium, yuv420p)."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 12,
        preset: str = "medium",
        pix_fmt: str = "yuv420p",
        thread_limit: int | None = None,
    ) -> None:
        options = resolve_video_encoder_options("ffmpeg_h264", {"crf": crf, "preset": preset, "pix_fmt": pix_fmt})
        self.crf: int = options["crf"]
        self.preset: str = options["preset"]
        self.pix_fmt: str = options["pix_fmt"]
        self._expected_shape = (height, width, 3)
        super().__init__(
            path,
            width,
            height,
            fps,
            input_pixel_format="bgr24",
            encoder_args=[
                "-c:v",
                "libx264",
                "-pix_fmt",
                self.pix_fmt,
                "-crf",
                str(self.crf),
                "-preset",
                self.preset,
            ],
            writer_name="ffmpeg-h264-rgb",
            thread_limit=thread_limit,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != self._expected_shape:
            raise ValueError(
                "RGB H.264 requires uint8 BGR frames with shape "
                f"{self._expected_shape}; got dtype={frame.dtype}, "
                f"shape={frame.shape}"
            )
        super().write(np.ascontiguousarray(frame))


class FFmpegH264NvencBackend(FFmpegPipeBackend):
    """Write uint8 BGR frames using NVIDIA NVENC H.264 (GPU).

    Same frame contract as :class:`FFmpegH264Backend`; the encode runs on the
    GPU's dedicated encoder block instead of the CPU. ``crf`` is passed as
    ``-cq`` (NVENC's constant-quality control, same 0..51 lower-is-better
    scale).

    Do NOT add ``-rc constqp``/``-qp`` here, however natural it looks for
    CRF-style constant quality: on ffmpeg 4.4.2 (Ubuntu 22.04) against an
    RTX 5090 it SEGFAULTS at 1920x1440 (exit 139) while ``-cq`` alone encodes
    cleanly and yields a byte-identical size to ``-rc vbr -cq``. Verified on
    this rig; revisit only with a newer ffmpeg.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 12,
        preset: str = "p4",
        pix_fmt: str = "yuv420p",
        thread_limit: int | None = None,
    ) -> None:
        options = resolve_video_encoder_options(
            "ffmpeg_h264_nvenc", {"crf": crf, "preset": preset, "pix_fmt": pix_fmt}
        )
        self.crf: int = options["crf"]
        self.preset: str = options["preset"]
        self.pix_fmt: str = options["pix_fmt"]
        self._expected_shape = (height, width, 3)
        super().__init__(
            path,
            width,
            height,
            fps,
            input_pixel_format="bgr24",
            encoder_args=[
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                self.pix_fmt,
                # -cq alone is SILENTLY IGNORED: NVENC only honours it under a
                # constant-quality rate control, and -b:v 0 is required to stop
                # a default target bitrate from overriding it. Without both, every
                # crf value produced byte-identical 88 Mbps output (measured on
                # real 1920x1440 head-cam footage) -- the quality knob did nothing
                # and episodes were ~2.4x larger than the libx264 crf 12 they
                # replaced. With them, cq 12/19/23 -> 50/21/12 Mbps as expected.
                "-rc",
                "vbr",
                "-b:v",
                "0",
                "-cq",
                str(self.crf),
                "-preset",
                self.preset,
            ],
            writer_name="ffmpeg-h264-nvenc-rgb",
            thread_limit=thread_limit,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != self._expected_shape:
            raise ValueError(
                "NVENC H.264 requires uint8 BGR frames with shape "
                f"{self._expected_shape}; got dtype={frame.dtype}, "
                f"shape={frame.shape}"
            )
        super().write(np.ascontiguousarray(frame))


def create_video_writer(
    encoder: VideoEncoderKind,
    path: str,
    width: int,
    height: int,
    fps: float,
    *,
    source_config: Mapping[str, Any] | None = None,
) -> VideoWriterBackend:
    """Create a video writer backend for the given encoder and parameters.

    Args:
        encoder: "opencv_mp4v", "ffmpeg_h264" (CPU/libx264), or
            "ffmpeg_h264_nvenc" (GPU/NVENC).
        path: Output .mp4 file path.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        source_config: Effective per-source configuration. Its optional
            "video_encoder_options" mapping ({crf, preset, pix_fmt}) tunes
            either ffmpeg encoder; it is rejected for other encoders.

    Returns:
        A backend instance with write(frame) and release().
    """
    options = resolve_video_encoder_options(encoder, (source_config or {}).get("video_encoder_options"))
    thread_limit = validate_ffmpeg_threads((source_config or {}).get("ffmpeg_threads"))
    if encoder == "opencv_mp4v":
        return OpenCVBackend(path, width, height, fps)
    if encoder == "ffmpeg_h264":
        return FFmpegH264Backend(
            path,
            width,
            height,
            fps,
            crf=options["crf"],
            preset=options["preset"],
            pix_fmt=options["pix_fmt"],
            thread_limit=thread_limit,
        )
    if encoder == "ffmpeg_h264_nvenc":
        return FFmpegH264NvencBackend(
            path,
            width,
            height,
            fps,
            crf=options["crf"],
            preset=options["preset"],
            pix_fmt=options["pix_fmt"],
            thread_limit=thread_limit,
        )
    raise ValueError(f"Unknown video encoder: {encoder}")
