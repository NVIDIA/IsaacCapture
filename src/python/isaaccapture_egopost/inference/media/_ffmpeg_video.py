# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Restricted FFmpeg backend for metadata and decode-order RGB frame streaming."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

H264_GPU_REQUIRED_MESSAGE = (
    "A GPU is required for H.264-encoded inputs because H.264 is decoded with the "
    "approved hardware decoder. Deploy on a GPU or provide a VP9- or MPEG-4 Part 2-encoded "
    "input instead."
)

_CODEC_NAME_ALIASES = {
    "h264_cuvid": "h264",
    "vp9_cuvid": "vp9",
}

# Maximum time to wait for FFmpeg to exit after its decoded-frame stream reaches EOF.
DECODE_TIMEOUT_SECS = 60


class UnsupportedVideoCodecError(ValueError):
    """Raised when a video does not use an approved input codec."""


class VideoDecoderUnavailableError(RuntimeError):
    """Raised when no approved decoder can decode a supported stream."""


class VideoDecodeError(RuntimeError):
    """Raised when probing or streaming a supported video fails."""


@dataclass(frozen=True)
class VideoStreamInfo:
    """Metadata needed to select and stream a video decoder."""

    codec_name: str
    width: int
    height: int
    fps: float | None
    frame_count: int | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoDecodePlan:
    """Validated decoder choice for a video stream."""

    stream: VideoStreamInfo
    decoder_name: str


def probe_video_stream(path: Path | str) -> VideoStreamInfo:
    """Read codec and geometry metadata with the external ``ffprobe`` executable."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:"
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoDecodeError(f"Could not probe video stream {path}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise VideoDecodeError(f"ffprobe failed for {path}{suffix}")

    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise VideoDecodeError(f"Input has no video stream: {path}")
        stream: dict[str, Any] = streams[0]
        reported_codec_name = str(stream.get("codec_name") or "").lower()
        codec_name = _CODEC_NAME_ALIASES.get(reported_codec_name, reported_codec_name)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise VideoDecodeError(f"Invalid ffprobe response for {path}") from exc

    if width <= 0 or height <= 0:
        raise VideoDecodeError(f"Could not determine video dimensions: {path}")
    fps = _parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    frame_count = _parse_positive_int(stream.get("nb_frames"))
    format_payload = payload.get("format")
    format_duration = (
        format_payload.get("duration") if isinstance(format_payload, dict) else None
    )
    duration_seconds = _parse_positive_float(stream.get("duration"))
    if duration_seconds is None:
        duration_seconds = _parse_positive_float(format_duration)
    if duration_seconds is None and frame_count is not None and fps is not None:
        duration_seconds = frame_count / fps
    return VideoStreamInfo(
        codec_name=codec_name,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def prepare_video_decode(path: Path | str) -> VideoDecodePlan:
    """Validate an H.264, VP9, or MPEG-4 Part 2 input and select its decoder."""
    stream = probe_video_stream(path)
    if stream.codec_name == "h264":
        if not _decoder_works(path, stream, "h264_cuvid"):
            raise VideoDecoderUnavailableError(H264_GPU_REQUIRED_MESSAGE)
        return VideoDecodePlan(stream=stream, decoder_name="h264_cuvid")

    if stream.codec_name == "vp9":
        for decoder_name in ("vp9_cuvid", "vp9"):
            if _decoder_works(path, stream, decoder_name):
                return VideoDecodePlan(stream=stream, decoder_name=decoder_name)
        raise VideoDecoderUnavailableError(
            f"VP9 input could not be decoded with an approved hardware or software decoder: {path}"
        )

    if stream.codec_name == "mpeg4":
        if _decoder_works(path, stream, "mpeg4"):
            return VideoDecodePlan(stream=stream, decoder_name="mpeg4")
        raise VideoDecoderUnavailableError(
            f"MPEG-4 Part 2 input could not be decoded with the approved software decoder: {path}"
        )

    rendered = stream.codec_name or "<unknown>"
    raise UnsupportedVideoCodecError(
        f"Unsupported video codec {rendered!r}; supported codecs: h264, vp9, mpeg4"
    )


def iter_decoded_rgb24_frames(
    path: Path | str,
    plan: VideoDecodePlan,
) -> Iterable[bytes]:
    """Yield exact RGB24 frame payloads from the selected FFmpeg decoder."""
    command = _decode_command(path, plan.decoder_name, frame_limit=None)
    frame_bytes = plan.stream.width * plan.stream.height * 3
    stderr_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
    except OSError as exc:
        stderr_file.close()
        raise VideoDecodeError(
            f"Could not start FFmpeg decoder {plan.decoder_name!r}: {exc}"
        ) from exc
    if process.stdout is None:
        process.kill()
        process.wait()
        stderr_file.close()
        raise VideoDecodeError("FFmpeg decoder did not expose a stdout frame stream.")

    try:
        while True:
            payload = _read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise VideoDecodeError(
                    f"FFmpeg returned a partial RGB frame ({len(payload)}/{frame_bytes} bytes)."
                )
            yield payload

        try:
            return_code = process.wait(timeout=DECODE_TIMEOUT_SECS)
        except subprocess.TimeoutExpired as exc:
            _terminate_and_reap(process)
            raise VideoDecodeError(
                _decode_timeout_message(path, plan.decoder_name, stderr_file)
            ) from exc
        if return_code != 0:
            raise VideoDecodeError(
                _decode_failure_message(path, plan.decoder_name, stderr_file)
            )
    finally:
        process.stdout.close()
        if process.poll() is None:
            _terminate_and_reap(process)
        stderr_file.close()


def _parse_frame_rate(value: object) -> float | None:
    if value in (None, "", "0/0"):
        return None
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _parse_positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_positive_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _decoder_works(
    path: Path | str, stream: VideoStreamInfo, decoder_name: str
) -> bool:
    try:
        result = subprocess.run(  # noqa: S603
            _decode_command(path, decoder_name, frame_limit=1),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    expected_bytes = stream.width * stream.height * 3
    return result.returncode == 0 and len(result.stdout) == expected_bytes


def _decode_command(
    path: Path | str,
    decoder_name: str,
    *,
    frame_limit: int | None,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-c:v",
        decoder_name,
        "-noautorotate",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        # One output frame per decoded frame. rawvideo output defaults to constant
        # frame rate, which duplicates or drops frames of variable-frame-rate
        # inputs and breaks alignment with saved bboxes and source frame indices
        # (it also makes the count disagree with get_video_meta's num_frames).
        "-fps_mode",
        "passthrough",
    ]
    if frame_limit is not None:
        command.extend(["-frames:v", str(frame_limit)])
    command.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    return command


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_failure_message(
    path: Path | str, decoder_name: str, stderr_file: Any
) -> str:
    detail = _decode_stderr_detail(stderr_file)
    suffix = f": {detail}" if detail else ""
    return f"FFmpeg decoder {decoder_name!r} failed for {path}{suffix}"


def _decode_timeout_message(
    path: Path | str, decoder_name: str, stderr_file: Any
) -> str:
    detail = _decode_stderr_detail(stderr_file)
    suffix = f": {detail}" if detail else ""
    return f"FFmpeg decoder {decoder_name!r} timed out decoding {path}{suffix}"


def _decode_stderr_detail(stderr_file: Any) -> str:
    stderr_file.seek(0)
    return str(stderr_file.read().decode("utf-8", errors="replace").strip())


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait()
        except ProcessLookupError:
            pass


__all__ = [
    "DECODE_TIMEOUT_SECS",
    "H264_GPU_REQUIRED_MESSAGE",
    "UnsupportedVideoCodecError",
    "VideoDecodeError",
    "VideoDecodePlan",
    "VideoDecoderUnavailableError",
    "VideoStreamInfo",
    "iter_decoded_rgb24_frames",
    "prepare_video_decode",
    "probe_video_stream",
]


# --- video_io API ---------------------------------------------------------

FPS_FALLBACK = 30.0


def get_video_meta(video_path: str) -> dict:
    """Return fps, width, height, num_frames without decoding any frames."""
    stream = probe_video_stream(video_path)
    return {
        "fps": stream.fps or FPS_FALLBACK,
        "width": stream.width,
        "height": stream.height,
        "num_frames": stream.frame_count or 0,
    }


def read_video_frames_batched(
    video_path: str,
    batch_size: int,
    max_frames: int | None = None,
):
    """Generator that yields lists of (frame_idx, BGR_frame), batch_size at a time.

    Each yielded list has exactly batch_size entries, except possibly the last
    which may be smaller. Stops early when max_frames is reached.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_frames is not None and max_frames <= 0:
        return
    plan = prepare_video_decode(video_path)
    width, height = plan.stream.width, plan.stream.height
    batch = []
    # Close the subprocess even when a caller stops consuming batches early.
    with closing(iter_decoded_rgb24_frames(video_path, plan)) as frames:
        for idx, payload in enumerate(frames):
            frame_rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            batch.append((idx, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)))
            if len(batch) == batch_size:
                yield batch
                batch = []
            if max_frames is not None and idx + 1 >= max_frames:
                break
        if batch:
            yield batch
