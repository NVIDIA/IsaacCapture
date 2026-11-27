# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-demand, lossless diagnostic capture from a running local media process."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import socketserver
import struct
import subprocess
import threading
import time
import zlib

logger = logging.getLogger("isaaccapture.remote_assistance.capture")


def write_png(path, data, width, height):
    """Store packed RGBA without resizing, color adjustment, or lossy encoding."""
    if len(data) != width * height * 4:
        raise ValueError("Capture requires tightly packed RGBA pixels")

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    stride = width * 4
    rows = b"".join(b"\0" + data[i : i + stride] for i in range(0, len(data), stride))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=1))
        + chunk(b"IEND", b"")
    )


def ffmpeg(*args):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False


class CaptureEndpoint:
    """A bounded side channel; producers never wait for image or disk encoding."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.tracks = {}
        self.lock = threading.Lock()
        self.job = None
        self.closed = threading.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        endpoint = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(10)
                try:
                    request = json.loads(self.rfile.readline(16384))
                    result = endpoint.capture(request)
                    response = {"ok": True, **result}
                except Exception as exc:
                    logger.warning("Diagnostic capture failed: %s", exc)
                    response = {"ok": False, "error": str(exc)}
                self.wfile.write(json.dumps(response).encode() + b"\n")

        # Refuse existing endpoints, including stale sockets; never unlink a live peer.
        self.server = _Server(str(self.path), Handler)
        os.chmod(self.path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info("Diagnostic capture endpoint: %s", self.path)

    def register(
        self, track, *, stage, width, height, fps, encoding="rgba", on_active=None
    ):
        with self.lock:
            self.tracks[track] = dict(
                stage=stage,
                width=width,
                height=height,
                fps=float(fps),
                encoding=encoding,
                on_active=on_active,
            )

    def wanted(self, track):
        with self.lock:
            return (
                self.job is not None
                and self.job["track"] == track
                and self.job["collecting"]
            )

    def offer(self, track, data, metadata, *, key=True):
        with self.lock:
            job = self.job
            if (
                job is None
                or not job["collecting"]
                or job["track"] != track
                or time.monotonic() > job["deadline"]
            ):
                return
            if job["wait_key"]:
                if not key:
                    return
                job["wait_key"] = False
            if job["queue"].full():
                job["dropped"] += 1
                return
            # Lazy copy only while requested. The source may reuse its buffer next frame.
            try:
                job["queue"].put_nowait(
                    (bytes(data()), dict(metadata), time.monotonic_ns())
                )
            except Exception as exc:
                job["error"] = str(exc)

    def capture(self, request):
        if request.get("list"):
            with self.lock:
                return {
                    "tracks": {
                        k: {p: v for p, v in spec.items() if p != "on_active"}
                        for k, spec in self.tracks.items()
                    }
                }
        seconds = float(request.get("seconds", 0))
        if not math.isfinite(seconds) or not 0 <= seconds <= 3600:
            raise ValueError("seconds must be finite and in 0..3600 (0: one frame)")
        output = Path(request["output"]).expanduser().absolute()
        track = request.get("track", "cam")
        video = bool(request.get("video", False))
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("Capture endpoint is stopping")
            if self.job is not None:
                raise RuntimeError("A capture is already in progress")
            if track not in self.tracks:
                raise ValueError(
                    f"Unknown track {track!r}; available: {list(self.tracks)}"
                )
            spec = self.tracks[track].copy()
            if (spec["encoding"] != "rgba" or video) and not shutil.which("ffmpeg"):
                raise RuntimeError(
                    "Install ffmpeg for encoded-camera captures or video assembly"
                )
            started = time.monotonic_ns()
            job = dict(
                track=track,
                queue=queue.Queue(maxsize=4),
                dropped=0,
                collecting=True,
                error=None,
                wait_key=spec["encoding"] != "rgba",
                deadline=time.monotonic() + (seconds or 5),
            )
            self.job = job
        manifest = dict(
            track=track,
            **{k: v for k, v in spec.items() if k != "on_active"},
            requested_seconds=seconds,
            requested_unix_ns=time.time_ns(),
            requested_monotonic_ns=started,
            frames=[],
        )
        output_created = False
        try:
            output.mkdir(parents=True, exist_ok=False)
            output_created = True
            if spec["on_active"]:
                spec["on_active"](True)
            self._write(job, spec, manifest, output, seconds)
            with self.lock:
                job["collecting"] = False
            if spec["on_active"]:
                spec["on_active"](False)
            if job["error"]:
                raise RuntimeError(job["error"])
            if not manifest["frames"]:
                raise TimeoutError("No fresh frame arrived during the capture window")
            if spec["encoding"] != "rgba":
                if job["dropped"]:
                    raise RuntimeError(
                        "Encoded capture overflowed; bitstream incomplete"
                    )
                source = output / ("source." + spec["encoding"])
                ffmpeg("-i", source, "-frames:v", 1, output / "frame-000000.png")
                if video:
                    ffmpeg(
                        "-r",
                        spec["fps"],
                        "-i",
                        source,
                        "-c",
                        "copy",
                        output / "capture.mkv",
                    )
            elif video:
                self._assemble(output, manifest)
            manifest["status"] = "complete"
            logger.info(
                "Captured %s: %d frames, %d diagnostic drops -> %s",
                track,
                len(manifest["frames"]),
                job["dropped"],
                output,
            )
            return {
                "output": str(output),
                "frames": len(manifest["frames"]),
                "dropped": job["dropped"],
            }
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            with self.lock:
                self.job = None
            if spec["on_active"]:
                spec["on_active"](False)
            if output_created:
                manifest["capture_dropped"] = job["dropped"]
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )

    def _write(self, job, spec, manifest, output, seconds):
        encoded = spec["encoding"] != "rgba"
        stream = (
            (output / ("source." + spec["encoding"])).open("xb") if encoded else None
        )
        try:
            while not self.closed.is_set():
                # Drain frames already collected inside the window after its deadline.
                if time.monotonic() > job["deadline"] and job["queue"].empty():
                    break
                try:
                    payload, meta, offered_ns = job["queue"].get(timeout=0.05)
                except queue.Empty:
                    continue
                index = len(manifest["frames"])
                if index == 0:
                    manifest.update(width=meta["width"], height=meta["height"])
                if encoded:
                    stream.write(payload)
                    meta.update(offset=stream.tell() - len(payload), size=len(payload))
                else:
                    filename = f"frame-{index:06d}.png"
                    write_png(output / filename, payload, meta["width"], meta["height"])
                    meta["file"] = filename
                meta["offered_monotonic_ns"] = offered_ns
                manifest["frames"].append(meta)
                if not seconds:
                    break
            if self.closed.is_set():
                raise RuntimeError("Media process stopped during diagnostic capture")
        finally:
            if stream:
                stream.close()

    @staticmethod
    def _assemble(output, manifest):
        frames = manifest["frames"]
        lines = ["ffconcat version 1.0"]
        for index, frame in enumerate(frames):
            following = frames[min(index + 1, len(frames) - 1)]
            duration = (
                following["offered_monotonic_ns"] - frame["offered_monotonic_ns"]
            ) / 1e9
            lines += [
                f"file '{frame['file']}'",
                f"duration {duration or 1 / manifest['fps']:.9f}",
            ]
        (output / "frames.ffconcat").write_text("\n".join(lines) + "\n")
        ffmpeg(
            "-f",
            "concat",
            "-safe",
            1,
            "-i",
            output / "frames.ffconcat",
            "-fps_mode",
            "vfr",
            "-c:v",
            "ffv1",
            output / "capture.mkv",
        )

    def close(self):
        self.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument(
        "--track",
        default="cam",
        help="Track name; stereo eyes use cam.left / cam.right",
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory, on the media process host"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="0: fresh snapshot; positive: timed frame sequence (max 3600)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also assemble capture.mkv (requires ffmpeg)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available capture tracks and stages"
    )
    args = parser.parse_args()
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is used")
    request = dict(
        track=args.track,
        output=str(args.output.expanduser().absolute()) if args.output else None,
        seconds=args.seconds,
        video=args.video,
        list=args.list,
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(
                max(0, args.seconds) + 180 if math.isfinite(args.seconds) else 10
            )
            client.connect(str(args.socket.expanduser()))
            client.sendall(json.dumps(request).encode() + b"\n")
            response = json.loads(client.makefile("rb").readline(65536))
        if not response["ok"]:
            parser.exit(1, f"Capture failed: {response['error']}\n")
        print(json.dumps(response, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Capture endpoint unavailable: {exc}\n")


if __name__ == "__main__":
    main()
