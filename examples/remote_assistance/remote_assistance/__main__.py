# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video and message examples using the transport contract; no robot controller dependencies."""

import argparse
import asyncio
import gc
import json
import logging
import math
import signal
import threading
import time
from pathlib import Path
import yaml
from isaac_media_transport.latency import for_process, get_recorder, ViewerTiming
from isaac_media_transport.stereo import stereo_tracks
from isaac_media_transport.transport import create_receiver, get_transport, ReceivedPair


def load_config(path):
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("livekit"), dict):
        raise ValueError("Config requires a livekit settings mapping")
    if cfg.get("latency_config"):
        policy = yaml.safe_load((path.parent / cfg["latency_config"]).read_text())
        if not isinstance(policy, dict):
            raise ValueError("Latency config must be a YAML mapping")
        cfg["livekit"]["latency"] = policy
    return cfg


def select_camera(cfg):
    cameras = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]
    if len(cameras) != 1:
        raise ValueError("Select exactly one camera (mono or stereo) in the config")
    return cameras[0]


async def sender(settings, cam, seconds):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    timer = loop.call_later(seconds, stop.set) if seconds else None
    try:
        if cam["type"] != "synthetic":
            if settings.get("video_encoder") != "gstreamer":
                raise ValueError(
                    "This camera example requires video_encoder: gstreamer"
                )
            from .gst_camera import publish_encoded_camera

            await publish_encoded_camera(settings, cam, stop)
            return
        import numpy as np

        session = get_transport(settings).video_sender(settings, stop)
        publication = None
        try:
            await session.start()
            publication = await session.publish(cam, encoded=False)
            image = np.zeros((cam["height"], cam["width"], 4), np.uint8)
            image[..., 3] = 255
            index = 0
            timing = get_recorder(settings)
            while not stop.is_set():
                image[..., 0] = 220 if (index // round(cam["fps"])) % 2 else 30
                stamp = time.monotonic_ns()
                timing.record("capture_available", cam["name"], stamp, at_ns=stamp)
                publication.capture_rgba(image, stamp, index)
                index += 1
                await asyncio.sleep(1 / cam["fps"])
        finally:
            try:
                if publication:
                    await publication.close()
            finally:
                await session.close()
    finally:
        if timer:
            timer.cancel()


def receive(settings, cam, args):
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    tracks = stereo_tracks(cam) if cam.get("stereo") else (cam["name"],)
    if args.command == "receiver":
        if settings.get("decoded_memory") != "cuda":
            raise ValueError(
                "Separate receiver requires decoded_memory: cuda; CPU profiles use viewer directly"
            )
        settings = {**settings, "decoded_handoff": "inprocess"}
    elif args.ipc:
        settings = {**settings, "decoded_handoff": "ipc"}
    live_timing = ViewerTiming(settings if args.command == "viewer" else {})
    receiver = create_receiver(
        settings,
        identity=f"{settings.get('viewer_identity', 'host-video')}-{cam['name']}",
        publisher=settings.get("publisher", "client-camera"),
        track=tracks if len(tracks) > 1 else tracks[0],
        pairing=cam.get("stereo_pairing"),
    )
    timer = threading.Timer(args.seconds, stop.set) if args.seconds else None
    if timer:
        timer.start()
    timing = get_recorder(settings)
    cv2 = None
    try:
        receiver.start()
        if args.command == "receiver":
            from isaac_media_transport.gpu_ipc import IpcServer, socket_path

            IpcServer(
                socket_path(settings),
                receiver,
                tracks,
                int(settings.get("decode_gpu", 0)),
            ).run(stop)
            return
        if not args.headless:
            import cv2

            cv2.namedWindow("Media bridge", cv2.WINDOW_NORMAL)
        count, report = 0, time.monotonic()
        while not stop.is_set():
            sample = receiver.latest()
            if sample:
                count += 1
                if cv2:
                    import numpy as np

                    frames = (
                        (sample.left, sample.right)
                        if isinstance(sample, ReceivedPair)
                        else (sample,)
                    )
                    for track, frame in zip(tracks, frames, strict=True):
                        if frame.metadata:
                            timing.record(
                                "viewer_take", track, frame.metadata.user_timestamp
                            )
                    images = []
                    for frame in frames:
                        f = frame.frame
                        pixels = np.frombuffer(f.to_bytes(), np.uint8).reshape(
                            f.height, f.width, 4
                        )
                        images.append(cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR))
                    joined = np.concatenate(images, axis=1)
                    for track, frame in zip(tracks, frames, strict=True):
                        if frame.metadata:
                            timing.record(
                                "viewer_ready",
                                track,
                                frame.metadata.user_timestamp,
                            )
                            timing.record(
                                "viewer_submit_begin",
                                track,
                                frame.metadata.user_timestamp,
                            )
                    submit_begin_ns = time.monotonic_ns() if live_timing.enabled else 0
                    cv2.imshow(
                        "Media bridge",
                        cv2.resize(joined, None, fx=args.scale, fy=args.scale),
                    )
                    if live_timing.enabled:
                        live_timing.observe(
                            sample.received_ns, submit_begin_ns, time.monotonic_ns()
                        )
                    for track, frame in zip(tracks, frames, strict=True):
                        if frame.metadata:
                            timing.record(
                                "viewer_submit_end",
                                track,
                                frame.metadata.user_timestamp,
                            )
            if cv2 and (
                cv2.waitKey(1) & 255 == ord("q")
                or cv2.getWindowProperty("Media bridge", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            now = time.monotonic()
            if now - report >= 5:
                print(
                    f"Viewer: {count / (now - report):.1f} samples/s, {len(tracks)} video track(s)",
                    flush=True,
                )
                count, report = 0, now
            live_timing.report()
            stop.wait(0.001)
    finally:
        if timer:
            timer.cancel()
        receiver.stop()
        if cv2:
            cv2.destroyAllWindows()
        live_timing.close()
        gc.collect()


async def messages(cfg, role, seconds):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    settings = {**cfg["livekit"], **cfg[role].get("livekit", {})}
    peer = "client" if role == "host" else "host"

    def received(topic, payload):
        print(f"{topic}: {payload.decode(errors='replace')}", flush=True)

    session = get_transport(settings).messages(
        settings,
        cfg[role]["identity"],
        cfg[peer]["identity"],
        ["ra/v1/demo/status"],
        received,
    )
    start = time.monotonic()
    try:
        await session.start()
        while not stop.is_set() and (not seconds or time.monotonic() - start < seconds):
            if session.error:
                raise session.error
            if session.ready:
                await session.send(
                    "ra/v1/demo/status",
                    json.dumps({"site": role}).encode(),
                    reliable=True,
                )
            await asyncio.sleep(1)
    finally:
        await session.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("camera", "receiver", "viewer", "messages"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--role", choices=("host", "client"), default="host")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument(
        "--ipc", action="store_true", help="Viewer reads the separate CUDA receiver"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Receive and count without a display"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Diagnostic window scale"
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.seconds)
        or args.seconds < 0
        or not math.isfinite(args.scale)
        or args.scale <= 0
    ):
        parser.error(
            "seconds must be finite and nonnegative; scale must be finite and positive"
        )
    # This standalone example deliberately does not import isaaccapture.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        cfg = load_config(args.config)
        if args.command == "messages":
            asyncio.run(messages(cfg, args.role, args.seconds))
        else:
            cam = select_camera(cfg)
            settings = for_process(
                cfg["livekit"], "sender" if args.command == "camera" else args.command
            )
            if args.command == "camera":
                asyncio.run(sender(settings, cam, args.seconds))
            else:
                receive(settings, cam, args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Media bridge: {exc}\n")
    finally:
        # Room callbacks form cycles. Release their native handles before the
        # LiveKit atexit hook destroys the FFI handle registry.
        gc.collect()


if __name__ == "__main__":
    main()
