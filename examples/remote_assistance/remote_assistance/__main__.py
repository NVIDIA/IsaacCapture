# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone sender/bridge CLI; deliberately independent of isaaccapture/CUDA."""

import argparse
import asyncio
import json
import logging
import signal
import time
from pathlib import Path

import yaml


async def run(args, cfg):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    timer = loop.call_later(args.seconds, stop.set) if args.seconds else None
    try:
        if args.command == "camera":
            from .camera import publish_camera

            cameras = [c for c in cfg["cameras"] if c.get("enabled", True)]
            if len(cameras) != 1:
                raise ValueError(
                    "This sender currently supports exactly one enabled camera"
                )
            spec = dict(cameras[0])
            if args.synthetic:
                spec["type"] = "synthetic"
            await publish_camera(cfg["livekit"], spec, stop)
        elif args.command == "bridge":
            from .bridge import run_bridge

            await run_bridge(cfg, args.role, stop)
        else:
            from .video import VideoReceiver

            spec = next(c for c in cfg["cameras"] if c.get("enabled", True))
            settings = cfg["livekit"]
            receiver = VideoReceiver(
                settings,
                identity=settings.get("probe_identity", "edge-probe"),
                publisher=settings.get("publisher", "robot-camera"),
                track=spec["name"],
            )
            count, dimensions, first, last, metadata_count = 0, set(), None, None, 0
            pulse_levels = set()
            await asyncio.to_thread(receiver.start)
            try:
                while not stop.is_set():
                    frame = receiver.latest()
                    if frame:
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                        count += 1
                        metadata_count += frame.metadata is not None
                        dimensions.add((frame.frame.width, frame.frame.height))
                        if args.synthetic:
                            import numpy as np

                            if frame.metadata is None:
                                raise RuntimeError(
                                    "Synthetic validation requires frame metadata"
                                )
                            index = frame.metadata.frame_id
                            expected = (
                                220 if (index // max(1, round(spec["fps"]))) % 2 else 30
                            )
                            pixels = np.frombuffer(
                                frame.frame.data, dtype=np.uint8
                            ).reshape(frame.frame.height, frame.frame.width, 4)
                            mean_red = float(
                                pixels[frame.frame.height // 2, :, 0].mean()
                            )
                            if abs(mean_red - expected) >= 5:
                                raise RuntimeError(
                                    f"Decoded pixels do not match frame {index}: red={mean_red:.1f}, expected={expected}"
                                )
                            pulse_levels.add(expected)
                    await asyncio.sleep(0.001)
            finally:
                await asyncio.to_thread(receiver.stop)
            if count < 2:
                raise RuntimeError("Probe did not receive enough frames")
            if args.synthetic and pulse_levels != {30, 220}:
                raise RuntimeError(
                    "Probe did not observe both synthetic pulse levels; run longer"
                )
            print(
                json.dumps(
                    {
                        "frames": count,
                        "fps": (count - 1) / (last - first),
                        "dimensions": sorted(dimensions),
                        "frames_with_metadata": metadata_count,
                        "mailbox_overwritten": receiver.overwritten,
                        "synthetic_pixels_verified": args.synthetic,
                    },
                    indent=2,
                )
            )
    finally:
        if timer:
            timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("camera", "bridge", "probe"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--role", choices=("edge", "robot"), default="edge")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate camera frames, or validate their pixel content in the probe",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after this duration (0: until Ctrl-C)",
    )
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be nonnegative")
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    # This standalone program never imports isaaccapture (including its logging setup).
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    asyncio.run(run(args, cfg))


if __name__ == "__main__":
    main()
