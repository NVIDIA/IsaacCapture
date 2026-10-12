# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the internal tactile schema end to end against tactile_pusher (internal only).

live:   read the pusher's streams through TactileShapeSource / TactileSource, optionally
        recording them to --mcap.
replay: read the same streams back from --mcap, with no OpenXR runtime.

Prints a PASS/FAIL report and exits 0 only on PASS.
"""

import argparse
import logging
import sys
import time

import numpy as np

from isaaccapture.deviceio_session import (
    DeviceIOSession,
    McapRecordingConfig,
    McapReplayConfig,
    ReplaySession,
)
from isaaccapture.oxr import OpenXRSession
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    TactileShapeIndex,
    TactileShapeSource,
    TactileSource,
)
from isaaccapture.schema import TactileShape

log = logging.getLogger("isaaccapture.examples.tactile_reader")

NUM_JOINTS = 26
WRIST = 1
SHAPE_VERSION = 1


class Checks:
    """What the run saw, judged against what tactile_pusher sends."""

    def __init__(self, side: str, num_taxels: int) -> None:
        self.device_id = f"mock_tactile_{side}"
        self.num_taxels = num_taxels
        self.updates = 0
        self.frames = 0
        self.shape_frames = 0
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        if message not in self.failures:
            self.failures.append(message)

    def check_shape(self, payload, layout) -> None:
        self.shape_frames += 1
        if payload.device_id != self.device_id:
            self.fail(f"shape device_id {payload.device_id!r} != {self.device_id!r}")
        if payload.shape_version != SHAPE_VERSION:
            self.fail(f"shape_version {payload.shape_version} != {SHAPE_VERSION}")
        parents = np.asarray(layout[TactileShapeIndex.JOINT_PARENTS])
        if parents[WRIST] != TactileShape.ROOT_PARENT:
            self.fail("wrist is not the skeleton root")
        weights = np.asarray(layout[TactileShapeIndex.TAXEL_JOINT_WEIGHTS])
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
            self.fail("skinning weights do not sum to 1")

    def check_frame(self, payload, forces: np.ndarray) -> None:
        self.frames += 1
        if payload.device_id != self.device_id:
            self.fail(f"frame device_id {payload.device_id!r} != {self.device_id!r}")
        if payload.shape_version != SHAPE_VERSION:
            self.fail("frame shape_version does not match the shape")
        if not np.all(np.isfinite(forces)):
            self.fail("non-finite force")
        if forces[:, 2].min() < -1e-4 or forces[:, 2].max() > 2.0 + 1e-4:
            self.fail("normal force outside the pusher's [0, 2] N range")

    def report(self, mode: str) -> bool:
        if self.frames == 0:
            self.fail("no TactileData frame received")
        if self.shape_frames == 0:
            self.fail("no TactileShape received")
        elif mode == "replay" and self.shape_frames < self.frames:
            self.fail("shape not available on every replayed update with data")
        print(f"\n=== tactile_reader ({mode}) ===")
        print(f"updates:                    {self.updates}")
        print(f"updates with TactileData:   {self.frames}")
        print(f"updates with TactileShape:  {self.shape_frames}")
        for failure in self.failures:
            print(f"FAIL: {failure}")
        print("RESULT: PASS" if not self.failures else "RESULT: FAIL")
        return not self.failures


def step(
    session, shape_src: TactileShapeSource, data_src: TactileSource, checks: Checks
):
    """Advance one update and check what both sources produced. Returns the data payload."""
    session.update()
    checks.updates += 1

    polled_shape = shape_src.poll_tracker(session)
    shape_payload = polled_shape["deviceio_tactile_shape"][0]
    layout = shape_src(polled_shape)[TactileShapeSource.SHAPE]
    if not layout.is_none:
        checks.check_shape(shape_payload, layout)
        if checks.shape_frames == 1:
            print(shape_payload)
            for index in TactileShapeIndex:
                print(f"{index.name.lower()}:\n{np.asarray(layout[index])}")

    polled_data = data_src.poll_tracker(session)
    data_payload = polled_data["deviceio_tactile"][0]
    forces = data_src(polled_data)[TactileSource.FORCES]
    if not forces.is_none:
        checks.check_frame(data_payload, np.asarray(forces[0]))
        print(f"update {checks.updates}: {data_payload}\n{np.asarray(forces[0])}")
    return data_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--side", default="right")
    parser.add_argument("--taxels", type=int, default=64)
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="live mode duration"
    )
    parser.add_argument("--mcap", help="record here (live) or read from here (replay)")
    args = parser.parse_args()
    if args.mode == "replay" and not args.mcap:
        parser.error("--mcap is required in replay mode")

    shape_src = TactileShapeSource(
        "tactile_shape", f"tactile_shape/{args.side}", NUM_JOINTS, args.taxels
    )
    data_src = TactileSource("tactile", f"tactile/{args.side}", args.taxels)
    trackers = [shape_src.get_tracker(), data_src.get_tracker()]
    channels = [(trackers[0], "tactile_shape"), (trackers[1], "tactile")]
    checks = Checks(args.side, args.taxels)

    if args.mode == "live":
        extensions = DeviceIOSession.get_required_extensions(trackers)
        log.info("Creating the OpenXR session; this waits until a client is connected")
        with OpenXRSession("TactileReader", extensions) as oxr_session:
            recording = McapRecordingConfig(args.mcap, channels) if args.mcap else None
            with DeviceIOSession.run(
                trackers, oxr_session.get_handles(), recording
            ) as session:
                deadline = time.monotonic() + args.seconds
                next_report = time.monotonic()
                while time.monotonic() < deadline:
                    step(session, shape_src, data_src, checks)
                    if time.monotonic() >= next_report:
                        log.info(
                            "%d frames, shape seen on %d updates",
                            checks.frames,
                            checks.shape_frames,
                        )
                        next_report += 1.0
                    time.sleep(1.0 / 90.0)
        if args.mcap:
            log.info("Recorded to %s", args.mcap)
    else:
        with ReplaySession.run(McapReplayConfig(args.mcap, channels)) as session:
            empty_in_a_row = 0
            # Stop a few updates after the data channel runs out.
            while empty_in_a_row < 5:
                payload = step(session, shape_src, data_src, checks)
                empty_in_a_row = 0 if payload is not None else empty_in_a_row + 1
            checks.updates -= empty_in_a_row

    return 0 if checks.report(args.mode) else 1


if __name__ == "__main__":
    sys.exit(main())
