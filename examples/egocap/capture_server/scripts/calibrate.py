# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrate Manus gloves through a CaptureServer.

    python calibrate.py ws://station:8082

Prompts are printed, and a step starts only when the operator presses Enter. The calibration files are written
into the server's --manus-calibration-dir and stay on the server host; only their basenames come back.
"""

from __future__ import annotations

import asyncio
import sys

from capture_server.client import CalibrationStopped, CaptureClient
from capture_server.protocol import CalibrationPhase, CalibrationStopReason, HandSide


async def calibrate(url: str) -> CalibrationStopped:
    """Run one calibration for both gloves; the operator confirms each step at the keyboard."""
    client = CaptureClient(url)
    loop = asyncio.get_running_loop()
    finished: asyncio.Future[CalibrationStopped] = loop.create_future()
    offered: set[str] = set()

    async def ask(token: str) -> None:
        await loop.run_in_executor(None, input, "  press Enter to start this step (Ctrl-C cancels) ")
        await client.confirm_calibration_step(step_token=token)

    def on_status(status) -> None:
        step = status.step
        if status.phase is CalibrationPhase.AWAITING_STEP and step is not None and step.token not in offered:
            offered.add(step.token)
            print(f"{step.side.name} glove, step {step.index + 1} of {step.count}: {step.title}")
            print(f"  {step.instruction}")
            if step.continuous:
                print(f"  (about {step.estimated_seconds:.0f} s; the SDK gives no completion signal)")
            # Scheduled, not awaited: a callback must never wait on a later callback.
            asyncio.ensure_future(ask(step.token))

    await client.start_calibration(
        sides=(HandSide.LEFT, HandSide.RIGHT),
        on_status=on_status,
        on_stopped=lambda stopped: finished.set_result(stopped),
    )
    try:
        return await finished
    except asyncio.CancelledError:
        await client.stop_calibration()
        raise


async def main(url: str) -> int:
    stopped = await calibrate(url)
    if stopped.reason is not CalibrationStopReason.COMPLETED or stopped.result is None:
        print(f"calibration did not complete: {stopped.reason!r} {stopped.error_message}")
        return 1
    files = {f.side: f.filename for f in stopped.result.files}
    print(f"calibration files on the server: {files}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1])))
