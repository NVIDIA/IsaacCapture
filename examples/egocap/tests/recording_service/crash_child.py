# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Child process for crash tests: records real samples through the real service, then dies mid-episode.

Usage: crash_child.py OUTPUT_ROOT MODE [SESSION_ID]
  MODE ``writing``       -- wait until samples are written, announce READY, then sleep (parent SIGKILLs us)
  MODE ``after_checked`` -- stop normally but exit the process inside the finalize rename step
  MODE ``after_rename``  -- stop normally but exit right after the rename, before the FINALIZED checkpoint
"""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import real  # noqa: E402

import recording_service.recording_service as impl  # noqa: E402
from recording_service import CalibrationAsset, RecordingService  # noqa: E402


async def main(root: str, mode: str, session_id: str | None) -> None:
    """Run one episode to the requested point of failure."""
    if mode == "after_checked":
        impl.finalize_episode_dir = lambda *_a, **_k: os._exit(9)
    if mode == "after_rename":
        real_write = impl.write_manifest

        def exit_before_finalized_checkpoint(session_path, manifest):
            if manifest.storage_phase == "FINALIZED":
                os._exit(9)
            real_write(session_path, manifest)

        impl.write_manifest = exit_before_finalized_checkpoint
    config = real.make_config(Path(root))
    calibration = os.environ.get("CRASH_CALIBRATION")
    if calibration:
        config = replace(
            config, calibration_assets=(CalibrationAsset("head_camera_intrinsics.json", Path(calibration)),)
        )
    service = RecordingService(config)
    session = await service.open_session(session_id)
    print(session.session_id, flush=True)
    episode = await service.start_episode("crash test")
    await real.submit_all(service, episode.episode_id, real.samples(15))
    if mode == "writing":
        while any(c.written != c.accepted for c in service.snapshot().counts):
            await asyncio.sleep(0.05)
        print("READY", flush=True)
        await asyncio.sleep(120)
    else:
        await service.stop_episode(episode.episode_id)


asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
