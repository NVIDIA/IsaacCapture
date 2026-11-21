# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the operator's body pose to a standard Isaac Capture full-body MCAP.

The recording is the standard ``core.FullBodyPoseRecord`` schema on the standard ``full_body``
channel, so ``python -m isaaccapture_examples.mcap_record_replay.replay_full_body`` plays it back
unmodified.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio import McapRecordingConfig
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    FullBodySource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types import FullBodyInputIndex
from isaaccapture.schema import BodyJoint
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

from .plugin import (
    DEFAULT_CHANNEL,
    DEFAULT_COLLECTION_ID,
    full_body_source,
    plugin_config,
)


def _build_pipeline(collection_id: str) -> OutputCombiner:
    controllers = ControllersSource(name="controllers")
    full_body = full_body_source(collection_id)
    # The outputs common.build_full_body_pipeline() declares, so the pipeline the replay script
    # builds finds every channel this recorded.
    return OutputCombiner(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "full_body": full_body.output(FullBodySource.FULL_BODY),
        }
    )


def _resolve_output(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
    else:
        # Under the working directory, as mcap_record_replay does: next to this file is inside
        # the package, and inside site-packages once installed.
        out_dir = Path.cwd() / "recordings"
        path = out_dir / f"jmex_full_body_{datetime.now():%Y%m%d_%H%M%S}.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str]) -> int:
    """Parse ``argv``, record for the requested duration, and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=5.0, help="Recording duration (s)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .mcap path (default: ./recordings/jmex_full_body_<timestamp>.mcap)",
    )
    parser.add_argument(
        "--collection-id",
        default=DEFAULT_COLLECTION_ID,
        help=(
            "The collection the plugin publishes the body pose on, and the one this "
            f"reads from (default: {DEFAULT_COLLECTION_ID})"
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help=(
            f"MOXI Player channel the plugin pairs on (default: {DEFAULT_CHANNEL}). "
            "Running in src/plugins/jmex/README.md names the one Player's file replay uses."
        ),
    )
    parser.add_argument(
        "--no-plugin",
        action="store_true",
        help=(
            "Do not auto-launch jmex_plugin; use an already running one, started with the "
            "flags you need (src/plugins/jmex/README.md, or jmex_plugin --help)."
        ),
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])

    mcap_path = _resolve_output(args.output)
    plugins: list[PluginConfig] = []
    if not args.no_plugin:
        try:
            plugins.append(
                plugin_config(
                    args.channel,
                    args.collection_id,
                    otherwise="pass `--no-plugin` for a manually started plugin.",
                )
            )
        except RuntimeError as e:
            sys.exit(f"[record] error: {e}")

    config = TeleopSessionConfig(
        app_name="JmexFullBodyRecordExample",
        pipeline=_build_pipeline(args.collection_id),
        plugins=plugins,
        mcap_config=McapRecordingConfig(str(mcap_path)),
    )

    joint_count = int(BodyJoint.NUM_JOINTS)
    print(f"[record] writing {mcap_path} for {args.duration:.1f}s")
    print(f"[record] collection_id={args.collection_id}")

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(
                f"[record] CloudXR runtime started (WSS log: {launcher.wss_log_path})"
            )
        with TeleopSession(config) as session:
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                result = session.step()
                if session.frame_count % 60 == 0:
                    full_body = result["full_body"]
                    n_valid = (
                        0
                        if full_body.is_none
                        else int(
                            np.count_nonzero(
                                np.asarray(
                                    full_body[FullBodyInputIndex.JOINT_VALID],
                                    dtype=np.uint8,
                                )
                            )
                        )
                    )
                    print(
                        f"[record] t={time.monotonic() - start:5.2f}s  "
                        f"frame={session.frame_count}  "
                        f"joints={n_valid:02d}/{joint_count}"
                    )
                time.sleep(1 / 60)

    print(f"[record] done - {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
