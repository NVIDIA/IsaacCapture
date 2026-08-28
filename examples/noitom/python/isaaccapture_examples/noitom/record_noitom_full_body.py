# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record Noitom full-body samples to a standard IsaacTeleop full-body MCAP."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio import McapRecordingConfig, TrackerVendor
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


DEFAULT_COLLECTION_ID = "noitom_mocap"
DEFAULT_MAX_FLATBUFFER_SIZE = 16 * 1024
PLUGIN_NAME = "noitom_mocap"
PLUGIN_ROOT_ID = "noitom_mocap"
NOITOM_VENDOR_ID = "body.noitom"


def _plugin_search_paths() -> list[Path]:
    # Six levels up is the repo root -- or the install prefix, which has the
    # same shape under examples/<name>/python/isaaccapture_examples/<name>/.
    base = Path(__file__).resolve().parents[5]
    candidates = [
        base / "plugins",
        base / "install" / "plugins",
    ]
    return [path for path in candidates if path.exists()]


def _build_pipeline(collection_id: str, max_flatbuffer_size: int) -> OutputCombiner:
    controllers = ControllersSource(name="controllers")
    full_body = FullBodySource(
        name="full_body",
        vendor=TrackerVendor(
            NOITOM_VENDOR_ID,
            {
                "collection_id": collection_id,
                "max_flatbuffer_size": str(max_flatbuffer_size),
            },
        ),
    )
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
        # CWD-relative, matching mcap_record_replay: anchoring to __file__ would
        # write inside the package, and into site-packages once installed.
        out_dir = Path.cwd() / "recordings"
        path = out_dir / f"noitom_full_body_{datetime.now():%Y%m%d_%H%M%S}.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=5.0, help="Recording duration (s)"
    )
    parser.add_argument("output", nargs="?", help="Output .mcap path")
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument(
        "--max-flatbuffer-size", type=int, default=DEFAULT_MAX_FLATBUFFER_SIZE
    )
    parser.add_argument(
        "--no-plugin",
        action="store_true",
        help="Do not auto-launch noitom_mocap_plugin; use an already running plugin.",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])

    mcap_path = _resolve_output(args.output)
    plugins: list[PluginConfig] = []
    if not args.no_plugin:
        search_paths = _plugin_search_paths()
        if not search_paths:
            sys.exit(
                "[record] error: no installed plugin directory found. "
                "Run `cmake --install build` or pass `--no-plugin` for a manually "
                "started plugin."
            )
        plugins.append(
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=search_paths,
            )
        )

    config = TeleopSessionConfig(
        app_name="NoitomFullBodyRecordExample",
        pipeline=_build_pipeline(args.collection_id, args.max_flatbuffer_size),
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
            start = time.time()
            while time.time() - start < args.duration:
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
                        f"[record] t={time.time() - start:5.2f}s  "
                        f"frame={session.frame_count}  "
                        f"joints={n_valid:02d}/{joint_count}"
                    )
                time.sleep(1 / 60)

    print(f"[record] done - {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
