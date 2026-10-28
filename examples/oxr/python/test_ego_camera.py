# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record every EGO data class in one of the three supported modes."""

import argparse
import time
from pathlib import Path

import isaaccapture.deviceio_trackers as deviceio_trackers
from isaaccapture.deviceio_session import DeviceIOSession, McapRecordingConfig
import isaaccapture.oxr as oxr
import isaaccapture.plugin_manager as plugin_manager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["no-metadata", "plugin-mcap", "schema-pusher"],
        default="schema-pusher",
    )
    parser.add_argument("--format", choices=["mjpg", "h264", "h265"], default="h264")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--mcap", type=Path, default=Path("ego.mcap"))
    args = parser.parse_args()

    prefix = "ego"
    # The plugin publishes each stream as "{prefix}/{StreamName}"; one tracker per stream.
    frames = {
        name: deviceio_trackers.OrbbecEgoFrameMetadataTracker(f"{prefix}/{name}")
        for name in ("ColorLeft", "ColorRight")
    }
    imus = {
        name: deviceio_trackers.OrbbecEgoImuTracker(f"{prefix}/{name}")
        for name in ("Accel", "Gyro")
    }
    audio = deviceio_trackers.OrbbecEgoAudioTracker(prefix + "/Audio")
    calibration = deviceio_trackers.OrbbecEgoCalibrationTracker(prefix + "/Calibration")
    state = deviceio_trackers.OrbbecEgoDeviceStateTracker(prefix + "/DeviceState")
    trackers = [*frames.values(), *imus.values(), audio, calibration, state]

    suffix = {"mjpg": "mjpg", "h264": "h264", "h265": "h265"}[args.format]
    plugin_args = [
        f"--add-stream=camera=ColorLeft,output=recordings/left.{suffix},format={args.format},width=1600,height=1300,fps=30",
        f"--add-stream=camera=ColorRight,output=recordings/right.{suffix},format={args.format},width=1600,height=1300,fps=30",
        "--enable-imu",
        "--imu-rate=1000",
        "--audio-output=recordings/audio.wav",
        "--calibration-output=recordings/calibration.json",
    ]
    if args.preview:
        plugin_args.append("--preview")
    if args.mode == "plugin-mcap":
        plugin_args.append(f"--mcap-filename={args.mcap.resolve()}")
    elif args.mode == "schema-pusher":
        plugin_args.append(f"--collection-prefix={prefix}")

    manager = plugin_manager.PluginManager([str(args.plugin_root)])
    if "orbbec_ego_camera" not in manager.get_plugin_names():
        raise RuntimeError("orbbec_ego_camera plugin was not discovered")

    with manager.start("orbbec_ego_camera", "ego_camera", plugin_args) as plugin:
        if args.mode != "schema-pusher":
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                plugin.check_health()
                time.sleep(0.1)
            return 0

        extensions = DeviceIOSession.get_required_extensions(trackers)
        recording = McapRecordingConfig(
            str(args.mcap),
            [
                *(
                    (tracker, f"orbbec_ego_metadata_{name.lower()}")
                    for name, tracker in frames.items()
                ),
                *(
                    (tracker, f"orbbec_ego_imu_{name.lower()}")
                    for name, tracker in imus.items()
                ),
                (audio, "orbbec_ego_audio"),
                (calibration, "orbbec_ego_calibration"),
                (state, "orbbec_ego_device"),
            ],
        )
        with oxr.OpenXRSession("EgoCameraTest", extensions) as oxr_session:
            with DeviceIOSession.run(
                trackers, oxr_session.get_handles(), recording
            ) as session:

                def all_data_received() -> bool:
                    return (
                        all(
                            tracker.get_data(session) is not None
                            for tracker in (*frames.values(), *imus.values())
                        )
                        and audio.get_data(session) is not None
                        and calibration.get_data(session) is not None
                        and state.get_data(session) is not None
                    )

                startup_deadline = time.monotonic() + 30.0
                while not all_data_received() and time.monotonic() < startup_deadline:
                    plugin.check_health()
                    session.update()
                    time.sleep(0.016)
                if not all_data_received():
                    raise RuntimeError(
                        "did not receive all EGO data classes within 30 seconds"
                    )

                deadline = time.monotonic() + args.duration
                while time.monotonic() < deadline:
                    plugin.check_health()
                    session.update()
                    time.sleep(0.016)

                if not all_data_received():
                    raise RuntimeError("lost an EGO data class during recording")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
