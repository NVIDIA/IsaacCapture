<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Orbbec EGO Camera Plugin

C++ plugin that records EGO ColorLeft and ColorRight video and optionally captures
IMU, calibration, and device diagnostics in Isaac Capture.

The examples below use the native `orbbec_ego_camera_plugin` executable. The optional
`orbbec_ego.sh` helper provides a quick way to build, record, and verify a dataset.

## Features

- Record MJPEG, H.264, or H.265 video as elementary streams.
- Capture accelerometer and gyroscope data.
- Write structured data to a local MCAP or publish it to a multi-device TeleopSession.
- Inspect device profiles, configure controls, and preview stereo video.
- Recover the same physical camera after a disconnect.

The plugin does not provide depth, IR, D2C, point clouds, or YUYV streams.

## Install and Build

### One-command Installation

On Linux x86_64, complete the [source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html)
first, then run from the Isaac Capture repository root:

```bash
./src/plugins/orbbec_ego/install.sh
```

The installer installs the packages EGO adds (see Manual Prerequisites),
downloads the pinned EGO SDK below, verifies its SHA-256, installs the SDK's
udev rules, and builds the plugin with `orbbec_ego.sh build`. Reconnect the camera after installation.
Repeated runs reuse the verified download. Add `--preview` for SDL preview.

After reconnecting, inspect the camera with the native executable:

```bash
export ORBBEC_EGO_BIN="$PWD/build-orbbec-ego-py3.11/src/plugins/orbbec_ego_camera"
export ORBBEC_EGO_PLUGIN="$ORBBEC_EGO_BIN/orbbec_ego_camera_plugin"
"$ORBBEC_EGO_PLUGIN" --list-capabilities
```

For an already extracted SDK, use `--sdk-root /path/to/OrbbecSDK` or set
`ORBBEC_SDK_ROOT`. Use `--skip-deps` or `--skip-udev` on a prepared host.
Containers skip udev setup; install the rules on the host and reconnect the
camera before running in a container. Use `--container` when container
detection is unavailable. Run `install.sh --help` for other build options.

### Manual Prerequisites

Use the EGO-specific
[OrbbecSDK_v2-EGO v2.10.6-EGO-20261019 release](https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/tag/v2.10.6-EGO-20261019).
For Linux x86_64, download
[`OrbbecSDK_v2.10.6_202610191449_d49af169_linux_x86_64.tar.gz`](https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/download/v2.10.6-EGO-20261019/OrbbecSDK_v2.10.6_202610191449_d49af169_linux_x86_64.tar.gz).
General OrbbecSDK v2 releases, including the ordinary v2.10.6 package, do not
provide the required EGO APIs and cannot replace this EGO-specific build.

From the directory containing the downloaded archive, extract it and set the SDK path:

```bash
tar -xzf OrbbecSDK_v2.10.6_202610191449_d49af169_linux_x86_64.tar.gz
export ORBBEC_SDK="$PWD/OrbbecSDK_v2.10.6_202610191449_d49af169_linux_x86_64"
```

The extracted SDK root must contain:

```text
include/libobsensor/ObSensor.hpp
lib/OrbbecSDKConfig.cmake
lib/libOrbbecSDK.so.2
lib/extensions/
```

Install the toolchain and system packages in the
[source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html),
then the packages EGO adds: `curl` for the SDK download, `ffmpeg` for
`orbbec_ego.sh verify`, and `usbutils` for its `doctor` camera check:

```bash
sudo apt update
sudo apt install -y ca-certificates curl ffmpeg usbutils
```

EGO uses USB VID `0x2bc5`, PID `0x1201`. The SDK's
`shared/99-obsensor-libusb.rules` includes this rule:

```udev
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2bc5", ATTRS{idProduct}=="1201", MODE:="0666", OWNER:="root", GROUP:="video", SYMLINK+="Ego"
```

Install the rules from the same SDK package, reload them, and reconnect the
camera for non-root access:

```bash
sudo "$ORBBEC_SDK/shared/install_udev_rules.sh"
sudo udevadm control --reload-rules
sudo udevadm trigger
```

SDL preview is optional. To build it, also install:

```bash
sudo apt install -y libsdl2-dev libavcodec-dev libavutil-dev libswscale-dev
```

### Manual Configure and Build

Run from the Isaac Capture repository root, keeping `ORBBEC_SDK` set to the
extracted SDK path above:

```bash
cmake -S . -B build-orbbec-ego-py3.11 \
  -DISAAC_TELEOP_PYTHON_VERSION=3.11 \
  -DBUILD_VIZ=OFF -DBUILD_PLUGIN_ORBBEC_EGO_CAMERA=ON \
  -DORBBEC_SDK_ROOT="$ORBBEC_SDK"
cmake --build build-orbbec-ego-py3.11 \
  --target orbbec_ego_camera_plugin --parallel

export ORBBEC_EGO_BIN="$PWD/build-orbbec-ego-py3.11/src/plugins/orbbec_ego_camera"
export ORBBEC_EGO_PLUGIN="$ORBBEC_EGO_BIN/orbbec_ego_camera_plugin"
"$ORBBEC_EGO_PLUGIN" --help
```

Add `-DORBBEC_EGO_ENABLE_PREVIEW=ON` to the configure command for SDL preview.
Recording alone does not require SDL2 or FFmpeg development packages.
The plugin installs under `plugins/orbbec_ego`; after installation,
set `ORBBEC_EGO_BIN` and `ORBBEC_EGO_PLUGIN` to that directory and executable instead.

## Usage

### Inspect the Camera

Before recording, inspect the connected device and its supported profiles:

```bash
"$ORBBEC_EGO_PLUGIN" --list-capabilities
```

This lists firmware, SDK identity, serial/UID, connection, sensor profiles, and
property ranges. Profiles are advertised per sensor; starting capture checks
the exact simultaneous combination. Unsupported combinations fail without a
fallback. H.264/H.265 profiles above 30 FPS are rejected, including profiles
resolved through `fps=0`.

### Record Stereo Video

Create a new output directory and request both eyes explicitly:

```bash
RUN="recordings/ego_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/raw"
"$ORBBEC_EGO_PLUGIN" --width=1600 --height=1300 --fps=30 \
  --add-stream=camera=ColorLeft,output="$RUN/raw/ColorLeft.h264",format=h264 \
  --add-stream=camera=ColorRight,output="$RUN/raw/ColorRight.h264",format=h264
```

Press `Ctrl+C` once to stop. Wait for final statistics and the shell prompt so
the plugin can drain queues, close outputs, and restore changed controls.
Use fresh output paths: recording refuses to overwrite existing files.

This command records raw video only. Native defaults are MJPEG, SDK-selected
resolution/FPS, IMU off, and no MCAP. H.264 `1600x1300@30` above is
an explicit example; use a profile advertised by your camera.

### Select a Profile or Device

Stream fields override global dimensions/FPS. For example, to record one eye
from a selected device with an advertised H.265 profile:

```bash
"$ORBBEC_EGO_PLUGIN" --device-uid=DEVICE_UID \
  --add-stream=camera=ColorLeft,output=recordings/left.h265,format=h265,width=1600,height=1300,fps=30
```

Use `format=mjpg` with a `.mjpg` output for MJPEG. The native executable permits
one or both eyes; it never adds a second stream implicitly.

### Optional IMU and Preview

Add these flags to a recording command as needed:

| Option | Effect |
|---|---|
| `--enable-imu --imu-rate=400` | Request available accelerometer and gyroscope streams. Use MCAP or TeleopSession to retain their records. |
| `--preview` | Show SDL side-by-side preview; requires a preview-enabled build and desktop session. |

The SDL preview decodes on the CPU and may drop stale display frames without
changing the recording.

### Record Data in MCAP

Choose where structured data is recorded:

| Workflow | How to select it | Result |
|---|---|---|
| Raw media only | Omit both metadata destination options. | Video files only; calibration, IMU, and diagnostics need MCAP or TeleopSession. |
| Local MCAP | `--mcap-filename=PATH` | Plugin-owned metadata, enabled sensors, calibration, and diagnostics. |
| TeleopSession | `--collection-prefix=PREFIX` | Structured data published through OpenXR and recorded with other trackers. |

`--mcap-filename` and `--collection-prefix` are mutually exclusive.

#### Local MCAP

To record stereo video, IMU, and calibration with a local MCAP:

```bash
RUN="recordings/ego_mcap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/raw"
"$ORBBEC_EGO_PLUGIN" --width=1600 --height=1300 --fps=30 \
  --add-stream=camera=ColorLeft,output="$RUN/raw/ColorLeft.h264",format=h264 \
  --add-stream=camera=ColorRight,output="$RUN/raw/ColorRight.h264",format=h264 \
  --enable-imu --mcap-filename="$RUN/metadata.mcap"
```

No OpenXR runtime or host tracker is needed here. Video files are sidecars;
MCAP holds structured records and media indices.

#### TeleopSession: one MCAP with other trackers

Live publication requires the CloudXR runtime (see the
[CloudXR guide](https://nvidia.github.io/IsaacCapture/main/references/cloudxr.html)) and the built/installed
Python package. The plugin waits for the runtime's OpenXR system before it opens
the camera. Complete the full build and install it to the default `install/` prefix:

```bash
cmake --build build-orbbec-ego-py3.11 --parallel
cmake --install build-orbbec-ego-py3.11
```

Then install the `isaaccapture` wheel from `install/wheels/` into a Python 3.11
environment
([Build from source, step 4](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html#install-the-isaaccapture-pip-package)).

This extends an existing live `TeleopSessionConfig` named `config`, with its
pipeline and other devices already configured, but no EGO plugin or trackers
added yet. It requests stereo RGB and IMU and preserves existing manual recording entries:

```python
from pathlib import Path
import isaaccapture.deviceio_trackers as dt
from isaaccapture.deviceio_session import McapRecordingConfig
from isaaccapture.teleop_session_manager import PluginConfig, TeleopSession

run = Path("recordings/teleop_ego").resolve()
(run / "raw").mkdir(parents=True, exist_ok=True)
ego_entries = [
    *((dt.OrbbecEgoFrameMetadataTracker(f"ego/{stream}"), f"orbbec_ego_metadata_{stream.lower()}")
      for stream in ("ColorLeft", "ColorRight")),
    *((dt.OrbbecEgoImuTracker(f"ego/{sensor}"), f"orbbec_ego_imu_{sensor.lower()}")
      for sensor in ("Accel", "Gyro")),
    (dt.OrbbecEgoCalibrationTracker("ego/Calibration"), "orbbec_ego_calibration"),
    (dt.OrbbecEgoDeviceStateTracker("ego/DeviceState"), "orbbec_ego_device"),
]
existing = config.mcap_config.get_tracker_names() if config.mcap_config else []
config.trackers.extend(tracker for tracker, _ in ego_entries)
config.mcap_config = McapRecordingConfig(str(run / "metadata.mcap"), existing + ego_entries)
config.plugins.append(PluginConfig(
    plugin_name="orbbec_ego_camera",
    plugin_root_id="ego_camera",
    search_paths=[Path("build-orbbec-ego-py3.11/src/plugins").resolve()],
    required=True,
    plugin_args=[
        "--collection-prefix=ego", "--width=1600", "--height=1300", "--fps=30",
        f"--add-stream=camera=ColorLeft,output={run}/raw/ColorLeft.h264,format=h264",
        f"--add-stream=camera=ColorRight,output={run}/raw/ColorRight.h264,format=h264",
        "--enable-imu",
    ],
))
with TeleopSession(config) as session:
    try:
        while True:
            session.step()
    except KeyboardInterrupt:
        pass
```

Plugin search paths point to the parent containing plugin folders; for an
installation, use `install/plugins`. Keep absolute output paths; the
plugin's `plugin.yaml` sets its 15-second shutdown timeout. Collection prefixes and MCAP source names
serve different purposes; retain these canonical names for the EGO replay example.

Query current availability through
`session.get_device_status("ego_camera/camera/ego")` after `session.step()`;
`OrbbecEgoDeviceStateTracker` carries historical diagnostic events.

## Configuration

### Native CLI Options

These entries describe the native executable, not wrapper presets:

| Option | Native default | Purpose |
|---|---|---|
| `--add-stream=camera=...,output=...[,format=...,width=...,height=...,fps=...]` | None; at least one required | Select `ColorLeft` or `ColorRight`. Format defaults to `mjpg`; output is required. |
| `--width=N --height=N --fps=N` | `0` | Global profile fields; zero selects an SDK default. Nonzero stream fields take precedence. |
| `--device-uid=UID` | Automatic selection | Select a device reported by `--list-capabilities`. |
| `--enable-imu --imu-rate=400\|1000` | Off; rate `400` | Enable available IMU. Setting rate alone does not enable it. |
| `--accel-full-scale=N --gyro-full-scale=N` | `24` g / `2000` dps | Requested IMU ranges, subject to advertised support. |
| `--preview` | Off | Open SDL preview. |
| `--mcap-filename=PATH` / `--collection-prefix=PREFIX` | Neither | Choose local MCAP or live publication. |
| `--record-device-diagnostics=on\|off` | `on` | Include diagnostic events in MCAP. |
| `--bitrate=N --dynamic-bitrate=on\|off` | Leave device values unchanged | Set SDK bitrate properties; inspect the advertised range. |
| `--reconnect-timeout=N` | `30` seconds | Wait for the same physical camera; `0` disables recovery. |
| `--reconnect-interval-ms=N` | `1000` | Re-enumeration interval; must be positive. |
| `--persist-controls` | Off | Keep modified camera properties after stopping. |

### Device Controls

Inspect `--list-capabilities` before requesting exposure, gain, white balance,
brightness, sharpness, saturation, contrast, or power-frequency values. For example,
after confirming the property ranges for the intended profile:

```bash
"$ORBBEC_EGO_PLUGIN" \
  --add-stream=camera=ColorLeft,output=recordings/manual.h264,format=h264,width=1600,height=1300,fps=30 \
  --set-property=OB_PROP_COLOR_AUTO_EXPOSURE_BOOL=0 --exposure=1000 --gain=100
```

Controls are validated against the active profile and restored after capture
stops. Use `--persist-controls` only when device changes should remain.
`--set-property=SDK_PROPERTY_NAME=VALUE` supports other writable SDK properties.
EGO has no `--quality` option; bitrate values are SDK property values, not OAK's units.

To omit stored diagnostic events, use `--record-device-diagnostics=off`.
Capture checks, recovery, logs, live diagnostics, and Isaac Capture device
monitoring remain active. The option applies only to the plugin's local MCAP.
For TeleopSession, leave `orbbec_ego_device` out of the recording list and
keep `OrbbecEgoDeviceStateTracker` out of pipeline sources, which the session
records automatically.

## Architecture

```text
EGO camera -> OrbbecSDK -> orbbec_ego_camera_plugin
                          |-> video sidecars
                          |-> local MCAP: structured data
                          `-> OpenXR publication -> DeviceIO trackers -> session MCAP
```

## Output and Verification

### Recording Files

The local MCAP example produces:

```text
recordings/ego_mcap_<timestamp>/
├── raw/ColorLeft.h264
├── raw/ColorRight.h264
└── metadata.mcap
```

Schema and tracker types use `OrbbecEgo*`; the logical plugin name is
`orbbec_ego_camera`.

Canonical structured topics are `orbbec_ego_metadata_colorleft/FrameMetadata`,
`orbbec_ego_metadata_colorright/FrameMetadata`, `orbbec_ego_imu_accel/Imu`,
`orbbec_ego_imu_gyro/Imu`, `orbbec_ego_calibration/Calibration`, and
`orbbec_ego_device/DeviceState`. Host recordings also carry a `_tracked` channel
beside each one. IMU topics depend on enabled and available sensors, and
diagnostics can be omitted.

### Calibration Data

`orbbec_ego_calibration/Calibration` stores the camera intrinsics, stereo extrinsics,
IMU-to-left-camera extrinsics, and original alignment/IMU YAML in MCAP.

SDK calibration takes priority. When it is unavailable, the plugin fills missing
camera fields from the alignment YAML and missing `accel_to_left` / `gyro_to_left`
from the IMU YAML's `cam0.T_cam_imu`. Here `cam0` is `ColorLeft`; the rotation is
row-major and the translation is in millimeters. This follows the pinned
[EGO SDK's registration of IMU-to-camera and coincident gyro/accel extrinsics](https://github.com/orbbec/OrbbecSDK_v2-EGO/blob/d49af1694a05f533502e4697c5c7a7861e408973/src/device/ego/DataAcquisitionAlgParamManager.cpp#L597-L605).
The fallback validates the rigid transform and preserves both original YAML
strings. It does not change IMU samples, apply intrinsic corrections, or apply
the YAML's time offset.

SDK warnings about unavailable profile calibration can remain when the fallback
has supplied all fields. Check the recorded fields with the strict verification
below. Unsupported or invalid calibration remains missing and is reported.

MCAP is written as `PATH.partial` and promoted to `PATH` only after capture and
cleanup finish. A remaining `.partial` or a nonzero exit is a failed recording,
for example a write failure, metadata queue loss, or a reconnect timeout;
preserve it with its logs for diagnosis. Dropped video frame sets and sequence
gaps are logged as warnings and the recording is still finalized; H.264/H.265
resume at the next IDR. Recovery opens a new `capture_epoch`; sequence and
timestamp order are checked within each epoch, and a different physical device
is never substituted.

### Verify a Recording

Decode both H.264 streams completely and inspect the actual profile:

```bash
ffmpeg -v error -xerror -f h264 -i "$RUN/raw/ColorLeft.h264" -f null -
ffmpeg -v error -xerror -f h264 -i "$RUN/raw/ColorRight.h264" -f null -
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 \
  "$RUN/raw/ColorLeft.h264"
```

For the two-eye local MCAP layout above, the helper also checks MCAP topics,
video counts/cadence, enabled sensors, and capture health; it reports video
sequence gaps, dropped video frame sets, and capture epochs that ended before
their first video frame without failing:

When the active Python has no MCAP reader, `verify` runs these checks through uv.

```bash
./src/plugins/orbbec_ego/orbbec_ego.sh verify "$RUN" --expect-imu
```

To require complete calibration as well, add `--require-calibration` to `record`
or `verify`:

```bash
./src/plugins/orbbec_ego/orbbec_ego.sh record --duration 30 --require-calibration
./src/plugins/orbbec_ego/orbbec_ego.sh verify "$RUN" --expect-imu --require-calibration
```

This checks the MCAP calibration record for each video epoch, and IMU extrinsics
when IMU is enabled, including profile dimensions, finite values, and proper rotations.
Without `--require-calibration`, verification reports calibration completeness while
retaining compatibility with recordings that lack calibration fields.

Omit `--expect-imu` if IMU was not requested. This verifier expects
`metadata.mcap` and two video streams; it is not a verifier for raw-only or
single-eye capture. Native recording does not
create `requested_profile.json` or `logs/capture.log`: without those files,
request matching or log-based loss reporting is skipped with a warning.
Success prints `Verification passed` and writes `verification_report.txt`;
failure writes `verification_report.failed.txt` and invalidates an old success report.
These checks establish recording integrity, not physical cross-stream synchronization.

### Play Media

Elementary streams are not MP4 files. To play or remux H.264 sidecars:

```bash
ffplay -f h264 -framerate 30 "$RUN/raw/ColorLeft.h264"
ffmpeg -f h264 -framerate 30 -i "$RUN/raw/ColorLeft.h264" -c:v copy "$RUN/ColorLeft.mp4"
```

For H.265, use `-f hevc` and the `.h265` file. For MJPEG, use `-f mjpeg` and
transcode with `-c:v libx264 -pix_fmt yuv420p`.

### Structured Replay

With the installed Python package, inspect a recording using:

```bash
uv pip install -e ./examples/orbbec_ego
python -m isaaccapture_examples.orbbec_ego.replay \
  /absolute/path/to/recording/metadata.mcap
```

This example requires RGB, IMU, calibration, and enabled diagnostic channels.
Replay reads structured records, not decoded video, and one update is not a
synchronized bundle.

## Troubleshooting

| Symptom | What to check |
|---|---|
| No camera or permission denied | Install the selected SDK's udev rules, reconnect, and run `--list-capabilities`. |
| Requested profile rejected | Choose advertised profiles for each eye; the simultaneous combination must also start successfully. H.264/H.265 are limited to 30 FPS. |
| `--preview` rejected | Reconfigure with `-DORBBEC_EGO_ENABLE_PREVIEW=ON`, install SDL/FFmpeg development packages, and use a desktop session. |
| Camera busy or preview cannot open | Stop other processes using that EGO. |
| Recording waits after disconnect | Reconnect the same physical camera within `--reconnect-timeout`; UID may change after USB re-enumeration. |
| Nonzero exit or `.partial` remains | Preserve capture and logs; check metadata queue loss, write/output failures, reconnect timeout, and control restoration. |
| Warnings about dropped video frame sets or sequence gaps | The recording was kept; frames are missing between the logged sequence numbers. Reduce host CPU/disk load or bitrate if they recur. Gap counts are in `logs/capture.log` and the verification report; dropped frame sets are also in the capture summary. |
| MCAP verifier cannot read a native run | Use the two-eye `metadata.mcap` layout; install uv, or the Python MCAP reader in the active environment. Missing wrapper manifests/logs reduce the checks performed. |
| Video does not open in a desktop player | Decode it with FFmpeg, then remux/transcode to MP4. Keep other SEI data intact when diagnosing decode errors. |
| Live plugin stalls at `waiting for a system...` | The runtime has no OpenXR system yet; see `XR_ERROR_FORM_FACTOR_UNAVAILABLE` in the [CloudXR guide](https://nvidia.github.io/IsaacCapture/main/references/cloudxr.html#cloudxr-form-factor-unavailable). Ctrl-C does not interrupt this wait; PluginManager stops the plugin after its shutdown timeout. |
| TeleopSession cannot publish EGO data | Check the CloudXR environment, installed Python bindings, plugin search path, and collection prefix. |
| Local recording logs an `NV_CXR_RUNTIME_DIR` error | Expected under PluginManager without the CloudXR environment: device status publication is skipped and recording continues. |

For SDK API background, consult the [Orbbec SDK v2 API guide](https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/)
and [timestamp guide](https://orbbec.github.io/OrbbecSDK_v2/docs/tutorial/timestamp.html).
These describe SDK behavior; EGO profiles and control ranges must be checked on the connected camera.

## Quick Start with orbbec_ego.sh (Optional)

After `install.sh` or the manual build above, the helper automates dependency
checks, native rebuilds, run-directory creation, and post-recording verification.
Unlike the native executable, `record` requests both eyes at H.264
`1600x1300@30`, available IMU, calibration, and local MCAP by default.

```bash
./src/plugins/orbbec_ego/orbbec_ego.sh capabilities
./src/plugins/orbbec_ego/orbbec_ego.sh record --duration 30
./src/plugins/orbbec_ego/orbbec_ego.sh record --help
```

For repeatable settings, copy `src/plugins/orbbec_ego/orbbec_ego.example.toml` outside the
source tree and run `orbbec_ego.sh record --config /path/to/run.toml`. Later CLI options
override TOML values; arguments after `--` pass through to the native plugin:

```bash
./src/plugins/orbbec_ego/orbbec_ego.sh record --config /path/to/run.toml \
  --duration 60 -- --bitrate=8 --dynamic-bitrate=on
```

Helper runs additionally save capabilities/provenance, `requested_profile.json`,
`logs/capture.log`, and a verification report. `--duration` includes startup
time; omit it for recording until `Ctrl+C`. The native executable does not
accept `--duration`, `--config`, or wrapper-style `--format` options.
