<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EGO Camera Plugin

C++ plugin that records EGO ColorLeft and ColorRight video and optionally captures
IMU, audio, calibration, and device diagnostics in Isaac Capture.

The examples below use the native `ego_camera_plugin` executable. The optional
`ego.sh` helper provides a quick way to build, record, and verify a dataset.

## Features

- Record MJPEG, H.264, or H.265 video as elementary streams.
- Capture accelerometer and gyroscope data, plus optional 48 kHz mono S16_LE audio.
- Write structured data to a local MCAP or publish it to a multi-device TeleopSession.
- Embed encoded video and PCM audio in a private MCAP and export without transcoding.
- Inspect device profiles, configure controls, and preview stereo video.
- Recover the same physical camera after a disconnect.

The plugin does not provide depth, IR, D2C, point clouds, or YUYV streams.

## Install and Build

### One-command Installation

On Linux x86_64, install CMake 3.24 or newer and `uv`, then run from the
Isaac Capture repository root:

```bash
./src/plugins/ego/install.sh
```

The installer installs native build/media dependencies, downloads the pinned
EGO SDK below, verifies its SHA-256, installs the SDK's udev rules, and builds
the native tools with `ego.sh build`. Reconnect the camera after installation.
Repeated runs reuse the verified download. Add `--preview` for SDL preview.

After reconnecting, inspect the camera with the native executable:

```bash
export EGO_BIN="$PWD/build-ego-py3.11/src/plugins/ego_camera"
export EGO_PLUGIN="$EGO_BIN/ego_camera_plugin"
"$EGO_PLUGIN" --list-capabilities
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

Install CMake 3.24 or newer, Python 3.11, `uv`, and the normal Isaac Capture
build dependencies. These packages cover the common native build and media tools:

```bash
sudo apt update
sudo apt install -y build-essential cmake ccache clang-format-14 patchelf \
  pkg-config libudev-dev nlohmann-json3-dev libssl-dev ffmpeg jq usbutils
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
cmake -S . -B build-ego-py3.11 \
  -DISAAC_TELEOP_PYTHON_VERSION=3.11 \
  -DBUILD_VIZ=OFF -DBUILD_PLUGIN_EGO_CAMERA=ON \
  -DORBBEC_SDK_ROOT="$ORBBEC_SDK"
cmake --build build-ego-py3.11 \
  --target ego_camera_plugin ego_mcap_export_media ego_mcap_merge --parallel

export EGO_BIN="$PWD/build-ego-py3.11/src/plugins/ego_camera"
export EGO_PLUGIN="$EGO_BIN/ego_camera_plugin"
"$EGO_PLUGIN" --help
```

Add `-DEGO_ENABLE_PREVIEW=ON` to the configure command for SDL preview.
Recording alone does not require SDL2 or FFmpeg development packages.
The native tools install together under `plugins/ego`; after installation,
set `EGO_BIN` and `EGO_PLUGIN` to that directory and executable instead.

## Usage

### Inspect the Camera

Before recording, inspect the connected device and its supported profiles:

```bash
"$EGO_PLUGIN" --list-capabilities
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
"$EGO_PLUGIN" --width=1600 --height=1300 --fps=30 \
  --add-stream=camera=ColorLeft,output="$RUN/raw/ColorLeft.h264",format=h264 \
  --add-stream=camera=ColorRight,output="$RUN/raw/ColorRight.h264",format=h264
```

Press `Ctrl+C` once to stop. Wait for final statistics and the shell prompt so
the plugin can drain queues, close outputs, and restore changed controls.
Use fresh output paths: recording refuses to overwrite existing files.

This command records raw video only. Native defaults are MJPEG, SDK-selected
resolution/FPS, IMU off, audio off, and no MCAP. H.264 `1600x1300@30` above is
an explicit example; use a profile advertised by your camera.

### Select a Profile or Device

Stream fields override global dimensions/FPS. For example, to record one eye
from a selected device with an advertised H.265 profile:

```bash
"$EGO_PLUGIN" --device-uid=DEVICE_UID \
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
| `--calibration-output="$RUN/calibration.json"` | Export calibration and the original SDK YAML. |

The SDL preview decodes on the CPU and may drop stale display frames without
changing the recording. For GPU preview, use the independent viewer:

```bash
examples/camera_viz/camera_viz.sh setup --with-ego --sdk-root "$ORBBEC_SDK"
examples/camera_viz/camera_viz.sh run \
  examples/camera_viz/configs/ego.yaml --mode window --stereo-debug sbs
```

H.264/H.265 GPU preview requires CUDA/NVDEC; MJPEG uses SDK conversion and GPU
upload. The viewer and recording plugin each open the camera; run one at a time.

### Optional audio and synchronization limits

For a WAV sidecar, add `--audio-output="$RUN/Audio.wav"` to the recording command.
For audio embedded in MCAP without a WAV sidecar, use `--enable-audio` instead.
Audio is disabled by default; enabling IMU does not enable audio.

Audio capture, storage, and export are supported; synchronization with the
camera and IMU requires separate validation.

### Record Data in MCAP

Choose where structured data is recorded:

| Workflow | How to select it | Result |
|---|---|---|
| Raw media only | Omit both metadata destination options. | Video files and optional WAV/calibration sidecars. |
| Local MCAP | `--mcap-filename=PATH` | Plugin-owned metadata, enabled sensors, calibration, and diagnostics. |
| TeleopSession | `--collection-prefix=PREFIX` | Structured data published through OpenXR and recorded with other trackers. |

`--mcap-filename` and `--collection-prefix` are mutually exclusive.

#### Local MCAP

To record stereo video, IMU, and calibration with a local MCAP:

```bash
RUN="recordings/ego_mcap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/raw"
"$EGO_PLUGIN" --width=1600 --height=1300 --fps=30 \
  --add-stream=camera=ColorLeft,output="$RUN/raw/ColorLeft.h264",format=h264 \
  --add-stream=camera=ColorRight,output="$RUN/raw/ColorRight.h264",format=h264 \
  --enable-imu --mcap-filename="$RUN/metadata.mcap" \
  --calibration-output="$RUN/calibration.json"
```

No OpenXR runtime or host tracker is needed here. By default, video and WAV are
sidecars; MCAP holds structured records and media indices.

#### TeleopSession: one MCAP with other trackers

Live publication requires a running OpenXR runtime and the built/installed
Python package. Complete the full build and install the wheel first:

```bash
cmake --build build-ego-py3.11 --parallel
cmake --install build-ego-py3.11 --prefix install-ego-py3.11
uv pip install --reinstall install-ego-py3.11/wheels/isaaccapture-*.whl
```

This extends an existing live `TeleopSessionConfig` named `config`, with its
pipeline and other devices already configured, but no EGO plugin or trackers
added yet. It requests stereo RGB and IMU and preserves existing manual recording entries:

```python
from pathlib import Path
import isaaccapture.deviceio_trackers as dt
from isaaccapture.deviceio_session import McapRecordingConfig
from isaaccapture.ego import configure_recording
from isaaccapture.schema import OrbbecEgoCameraStream
from isaaccapture.teleop_session_manager import PluginConfig, TeleopSession

run = Path("recordings/teleop_ego").resolve()
(run / "raw").mkdir(parents=True, exist_ok=True)
ego_entries = [
    (dt.OrbbecEgoFrameMetadataTracker(
        "ego", [OrbbecEgoCameraStream.ColorLeft, OrbbecEgoCameraStream.ColorRight]
    ), "orbbec_ego_metadata"),
    (dt.OrbbecEgoImuTracker("ego"), "orbbec_ego_imu"),
    (dt.OrbbecEgoCalibrationTracker("ego/Calibration"), "orbbec_ego_calibration"),
    (dt.OrbbecEgoDeviceStateTracker("ego/DeviceState"), "orbbec_ego_device"),
]
existing = config.mcap_config.get_tracker_names() if config.mcap_config else []
config.trackers.extend(tracker for tracker, _ in ego_entries)
config.mcap_config = McapRecordingConfig(str(run / "metadata.mcap"), existing + ego_entries)
config.plugins.append(PluginConfig(
    plugin_name="orbbec_ego_camera",
    plugin_root_id="ego_camera",
    search_paths=[Path("build-ego-py3.11/src/plugins").resolve()],
    required=True,
    shutdown_timeout_seconds=15.0,
    plugin_args=[
        "--collection-prefix=ego", "--width=1600", "--height=1300", "--fps=30",
        f"--add-stream=camera=ColorLeft,output={run}/raw/ColorLeft.h264,format=h264",
        f"--add-stream=camera=ColorRight,output={run}/raw/ColorRight.h264,format=h264",
        "--enable-imu",
    ],
))
config = configure_recording(config)
with TeleopSession(config) as session:
    try:
        while True:
            session.step()
    except KeyboardInterrupt:
        pass
```

Plugin search paths point to the parent containing plugin folders; for an
installation, use `install-ego-py3.11/plugins`. Keep the 15-second EGO shutdown
budget and absolute output paths. Collection prefixes and MCAP source names
serve different purposes; retain these canonical names for the EGO offline reader.

For audio, add `(dt.OrbbecEgoAudioTracker("ego/Audio"), "orbbec_ego_audio")` to `ego_entries`
before extending `config.trackers` and creating `McapRecordingConfig`, and add
`f"--audio-output={run / 'Audio.wav'}"` to `plugin_args`. The offline reader
requires this filename beside the session MCAP. Query current availability through
`session.get_device_status("ego_camera/camera/ego")` after `session.step()`;
`OrbbecEgoDeviceStateTracker` carries historical diagnostic events in version-2 payloads.

### Store Media Inside MCAP

Media layout is separate from the recording workflow:

| Layout | Option | Media location |
|---|---|---|
| Sidecar (default) | `--mcap-media=sidecar` | Video/WAV files beside structured MCAP data. |
| Embedded | `--mcap-media=embedded` | Encoded video and enabled PCM audio inside MCAP, without transcoding. |

Both layouts record the selected video streams and optional audio; the option
controls where media is stored. Only `sidecar` and `embedded` are accepted.

For a self-contained local recording, omit video output paths:

```bash
RUN="recordings/ego_embedded_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
"$EGO_PLUGIN" --width=1600 --height=1300 --fps=30 \
  --add-stream=camera=ColorLeft,format=h264 \
  --add-stream=camera=ColorRight,format=h264 \
  --enable-imu --mcap-filename="$RUN/metadata.mcap" --mcap-media=embedded
```

Add `--enable-audio` to embed PCM. To retain sidecars too, add
`--keep-media-sidecars`, supply each stream's `output=...`, and use
`--audio-output=PATH.wav` if a WAV sidecar is wanted.

For embedded multi-device recording, replace the execution block in the
TeleopSession example with the following, before opening the session:

If that configuration includes audio, replace `--audio-output=...` with
`--enable-audio` for embedded-only PCM, or explicitly retain media sidecars.

```python
from isaaccapture.ego import embedded_recording

fragment = run / "ego_fragment.mcap"
config.mcap_config.filename = str(run / "final.mcap")
config.plugins[-1].plugin_args.extend([
    "--mcap-media=embedded", f"--mcap-media-spool={fragment}",
])
with embedded_recording(config, media_fragment=fragment) as session:
    try:
        while True:
            session.step()
    except KeyboardInterrupt:
        pass
```

The helper accepts exactly one enabled EGO plugin and merges its fragment with
the other devices' session MCAP after clean shutdown. The fragment must match
`--mcap-media-spool`; `final.mcap`, `final.session.mcap`, the fragment, and their
`.partial` paths must be unused. `ego_mcap_merge` must be beside the plugin.
Cleanup/merge failures propagate and preserve inputs for diagnosis. Catch
`Ctrl+C` inside the context to request normal finalization.

## Configuration

### Native CLI Options

These entries describe the native executable, not wrapper presets:

| Option | Native default | Purpose |
|---|---|---|
| `--add-stream=camera=...,output=...[,format=...,width=...,height=...,fps=...]` | None; at least one required | Select `ColorLeft` or `ColorRight`. Format defaults to `mjpg`; output is required unless embedded media has no sidecars. |
| `--width=N --height=N --fps=N` | `0` | Global profile fields; zero selects an SDK default. Nonzero stream fields take precedence. |
| `--device-uid=UID` | Automatic selection | Select a device reported by `--list-capabilities`. |
| `--enable-imu --imu-rate=400\|1000` | Off; rate `400` | Enable available IMU. Setting rate alone does not enable it. |
| `--accel-full-scale=N --gyro-full-scale=N` | `24` g / `2000` dps | Requested IMU ranges, subject to advertised support. |
| `--enable-audio` / `--audio-output=PATH.wav` | Off | Enable PCM; the output option also writes WAV. |
| `--preview` | Off | Open SDL preview. |
| `--mcap-filename=PATH` / `--collection-prefix=PREFIX` | Neither | Choose local MCAP or live publication. |
| `--mcap-media=sidecar\|embedded` | `sidecar` | Choose media layout; embedded requires an MCAP/publication destination. |
| `--keep-media-sidecars` | Off | Retain sidecars in embedded mode. |
| `--mcap-media-spool=PATH` | None | Required fragment for embedded TeleopSession. |
| `--record-device-diagnostics=on\|off` | `on` | Include diagnostic events in MCAP. |
| `--calibration-output=PATH.json` | None | Export calibration as a sidecar. |
| `--time-alignment=PATH` | None | Load a validated reusable RGB/IMU capture profile. |
| `--bitrate=N --dynamic-bitrate=on\|off` | Leave device values unchanged | Set SDK bitrate properties; inspect the advertised range. |
| `--reconnect-timeout=N` | `30` seconds | Wait for the same physical camera; `0` disables recovery. |
| `--reconnect-interval-ms=N` | `1000` | Re-enumeration interval; must be positive. |
| `--persist-controls` | Off | Keep modified camera properties after stopping. |

### Device Controls

Inspect `--list-capabilities` before requesting exposure, gain, white balance,
brightness, sharpness, saturation, contrast, or power-frequency values. For example,
after confirming the property ranges for the intended profile:

```bash
"$EGO_PLUGIN" \
  --add-stream=camera=ColorLeft,output=recordings/manual.h264,format=h264,width=1600,height=1300,fps=30 \
  --set-property=OB_PROP_COLOR_AUTO_EXPOSURE_BOOL=0 --exposure=1000 --gain=100
```

Controls are validated against the active profile and restored after capture
stops. Use `--persist-controls` only when device changes should remain.
`--set-property=SDK_PROPERTY_NAME=VALUE` supports other writable SDK properties.
EGO has no `--quality` option; bitrate values are SDK property values, not OAK's units.

To omit stored diagnostic events, use `--record-device-diagnostics=off`.
Capture checks, recovery, logs, live diagnostics, and Isaac Capture device
monitoring remain active.
For TeleopSession, use `config = configure_recording(config,
record_device_diagnostics=False)` or the same keyword on `embedded_recording`.
Remove diagnostic trackers from explicit recording lists and automatically
recorded pipeline/control/sink sources; conflicts fail before capture. The
merger also rejects diagnostic channels in either input under this policy.

## Architecture

```text
EGO camera -> OrbbecSDK -> ego_camera_plugin
                          |-> video/WAV/calibration sidecars
                          |-> local MCAP: structured data + optional media
                          `-> OpenXR publication -> DeviceIO trackers -> session MCAP
```

Embedded TeleopSession additionally writes an EGO capture fragment, then merges
it with the session MCAP into one final file after both producers stop.

## Output and Verification

### Recording Files

The local MCAP example produces:

```text
recordings/ego_mcap_<timestamp>/
├── raw/ColorLeft.h264
├── raw/ColorRight.h264
├── Audio.wav              # When audio output is enabled.
├── calibration.json
└── metadata.mcap
```

Audio adds `Audio.wav` only when requested with an output path. Embedded mode
stores media in MCAP and omits video/WAV sidecars unless explicitly retained.

Schema and tracker types use `OrbbecEgo*`; the logical plugin name is
`orbbec_ego_camera`. This is a breaking rename of the former `Ego*` Python APIs,
`core.Ego*` schema names, and `ego_*` MCAP topics.

Canonical structured topics are `orbbec_ego_metadata/ColorLeft`, `orbbec_ego_metadata/ColorRight`,
`orbbec_ego_imu/Accel`, `orbbec_ego_imu/Gyro`, `orbbec_ego_calibration/Calibration`, and
`orbbec_ego_device/DeviceState`; audio adds `orbbec_ego_audio/Audio`. IMU/audio topics depend
on enabled and available sensors, and diagnostics can be omitted. Embedded
media uses `orbbec_ego_media/ColorLeft`, `orbbec_ego_media/ColorRight`, and optional `orbbec_ego_media/Audio`.

### Calibration Data

`orbbec_ego_calibration/Calibration` stores the camera intrinsics, stereo extrinsics,
IMU-to-left-camera extrinsics, and original alignment/IMU YAML in MCAP.
`--calibration-output=PATH.json` exports the same calibration as a sidecar;
`ego.sh record` creates `calibration.json` automatically.

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

MCAP is written as `PATH.partial` and promoted to `PATH` only after successful
capture and cleanup. A remaining `.partial`, nonzero exit, or queue/sequence
loss is a failed recording; preserve it with its logs for diagnosis.
Recovery opens a new `capture_epoch`; continuity is checked within each epoch,
and a different physical device is never substituted.

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
video counts/cadence, enabled sensors, and capture health:

```bash
python3 -m pip install --user mcap  # If the MCAP reader is not installed.
./src/plugins/ego/ego.sh verify "$RUN" --plugin "$EGO_PLUGIN" --expect-imu
```

To require complete calibration as well, add `--require-calibration` to `record`
or `verify`:

```bash
./src/plugins/ego/ego.sh record --duration 30 --require-calibration
./src/plugins/ego/ego.sh verify "$RUN" --plugin "$EGO_PLUGIN" --expect-imu --require-calibration
```

This checks camera calibration for each video epoch and IMU extrinsics when IMU
is enabled, including profile dimensions, finite values, proper rotations,
and agreement between MCAP and the final `calibration.json`. Native recordings
must supply `--calibration-output="$RUN/calibration.json"` for that sidecar check.
Without `--require-calibration`, verification reports calibration completeness while
retaining compatibility with recordings that lack calibration fields.

Omit `--expect-imu` if IMU was not requested; add `--expect-audio` when audio was
requested. This verifier expects `metadata.mcap` and two video streams; it is
not a verifier for raw-only or single-eye capture. Native recording does not
create `requested_profile.json` or `logs/capture.log`: without those files,
request matching or log-based loss checks are skipped with a warning.
Success prints `Verification passed` and writes `verification_report.txt`;
failure writes `verification_report.failed.txt` and invalidates an old success report.
These checks establish recording integrity, not physical cross-stream synchronization.

### Export and Play Media

Export a completed embedded MCAP with the native tool. Choose a fresh output
directory because the exporter overwrites same-name media files:

```bash
"$EGO_BIN/ego_mcap_export_media" "$RUN/metadata.mcap" "$RUN/exported_media"
```

It writes `ColorLeft.<format>`, `ColorRight.<format>`, and `Audio.wav` when PCM
was recorded. Do not export a `.partial` capture as a completed dataset.

Elementary streams are not MP4 files. To play or remux H.264 sidecars:

```bash
ffplay -f h264 -framerate 30 "$RUN/raw/ColorLeft.h264"
ffmpeg -f h264 -framerate 30 -i "$RUN/raw/ColorLeft.h264" -c:v copy "$RUN/ColorLeft.mp4"
```

For H.265, use `-f hevc` and the `.h265` file. For MJPEG, use `-f mjpeg` and
transcode with `-c:v libx264 -pix_fmt yuv420p`. For embedded captures, use the
exported paths. Opening video and WAV separately does not demonstrate synchronized playback.

### Structured Replay and Offline Reading

With the installed Python package, inspect an audio-enabled recording using:

```bash
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.replay_ego \
  /absolute/path/to/audio_enabled_recording/metadata.mcap
```

This example requires RGB, IMU, Audio, calibration, and enabled diagnostic
channels. A missing Audio channel is expected for audio-disabled recordings;
it does not make them invalid. Replay reads structured records, not decoded
video or synchronized sound, and one update is not a synchronized bundle.

For sample-time association, install the offline extra and select only the
streams needed, supplying a validated model when none was recorded:

```bash
uv pip install 'isaaccapture[ego]'
python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
  "$RUN" --required-streams left_rgb right_rgb accel gyro \
  --alignment /absolute/path/to/alignment.json --output "$RUN/sync_audit.json"
```

In sidecar mode, the reader requires `raw/ColorLeft.<ext>`,
`raw/ColorRight.<ext>`, and, when audio is recorded, `Audio.wav`, all relative
to the MCAP's directory. The extension is `mjpg`, `h264`, or `h265`, matching
the recorded format. Other output paths are valid for native capture, but
sidecars must follow this layout for offline reading. Passing a directory
selects `metadata.mcap`; pass the MCAP file explicitly if it has another name.
A completed embedded MCAP carries its own media and does not need sidecars.

Unindexed video bytes after the metadata-indexed prefix are reported as
`sidecar_trailing_bytes` for each video stream and excluded from associations.
Missing bytes referenced by metadata remain an error.
Missing required streams/models fail; missing samples remain incomplete.
Add audio only with a validated audio model and continuity evidence. The audit
does not establish synchronization with other devices or absolute host sampling accuracy.

<details>
<summary>Advanced: capture profiles and recording-bound time models</summary>

A reusable profile supplied through `--time-alignment=PATH` has version 1,
`scope="capture_profile"`, `status="validated"`, `model_id`, and
`reference_stream="left_rgb"`. Its identity binds serial, firmware, SDK version,
and the loaded SDK library SHA-256; its profile binds video format/dimensions/FPS,
IMU rate, and audio format. Enabled RGB/IMU mappings provide `offset_ns` and
`scale=1`, with left RGB as the identity mapping. Audio remains an arrival-time
fallback during capture. The file is limited to 8 KiB.

Recording-bound models use `scope="recording"` and bind identity/profile,
MCAP/media fingerprints, and epoch validity intervals. RGB/IMU mappings include
`source_anchor_ns`, `reference_anchor_ns`, positive `scale`, `validation_start_ns`,
and `validation_end_ns`. Audio segments include `segment_id`, `first_sample_index`,
`sample_count`, `reference_anchor_ns`, `period_ns`, `continuity_verified`, and
the same validation bounds. Supply these models explicitly to offline reads;
they do not rewrite MCAP or automatically become reusable capture profiles.

To fit such a model from independently annotated events:

```bash
python -m isaaccapture_examples.mcap_record_replay.calibrate_ego_sync \
  "$RUN/events.json" --output "$RUN/alignment.json"
```

The version-1 event manifest contains `model_id`, `recording`, optional
`media_paths`, `identity`, `profile`, and `epochs`. Each epoch defines `epoch`,
`valid_from_ns`, `valid_to_ns`, and `streams`. Each fitted stream supplies
`calibration_events`, separate `validation_events`, and validation start/end
nanoseconds. Events contain `event_id`, `source_value` (raw nanoseconds or audio
sample index), `reference_time_ns`, `uncertainty_ns`, and an `evidence` path.
Audio uses `streams.audio.segments` with `segment_id`, `first_sample_index`,
`sample_count`, and `continuity_verified` (default false). Paths resolve relative
to the manifest; evidence files are hashed.

Validation requires at least three calibration events and three separate
validation events spanning the beginning, middle, and end of at least 15 minutes.
Maximum error plus uncertainty must be within half a configured video frame.
Reusable profiles declare `validation` evidence and `max_capture_duration_ns`
within the validated duration; expiry returns capture to an unverified
arrival-time fallback. Reconnects and clock discontinuities form new epochs;
models do not interpolate across them. Calibration/audit tools refuse to
overwrite outputs and return nonzero for unverified or incomplete results.

</details>

## Troubleshooting

| Symptom | What to check |
|---|---|
| No camera or permission denied | Install the selected SDK's udev rules, reconnect, and run `--list-capabilities`. |
| Requested profile rejected | Choose advertised profiles for each eye; the simultaneous combination must also start successfully. H.264/H.265 are limited to 30 FPS. |
| `--preview` rejected | Reconfigure with `-DEGO_ENABLE_PREVIEW=ON`, install SDL/FFmpeg development packages, and use a desktop session. |
| Camera busy or preview cannot open | Stop other processes using that EGO, including `camera_viz`. |
| Recording waits after disconnect | Reconnect the same physical camera within `--reconnect-timeout`; UID may change after USB re-enumeration. |
| Nonzero exit or `.partial` remains | Preserve capture and logs; check queue loss, output failures, reconnect timeout, and control restoration. |
| MCAP verifier cannot read a native run | Use the two-eye `metadata.mcap` layout; install the Python MCAP reader. Missing wrapper manifests/logs reduce the checks performed. |
| Video does not open in a desktop player | Decode it with FFmpeg, then remux/transcode to MP4. Keep other SEI data intact when diagnosing decode errors. |
| TeleopSession cannot publish EGO data | Check the running OpenXR runtime, installed Python bindings, plugin search path, and collection prefix. |
| Embedded session has no final file | Check clean context exit, matching spool/fragment paths, fresh output paths, and `ego_mcap_merge` beside the plugin. |
| Audio cadence or sync audit fails | Preserve data and logs; audio requires a validated time model and continuity evidence. |

For SDK API background, consult the [Orbbec SDK v2 API guide](https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/)
and [timestamp guide](https://orbbec.github.io/OrbbecSDK_v2/docs/tutorial/timestamp.html).
These describe SDK behavior; EGO profiles and control ranges must be checked on the connected camera.

## Quick Start with ego.sh (Optional)

After `install.sh` or the manual build above, the helper automates dependency
checks, native rebuilds, run-directory creation, and post-recording verification.
Unlike the native executable, `record` requests both eyes at H.264
`1600x1300@30`, available IMU, calibration, and local MCAP by default; audio is off.

```bash
./src/plugins/ego/ego.sh capabilities
./src/plugins/ego/ego.sh record --duration 30
./src/plugins/ego/ego.sh record --help
```

For repeatable settings, copy `src/plugins/ego/ego.example.toml` outside the
source tree and run `ego.sh record --config /path/to/run.toml`. Later CLI options
override TOML values; arguments after `--` pass through to the native plugin:

```bash
./src/plugins/ego/ego.sh record --config /path/to/run.toml \
  --duration 60 --no-audio -- --bitrate=8 --dynamic-bitrate=on
```

Helper runs additionally save capabilities/provenance, `requested_profile.json`,
`logs/capture.log`, and a verification report. `--duration` includes startup
time; omit it for recording until `Ctrl+C`. The native executable does not
accept `--duration`, `--config`, or wrapper-style `--format` options.
