<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EGO Camera Plugin

This plugin connects EGO stereo cameras to Isaac Capture. It captures
`ColorLeft` and `ColorRight` video, available accelerometer and gyroscope data,
calibration, and device state. Microphone capture is optional.

The plugin selects a device by its left/right color sensors and advertised
profiles. Run `ego.sh capabilities` for the connected device's firmware, USB
connection, and profiles. This plugin does not provide depth, IR, D2C,
point-cloud, or YUYV streams.

## What you can do

- Record stereo MJPEG, H.264, or H.265 elementary streams without a container.
- Optionally record 48 kHz, mono, S16_LE PCM audio as WAV.
- Record frame metadata, IMU batches, audio offsets, calibration, and device
  state in MCAP.
- Choose raw-media-only, plugin-local MCAP, or TeleopSession MCAP workflows.
- Create a self-contained private `embedded` MCAP containing encoded video and
  optional PCM audio, then export the recorded media.
- Inspect device capabilities and apply validated exposure, gain, white-balance,
  image, bitrate, and IMU controls.
- Preview stereo video with the plugin's SDL window or use the independent
  GPU `camera_viz` source.
- Associate selected RGB, IMU, and audio samples offline using an independently
  validated EGO time model, without changing the NV replay lifecycle.

Capabilities have two levels. `--list-capabilities` reports profiles advertised
for each sensor; starting the pipeline validates the exact simultaneous stream
combination. An unsupported combination fails with its requested profiles and
SDK error instead of silently falling back.

The wrapper defaults to stereo H.264 `1600x1300@30`. The plugin rejects
resolved H.264/H.265 profiles above 30 FPS, even if the device advertises them.
`fps=0` cannot bypass this check, and no lower-FPS fallback is selected.

Available IMU is requested by default; audio is disabled unless explicitly
enabled. See [Optional audio and synchronization limits](#optional-audio-and-synchronization-limits)
before adding audio to a recording.

## Prerequisites

For a Linux x86_64 host, use an extracted OrbbecSDK v2 package for the same
architecture and the normal Isaac Capture build dependencies. The SDK is
external and is not distributed by this repository. Its root must contain:

```text
include/libobsensor/ObSensor.hpp
lib/OrbbecSDKConfig.cmake
lib/libOrbbecSDK.so.2
lib/extensions/
```

The standard host tools are CMake **3.24 or newer**, a C++ compiler, Python,
`uv`, FFmpeg, and the Isaac Capture build dependencies. Reading a TOML run
configuration requires Python 3.11 or newer; it uses the standard `tomllib`
module and adds no package dependency. A recording-only build does **not** need
SDL2 or FFmpeg development packages.

```bash
sudo apt update
sudo apt install -y build-essential cmake ccache clang-format-14 patchelf \
  pkg-config libudev-dev nlohmann-json3-dev libssl-dev ffmpeg jq usbutils
```

The SDL preview is an optional build feature. Install its development packages
only when building with `--preview`:

```bash
sudo apt install -y libsdl2-dev libavcodec-dev libavutil-dev libswscale-dev
```

Install the udev rules provided by the **same SDK build** before using the
camera as a non-root user. Use the script supplied with that package:

```bash
sudo /path/to/OrbbecSDK/shared/install_udev_rules.sh
```

Reconnect the camera after installing rules. A desktop session is required for
`--preview`.

Do not identify an SDK binary by semantic version alone: different SDK builds
can report the same version. The workflow records the executable and resolved
`libOrbbecSDK.so.2` paths and SHA-256 hashes alongside the firmware, serial,
UID, USB connection, and timestamp capability reported by the native plugin.

## Quick start: build, record, verify

The recommended entry point is `ego.sh`. It never installs packages,
downloads SDKs, or runs `sudo`; `doctor` reports missing requirements and shows
the appropriate command instead.

Open a fresh terminal at the repository root:

```bash
cd /absolute/path/to/IsaacCapture
export ORBBEC_SDK=/absolute/path/to/OrbbecSDK_v2_linux_x86_64

# 1. Inspect host dependencies, SDK layout, USB access, and optional preview libraries.
./src/plugins/ego/ego.sh doctor --sdk-root "$ORBBEC_SDK"

# 2. Configure and build only the Ego plugin and embedded-media exporter.
./src/plugins/ego/ego.sh build --sdk-root "$ORBBEC_SDK" --jobs 8

# Add --preview only after installing the optional SDL/FFmpeg development packages.
# ./src/plugins/ego/ego.sh build --sdk-root "$ORBBEC_SDK" --preview --jobs 8

# 3. Save the software provenance and connected camera's advertised profiles.
./src/plugins/ego/ego.sh capabilities | tee capabilities.txt

# 4. Record stereo H.264 1600x1300@30, available IMU, calibration,
#    and a local metadata MCAP. Audio is off by default.
./src/plugins/ego/ego.sh record --duration 30
```

The final command creates a Git-ignored directory:

```text
recordings/ego_<timestamp>/
├── raw/ColorLeft.h264
├── raw/ColorRight.h264
├── calibration.json
├── metadata.mcap
├── capabilities.txt
├── requested_profile.json
├── verification_report.txt
└── logs/capture.log
```

Enabling audio adds `Audio.wav` in this sidecar workflow.

`verification_report.txt` is created after the automatic post-recording
verification succeeds. A standalone `verify` command invalidates the previous
success report before checking and refreshes it only after success; a failed
run writes `verification_report.failed.txt` instead so incomplete evidence is
not confused with an accepted report.

`requested_profile.json` is a machine-readable record of the requested
format, width, height, and FPS for both `ColorLeft` and `ColorRight`. It records
the request, while frame metadata and FFprobe describe what was actually
captured.

For an unlimited recording, omit `--duration` and stop with `Ctrl-C`. Wait for
the final statistics and the shell prompt; this finalizes the media and MCAP
output.

Verify the run before sharing it. Do not type angle-bracket placeholders such
as `<timestamp>` literally: replace them with the real directory name.

```bash
python3 -m pip install --user mcap  # only if verify asks for it

RUN="$(ls -dt recordings/ego_*/ | head -n 1)"
./src/plugins/ego/ego.sh verify "$RUN"
```

Success prints `Verification passed`, writes `verification_report.txt`, and
reports the delivery MCAP path and SHA-256 fingerprints for the MCAP and media.
Both H.264 streams decode completely with FFmpeg, and the MCAP has a Footer,
its required topics, and monotonic per-topic log times. It also rejects any remaining
`.partial`, rejects nonzero sequence-gap or queue-drop counters, confirms both
video profiles match, checks each FFprobe codec/resolution against
`requested_profile.json`, matches each decoded count to that stream's MCAP
video topics, and checks raw device timestamps for strict monotonicity and the
declared frame cadence within each capture epoch. Left and Right may begin at
different parameterized IDRs, so unequal totals are reported but are not by
themselves a failure. When
IMU or audio was requested, it also validates nonempty contiguous batches or
chunks, the declared profile, raw-device cadence, and at least 90% coverage of
the video timeline. Audio sample and byte counts must agree, embedded PCM must
match its metadata timeline, and the WAV sidecar or export must be 48 kHz,
mono, S16_LE. Embedded verification always exports and decodes the MCAP payload;
if retained sidecars exist, their video bytes must match the embedded payload
exactly.
If `requested_profile.json` is absent, `verify` warns that the
requested-versus-recorded profile check was skipped.
These integrity checks do not measure physical cross-stream synchronization.
Use the [time-alignment workflow](#sample-time-alignment-and-offline-reading)
and an independent physical error measurement for that conclusion.

### Optional audio and synchronization limits

For raw sound capture or timestamp investigation, request audio explicitly:

```bash
./src/plugins/ego/ego.sh record --duration 30 --audio
```

The TOML equivalent is `audio = true`. `--no-audio` overrides an audio-enabled
configuration; omitting both the flag and the configuration setting leaves
audio off. Native `--enable-audio` or `--audio-output=PATH.wav` arguments also
explicitly enable capture.

PCM capture, storage, and export remain available, but **audio cross-stream
synchronization has not passed physical acceptance**. A tested Ethernet
configuration with firmware 1.1.10 and an SDK 2.9.0 build returned raw audio
timestamps advancing about 1.875 times faster than the duration implied by its
48 kHz PCM samples. This observation does not establish the same fault for
other devices, SDK builds, or USB capture paths. The adapter preserves the
SDK values; neither a different source label nor successful WAV playback
corrects their time scale or aligns audio to video/IMU.

The recording verifier still checks audio when it is present and can reject
recordings with this anomaly. Preserve failed recordings and logs for diagnosis.
Include audio in an offline synchronization result only with an independently
validated audio time model and continuity evidence. Disabling audio
does not establish that the remaining RGB/IMU streams are physically synchronized.

### One-command script reference

```bash
./src/plugins/ego/ego.sh doctor --sdk-root "$ORBBEC_SDK"
./src/plugins/ego/ego.sh build --sdk-root "$ORBBEC_SDK" [--preview] [--jobs N] [--clean]
./src/plugins/ego/ego.sh capabilities [--device-uid UID]
./src/plugins/ego/ego.sh record [options] [-- PLUGIN_OPTIONS...]
./src/plugins/ego/ego.sh verify RUN_DIRECTORY
./src/plugins/ego/ego.sh export-media RUN_DIRECTORY [--output DIRECTORY]
```

Useful recording options:

```bash
# Select format/profile or output location.
./src/plugins/ego/ego.sh record --duration 30 \
  --format h265 --width 1600 --height 1300 --fps 30 \
  --output recordings/demo_h265

# Inspect the connected device's advertised profiles.
./src/plugins/ego/ego.sh capabilities

# Disable optional sensors, select a device, or show the SDL preview (after a --preview build).
./src/plugins/ego/ego.sh capabilities --device-uid DEVICE_UID
./src/plugins/ego/ego.sh record --duration 30 \
  --device-uid DEVICE_UID --no-imu --preview

# Change the same-device recovery window, or use 0 to retain fail-fast behavior.
./src/plugins/ego/ego.sh record --duration 120 \
  --reconnect-timeout 60 --reconnect-interval-ms 1000

# Pass advanced native controls unchanged after --.
./src/plugins/ego/ego.sh record --duration 30 -- \
  --bitrate=8 --dynamic-bitrate=on
```

### Reusable TOML recording configuration

Use a TOML file for repeatable recordings or deployment settings. Start
from [`ego.example.toml`](ego.example.toml), copy it outside the
source tree if desired, and run from the repository root:

```bash
cp src/plugins/ego/ego.example.toml recordings/ego_run.toml
./src/plugins/ego/ego.sh record \
  --config recordings/ego_run.toml
```

The `[record]` table supports:

| TOML key | Type | Meaning |
|---|---|---|
| `duration` | integer | Timed capture in seconds; omit for `Ctrl-C`. |
| `output` | string | New run directory. Relative paths use the command's working directory. |
| `format` | string | `mjpg`, `h264`, or `h265`. |
| `width`, `height`, `fps` | integer | Exact stereo profile requested for both color streams. |
| `device_uid` | string | UID reported by `capabilities`; omit to use the only connected compatible device. |
| `imu` | boolean | Request available IMU; default `true`. |
| `audio` | boolean | Request optional audio; default `false`. See the audio synchronization limits above. |
| `preview` | boolean | Enable the SDL stereo preview. |
| `mcap_media` | string | `metadata-only` or `embedded`. |
| `keep_media_sidecars` | boolean | Retain video/WAV sidecars in embedded mode. |
| `record_device_diagnostics` | boolean | Save diagnostic events in MCAP; default `true`. Monitoring and live diagnostics remain active. |
| `reconnect_timeout` | integer | Same-device recovery window in seconds; zero disables recovery. |
| `reconnect_interval_ms` | integer | Re-enumeration interval in milliseconds. |
| `preset`, `plugin` | string | Select a build preset or an explicit plugin executable. |
| `plugin_options` | string array | Native controls such as `--bitrate=8` and `--exposure=1000`. |

Stereo `ColorLeft` and `ColorRight` are mandatory in the wrapper
workflow; `imu` and `audio` select the optional data streams. Unknown keys,
wrong TOML types, missing files, and malformed TOML fail before the output
directory or camera is touched. Explicit CLI options after `--config` override
the corresponding file values, while options after `--` are appended after
`plugin_options`. For example:

```bash
# Reuse the file but override this run's duration, output, and audio choice.
./src/plugins/ego/ego.sh record \
  --config recordings/ego_run.toml \
  --duration 120 --output recordings/ego_run --no-audio
```

`record` requests available IMU by default. Audio requires `--audio` or
`audio = true` and is started only if the device advertises it. The wrapper
never silently changes an unsupported requested profile to another profile.
Advertising Left and Right profiles separately does not prove that a
particular video/IMU/audio combination can run simultaneously; pipeline start
and the completed `verify` run are the combination check.

For `--duration`, the wrapper sends `SIGINT`, allows up to 30 seconds for
draining and finalization, and preserves the plugin's real exit status. A
shutdown, control-restore, or MCAP-close failure therefore cannot be mistaken
for an intentional timeout. The duration includes SDK startup and readiness;
allow enough time for both before evaluating the recorded interval.

### Disconnect recovery

Recording waits up to 30 seconds by default for a disconnected camera to
reappear with the same serial number and USB VID/PID. A different Ego device
is never substituted. SDK 2.9.0 EGO UIDs contain the USB enumeration address
and can change after a physical reconnect, so UID is treated as an enumeration
instance and may differ between capture epochs.
The SDK pipelines, callbacks, exact stream profiles, calibration, and requested
controls are rebuilt before capture resumes. After stream readiness, recovery
waits five seconds, then requires each requested auxiliary stream to have
arrived within the preceding 500 ms before committing the new epoch.
Set `--reconnect-timeout 0` to disable recovery;
`--reconnect-interval-ms` controls re-enumeration frequency.

Every successful restart opens a new `capture_epoch` in video, IMU, audio,
calibration, and device-state records. Hardware sequence numbers and raw device
timestamps may restart only across that boundary; local common-clock time stays
monotonic. H.264/H.265 output discards post-reconnect prediction frames until a
new parameterized IDR arrives. The final statistics report the epoch, attempts,
and successful reconnects, and `verify` checks sequence and timestamp
continuity independently inside each epoch.

Queue loss, output/MCAP/SchemaPusher failure, changed firmware or profiles, and
control-restore failure remain fatal. If the same device does not return before
the timeout, the process exits nonzero and preserves the `.partial` MCAP. A
`Ctrl-C` while the device is absent also preserves the incomplete recording.
Video, IMU, and audio are each monitored after recovery; a stream that stops
for five seconds triggers another recovery instead of leaving a partial sensor
set marked healthy.

## Current availability and diagnostic events

`TeleopSession.get_status()` is the current availability API. The EGO plugin
uses NVIDIA's `PluginDeviceStatusPublisher` and the existing OpenXR session;
it publishes one manifest device, `/camera/ego`, through
`<plugin_root_id>/device_status`. Its observations come from the existing
capture readiness, removal, liveness, and recovery checks, not a second probe.

```python
# Inside an active TeleopSession; plugin_root_id is your EGO PluginConfig value.
session.step()
device = session.get_device_status(plugin_root_id + "/camera/ego")
provider = session.get_provider_status("plugin/" + plugin_root_id)
```

| Capture observation | NV device state / reason |
|---|---|
| No authoritative current conclusion | `UNKNOWN / NO_CURRENT_DATA` |
| Target device confirmed absent | `DISCONNECTED / NO_HARDWARE_SIGNAL` |
| Present but starting or recovering | `DEGRADED / NO_CURRENT_DATA` |
| All requested streams ready and existing liveness checks pass | `CONNECTED / NONE` |
| Nonrecoverable device failure while the provider is running | `FAILED / DEVICE_ERROR` |

NVIDIA owns provider startup, failure, and teardown; a failed or stopped
provider makes its device `UNAVAILABLE`. The official heartbeat is one second,
with reports stale after three seconds. A blocked capture owner cannot be kept
online by refreshing a cached report's timestamp. State publishing runs on the
existing publishing thread, outside SDK callbacks. Status transport failures
are logged and retried without independently failing recording; ordinary data
publication and archive failures still prevent a successful recording.
Standalone capture without `--plugin-root-id` has no monitoring dependency.
Local recording launched with a root ID can continue if its status-only OpenXR
session cannot start; the host then has no current report.

`EgoDeviceStateTracker` serves a different purpose: it delivers individual
**diagnostic events**, not the latest complete device condition. The collection
remains `<collection-prefix>/DeviceState`, and its canonical recording topic
remains `ego_device/DeviceState`. Each version-2 `EgoDeviceState` has a device UID,
capture epoch, observation time in local common-clock nanoseconds, and exactly
one typed `event`:

| Python event type | Observations |
|---|---|
| `EgoSdkReport` | SDK report sequence, work mode, raw status/error flags, free storage bytes. |
| `EgoCaptureSummary` | Capture health and reason, event queue capacity/peak, dropped event/video-frame-set totals, and `is_final`. |
| `EgoRecoveryEvent` | `Started`, `Recovered`, or `Failed` stage, reason, and cumulative reconnect-attempt count. |
| `EgoDeviceSnapshot` | Successfully read startup properties and optional `imu_temperature_c`. |

Queue peaks, losses, and reconnect attempts accumulate for the capture instance,
not separately for each epoch. A `Started` event carries the number of attempts
already made; `Recovered`/`Failed` carry the count reached at that transition.
SDK report sequences belong only to `EgoSdkReport`. Raw SDK flags are diagnostic
values, not an independent USB connectivity decision.

Temperature `None` means unavailable; `0.0` means a measured zero. Missing
properties are omitted. Startup reads are timestamped when that batch finishes;
they are not simultaneous periodic measurements, and recovery does not relabel
them as new samples. Capture summaries retain the existing cadence and include
a final event after producer drain. A final summary does not prove that the
subsequent archive close and atomic publication succeeded.

```python
from isaaccapture.schema import EgoCaptureSummary, EgoDeviceSnapshot

state = state_tracker.get_data(session.deviceio_session)
if state is not None:
    event = state.event
    if isinstance(event, EgoCaptureSummary):
        print(event.capture_health, event.reason, event.dropped_video_frame_sets)
    elif isinstance(event, EgoDeviceSnapshot):
        print(event.imu_temperature_c, event.properties)
```

Use the active DeviceIO session accepted by your tracker (`session` when working
directly with DeviceIO). Event views are immutable and retain their record
buffer after the parent view is released. There is no `valid_fields` mask or
flat-field compatibility API.

**Development protocol break:** version 2 replaces the former EGO diagnostic
layout. The EGO device-state golden is intentionally renewed; other schema
goldens and NVIDIA's compatibility checks are unchanged. Old files are retained,
but requesting their diagnostic tracker can fail ReplaySession setup. No old
diagnostic conversion is provided. The official monitoring schema remains
version 1 and is independent of the EGO diagnostic version.

## Recording modes

There are three recording workflows. `--mcap-filename` and
`--collection-prefix` are mutually exclusive.

| Workflow | Use it when | Media location |
|---|---|---|
| Raw media only | You only need video and/or WAV. | `.mjpg`, `.h264`, `.h265`, `.wav` files. |
| Plugin-local MCAP | You need camera timing and sensor data in one local MCAP. | Default: sidecar video, optional WAV, and `metadata.mcap`. |
| TeleopSession MCAP | You need camera data alongside hand, head, controller, or other trackers. | DeviceIO writes the shared MCAP; raw media remains sidecar by default. |

### Raw media only

Use the native plugin directly when no MCAP is needed:

```bash
EGO_PLUGIN="$PWD/build-ego-py3.11/src/plugins/ego_camera/ego_camera_plugin"
RUN="recordings/raw_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/raw"

timeout -s INT 30 "$EGO_PLUGIN" \
  --add-stream=camera=ColorLeft,output="$RUN/raw/ColorLeft.h264",format=h264,width=1600,height=1300,fps=30 \
  --add-stream=camera=ColorRight,output="$RUN/raw/ColorRight.h264",format=h264,width=1600,height=1300,fps=30
```

The command ends normally after 30 seconds. Check `ColorLeft.h264` and
`ColorRight.h264` are non-empty, then use the media checks in
[Inspect and play recordings](#inspect-and-play-recordings).
To capture optional sound, add `--audio-output="$RUN/Audio.wav"` to the native
command after reviewing the [audio limits](#optional-audio-and-synchronization-limits).

### Local metadata MCAP (default script behavior)

The quick-start `record` command uses this mode. It writes the lightweight
structured topics below while leaving high-throughput video as regular files:

```text
ego_metadata/ColorLeft, ego_metadata/ColorRight
ego_imu/Accel,           ego_imu/Gyro
ego_calibration/Calibration
ego_device/DeviceState
```

This mode avoids copying media bytes into an MCAP and keeps video playable
through standard tools. IMU topics require an available, enabled IMU.
Optional audio adds `ego_audio/Audio` indices and the separate `Audio.wav`.

### Self-contained private embedded MCAP

Use `embedded` when one private Ego MCAP must contain video bytes, metadata,
enabled IMU and optional PCM audio, calibration, and state. It does not
transcode the data.
During recording the file has a `.partial` suffix and is renamed only after a
valid MCAP Footer is written and all capture, queue drain, media close, control
restoration, and SchemaPusher cleanup steps succeed. A Footer alone does not
prove a successful capture; cleanup failures retain the `.partial` file.

```bash
RUN="recordings/embedded_$(date +%Y%m%d_%H%M%S)"
./src/plugins/ego/ego.sh record --duration 30 \
  --output "$RUN" --mcap-media embedded
./src/plugins/ego/ego.sh verify "$RUN"
```

In this mode no sidecar video/WAV is written unless you add
`--keep-media-sidecars`. The private media channels are:

```text
ego_media/ColorLeft
ego_media/ColorRight
```

Enabling audio adds `ego_media/Audio` with PCM payloads; audio is not enabled
by selecting `embedded`.

Export a completed embedded MCAP for ordinary playback:

```bash
./src/plugins/ego/ego.sh export-media "$RUN"
```

When `verify --plugin PATH` is used, the script requires
`ego_mcap_export_media` beside that plugin or in the same build tree's
`ego/mcap/export_media_tool` directory. It does not fall back to an exporter from
another preset or build tree.

The output is `ColorLeft.<format>` and `ColorRight.<format>`, plus `Audio.wav`
when audio was recorded.
A remaining `metadata.mcap.partial` means the capture was interrupted or
failed; do not treat it as a deliverable.

### TeleopSession: one MCAP with other trackers

Complete the normal full build and install its Python wheel
before using the EGO trackers or `isaaccapture.ego` helper. The wrapper's
`build` command builds the native plugin tools only:

```bash
cmake --build build-ego-py3.11 --parallel
cmake --install build-ego-py3.11 --prefix install-ego-py3.11
uv pip install --reinstall install-ego-py3.11/wheels/isaaccapture-*.whl
```

For multi-device recording, create trackers for the enabled EGO streams together
with your existing hand/head/controller Trackers, and give them channels in the
same `McapRecordingConfig`:

```python
from isaaccapture.deviceio_trackers import (
    EgoFrameMetadataTracker, EgoImuTracker,
    EgoCalibrationTracker, EgoDeviceStateTracker,
)
from isaaccapture.deviceio_session import McapRecordingConfig
from isaaccapture.schema import EgoCameraStream

prefix = "ego"
trackers = [
    EgoFrameMetadataTracker(
        prefix,
        [EgoCameraStream.ColorLeft, EgoCameraStream.ColorRight],
    ),
    EgoImuTracker(prefix),
    EgoCalibrationTracker(prefix + "/Calibration"),
    EgoDeviceStateTracker(prefix + "/DeviceState"),
]
tracker_names = list(zip(trackers, [
    "ego_metadata", "ego_imu", "ego_calibration", "ego_device",
]))

# Add these trackers to the same TeleopSession and McapRecordingConfig as the
# configured hand, head, and controller trackers.
```

When a pipeline supplies these trackers through DeviceIO sources, use the same
`ego_metadata`, `ego_imu`, `ego_calibration`, and `ego_device` source
names. Source names determine MCAP topic bases; `--collection-prefix` determines
OpenXR collection IDs independently.

Use plugin arguments such as:

```text
--collection-prefix=ego
--add-stream=camera=ColorLeft,output=<run>/raw/ColorLeft.h264,format=h264,width=1600,height=1300,fps=30
--add-stream=camera=ColorRight,output=<run>/raw/ColorRight.h264,format=h264,width=1600,height=1300,fps=30
--enable-imu
```

For optional audio, also import and configure `EgoAudioTracker(prefix + "/Audio")`
with recording/source name `ego_audio`, and add `--audio-output=<run>/Audio.wav`.
The tracker carries indices; the WAV contains the sound samples. The
[audio synchronization limits](#optional-audio-and-synchronization-limits) also
apply to TeleopSession recordings.

Set `shutdown_timeout_seconds=15.0` on this EGO `PluginConfig` to allow SDK stop,
queue drain, and control restoration before the process deadline. Other plugins
retain the default of two seconds.

This publishes structured Ego data through SchemaPusher; the TeleopSession
MCAP then contains Ego channels and any configured hand/head/controller
channels. It requires a running OpenXR runtime. See
`examples/oxr/python/test_ego_camera.py` for a camera-only session example.

Periodic device-state publication uses the identity captured when the device
was opened. It does not perform a synchronous SDK identity query in the capture
drain loop; disconnect detection remains covered by the SDK removal callback
and per-stream liveness checks. This avoids periodically stalling every
SchemaPusher stream while retaining same-device recovery behavior.

For a self-contained multi-device recording, use the EGO-only finalization
context around the ordinary `TeleopSessionConfig`. The existing pipeline and
hand/head/controller trackers stay in that config. Add exactly one enabled
`ego_camera` plugin with these arguments (absolute paths avoid the plugin
working-directory ambiguity):

```text
--collection-prefix=ego
--mcap-media=embedded
--mcap-media-spool=/absolute/run/ego_fragment.mcap
--add-stream=camera=ColorLeft,format=h264,width=1600,height=1300,fps=30
--add-stream=camera=ColorRight,format=h264,width=1600,height=1300,fps=30
--enable-imu
```

For optional embedded audio, add `--enable-audio` and the audio tracker described
above. It does not require a WAV sidecar.

```python
from pathlib import Path
from isaaccapture.ego import embedded_recording

run = Path("recordings/teleop_ego").resolve()
run.mkdir(parents=True, exist_ok=True)
# config is your live TeleopSessionConfig, with trackers/plugin args above.
config.mcap_config = McapRecordingConfig(
    str(run / "final.mcap"),
    tracker_names,
)
with embedded_recording(config, media_fragment=run / "ego_fragment.mcap") as session:
    try:
        while True:
            session.step()
    except KeyboardInterrupt:
        pass  # Exit the context normally to stop and finalize the recording.
```

The final name comes from `config.mcap_config.filename`. The context records
host tracker observations to `final.session.mcap`; the plugin writes a complete
structured-data plus media fragment. Only after the session and plugin
have stopped successfully does the adjacent `ego_mcap_merge` tool create
`final.mcap`. Canonical EGO topics come from the fragment, so the last captured
records do not depend on the host's final `session.step()`; overlapping host
observations use `session_observed/` topics in the merged file.

The final, session, and fragment paths must differ, their parent directories
must exist, and none of them or their `.partial` siblings may already exist.
`media_fragment` must match the plugin's `--mcap-media-spool`, resolved against
its working directory. Build/install `ego_mcap_merge` beside the plugin
before starting. A body exception, failed cleanup, or failed merge propagates
and leaves inputs or `.partial` output for diagnosis. Catch Ctrl+C inside the
context when you intend a clean stop; do not force-kill it.

Camera and IMU `get_stream_data(session, index)` return a payload or `None`.
Audio, calibration, and state use `get_data(session)` with the same contract.
Their collection IDs are the complete `prefix/Audio`, `prefix/Calibration`, and
`prefix/DeviceState` strings; payloads have no `.data` wrapper.

### Optional diagnostic recording

Diagnostics are recorded by default. To omit them from plugin-local MCAP:

```bash
./src/plugins/ego/ego.sh record --duration 30 \
  --output recordings/no_diagnostics --record-device-diagnostics=off
```

The TOML equivalent is `record_device_diagnostics = false` under `[record]`.
The choice is recorded in `requested_profile.json` and the MCAP `ego.recording`
Metadata, together with `diagnostic_version=2`. It removes only the diagnostic
schema/channel/messages; capture checks, recovery, logs, NV monitoring, and
live diagnostic publication remain active.

For a normal NV session, validate a copied configuration before opening it:

```python
from isaaccapture.ego import configure_recording
from isaaccapture.teleop_session_manager import TeleopSession

config = configure_recording(config, record_device_diagnostics=False)
with TeleopSession(config) as session:
    session.step()
```

For embedded recording, pass the same keyword to `embedded_recording`.
The helper accepts exactly one enabled EGO plugin. An explicit Python choice
wins over effective manifest/instance plugin arguments; otherwise the default
is on. With diagnostics off, remove `EgoDeviceStateTracker` from the explicit
MCAP tracker-name list and from automatically recorded DeviceIO sources in the
main pipeline, control pipeline, and sinks. The helper rejects these conflicts
before capture instead of changing NVIDIA's automatic recording behavior.
A manual live-only tracker can remain. Raw `TeleopSession` does not run this
EGO-specific precheck unless you call `configure_recording` first.

The helper also retains a recorded `EgoCalibrationTracker`, or adds one through
the existing `McapRecordingConfig` interface when it is missing. Calibration
contains the optional timing snapshot as well as spatial calibration. Use
`--collection-prefix` and the canonical recording names shown above for EGO
offline reading; custom calibration names remain valid for NV recording but
are outside the offline reader's canonical topic layout.

The merger checks both the parent session and the capture fragment. An off
policy rejects diagnostic channels by schema identity even under custom topic
names or with no messages. Conflicting policies fail too; inputs and partial
outputs remain available for diagnosis. Canonical fragment tail data and
`session_observed/` host observations retain their existing merge behavior.
The replay example reads the recorded policy and omits the diagnostic tracker
when it is off; `--record-device-diagnostics on|off` checks an expected policy.

### Replay structured Ego data

Recordings use `Ego` schema types and `ego` topics; former Orbbec development
interfaces and recordings are unsupported.

From an environment containing the built or installed `isaaccapture` Python
package, inspect a recording containing RGB, IMU, and audio headlessly through
`ReplaySession`. This example requires an Audio channel; point `AUDIO_RUN` to
an audio-enabled recording rather than the default quick-start recording:

```bash
uv pip install -e ./examples/mcap_record_replay
AUDIO_RUN=/absolute/path/to/recording_with_audio
python -m isaaccapture_examples.mcap_record_replay.replay_ego \
  "$AUDIO_RUN/metadata.mcap"
```

The example reads `ColorLeft`/`ColorRight`, `Accel`/`Gyro`, audio,
calibration, and enabled diagnostic events, prints their capture epochs and key
fields, and exits nonzero if any expected channel is empty. Use
`--max-updates N` for a short inspection. This replays structured metadata and
sensor records; use FFmpeg or `export-media` to decode and view video payloads.
The final MCAP produced by `isaaccapture.ego.embedded_recording` can also be
passed to this command.

The example reports a missing Audio channel for an audio-disabled recording.
That result does not make the dataset incomplete for its selected streams.
Use `verify` for recording integrity and the selected-stream offline interface
below for sample association.

### Sample-time alignment and offline reading

NV `ReplaySession.update()` advances each configured tracker to its next
record. One update is not a synchronized video/IMU/audio bundle. Keep
`replay_ego` for structured replay and use `isaaccapture.ego.sync` when a
consumer needs samples associated by capture time. Stereo preview and XR
display continue using their existing left/right pairing.

Streams are recorded independently at their own rates. Raw timestamps and
host arrival times remain distinct. The optional `EgoCalibration.time_alignment`
snapshot records clock sources, device/profile identity, model version, epoch
and mapping provenance through the normal calibration channel. Optional
`EgoAudioChunk.timing` and `EgoPcmAudioChunk.timing` record a continuous segment
ID, the retained block's first sample index, and observed discontinuities.
These are EGO payload extensions within NV's existing transport and MCAP
format; the public timestamp and IMU sample layout are unchanged.

Without a validated model, local sample times retain the arrival-time fallback
and the recording is marked unverified. `time_alignment.sources.audio` is
`sdk_audio_raw`: it identifies the raw `AudioFrame.getTimeStampUs()` value
returned by the SDK, without claiming a hardware clock or a block-start reference.
Its semantics depend on the transport and SDK version. The audited USB path
estimates audio time from host time and sample count; the audited Ethernet path
converts an RTP timestamp. The network clock rate, sample reference and relationship
to camera/IMU time still require verification. Existing recordings can retain the
older `sdk_host_estimated_block_start` label; that label alone does not establish
their transport or clock semantics.

Retained sample indices do not prove that the device lost no samples. The adapter
therefore records `continuity_known=false`; an offline audio model needs separate
continuity evidence before its windows can be complete.

For an already validated reusable RGB/IMU calibration, pass it before capture:

```bash
./src/plugins/ego/ego.sh record --duration 30 -- \
  --time-alignment=/absolute/path/capture_profile.json
```

The version-1 file has `scope="capture_profile"`, `status="validated"`, a
`model_id`, `reference_stream="left_rgb"`, exact `identity` (`serial`,
`firmware`, `sdk_version`, `sdk_binary_sha256`), and exact `profile` (`video` format/width/height/fps,
`imu_rate_hz`, `audio` sample_rate_hz/channels/bits_per_sample). Each enabled
RGB/IMU entry in `streams` contains `offset_ns` and `scale=1`; left RGB
must be the identity mapping. Recording-bound models can fit drift; their
nonunit scales cannot be rebased onto a later capture's unknown clock origin.
Relative time is:

```text
t_ref = left_epoch_origin + scale * (raw_time - left_epoch_origin) + offset_ns
```

`validation` supplies `method`, `evidence`, `max_abs_error_ns`, `uncertainty_ns`
and `held_out_duration_ns`. The independent duration must be at least 15 minutes
and error plus uncertainty must be at most half a configured video frame.
`max_capture_duration_ns` limits use from the first archived left frame and
cannot exceed the validated duration. After expiry the adapter preserves raw
data, records the boundary and returns to unverified arrival-time fallback;
offline reads do not extend the calibrated interval into that tail. The plugin
hashes the SDK library actually loaded by its process before validating identity.
The plugin rejects malformed files before creating capture outputs and rejects
identity/profile mismatches before starting capture pipelines. Files are
limited to 8 KiB. A recording-bound file with old epoch/audio anchors cannot be
reused as a capture profile; the recording calibration tool below does not
automatically produce a reusable profile.

The adapter anchors NV local time at the first archived left frame's host
arrival. This provides a common local representation of the validated EGO
relative clock; its absolute host-clock accuracy remains unverified. It does
not validate alignment to another device. Audio still uses arrival-time
fallback during capture. Updated calibration snapshots remain within the
existing 32 KiB transport limit. A reconnect or observed clock discontinuity
opens another epoch/segment instead of interpolating across it.
Snapshots also identify the first mapped raw sample in each stream; an earlier
arrival-time fallback belongs to a different timestamp mode.

Install the offline dependency into an environment containing the built
`isaaccapture` package, then select only the streams your consumer needs:

```bash
uv pip install 'isaaccapture[ego]'
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
  "$RUN" --required-streams left_rgb right_rgb accel gyro \
  --output "$RUN/sync_audit.json"
```

The reader accepts the standard sidecar directory (`metadata.mcap`,
`raw/ColorLeft.<format>`, `raw/ColorRight.<format>`, and `Audio.wav` when audio
was recorded) or a completed embedded MCAP. Metadata-only MCAP without
accessible media is not a complete
dataset. Merged recordings consume canonical EGO channels and ignore
`session_observed/` duplicates. MCAP `logTime` and latest-value tracker getters
are not sample-time sources.

```python
from isaaccapture.ego.sync import iter_sync_windows

for window in iter_sync_windows(
    run,
    required_streams=("left_rgb", "right_rgb", "accel", "gyro"),
    alignment=run / "alignment.json",  # Explicit recording-bound calibration.
):
    if window.complete:
        consume(window.left_rgb, window.right_rgb, window.accel, window.gyro)
    else:
        record_gap(window.start_time_ns, window.reasons)
```

For a recording with audio and an independently validated audio model, add
`"audio"` to `required_streams` to receive `window.audio` sample slices. The
known timestamp anomaly does not make such a model available automatically;
see the [audio limits](#optional-audio-and-synchronization-limits).

Consecutive continuous reference frames define half-open windows
`[t_k, t_{k+1})`. Windows contain video media references, every IMU sample in
the interval, audio sample slices, aligned time, model ID and completeness
reasons. `reference_stream="right_rgb"` selects right-frame windows while the
model still uses the left camera's canonical clock. `epoch`, `start_time_ns`
and `end_time_ns` limit selection; only complete reference-frame intervals
inside a requested range are returned. These bounds are aligned nanoseconds.
Missing data returns incomplete windows without stale samples or fabricated
fills. Missing required streams or a trusted model raises an error. The audit
also lists tails, unpaired frames, unassigned sample ranges and gaps.

A valid model captured with the recording is the default. After-recording
calibration must be supplied explicitly as `alignment`; it does not rewrite
the original MCAP. Recording-bound models use `scope="recording"` and contain
device/profile identity, MCAP/media SHA-256 fingerprints, independent
validation and nonoverlapping epoch validity ranges. RGB/IMU mappings contain
integer `source_anchor_ns`, `reference_anchor_ns`, positive `scale`, and
`validation_start_ns`/`validation_end_ns` in the reference clock. Effective
RGB/IMU intervals must stay within those independent validation bounds. Audio
segments contain `segment_id`, `first_sample_index`, `sample_count`,
`reference_anchor_ns`, fitted `period_ns` and an evidence-backed
`continuity_verified` boolean, and the same validation bounds. Audio slices
outside the validated segment interval remain incomplete. They express:

```text
t_sample = reference_anchor_ns + period_ns * (sample_index - first_sample_index)
```

Prepare independently annotated physical events, then fit a recording-bound
model without opening a device:

```bash
python -m isaaccapture_examples.mcap_record_replay.calibrate_ego_sync \
  "$RUN/events.json" --output "$RUN/alignment.json"
python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
  "$RUN" --required-streams left_rgb right_rgb accel gyro \
  --alignment "$RUN/alignment.json" --output "$RUN/rgb_imu_audit.json"
```

The version-1 event manifest names `model_id`, `recording`, optional
`media_paths`, `identity`, `profile` and `epochs`. Each epoch names `epoch`,
`valid_from_ns`, `valid_to_ns` and `streams`. A fitted RGB/IMU stream provides
`calibration_events`, separate `validation_events`, `validation_start_ns` and
`validation_end_ns`. Each event provides `event_id`, `source_value` (raw ns for
RGB/IMU, sample index for audio), `reference_time_ns`, `uncertainty_ns`, and an
`evidence` file path. Audio uses `streams.audio.segments`, adding
`segment_id`, `first_sample_index`, `sample_count`, and optional
`continuity_verified` (default false) to those event fields. Manifest paths
resolve relative to the manifest directory; evidence files are hashed.

At least three calibration events and three distinct validation events are
required per fit. Held-out events cover the beginning, middle, and end of a
15-minute interval. Results apply to the validated duration; longer recordings
need longer independent evidence. Uncertainty includes exposure reference, event detection
and light/sound source delay. No fixed observed offset or factory `time_offset`
is applied without verifying its units, direction and prior compensation.
The tools do not detect physical events or manufacture acceptance evidence.
The Python calibration helpers are also available from `isaaccapture.ego`.

Both commands refuse to overwrite outputs. They preserve an unverified or
incomplete JSON result and exit nonzero; a successful selected-stream audit
applies only to its returned windows. The audit stores compact record/sample
ranges without copying all IMU XYZ values into JSON. Physical measurements
must still accompany the model; software association alone is not physical
acceptance.

## Inspect and play recordings

Elementary video has no container index. Validate it first, then remux it into
MP4 for a desktop player:

```bash
# Replace RUN with a real recording directory.
ffmpeg -v error -f h264 -i "$RUN/raw/ColorLeft.h264" -f null -
ffmpeg -v error -f h264 -i "$RUN/raw/ColorRight.h264" -f null -
ffmpeg -y -f h264 -framerate 30 -i "$RUN/raw/ColorLeft.h264" -c:v copy "$RUN/ColorLeft.mp4"
xdg-open "$RUN/ColorLeft.mp4"
```

If audio was recorded, inspect and listen to the WAV separately:

```bash
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
  -of default=noprint_wrappers=1 "$RUN/Audio.wav"
xdg-open "$RUN/Audio.wav"
```

Opening video and WAV separately does not demonstrate synchronized playback.

The plugin removes only recognized Ego timestamp-only SEI units, including
the legacy `ORBBEC,EGO_` form and firmware 1.1.1's `EGO..._[LR],timestamp_us=...`
form. Other SEI data is preserved. Do not work around a decode failure by
stripping all SEI; preserve the run and capture log for diagnosis.

For H.265 replace `h264` with `hevc`; for MJPEG use:

```bash
ffmpeg -f mjpeg -framerate 30 -i "$RUN/raw/ColorLeft.mjpg" \
  -c:v libx264 -pix_fmt yuv420p "$RUN/ColorLeft.mp4"
```

MCAP is structured recording data, not a generic video player. List its topics
with the standard reader:

```bash
python3 - "$RUN/metadata.mcap" <<'PY'
from collections import Counter
from pathlib import Path
import sys
from mcap.reader import make_reader

path = Path(sys.argv[1])
with path.open("rb") as stream:
    counts = Counter(channel.topic for _, channel, _ in make_reader(stream).iter_messages())
for topic in sorted(counts):
    print(f"{topic}: {counts[topic]}")
PY
```

## Device controls

Always inspect the connected device first. Property availability, range, and
step are firmware- and active-profile-specific:

```bash
./src/plugins/ego/ego.sh capabilities | tee capabilities.txt
```

The capability listing is a discovery snapshot, not a guarantee that a range
is unchanged after another profile is activated. When controls are requested,
the plugin resolves the target profiles, establishes their range context, and
then validates permissions, type, range, step, and readback. It restores
changed properties only after capture pipelines have stopped unless
`--persist-controls` is explicitly supplied.

```bash
./src/plugins/ego/ego.sh record --duration 20 -- \
  --set-property=OB_PROP_COLOR_AUTO_EXPOSURE_BOOL=0 \
  --exposure=1000 --gain=100 \
  --set-property=OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL=0 \
  --white-balance=5000 --sharpness=20 --saturation=128 --contrast=50 \
  --power-frequency=1
```

Supported friendly options are `--exposure`, `--gain`, `--white-balance`,
`--brightness`, `--sharpness`, `--saturation`, `--contrast`, and
`--power-frequency`. `--set-property=SDK_PROPERTY_NAME=VALUE` is the escape
hatch for other writable SDK properties. Ego has no OAK-equivalent `--quality`;
use `--bitrate` and `--dynamic-bitrate` instead.

Do not use `--persist-controls` for routine testing. It intentionally leaves
the selected values on the physical device.

## Preview and GPU stereo

Add `--preview` to a native plugin command or the one-command script to open a
side-by-side SDL preview while recording:

```bash
./src/plugins/ego/ego.sh record --duration 30 --preview
```

The preview decodes on the CPU and can drop stale display frames without
changing recording timestamps.

For GPU-resident frames, use the independent `camera_viz` source:

```bash
examples/camera_viz/camera_viz.sh setup \
  --with-ego --sdk-root "$ORBBEC_SDK"
examples/camera_viz/camera_viz.sh run \
  examples/camera_viz/configs/ego.yaml --mode window
```

The config defaults to H.264 `1600x1300@30`. Set `format` to `h265` or `mjpg`
and choose an advertised stereo profile. H.264/H.265 requires a working CUDA
toolkit and the native NVDEC codec; MJPEG uses SDK RGB conversion and CuPy for
GPU upload.
EGO's `gpu_id: 0` means automatic selection: the viewer supplies its actual
CUDA device ID, shared by both eyes and decoders. A nonzero pin must match that
device or startup fails before opening EGO. Capture-open/read failures retain
their retry behavior; unhandled producer failures propagate through NVIDIA's
`latest()` / `VizRunner.wait()` interfaces and stop the viewer with an error.
The SDK's udev rules must be installed before opening the camera as a non-root
user. See the [EGO preview binding guide](../../../examples/camera_viz/ego_preview/README.md)
for binding builds, API, and GPU data flow.

`camera_viz` and the recording plugin both open the physical camera; do not run
them against the same EGO at the same time. For desktop stereo inspection and
XR output options, see the [viewer guide](../../../examples/camera_viz/README.md#mode-1--direct).

## Source layout

| Directory | Responsibility | CMake target |
|---|---|---|
| `core/` | Camera capture, device controls, recording sinks, and optional preview. | `ego::camera_core` |
| `app/` | Camera plugin command-line entry point. | `ego_camera_plugin` |
| `mcap/io/` | Shared MCAP reader/writer implementation and checked writes. | `ego::mcap_support` |
| `mcap/merge/` | Reusable session and capture-fragment merge logic. | `ego::mcap_merge` |
| `mcap/merge_tool/` | MCAP merge command-line entry point. | `ego_mcap_merge` |
| `mcap/export_media_tool/` | Embedded video and audio export tool. | `ego_mcap_export_media` |
| `tools/` | Python configuration and recording-verification helpers. | — |

`mcap/CMakeLists.txt` groups the four MCAP targets, which build without the
Orbbec SDK. The plugin and both native tools are installed together under
`plugins/ego`. Library `inc/` directories contain headers shared with other
targets; private headers stay beside their sources.

## Native CLI reference

The script covers normal work. Use the executable directly only for custom
workflows:

```bash
EGO_PLUGIN="$PWD/build-ego-py3.11/src/plugins/ego_camera/ego_camera_plugin"
"$EGO_PLUGIN" --help
"$EGO_PLUGIN" --list-capabilities
```

Important options:

| Option | Meaning |
|---|---|
| `--add-stream=camera=...,output=...[,format=...,width=...,height=...,fps=...]` | Add one ColorLeft or ColorRight stream. Stream values override global values. |
| `--device-uid=UID` | Select one enumerated camera. |
| `--reconnect-timeout=N` | Wait N seconds for the same physical device after disconnect; 0 disables recovery. |
| `--reconnect-interval-ms=N` | Set the device re-enumeration interval; default 1000 ms. |
| `--enable-imu --imu-rate=400\|1000` | Capture accelerometer and gyroscope when available. |
| `--enable-audio` | Explicitly enable optional PCM audio; no WAV path is needed for embedded capture. |
| `--audio-output=PATH.wav` | Explicitly enable optional PCM audio and write a WAV sidecar. |
| `--calibration-output=PATH.json` | Export structured calibration and original SDK YAML. |
| `--time-alignment=PATH` | Load a validated, identity/profile-matched reusable RGB/IMU capture profile. |
| `--mcap-filename=PATH` | Write a plugin-local MCAP. |
| `--collection-prefix=PREFIX` | Publish data for a TeleopSession. |
| `--mcap-media=metadata-only\|embedded` | Select sidecar or private self-contained local media handling. |
| `--keep-media-sidecars` | Also write raw video/WAV in embedded mode. |
| `--mcap-media-spool=PATH` | Required for embedded TeleopSession media merging. |
| `--record-device-diagnostics=on\|off` | Record EGO diagnostic events; default on. |
| `--plugin-root-id=ID` | Launcher-owned identity for NV device monitoring. |

`--list-capabilities` reports SDK and firmware identity, serial and UID, USB
connection, global-timestamp support, per-sensor advertised profiles, and
property ranges. The wrapper adds plugin and resolved SDK-library hashes so a
recording can distinguish SDK builds that share a version number. Global
timestamp is not synthesized when the device reports it unsupported; recorded
frames retain host monotonic arrival time and raw device timestamp separately.
Validated EGO relative mappings may produce local sample times through the
adapter; neither global time nor absolute host sampling accuracy is inferred.

## Build, installation, and review checks

The script uses the equivalent CMake commands below. Use them when integrating
the plugin into an existing build or CI job:

```bash
cmake -S . -B build-ego-py3.11 \
  -DISAAC_TELEOP_PYTHON_VERSION=3.11 \
  -DBUILD_VIZ=OFF \
  -DBUILD_PLUGIN_EGO_CAMERA=ON \
  -DORBBEC_SDK_ROOT="$ORBBEC_SDK"
cmake --build build-ego-py3.11 \
  --target ego_camera_plugin ego_mcap_export_media ego_mcap_merge --parallel
```

After installation, `ego.sh` is installed beside the Ego plugin and
supports `capabilities`, `record`, and `verify`. Its `build` command is only
for a source checkout.

The plugin uses the ordinary source build and install layout. The SDK remains
an explicit external dependency.

For SDK-free media tools and synthetic MCAP tests, configure separately:

```bash
cmake -S . -B build-ego-media-py3.11 \
  -DISAAC_TELEOP_PYTHON_VERSION=3.11 -DBUILD_VIZ=OFF \
  -DBUILD_PLUGIN_EGO_CAMERA=OFF -DBUILD_EGO_MEDIA_TOOLS=ON
cmake --build build-ego-media-py3.11 \
  --target ego_mcap_export_media ego_mcap_merge ego_mcap_tests --parallel
```

### Hardware-free software CI

The [software CI workflow](../../../.github/workflows/test-ego-software.yaml)
builds with an external SDK and runs without an attached camera. After building
with the SDK, run the same CTests locally:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  UV_CACHE_DIR=/tmp/isaaccapture-uv-cache \
  ctest --test-dir build-ego-py3.11 --output-on-failure --parallel \
  --no-tests=error -R '[Ee]go'
```

For viewer configuration tests, run
`uv run --python 3.11 --extra dev pytest` from
`tests/python/examples/camera_viz`.

From the repository root, run the review checks:

```bash
# Stage newly created project files first; --all-files enumerates tracked files.
SKIP=check-copyright-year pre-commit run --all-files
git diff --check
```

Software tests do not validate physical capture or XR. On the intended SDK,
firmware, and profile, verify completed recordings and exercise clean stop,
disconnect recovery, preview, and multi-device sessions.

Do not commit `recordings/`, `build/`, SDK extracts, `Log/`, or IDE files.

## Troubleshooting

| Symptom | Check |
|---|---|
| `doctor` reports no camera or no USB access | Install the selected SDK's udev rules, reconnect the camera, then run `doctor` and `capabilities` again. |
| Requested profile fails | Run `capabilities`; request one exact advertised profile per sensor. A profile may still fail as a simultaneous combination; the plugin does not fall back. |
| Recorded profile does not match `requested_profile.json` | Preserve the run and log. The requested profile was not delivered exactly; do not accept it as a silent substitute. |
| Embedded `verify --plugin PATH` cannot find an exporter | Build or install `ego_mcap_export_media` beside that plugin or in the same CMake build tree; do not mix artifacts from different builds. |
| `verify` reports no MCAP reader | Run `python3 -m pip install --user mcap`. |
| Video does not open in a desktop player | Validate/remux it with the FFmpeg commands above; elementary streams are not MP4 files. |
| `.partial` MCAP remains | The recording did not close cleanly. Preserve its log for diagnosis and do not publish it as a completed capture. |
| Capture waits after a disconnect | Reconnect the same physical device before `--reconnect-timeout` expires. The log reports any UID change, each attempt, and the resumed capture epoch. |
| Reconnect times out | Preserve the log and `.partial` MCAP for diagnosis, reconnect the camera, and start a new run. |
| An advertised encoded profile above 30 FPS is rejected | H.264/H.265 above 30 FPS is disabled. Choose an advertised profile at or below 30 FPS. |
| Preview is delayed | It is a CPU convenience preview. Close other GPU/camera applications or use `camera_viz` for the GPU path. |
| SchemaPusher exits immediately | A running OpenXR runtime is required; an old `cloudxr.env` file alone is insufficient. |
| SchemaPusher latency spikes at a regular multi-second interval | Confirm the plugin includes the nonblocking periodic device-state path; synchronous SDK identity polling in the capture drain loop can stall all streams. |
