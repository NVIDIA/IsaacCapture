.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

EGO Camera Plugin
=================

The ``ego_camera`` plugin connects EGO stereo cameras to Isaac Capture.
It records ColorLeft and ColorRight video, and captures accelerometer,
gyroscope, microphone, calibration, and device state when the device exposes
them. Detailed operator commands live in
:code-file:`src/plugins/ego/README.md`.

.. contents:: On this page
   :local:
   :depth: 2

Supported streams and profiles
------------------------------

The plugin captures ``ColorLeft`` and ``ColorRight`` in MJPEG, H.264, or H.265.
The recording wrapper defaults to stereo H.264 ``1600x1300@30``. Use
``capabilities`` to inspect the connected device's available profiles;
unsupported simultaneous combinations fail without a fallback. The current
plugin rejects H.264/H.265 profiles above 30 FPS and does not capture depth or
infrared streams.

Installation
------------

Use an extracted Linux OrbbecSDK v2 package and the normal Isaac Capture source
build dependencies, including ``nlohmann-json3-dev`` for the EGO adapter.
Install the SDK package's udev rules and reconnect the
camera before using it as a non-root user. The SDK is an external dependency;
the :code-file:`plugin README <src/plugins/ego/README.md>` describes its expected
layout and advanced build options.

From the repository root:

.. code-block:: bash

   export ORBBEC_SDK_ROOT=/path/to/OrbbecSDK
   ./src/plugins/ego/ego.sh doctor --sdk-root "$ORBBEC_SDK_ROOT"
   ./src/plugins/ego/ego.sh build --sdk-root "$ORBBEC_SDK_ROOT" --jobs 8
   ./src/plugins/ego/ego.sh capabilities

Preview is optional and requires SDL2 and FFmpeg development libraries; enable
it with the wrapper's ``build --preview`` flag. Recording alone needs neither.

Capture and verify
------------------

The wrapper records the default stereo profile with available accelerometer
and gyroscope streams. Audio is off by default. Give each recording a fresh
output directory:

.. code-block:: bash

   RUN="recordings/ego_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/ego/ego.sh record --duration 30 --output "$RUN"
   ./src/plugins/ego/ego.sh verify "$RUN"

Raw video is an elementary MJPEG, H.264, or H.265 stream. Metadata, IMU,
calibration, and state are recorded in MCAP. The wrapper also preserves profile
and software/device provenance.

Enable optional audio with ``--audio`` or ``audio = true`` in a TOML config;
``--no-audio`` overrides a config that enables it. Audio is saved as 48 kHz mono
S16_LE WAV, with audio offsets recorded in MCAP:

.. code-block:: bash

   ./src/plugins/ego/ego.sh record --duration 30 --audio \
     --output recordings/ego_with_audio

Raw PCM capture, storage and export are supported. The tested Ethernet path
has an SDK audio timestamp cadence anomaly, so audio/video and five-stream
physical synchronization have not passed acceptance. The recommended capture
workflow uses stereo RGB and IMU; including audio in synchronization requires
an independently validated audio time model. See "Optional audio and
synchronization limits" in the
:code-file:`plugin README <src/plugins/ego/README.md>`.

For one self-contained MCAP with encoded video and any enabled audio PCM:

.. code-block:: bash

   RUN="recordings/ego_embedded_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/ego/ego.sh record --duration 30 \
     --output "$RUN" --mcap-media embedded
   ./src/plugins/ego/ego.sh verify "$RUN"
   ./src/plugins/ego/ego.sh export-media "$RUN"

Embedded recording preserves the encoded media without transcoding. Export
produces ordinary video files and, when audio was enabled, WAV. A remaining
``.partial`` file means the capture did not finish successfully; preserve it
and the log for diagnosis.

Data flow and published collections
-----------------------------------

The plugin writes captured media and sensor records and can publish structured
data through SchemaPusher. DeviceIO trackers read the published collections
during session updates.

With ``--collection-prefix=ego``, the collections and Python read APIs are:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Collection
     - Python tracker
     - Read method
   * - ``ego/ColorLeft``, ``ego/ColorRight``
     - ``EgoFrameMetadataTracker``
     - ``get_stream_data``
   * - ``ego/Accel``, ``ego/Gyro``
     - ``EgoImuTracker``
     - ``get_stream_data``
   * - ``ego/Audio``
     - ``EgoAudioTracker``
     - ``get_data``
   * - ``ego/Calibration``
     - ``EgoCalibrationTracker``
     - ``get_data``
   * - ``ego/DeviceState``
     - ``EgoDeviceStateTracker``
     - ``get_data``

All reads return the schema payload directly or ``None`` when unavailable.
Audio records are available only when audio capture is enabled.
Camera and IMU readers select one configured stream by its index. Camera streams
use ``EgoCameraStream`` enums; IMU sensors default to Accel and Gyro. See the
:code-file:`plugin README <src/plugins/ego/README.md>` for the full tracker
configuration example.

Availability and diagnostics
----------------------------

The plugin integrates with :doc:`NV device monitoring <../references/device_provider_monitoring>`.
After ``session.step()``, query ``session.get_device_status(plugin_root_id +
"/camera/ego")`` for current availability. The provider ID is
``plugin/<plugin_root_id>``. The existing capture checks report unknown, absent,
starting/recovering, streaming, or failed conditions; NVIDIA owns provider
failure/teardown and the one-second heartbeat / three-second stale-report rule.
No additional device polling thread is introduced.

``EgoDeviceStateTracker`` carries historical diagnostic events, separately from
current availability. Version-2 payloads have ``device_uid``, ``capture_epoch``,
``observed_at_ns`` and one typed ``event``:

- ``EgoSdkReport``: original SDK mode, flags, storage and report sequence.
- ``EgoCaptureSummary``: capture health, queue/loss totals and final-summary flag.
- ``EgoRecoveryEvent``: recovery stage, cause and attempt count.
- ``EgoDeviceSnapshot``: successful startup property reads and optional IMU temperature.

An unavailable temperature is ``None``; a measured zero remains ``0.0``. Startup
samples are not repeated as live periodic measurements. These event views are
read-only and retain the backing buffer. This development-stage layout replaces
the previous diagnostic protocol: old diagnostic recordings are unsupported,
and the official Replay compatibility check remains enabled.

Diagnostic MCAP recording defaults to on. Use
``--record-device-diagnostics=off`` or the TOML
``record_device_diagnostics = false`` option to omit it. Monitoring, live
publication, capture checks, recovery and logs continue. For NV sessions use
``isaaccapture.ego.configure_recording(config, record_device_diagnostics=False)``
or the same keyword on ``embedded_recording``. Remove diagnostic trackers from
explicit recording lists and automatically recorded pipeline/control/sink
sources; the helper rejects conflicts before capture. The merger checks both
inputs, including custom or empty diagnostic channels. See the plugin README
for complete configuration and failure handling.

Multi-device sessions
---------------------

Add the EGO trackers to a normal
:doc:`TeleopSession <../getting_started/teleop_session>` configuration and its
``McapRecordingConfig`` tracker-name list, alongside the hand, head, or
controller trackers. Configure ``ego_camera`` through ``PluginConfig`` with
``--collection-prefix=ego`` and ``shutdown_timeout_seconds=15.0`` to allow
capture cleanup. Live publication requires a running OpenXR runtime.
Complete the normal full source build and install its Python wheel before
using the EGO tracker bindings and recording helper; the wrapper's
``build`` command builds only the native plugin tools.

Call ``isaaccapture.ego.configure_recording`` before opening a normal NV
recording session. It retains or adds the calibration tracker through the
existing recording configuration, so EGO time-model snapshots are recorded.
``embedded_recording`` performs this configuration as part of its existing
finalization workflow.

For one self-contained multi-device MCAP, use
``isaaccapture.ego.embedded_recording`` with exactly one EGO plugin configured
for embedded media. The full configuration, matching media-spool path, and
clean-stop procedure are documented under "TeleopSession: one MCAP with other
trackers" in the :code-file:`plugin README <src/plugins/ego/README.md>`.

Replay and visualization
------------------------

The current structured replay example requires camera, IMU, audio, calibration
and any enabled diagnostic channels. Use an audio-enabled recording such as the
one above:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   python -m isaaccapture_examples.mcap_record_replay.replay_ego \
     recordings/ego_with_audio/metadata.mcap

A recording without audio is valid; this example reports a missing Audio
channel because of its fixed tracker configuration. It reads structured
records and does not play sound or schedule synchronized media playback. Use
``export-media`` followed by FFmpeg for encoded media playback. For a live GPU
preview, build the standalone camera_viz binding:

.. code-block:: bash

   examples/camera_viz/camera_viz.sh setup --with-ego \
     --sdk-root "$ORBBEC_SDK_ROOT"
   examples/camera_viz/camera_viz.sh run \
     examples/camera_viz/configs/ego.yaml --mode window --stereo-debug sbs

MJPEG uses SDK conversion and GPU upload; H.264/H.265 uses native NVDEC. Window
SBS is an optional left/right inspection image and is refused in XR mode.
XR mode submits separate eye textures using the existing visualization API.
The viewer and recording plugin each open the camera; run them one at a time.

Sample-time association
-----------------------

NV replay advances each tracker to its next record; a single update does not
automatically align video, IMU and audio. The EGO offline API associates only
the streams required by a consumer. It preserves the existing replay lifecycle
and introduces no video/audio player.

Capture retains original timestamps and host arrival times separately. An
optional ``--time-alignment=PATH`` loads a validated reusable RGB/IMU model
bound to the device identity and actual stream profile. EGO calibration payloads
carry clock provenance and model snapshots through the normal NV recording
channel. Audio payloads carry retained sample indices and segment information.
Reusable profiles require the loaded SDK's SHA-256, unit scale and a verified
duration limit. Expiry preserves capture data with an explicit arrival-time
fallback; offline reads keep the validated interval separate from that tail.
Without a trustworthy model the recording remains unverified. An estimated
host-clock origin does not establish absolute host sampling accuracy or
synchronization with other devices.

The audio source label ``sdk_audio_raw`` identifies the SDK's original
timestamp; its sampling-clock semantics depend on the SDK transport and do
not alone establish a common camera/IMU clock. Audio alignment needs an
independently validated sample-time model, including offset and sample-clock
drift. Retained audio sample indices do not prove physical continuity; this
needs separate evidence. Reconnects and detected clock discontinuities are
segmented, and the reader does not interpolate across these boundaries.

Install the offline extra into an environment containing the built package:

.. code-block:: bash

   uv pip install 'isaaccapture[ego]'
   uv pip install -e ./examples/mcap_record_replay
   python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
     "$RUN" --required-streams left_rgb right_rgb accel gyro \
     --output "$RUN/sync_audit.json"

The recommended audit selects stereo RGB and IMU. Include ``audio`` only for
recordings with accessible audio media and an independently validated audio
time model; selecting the stream does not establish physical synchronization.

Use ``isaaccapture.ego.sync.iter_sync_windows`` for Python consumers. Consecutive
reference video frames define half-open time windows containing frame media
references, all interval IMU samples, audio sample slices when requested,
model identity and completeness reasons. The default reference is left RGB;
right RGB is also selectable. Missing data remains incomplete without stale
fills. An absent required stream or trusted time model is an error. The audit
retains unpaired frames, tails and gaps.

Standard sidecar recording directories and embedded MCAP are supported.
Metadata-only files need their accessible video sidecars and, when audio was
recorded, their WAV sidecar to form a complete dataset. Merged recordings use
canonical EGO topics and ignore duplicated
session observations. MCAP log times and latest-value tracker getters are not
used as sample times.

A model recorded during capture is selected by default. For calibration made
after recording, pass ``--alignment PATH`` explicitly; the model must match
the recording/media fingerprints, identity and valid intervals. The
``calibrate_ego_sync`` example fits such a model from separately annotated
calibration and validation events and preserves evidence hashes. It does not
detect physical events or rewrite the MCAP. Both offline commands refuse to
overwrite output files and return nonzero for unverified/incomplete results.

Physical acceptance requires independent beginning/middle/end measurements
over at least 15 minutes. Results apply to the validated duration. Corrected
maximum error plus measurement uncertainty
must be no more than half a configured video frame. Successful software
association alone does not establish that bound. Detailed model and event
manifest formats are in the
:code-file:`plugin README <src/plugins/ego/README.md>`.

Troubleshooting
---------------

- **No camera or USB access:** install the selected SDK's udev rules, reconnect,
  and re-run ``doctor`` and ``capabilities``.
- **Profile rejected:** choose an exact advertised left/right combination;
  the current plugin rejects H.264/H.265 profiles above 30 FPS.
- **Partial MCAP:** the recording did not finish successfully. Preserve it and
  the log for diagnosis before starting a new recording.
- **Publisher startup failure:** check the running OpenXR runtime and plugin
  search path.
- **No embedded final output:** check normal context exit, committed fragment,
  matching spool path, and the merge tool beside the selected plugin.
- **Native decoder unavailable:** build camera_viz's codec with CUDA, or choose
  an advertised MJPEG profile for that preview.
