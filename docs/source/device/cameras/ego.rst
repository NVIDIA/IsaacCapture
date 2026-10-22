.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

EGO Camera Plugin
=================

Orbbec EGO is a stereo camera module for egocentric data collection. The
``orbbec_ego_camera`` C++ plugin records its ``ColorLeft`` and ``ColorRight`` video
streams, with optional accelerometer, gyroscope, and microphone data. It also
records calibration and device diagnostics.

The plugin supports standalone recording and live data publication to an
Isaac Capture session. Source code and detailed configuration are in
:code-file:`src/plugins/ego/README.md`.

.. contents:: On this page
   :local:
   :depth: 2

Supported data
--------------

Video is recorded as MJPEG, H.264, or H.265 elementary streams. The ``ego.sh``
recording wrapper defaults to stereo H.264 at ``1600×1300@30``, enables available
accelerometer and gyroscope streams, and leaves audio disabled. These are
wrapper defaults; the native executable requires explicit stream configuration.

Inspect the connected camera's capabilities before selecting a profile. The
requested left/right combination must be supported simultaneously; the plugin
does not substitute another profile. H.264/H.265 capture is limited to 30 FPS.
Depth and infrared capture are not supported.

Installation
------------

On Linux x86_64, install CMake 3.24 or newer and ``uv``, then run from the
Isaac Capture repository root:

.. code-block:: bash

   ./src/plugins/ego/install.sh

The installer prepares native dependencies, downloads the pinned
`OrbbecSDK_v2-EGO v2.10.6-EGO-20261019 <https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/tag/v2.10.6-EGO-20261019>`_
package, verifies its SHA-256, installs udev rules, and builds the plugin.
General OrbbecSDK v2 packages do not provide the required EGO APIs.

Run the installer as your normal user; it requests elevated privileges for
system setup when needed. Add ``--preview`` for optional SDL2/FFmpeg preview
support. To use an extracted EGO SDK, pass ``--sdk-root /path/to/OrbbecSDK``
or set ``ORBBEC_SDK_ROOT``. Manual setup and build options are documented in
the :code-file:`plugin README <src/plugins/ego/README.md>`.

Reconnect the camera after installation, then inspect its profiles and controls:

.. code-block:: bash

   ./src/plugins/ego/ego.sh capabilities

For non-root USB access, the SDK's udev rules must include the EGO device
(``2bc5:1201``). In a container, install these rules on the host and reconnect
the camera; the installer skips udev setup inside containers.

Capture and verify
------------------

Record 30 seconds using the wrapper's default stereo and IMU configuration.
Use a fresh output directory for each recording:

.. code-block:: bash

   RUN="recordings/ego_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/ego/ego.sh record --duration 30 --output "$RUN"

The default ``sidecar`` mode writes video under ``raw/``, structured data to
``metadata.mcap``, and calibration to ``calibration.json``. It preserves the requested profile,
device/software identity, and capture log, then verifies the recording.
To repeat the integrity checks later:

.. code-block:: bash

   ./src/plugins/ego/ego.sh verify "$RUN"

Enable optional audio with ``--audio`` or ``audio = true`` in a TOML config;
``--no-audio`` overrides a config that enables it. Audio is saved as 48 kHz mono
S16_LE WAV, with audio offsets recorded in MCAP:

.. code-block:: bash

   ./src/plugins/ego/ego.sh record --duration 30 --audio \
     --output recordings/ego_with_audio

Audio capture and export are supported. Synchronizing audio with video and IMU
requires an independently validated time model and evidence of sample
continuity; SDK audio timestamps alone do not establish a shared sampling clock.

Use ``embedded`` mode for one self-contained MCAP with encoded video and any
enabled audio PCM:

.. code-block:: bash

   EMBEDDED_RUN="recordings/ego_embedded_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/ego/ego.sh record --duration 30 \
     --output "$EMBEDDED_RUN" --mcap-media embedded
   ./src/plugins/ego/ego.sh export-media "$EMBEDDED_RUN"

Embedded recording stores media without transcoding. Export produces
elementary video streams and, when audio was enabled, WAV. A remaining
``.partial`` file indicates that capture or cleanup did not complete
successfully; preserve it and the log for diagnosis. Integrity verification
does not establish physical synchronization between sensors.

Data flow and published collections
-----------------------------------

Standalone recording writes media and structured records locally and requires
no OpenXR runtime. For live sessions, ``--collection-prefix`` publishes
structured data through OpenXR SchemaPusher; DeviceIO trackers read it during
session updates. Live publication requires a running OpenXR runtime.
The native options ``--mcap-filename`` and ``--collection-prefix`` select these
two workflows and are mutually exclusive.

With ``--collection-prefix=ego``, the collections and Python read APIs are:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Collection
     - Python tracker
     - Read method
   * - ``ego/ColorLeft``, ``ego/ColorRight``
     - ``OrbbecEgoFrameMetadataTracker``
     - ``get_stream_data``
   * - ``ego/Accel``, ``ego/Gyro``
     - ``OrbbecEgoImuTracker``
     - ``get_stream_data``
   * - ``ego/Audio``
     - ``OrbbecEgoAudioTracker``
     - ``get_data``
   * - ``ego/Calibration``
     - ``OrbbecEgoCalibrationTracker``
     - ``get_data``
   * - ``ego/DeviceState``
     - ``OrbbecEgoDeviceStateTracker``
     - ``get_data``

All reads return the schema payload directly or ``None`` when unavailable.
Audio records are available only when audio capture is enabled.
Camera and IMU readers select one configured stream by its index. Camera streams
use ``OrbbecEgoCameraStream`` enums; IMU sensors default to Accel and Gyro. See the
:code-file:`plugin README <src/plugins/ego/README.md>` for the full tracker
configuration example.

Availability and diagnostics
----------------------------

The plugin integrates with
:doc:`Isaac Capture device monitoring <../../references/device_provider_monitoring>`.
After ``session.step()``, query
``session.get_device_status(f"{plugin_root_id}/camera/ego")`` for current
availability, using the root ID configured for the plugin.
``OrbbecEgoDeviceStateTracker`` provides historical SDK reports, capture summaries,
recovery events, and device snapshots for diagnosis.

Diagnostic recording is enabled by default. To omit these events from MCAP,
pass ``--record-device-diagnostics=off`` to the recording wrapper. For
``TeleopSession``, pass ``record_device_diagnostics=False`` to
``isaaccapture.ego.configure_recording`` or ``embedded_recording`` and exclude
diagnostic trackers from the recording configuration. This option leaves
availability monitoring, recovery, and logging enabled.

Multi-device sessions
---------------------

Add the EGO trackers to a
:doc:`TeleopSession <../../teleop/teleop_session>` configuration and its
``McapRecordingConfig``, alongside the hand, head, or
controller trackers. Configure ``orbbec_ego_camera`` through ``PluginConfig`` with
``--collection-prefix=ego`` and ``shutdown_timeout_seconds=15.0`` to allow
capture cleanup. Complete the full source build and install its Python wheel before
using the EGO tracker bindings and recording helper; the wrapper's
``build`` command builds only the native plugin tools.

Apply ``config = isaaccapture.ego.configure_recording(config)`` before opening
the session to include the calibration tracker and any EGO time-model snapshots.

For one self-contained multi-device MCAP, use
``isaaccapture.ego.embedded_recording`` with exactly one EGO plugin configured
for embedded media. The full configuration, matching media-spool path, and
clean-stop procedure are documented under "TeleopSession: one MCAP with other
trackers" in the :code-file:`plugin README <src/plugins/ego/README.md>`.

Replay and visualization
------------------------

With the built Python package installed, inspect structured EGO records using
the replay example. It requires stereo camera metadata, IMU, audio, calibration,
and any enabled diagnostic channels, so use an audio-enabled recording:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay
   python -m isaaccapture_examples.mcap_record_replay.replay_ego \
     recordings/ego_with_audio/metadata.mcap

A recording without audio is valid; this example reports a missing Audio
channel because of its fixed tracker configuration. The example reads
structured records without decoding video or playing audio. To play video,
use the sidecars or run ``export-media`` for an embedded recording, then open
the resulting elementary streams with FFmpeg.

For a live GPU preview, build the standalone camera_viz binding, specifying
the path to the extracted EGO SDK:

.. code-block:: bash

   examples/camera_viz/camera_viz.sh setup --with-ego \
     --sdk-root /path/to/OrbbecSDK
   examples/camera_viz/camera_viz.sh run \
     examples/camera_viz/configs/ego.yaml --mode window --stereo-debug sbs

The viewer uses GPU upload for MJPEG and NVDEC for H.264/H.265. The command
above displays the two eyes side by side in a desktop window; XR mode uses
separate eye textures. The viewer and recording plugin each open the camera;
run them one at a time. See :code-file:`examples/camera_viz/ego_preview/README.md`
for decoder requirements and XR configuration.

Sample-time association
-----------------------

Capture preserves device timestamps and host arrival times separately. A
session update or MCAP log time does not establish a common sampling instant
for video, IMU, and audio. For sample-time association, use the EGO offline
reader with a validated time model for the recording's device, profile, and
capture intervals.

Install the ``ego`` extra into an environment containing the built package.
If no validated model was recorded during capture, supply one explicitly:

.. code-block:: bash

   uv pip install 'isaaccapture[ego]'
   uv pip install -e ./examples/mcap_record_replay
   python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
     "$RUN" --required-streams left_rgb right_rgb accel gyro \
     --alignment /absolute/path/to/alignment.json \
     --output "$RUN/sync_audit.json"

The audit supports standard sidecar directories and embedded MCAP. It reports
coverage, gaps, and incomplete associations without decoding or playing media.
Python consumers can use ``isaaccapture.ego.sync.iter_sync_windows`` to obtain
frame references and sensor samples grouped into reference-frame intervals.
Missing required streams or a validated model cause an error.

Include ``audio`` only when its media is accessible and its sample-time model
and continuity have been independently validated. Associations remain limited
to validated intervals and do not bridge reconnects or clock discontinuities.
They do not establish synchronization with other devices or absolute host
sampling accuracy. Model preparation, validation requirements, and offline API
details are documented in the
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
