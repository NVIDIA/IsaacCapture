.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Orbbec EGO Camera Plugin
========================

Orbbec EGO is a stereo camera module for egocentric data collection. The
``orbbec_ego_camera`` C++ plugin records its ``ColorLeft`` and ``ColorRight`` video
streams, with optional accelerometer and gyroscope data. It also records
calibration and device diagnostics.

The plugin supports standalone recording and live data publication to an
Isaac Capture session. Source code and detailed configuration are in
:code-file:`src/plugins/orbbec_ego/README.md`.

.. contents:: On this page
   :local:
   :depth: 2

Supported data
--------------

Video is recorded as MJPEG, H.264, or H.265 elementary streams. The ``orbbec_ego.sh``
recording wrapper defaults to stereo H.264 at ``1600×1300@30`` and enables available
accelerometer and gyroscope streams. These are wrapper defaults; the native
executable requires explicit stream configuration.

Inspect the connected camera's capabilities before selecting a profile. The
requested left/right combination must be supported simultaneously; the plugin
does not substitute another profile. H.264/H.265 capture is limited to 30 FPS.
Depth and infrared capture are not supported.

Installation
------------

On Linux x86_64, install CMake 3.24 or newer and ``uv``, then run from the
Isaac Capture repository root:

.. code-block:: bash

   ./src/plugins/orbbec_ego/install.sh

The installer prepares native dependencies, downloads the pinned
`OrbbecSDK_v2-EGO v2.10.6-EGO-20261019 <https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/tag/v2.10.6-EGO-20261019>`_
package, verifies its SHA-256, installs udev rules, and builds the plugin.
General OrbbecSDK v2 packages do not provide the required EGO APIs.

Run the installer as your normal user; it requests elevated privileges for
system setup when needed. Add ``--preview`` for optional SDL2/FFmpeg preview
support. To use an extracted EGO SDK, pass ``--sdk-root /path/to/OrbbecSDK``
or set ``ORBBEC_SDK_ROOT``. Manual setup and build options are documented in
the :code-file:`plugin README <src/plugins/orbbec_ego/README.md>`.

Reconnect the camera after installation, then inspect its profiles and controls:

.. code-block:: bash

   ./src/plugins/orbbec_ego/orbbec_ego.sh capabilities

For non-root USB access, the SDK's udev rules must include the EGO device
(``2bc5:1201``). In a container, install these rules on the host and reconnect
the camera; the installer skips udev setup inside containers.

Capture and verify
------------------

Record 30 seconds using the wrapper's default stereo and IMU configuration.
Use a fresh output directory for each recording:

.. code-block:: bash

   RUN="recordings/ego_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/orbbec_ego/orbbec_ego.sh record --duration 30 --output "$RUN"

The default ``sidecar`` mode writes video under ``raw/``, structured data to
``metadata.mcap``, and calibration to ``calibration.json``. It preserves the requested profile,
device/software identity, and capture log, then verifies the recording.
To repeat the integrity checks later:

.. code-block:: bash

   ./src/plugins/orbbec_ego/orbbec_ego.sh verify "$RUN"

Use ``embedded`` mode for one self-contained MCAP with encoded video:

.. code-block:: bash

   EMBEDDED_RUN="recordings/ego_embedded_$(date +%Y%m%d_%H%M%S)"
   ./src/plugins/orbbec_ego/orbbec_ego.sh record --duration 30 \
     --output "$EMBEDDED_RUN" --mcap-media embedded
   ./src/plugins/orbbec_ego/orbbec_ego.sh export-media "$EMBEDDED_RUN"

Embedded recording stores media without transcoding. Export produces
elementary video streams. A remaining ``.partial`` file indicates that capture
or cleanup did not complete successfully; preserve it and the log for
diagnosis. Integrity verification does not establish physical synchronization
between sensors.

Data flow and published collections
-----------------------------------

Standalone recording writes media and structured records locally and requires
no OpenXR runtime. For live sessions, ``--collection-prefix`` publishes
structured data through OpenXR SchemaPusher; DeviceIO trackers read it during
session updates. Live publication requires the CloudXR runtime, and a process
that starts the plugin itself needs its environment loaded (see
:ref:`load-cloudxr-environment-variables`). The plugin waits until the runtime
provides an OpenXR system before it opens the camera.
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
     - ``get_data``
   * - ``ego/Accel``, ``ego/Gyro``
     - ``OrbbecEgoImuTracker``
     - ``get_data``
   * - ``ego/Calibration``
     - ``OrbbecEgoCalibrationTracker``
     - ``get_data``
   * - ``ego/DeviceState``
     - ``OrbbecEgoDeviceStateTracker``
     - ``get_data``

All reads return the schema payload directly or ``None`` when unavailable.
Create one tracker per collection, for example
``OrbbecEgoImuTracker("ego/Accel")``. To record a tracker, name it
``orbbec_ego_metadata_<stream>`` or ``orbbec_ego_imu_<stream>`` in lowercase, such as
``orbbec_ego_metadata_colorleft``; the replay example expects these names. See the
:code-file:`plugin README <src/plugins/orbbec_ego/README.md>` for the full tracker
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
``isaaccapture_examples.orbbec_ego.configure_recording`` or ``embedded_recording`` and exclude
diagnostic trackers from the recording configuration. This option leaves
availability monitoring, recovery, and logging enabled.

Multi-device sessions
---------------------

Add the EGO trackers to a
:doc:`TeleopSession <../../teleop/teleop_session>` configuration and its
``McapRecordingConfig``, alongside the hand, head, or
controller trackers. Configure ``orbbec_ego_camera`` through ``PluginConfig`` with
``--collection-prefix=ego``. Its ``plugin.yaml`` gives the plugin 15 seconds to
close its files when the session stops. Complete the full source build and install its Python wheel
before using the EGO tracker bindings, then ``uv pip install -e ./examples/orbbec_ego`` for the
recording helper; the wrapper's ``build`` command builds only the native plugin tools.

Apply ``config = isaaccapture_examples.orbbec_ego.configure_recording(config)`` before opening
the session to include the calibration tracker.

For one self-contained multi-device MCAP, use
``isaaccapture_examples.orbbec_ego.embedded_recording`` with exactly one EGO plugin configured
for embedded media. The full configuration, matching media-spool path, and
clean-stop procedure are documented under "TeleopSession: one MCAP with other
trackers" in the :code-file:`plugin README <src/plugins/orbbec_ego/README.md>`.

The test script :code-file:`tests/python/xr_runtime/test_orbbec_ego_camera.py` runs the
live flow with ``PluginManager`` and ``DeviceIOSession`` directly and checks that
every EGO collection keeps arriving. It loads the plugin from ``install/plugins``,
so it needs a full build and install with the EGO plugin enabled; for another
install prefix, pass ``--plugin-root <prefix>/plugins``. With
``isaaccapture`` installed in the active environment and the CloudXR environment
loaded, run from the repository root:

.. code-block:: bash

   python tests/python/xr_runtime/test_orbbec_ego_camera.py --mode schema-pusher --duration 30

Each run writes its media, calibration, and MCAP to a new
``recordings/ego_<time>/`` directory. ``--mode plugin-mcap`` and
``--mode no-metadata`` record without the CloudXR runtime.

Replay and visualization
------------------------

With the built Python package installed, inspect structured EGO records using
the replay example. It requires stereo camera metadata, IMU, calibration, and
any enabled diagnostic channels:

.. code-block:: bash

   uv pip install -e ./examples/orbbec_ego
   python -m isaaccapture_examples.orbbec_ego.replay "$RUN/metadata.mcap"

The example reads structured records without decoding video. To play video,
use the sidecars or run ``export-media`` for an embedded recording, then open
the resulting elementary streams with FFmpeg.

To check framing and exposure while recording, configure the plugin with
``-DORBBEC_EGO_ENABLE_PREVIEW=ON`` and pass ``--preview``: the plugin shows the two
eyes side by side in a desktop window without changing the recording.

Troubleshooting
---------------

- **No camera or USB access:** install the selected SDK's udev rules, reconnect,
  and re-run ``doctor`` and ``capabilities``.
- **Profile rejected:** choose an exact advertised left/right combination;
  the current plugin rejects H.264/H.265 profiles above 30 FPS.
- **Partial MCAP:** the recording did not finish successfully. Preserve it and
  the log for diagnosis before starting a new recording.
- **Live plugin stalls at** ``waiting for a system...``: the runtime has no
  OpenXR system yet; see :ref:`cloudxr-form-factor-unavailable`. Ctrl-C does
  not interrupt this wait; ``PluginManager`` stops the plugin after its
  shutdown timeout.
- **Publisher startup failure:** check that the CloudXR environment is loaded
  and the plugin search path is correct.
- **Local recording logs an** ``NV_CXR_RUNTIME_DIR`` **error:** under
  ``PluginManager`` without the CloudXR environment, the plugin skips device
  status publication and keeps recording, as the warning after it says.
- **No embedded final output:** check normal context exit, committed fragment,
  matching spool path, and the merge tool beside the selected plugin.
