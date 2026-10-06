.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Camera Streaming
================

Use :code-dir:`camera_viz <examples/camera_viz>` to stream live cameras or recordings to an XR
headset or desktop window. The sample uses :doc:`Televiz </getting_started/televiz>` and supports
cameras attached locally or streamed from a robot over RTP.

.. figure:: ../_static/televiz_2d.gif
   :alt: camera_viz with multiple camera feeds
   :width: 420px
   :class: no-image-zoom

   One plane per camera.

.. contents:: On this page
   :local:
   :depth: 1

Requirements
------------

- **Host:** Ubuntu with an NVIDIA GPU; see :doc:`system requirements </references/requirements>`.
  Jetson: Orin with JetPack 6.2.x or 7.2.1; Thor with JetPack 7.x.
- **CUDA:** runtime, NVRTC, and development headers matching the installed toolkit.
  JetPack 7.2.1 uses CUDA 13.2; see :ref:`camera-streaming-jetson-cuda`.
  A GPU is required even for recordings and window mode.
- **Display:** a :ref:`supported XR headset <connect-xr-headset>` with network access to the host
  and :ref:`CloudXR ports <whitelist-firewall-ports>` open, or a desktop display for window mode.
- **Camera — optional, recommended:** USB / UVC, OAK-D, or supported ZED camera for a live feed.
  Start with the bundled recording if no camera is available. ZED requires its SDK.
- **Controllers — optional:** for adjusting the view in XR.

Setup
-----

From a :ref:`repository checkout <check-out-code-base>`, run:

.. code-block:: bash

   cd examples/camera_viz
   ./camera_viz.sh setup

Setup creates a Python 3.12 ``.venv`` with ``isaaccapture[cloudxr]``, UVC, and OAK-D support.
It needs network access and may prompt for sudo to install system dependencies.
On Jetson, install the :ref:`CUDA prerequisites <camera-streaming-jetson-cuda>` first and add
``--jetson``. To reuse an environment, use :ref:`setup --venv <camera-streaming-external-venv>`.

All commands below run from ``examples/camera_viz``. No environment activation is needed.

First run — a recording, no camera required
-------------------------------------------

Fetch the bundled clip with Git LFS, then launch the viewer:

.. code-block:: bash

   git lfs pull --include="examples/camera_viz/test_data/recording.mp4"
   ./camera_viz.sh run configs/replay.yaml

The viewer starts CloudXR automatically; accept the EULA on first launch. Open the printed
client URL in your headset browser, accept the certificate, and click **Connect**
(:ref:`connection instructions <connect-xr-headset>`).

For a desktop preview instead, add ``--mode window`` to the run command.
On Orin, select **H.264** in the client and apply the
:ref:`black-plane workaround <camera-streaming-troubleshooting>` below.

.. _camera-streaming-first-session:

First-session checklist
^^^^^^^^^^^^^^^^^^^^^^^

With the default replay config:

- **Playback:** one plane is visible and the clip loops.
- **Head movement:** small turns and leaning change your viewpoint; the plane stays anchored.
- **Lazy recentering:** look more than 45° away for about a second; the plane moves smoothly
  back in front of you.
- **Image quality:** no freezing, flicker, torn regions, or unexpected colors, including while
  moving your head. This mono clip should show the same source image in both eyes.
- **Controllers, if available:** right-stick click recenters; ``A`` changes lock mode;
  ``Y`` restores the config.
- **Restart:** press ``Ctrl+C``, rerun, and reconnect if needed; playback resumes.

In window mode, check playback, image quality, and restart. If a check fails, see
:ref:`camera-streaming-troubleshooting`.

.. _camera-streaming-live-camera:

Running with a real camera
--------------------------

Attach the camera to the viewer host and keep ``source: local``. Edit the matching config's
device, resolution, and frame rate, then run:

.. code-block:: bash

   ./camera_viz.sh run configs/v4l2.yaml     # or oakd.yaml / zed.yaml

``v4l2.yaml`` defaults to a ZED Mini's 2560×720 UVC mode. For a regular webcam, use modes reported
by ``v4l2-ctl --list-formats-ext``. ZED SDK capture also needs ``setup --with-zed``.

.. _camera-streaming-sources:

Supported sources
^^^^^^^^^^^^^^^^^

Set ``type`` in each ``cameras`` entry. Multiple entries produce one plane per camera.
See :code-dir:`configs/ <examples/camera_viz/configs>` for complete examples.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Type
     - Source
   * - ``v4l2``
     - USB / UVC camera.
   * - ``oakd``
     - OAK-D RGB / LEFT / RIGHT; ``stereo: true`` for grayscale stereo.
   * - ``zed``
     - ZED 2 / Mini / X One through the ZED SDK; mono or stereo.
   * - ``video``
     - :ref:`Recorded video <recorded-camera-streaming>`, including side-by-side stereo.
   * - ``synthetic``
     - Animated GPU test pattern; no camera or file needed.

.. _recorded-camera-streaming:

Replaying your own recording
----------------------------

Edit :code-file:`configs/replay.yaml <examples/camera_viz/configs/replay.yaml>`:

.. code-block:: yaml

   cameras:
     - name: replay
       type: video
       path: /path/to/clip.mp4
       loop: true
       fps: 0          # use the file's frame rate
       stereo: false   # true for side-by-side recordings in direct mode

Relative paths resolve from the YAML's directory. OpenCV's FFmpeg backend must support the
file's format. Size and frame rate default to the recording's values; ``loop: false`` holds
the last frame. Keep setup's OpenCV dependency for replay (``--no-v4l2`` skips it).

Display modes
-------------

XR is the default. Use ``--mode window`` or ``display.mode: window`` for a desktop preview;
stereo sources show only the left eye in window mode.

In XR, set ``display.placements.<camera-name>.lock_mode``:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Mode
     - Behavior
   * - ``world``
     - Stays where initially placed.
   * - ``head``
     - Follows your head every frame; may trail fast motion.
   * - ``gimbal``
     - Follows your position while keeping its initial heading.
   * - ``lazy`` (default)
     - Stays anchored until you look or move away, then recenters smoothly.

Lazy-mode settings: ``look_away_angle_deg``, ``reposition_distance``, ``reposition_delay_s``,
and ``transition_duration_s`` under the camera's placement.

.. _camera-streaming-controls:

Controller bindings
^^^^^^^^^^^^^^^^^^^

Changes apply to all cameras and appear in the terminal and headset status panels.

.. Use concise titles and Televiz branding; name camera_viz in the prose.

.. figure:: ../_static/camera-viz-controls.png
   :alt: camera_viz XR controller bindings: left controller adjusts surface and shape;
         right controller adjusts depth and placement.
   :width: 100%

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Input
     - Effect
   * - Right stick ←/→
     - Adjust stereo plane gap; pan the panorama on ``equirect``.
   * - Right stick click
     - Recenter the surface or panorama.
   * - ``A``
     - Cycle ``world`` → ``head`` → ``gimbal`` → ``lazy``.
   * - ``B``
     - Toggle mono / stereo for stereo sources.
   * - ``X``
     - Cycle ``quad`` → ``cylinder`` → ``equirect``.
   * - ``Y``
     - Reset to YAML settings.
   * - Left stick
     - Adjust size / vertical position; arc width for cylinders, span for panoramas.

The stereo gap is limited by the headset's measured IPD. Configure bindings and rates under
``display.controls``; see the :code-file:`README <examples/camera_viz/README.md>`.

Display surfaces
^^^^^^^^^^^^^^^^

Set ``shape`` under ``display.placements.<camera-name>``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Shape
     - Use
   * - ``quad`` (default)
     - Flat camera feed. Set ``size: [width_m, height_m]``.
   * - ``cylinder``
     - Wide-FOV feed. Set ``cylinder_radius_m`` and ``cylinder_angle_deg``.
   * - ``equirect``
     - 360°×180° panorama. Set ``equirect_yaw_deg`` to adjust heading; lock modes do not apply.

Curved surfaces require XR. The OpenXR runtime composites surfaces by default. Flat planes can
use ``compositor: televiz`` instead; this is required with the experimental runtime on Orin.
See :ref:`openxr-composition-layers` for compositor details.

Setup options
-------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Flag
     - Effect
   * - ``--jetson``
     - Configure CUDA library discovery on Jetson; with RTP, also check for NVRTC.
   * - ``--venv PATH``
     - Install into ``PATH`` and link the sample's ``.venv`` to it.
   * - ``--with-rtp``
     - Install GStreamer, PyGObject, and native codec dependencies for split mode or loopback.
   * - ``--with-zed``
     - Install the ZED Python API. Requires the SDK at ``/usr/local/zed`` or ``--zed-sdk PATH``.
   * - ``--no-v4l2`` / ``--no-oakd``
     - Skip OpenCV / DepthAI respectively. Video replay also needs OpenCV.
   * - ``--sender-only``
     - Install only the robot sender's dependencies; implies ``--with-rtp``.
   * - ``--wheel PATH``
     - Install a locally built ``isaaccapture`` wheel.
   * - ``--build-from-source``
     - Build from this checkout; requires the :doc:`build environment </getting_started/build_from_source/index>`.

.. _camera-streaming-external-venv:

External virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reuse an existing Python 3.12 environment:

.. code-block:: bash

   ./camera_viz.sh setup --venv /absolute/path/to/venv

Setup installs dependencies there and links ``.venv`` to it. If ``.venv`` is already a real
directory, move it aside first. The wrapper always uses ``.venv/bin/python``, regardless of
the environment activated in your shell.

.. _camera-streaming-jetson-cuda:

Jetson CUDA prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^

JetPack 7.2.1 (L4T R39.2.1) needs CUDA 13.2 development components, including headers and NVRTC.
With the matching NVIDIA package repositories configured, install them using the
`JetPack CUDA development package
<https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html#install-specific-jetpack-components>`__:

.. code-block:: bash

   sudo apt update
   sudo apt install nvidia-cuda-dev
   readlink -f /usr/local/cuda
   /usr/local/cuda/bin/nvcc --version

On JetPack 7.2.1, verify that ``/usr/local/cuda`` selects CUDA 13.2. On JetPack 6.2.x, keep the
matching CUDA 12.x toolkit. Setup selects CuPy from this path; ``--jetson`` configures library
discovery but does not install the full toolkit. See NVIDIA's
`JetPack installation guide
<https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/latest/setup_jetpack.html>`__ for BSP setup.

CloudXR runtime flags
---------------------

The viewer attaches to an existing CloudXR service or starts one. Common run flags:

- ``--accept-eula`` — accept the EULA non-interactively.
- ``--cloudxr-device-profile PROFILE`` — device profile (default ``Quest3``).
- ``--no-host-client`` — disable the hosted client at ``https://<host>:48322/client/`` when
  starting a service. Stop an existing service first to change hosting.

Inspect or stop the service with the sample's interpreter:

.. code-block:: bash

   .venv/bin/python -m isaaccapture.cloudxr.service status
   .venv/bin/python -m isaaccapture.cloudxr.service stop

Set runtime overrides in YAML:

.. code-block:: yaml

   display:
     cloudxr:
       NV_DEVICE_PROFILE: apple-vision-pro
       NV_ENABLE_POSE_WAIT: null   # remove a viewer default

These settings override shell environment variables; an explicit ``--cloudxr-env-config``
takes precedence over the generated config. Resolved settings appear in
``~/.cloudxr/logs/cxr_server.*.log``. See :doc:`/references/cloudxr` and
``.venv/bin/python camera_viz.py --help`` for more options.

.. _split-mode:

Split mode — robot → workstation over RTP
-----------------------------------------

Use split mode when cameras are attached to a robot and the viewer runs on a workstation.
Prefer a direct connection when possible: RTP adds an encode/decode hop. Use a wired network;
there is no retransmission or FEC, so packet loss can corrupt video until the next IDR frame
(default interval: 5 seconds).

In the YAML, set ``source: rtp`` and specify ``width``, ``height``, and ``fps`` for each camera.
Then, on the workstation:

.. code-block:: bash

   ./camera_viz.sh setup --with-rtp
   export REMOTE_HOST=10.0.0.5 REMOTE_USER=nvidia
   export STREAMING_HOST=10.0.0.42                  # workstation IP
   ./camera_viz.sh deploy configs/v4l2.yaml
   ./camera_viz.sh run configs/v4l2.yaml

``deploy`` installs the sender and starts a ``camera-streamer`` systemd user service on the
robot. Use ``./camera_viz.sh service-{status,logs,restart}`` to manage it. The workstation's
UDP receive ports (default 5000+) must be reachable from the robot.

Loopback
^^^^^^^^

Test RTP on one machine with ``./camera_viz.sh loopback configs/v4l2.yaml``. This also supports
mono video recordings with explicit ``width``, ``height``, and ``fps``; stereo video replay is
unsupported over RTP.

Configuration
-------------

Start from a :code-dir:`source config <examples/camera_viz/configs>` and adjust the camera
settings for your device. Placement keys must match the camera's ``name``:

.. code-block:: yaml

   display:
     mode: xr
     placements:
       cam:
         lock_mode: lazy
         distance: 1.5
         shape: quad
         size: [1.0, 0.56]       # meters

Add entries to ``cameras`` and ``display.placements`` for multiple feeds. In split mode, assign
each stream its own RTP port; stereo sources also require ``rtp.port_right``.
The :code-file:`README <examples/camera_viz/README.md>` documents the full schema.

.. _camera-streaming-troubleshooting:

Troubleshooting
---------------

- **XR session does not start:** connect the client first. ``XR_ERROR_FORM_FACTOR_UNAVAILABLE``
  (-35) means no headset is connected. The viewer waits indefinitely by default; set
  ``display.xr.system_wait_seconds`` or ``--xr-wait`` to change the timeout.
  Check ``~/.cloudxr/logs/cxr_server.*.log`` and ``~/.cloudxr/logs/runtime_worker_stderr.log``
  (``runtime_stderr.log`` with ``ISAACCAPTURE_LOGGING=off``).
- **No video on Orin:** set the client's **Video Codec** to **H.264**.
- **Black camera plane on Orin:** set ``compositor: televiz`` under the camera's placement
  and restart. This workaround supports quads only; for the replay config:

  .. code-block:: yaml

     display:
       placements:
         replay:
           compositor: televiz

- **Corrupted or frozen video:** stop the viewer and run
  ``./camera_viz.sh run configs/synthetic.yaml`` to check the display path without a camera
  or recording. Add ``--mode window`` to check local rendering separately from XR streaming.
- **No window over SSH:** window mode needs a desktop display or video-capable remote desktop.
- **Video file fails to open:** paths are relative to the YAML directory. For the bundled clip,
  run the Git LFS command in the first-run instructions.
- **Python cannot find dependencies:** use ``.venv/bin/python`` or rerun ``setup --venv PATH``.
- **Split mode has no frames:** check ``./camera_viz.sh service-status``, ``STREAMING_HOST``,
  and the workstation's UDP ports. Set ``verbose: true`` in YAML for per-source logs.

CuPy / CUDA errors
^^^^^^^^^^^^^^^^^^

Setup checks imports, not GPU execution. Test a small kernel with the sample's interpreter:

.. code-block:: bash

   .venv/bin/python - <<'PY'
   import cupy as cp

   values = cp.arange(4, dtype=cp.float32)
   assert cp.asnumpy(values + 1).tolist() == [1.0, 2.0, 3.0, 4.0]
   print("CUDA kernel check passed")
   PY

Missing ``libnvrtc.so``, ``cuda_runtime.h``, or ``cuda_fp16.h`` indicates missing or mismatched
CUDA development components. Check ``/usr/local/cuda`` and rerun setup (``--jetson`` on Jetson).
The CUDA version reported by ``nvidia-smi`` describes driver support, not installed headers.
See :ref:`camera-streaming-jetson-cuda`.

Next steps
----------

- :ref:`Connect a live camera <camera-streaming-live-camera>` or
  :ref:`replay your own recording <recorded-camera-streaming>`.
- :ref:`Stream from a robot <split-mode>` when capture and viewing run on separate machines.
- :ref:`Integrate with teleoperation <sharing-the-xr-session>` by sharing Televiz's OpenXR
  session with ``TeleopSession``. Only one OpenXR session is allowed per process.
- :doc:`Build a custom viewer </getting_started/televiz>` using the Televiz API and
  :code-file:`camera_viz.py <examples/camera_viz/camera_viz.py>` as a reference.
