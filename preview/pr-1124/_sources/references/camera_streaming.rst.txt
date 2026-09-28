.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Camera Streaming
================

``camera_viz`` is the reference camera-streaming sample built on :doc:`Televiz
</getting_started/televiz>` (``isaacteleop.viz``). It captures frames from one or more cameras
and streams them to an XR headset — one plane per camera, aspect-fit — either directly or from a
robot to a workstation over the network (split mode). It can also stream :ref:`recorded camera
data <recorded-camera-streaming>` — a video file replayed in place of a live camera — and render
to a desktop window instead of the headset.

The sample lives at :code-dir:`examples/camera_viz/ <examples/camera_viz>`. This page walks you
from setup and a hardware-free first run to real cameras and the robot → workstation split mode.
For the exact command surface and flags, see the
:code-file:`README <examples/camera_viz/README.md>`.

.. figure:: ../_static/televiz_2d.gif
   :alt: camera_viz in window mode with synthetic multi-camera feeds
   :width: 420px
   :class: no-image-zoom

   ``camera_viz`` in XR mode — one plane per camera.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- A workstation meeting the :doc:`system requirements </references/requirements>` (Ubuntu, NVIDIA
  GPU, CUDA driver) — every source hands frames to the renderer GPU-resident via CuPy.
- For Jetson platforms:

  - **Orin:** `JetPack 6.2.x <https://developer.nvidia.com/embedded/jetpack-sdk-62>`__ or
    `7.2.x <https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2>`__
  - **Thor:** `JetPack 7.x <https://developer.nvidia.com/embedded/jetpack>`__
- For the default XR mode, a headset to connect as the CloudXR client — follow the
  :doc:`quick start </getting_started/quick_start>` step :ref:`connect-xr-headset`. The viewer
  launches the CloudXR runtime itself; nothing to start separately. No headset handy?
  ``--mode window`` renders to a desktop window instead.

Setup
-----

Clone the repository if you haven't already (quick start step :ref:`check-out-code-base`), then
run the sample's one-time setup:

.. code-block:: bash

   examples/camera_viz/camera_viz.sh setup
   source examples/camera_viz/.venv/bin/activate

There is no need to install the ``isaacteleop`` pip package yourself. ``setup`` builds the
sample's own environment: ``isaacteleop[cloudxr]`` — the ``cloudxr`` extra is not optional here,
since XR is the default display mode and the viewer launches the runtime itself — plus every
other Python dependency, into ``.venv/`` via ``uv``. It resolves a version new enough for the
sample on its own, falling back to a release candidate and then, after asking, to a source build
of the surrounding checkout (:code-file:`scripts/_install_deps.sh
<examples/camera_viz/scripts/_install_deps.sh>` holds the minimum; ``--wheel`` and
``--build-from-source`` override the choice). Finally it probes the system packages it needs and
prints the exact ``apt-get`` line to approve — declining, or a non-interactive run, aborts.

By default ``setup`` provisions the direct-mode path — USB / UVC and OAK-D camera support. Split
mode (RTP) and ZED support are opt-in, since both pull in dependencies the direct path never
needs. Flags trim or extend that:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Flag
     - Effect
   * - ``--no-v4l2``
     - Skip USB / UVC webcam support (``opencv-python``).
   * - ``--no-oakd``
     - Skip OAK-D support (``depthai``).
   * - ``--with-rtp``
     - Also install the split-mode dependencies: the GStreamer system packages, PyGObject, and the
       native NVENC/NVDEC codec build. Required for ``loopback`` and for any config with
       ``source: rtp``; implied by ``--sender-only``. Direct mode does not need it.
   * - ``--with-zed``
     - Also build + install the ZED SDK's Python API (``pyzed``). Requires the ZED SDK on the
       machine (default ``/usr/local/zed``; override with ``--zed-sdk PATH``).
   * - ``--sender-only``
     - Split mode only — robot-side install of just the sender's dependencies.
   * - ``--jetson``
     - Split mode only — extra CUDA wiring JetPack images need on the robot.
   * - ``--venv PATH``
     - Install into an existing virtual environment instead of creating ``.venv/``.
   * - ``--wheel PATH``
     - Install a locally built ``isaacteleop`` wheel instead of resolving one from the index — for
       developing Isaac Teleop itself.
   * - ``--build-from-source``
     - Skip the package index and build ``isaacteleop`` from the surrounding checkout without
       prompting — a full C++ / CUDA / Vulkan build (see
       :doc:`/getting_started/build_from_source/index`).

.. _recorded-camera-streaming:

First run — a recording, no camera required
-------------------------------------------

The recorded-camera source (``type: video``) replays a video file through exactly the same
source → layer → Televiz path a live camera uses. It is the supported way to stream recorded
camera data, it needs no extra setup, and it is the quickest end-to-end check. A test clip ships
with the repo, and :code-file:`configs/replay.yaml <examples/camera_viz/configs/replay.yaml>`
already points at it:

.. code-block:: bash

   cd examples/camera_viz
   ./camera_viz.sh run configs/replay.yaml                 # XR headset (default)
   ./camera_viz.sh run configs/replay.yaml --mode window   # desktop window instead

In XR mode the viewer first brings up the CloudXR runtime (accept the EULA on first launch, or
pass ``--accept-eula``), then **you should see** the terminal report the session and the source
coming up::

   camera_viz: source=local, mode=xr, xr=True, shapes=quad, 1 layer(s)
   [video] opening...
   [video] connected
   [video] streaming

and the clip looping on a plane in the headset once it connects — or in a desktop window
(``mode=window, xr=False``) with the ``--mode window`` override, which starts no runtime.

Replaying your own recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Point ``path`` at any file OpenCV's FFmpeg backend reads — mp4 / mkv / webm carrying H.264,
HEVC, AV1, and so on. The file is probed once at startup for its size and frame rate (a missing
or absurd rate falls back to 30 fps); a missing or unreadable file is a configuration error and
fails immediately instead of retrying like a camera would. A complete config:

.. code-block:: yaml

   source: local

   cameras:
     - name: replay
       enabled: true
       type: video
       path: ../test_data/recording.mp4   # required; relative to this YAML's directory
       loop: true                         # false = hold the last frame at end of file
       fps: 0                             # 0 = the file's native rate
       stereo: false                      # true = split a side-by-side recording into eyes
       # width/height default to the file's native size (per eye when stereo)

   display:
     mode: xr                             # or: window
     placements:
       replay: { lock_mode: lazy, distance: 1.5 }

With ``source: rtp``, ``width``, ``height``, and ``fps`` become required — the sender paces its
encode loop at ``fps`` and the receiver sizes its decoder from the config — and ``stereo`` is
rejected, since the sender needs per-eye streams. A mono recording is otherwise a fine capture
side for :ref:`split mode <split-mode>` and for ``loopback``, which exercises the whole
encode → UDP → decode path with no camera attached.

Supported sources
-----------------

The source kind is selected by the ``type`` field of each entry in the YAML ``cameras`` list:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - ``type:``
     - Notes
   * - ``v4l2``
     - USB / UVC cameras — anything ``v4l2-ctl --list-formats-ext`` reports.
   * - ``oakd``
     - OAK-D RGB / LEFT / RIGHT; mono or ``stereo: true``. Stereo ships GRAY8 over USB to halve
       bandwidth and is broadcast to RGBA on the GPU; ``stereo_rgb`` keeps color.
   * - ``zed``
     - ZED 2 / Mini / X One; mono or ``stereo: true`` (per-eye SDK retrieve, zero-copy on the GPU).
   * - ``video``
     - Recorded camera data — video-file replay (anything OpenCV's FFmpeg backend reads). Loops
       by default; ``stereo: true`` splits side-by-side recordings into eyes (viewer only). See
       :ref:`recorded-camera-streaming`.
   * - ``synthetic``
     - Debugging tool — GPU-generated test pattern, no hardware or file.

Running with a real camera
--------------------------

Attach the camera to the machine that runs the viewer, keep ``source: local`` in the config, and
run with the matching config:

.. code-block:: bash

   ./camera_viz.sh run configs/v4l2.yaml     # or oakd.yaml / zed.yaml

**You should see** the same startup lines as above with the camera's tag (``[v4l2]``,
``[oakd]``, ``[zed]``) and the live feed. Multiple entries in the ``cameras`` list render as one
plane each.

Display modes
-------------

XR is the default: each camera renders as its own plane in the headset via the active OpenXR
runtime, and stereo sources (``stereo: true``) render true side-by-side stereo. Pass
``--mode window`` (or set ``display.mode: window`` in the YAML) to render to a desktop window
instead — no headset or runtime needed; stereo shows the left eye.

In XR, how a plane follows the operator's head is the per-camera ``lock_mode`` under
``display.placements.<name>``:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Mode
     - Behavior
   * - ``world``
     - Placed once in front of you and stays put.
   * - ``head``
     - Follows your head every frame. Head-locked content updates its pose at application
       rate, so it trails fast head motion by roughly a frame — expected for any head-locked
       OpenXR layer; prefer ``lazy`` unless you need a true HUD.
   * - ``gimbal``
     - Follows your position but not your rotation: the surface stays pointed where you first
       looked, walks with you, and turning your head looks around it. The natural mode for
       wide cylinder feeds (a "virtual gimbal").
   * - ``lazy``
     - World-locked, but re-snaps in front of you when you look away (default).

Lazy-mode knobs live under ``placements.<name>``: ``look_away_angle_deg``,
``reposition_distance``, ``reposition_delay_s``, ``transition_duration_s``.

Controller bindings
^^^^^^^^^^^^^^^^^^^

In XR the controllers retune the view live, without editing the YAML and restarting. The right
hand changes how the feed looks; the left, what surface it is mapped onto:

.. figure:: ../_static/camera-viz-controls.svg
   :alt: camera_viz controller bindings, left and right
   :width: 100%

   Bindings at a glance. The table below is the same thing in words.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Input
     - Effect
   * - Right stick ←/→
     - Stereo plane gap — how far apart the two eyes' planes sit
       (``placements.<cam>.stereo_plane_distance_cm``). Widening it pushes the scene back
       instead of packing it into the space in front of the planes. Stereo cameras only; on
       ``equirect``, which has no gap to set, the same stick pans the panorama.
   * - Right stick click
     - Recenter on your view: an ``equirect`` yaws so the middle of the panorama lands dead
       ahead, and a placed surface re-snaps its anchor.
   * - ``A``
     - Cycle the lock mode: ``world`` → ``head`` → ``gimbal`` → ``lazy``.
   * - ``B``
     - Toggle mono / stereo. Stereo cameras only.
   * - ``X``
     - Cycle the shape: ``quad`` → ``cylinder`` → ``equirect``.
   * - ``Y``
     - Reset everything to the YAML values.
   * - Left stick
     - Retunes the active shape: ``quad`` size / vertical position, ``cylinder`` arc width /
       vertical position, ``equirect`` horizontal / vertical span.

Changes apply to every camera at once and appear both on the terminal status panel and on a
head-locked panel in the headset that auto-hides shortly after. Neither toggle reallocates:
``B`` sends the left frame to both eyes, and ``X`` flips visibility between shapes built at
startup, so the extra shapes cost VRAM (reported at startup) rather than a stall on the press.

The stereo gap is bounded below **divergent parallax** — a gap wider than your IPD would need
the eyes to splay outward — and the headset's measured IPD sets that ceiling, not the config.
The HUD suggests a gap derived from the plane distance and that IPD.

Bindings, rates and limits live under ``display.controls``; see the
:code-file:`README <examples/camera_viz/README.md>` for the full set.

Display surfaces
----------------

By default each camera renders on a flat plane, which suits normal-FOV feeds. Wide-FOV and
panoramic sources look better on a curved surface: set ``shape`` per camera under
``display.placements.<name>``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Shape
     - Behavior
   * - ``quad`` (default)
     - Flat plane. All lock modes apply.
   * - ``cylinder``
     - Curved arc facing the operator, so the image stays at a constant viewing distance edge
       to edge. ``cylinder_radius_m`` (default 2.0) sets that distance, ``cylinder_angle_deg``
       (default 90) the arc width. All lock modes apply (``head`` = gaze-tracking curved
       visor).
   * - ``equirect``
     - Full 360°×180° sphere around the operator, for equirectangular panorama / VR-video
       sources. The middle of the texture points along ``equirect_yaw_deg`` (0 = the
       direction the headset last recentered on, positive to the left), which is how a camera
       mounted facing another way gets aimed. Lock modes don't apply — the sphere is centred
       on the operator and follows them everywhere.

Curved shapes exist only in XR mode — the viewer exits with an error in window mode. Stereo
sources render per-eye textures on the same surface, and ``stereo_baseline_mm`` adds a per-eye
pose shift (no effect on the equirect sphere at its default infinite radius).

Surfaces are composited by the OpenXR runtime, which keeps them sharp under head motion and lets
CloudXR stream them efficiently (see
:ref:`OpenXR composition layers <openxr-composition-layers>`). For flat planes only,
``compositor: televiz`` opts a camera back into Televiz's built-in compositor. On Jetson Orin
(CloudXR experimental runtime), set ``compositor: televiz`` so the plane is not left black; see
Troubleshooting below.

CloudXR runtime flags
---------------------

In XR mode the viewer attaches to the running CloudXR runtime, or starts one if none is
serving. Useful flags:

- ``--accept-eula`` — accept the CloudXR EULA non-interactively (first run only).
- ``--cloudxr-device-profile PROFILE`` — ``NV_DEVICE_PROFILE`` (default ``Quest3``).

Run ``camera_viz.py --help`` for the rest (install dir, env-config file, WSS proxy toggle).

Runtime settings themselves are declared in the config rather than exported. The viewer ships
defaults (pose wait off, runtime foveation on), and ``display.cloudxr`` overrides them per
deployment — list only what you want to change:

.. code-block:: yaml

   display:
     cloudxr:
       NV_DEVICE_PROFILE: apple-vision-pro
       NV_ENABLE_POSE_WAIT: null   # null drops a viewer default

The viewer writes these to a generated ``--cloudxr-env-config`` file, which outranks the process
environment — so a value left in your shell (for example after sourcing
``~/.cloudxr/run/cloudxr.env``, as ``--no-launch-cloudxr-runtime`` suggests) cannot silently
override the config, and neither can it override ``--cloudxr-device-profile``. Booleans are
written in the lowercase spelling the runtime's parser recognises; the launcher-computed keys
(``XR_RUNTIME_JSON``, ``NV_CXR_RUNTIME_DIR``, ``NV_CXR_OUTPUT_DIR``, ``XRT_NO_STDIN``) are
rejected by name. Passing ``--cloudxr-env-config`` yourself takes precedence over the generated
file.

An unknown variable name warns with a suggestion before the runtime launches, and is passed
through regardless — the runtime has more settings than the viewer lists. To confirm what the
runtime resolved, read the settings dump at the top of the newest
``~/.cloudxr/logs/cxr_server.*.log``.

.. _split-mode:

Split mode — robot → workstation over RTP
-----------------------------------------

Split mode runs the capture side on the robot (``camera_streamer.py``) and ships RTP H.264 to
the viewer on the workstation (``source: rtp``).

.. warning::

   Split mode is **not recommended in most cases**. It exists for one situation: the cameras are
   on the robot, but Isaac Teleop runs on a workstation, so the frames must be streamed to where
   Isaac Teleop is running. That costs a full extra encode/decode hop — NVENC on the robot, UDP,
   NVDEC on the workstation — so whenever a camera can attach directly to the machine running
   Isaac Teleop, run direct mode instead. Wired networks only: there is no retransmit or FEC — one lost packet corrupts one
   frame until the next IDR (default every 5 s).

In split mode every camera entry must pin ``width``, ``height``, and ``fps`` in the YAML — the
receiver sizes its decoder from the config, not from the wire.

The workstation needs the RTP dependencies, which ``setup`` does not install by default — re-run
it with ``--with-rtp`` if you provisioned for direct mode first. The robot side is handled by
``deploy``, which implies the flag.

Set ``source: rtp`` in the config, export the robot/streaming credentials once per shell, then
deploy the sender and run the viewer:

.. code-block:: bash

   export REMOTE_HOST=10.0.0.5 REMOTE_USER=nvidia
   export STREAMING_HOST=10.0.0.42                  # workstation IP

   ./camera_viz.sh deploy configs/v4l2.yaml         # full deploy + systemd unit on the robot
   ./camera_viz.sh run    configs/v4l2.yaml         # viewer on the workstation

``deploy`` rsyncs the source to the robot, installs sender dependencies, renders a
``camera-streamer.service`` systemd user unit (injecting ``--host`` from ``$STREAMING_HOST``
without editing the YAML on disk), and enables it. Operate the running unit with
``./camera_viz.sh service-{status,logs,restart}``. The sender retries forever across unplug, SDK
errors, and network blips.

Loopback
^^^^^^^^

Loopback is a testing / debugging aid, not a deployment mode: ``./camera_viz.sh loopback
configs/v4l2.yaml`` runs the sender and viewer together on ``127.0.0.1`` — the quickest way to
smoke-test the RTP path on one machine. It also works camera-free with a mono ``type: video``
entry (set ``width`` / ``height`` / ``fps``).

Configuration
-------------

A single YAML drives both capture and visualization. Each entry in the ``cameras`` list becomes
its own plane (and, in split mode, its own RTP port). Abbreviated:

.. code-block:: yaml

   source: local | rtp
   streaming:
     host: 192.168.1.100         # workstation IP (overridden at deploy time)
   encoder: auto | native | gstreamer

   cameras:
     - name: cam
       enabled: true
       type: v4l2                # v4l2 | oakd | zed | video | synthetic
       width: 2560               # video: optional — defaults to the file's size
       height: 720               # (required when source: rtp)
       fps: 30
       stereo: false             # zed / video / synthetic — per-eye capture + SBS in XR
       path: clip.mp4            # video only — file to replay, relative to this YAML
       loop: true                # video only — rewind at end of file
       rtp:
         port: 5000              # left eye when stereo
         port_right: 5001        # required when stereo + source: rtp
         bitrate_mbps: 15

   display:
     mode: xr | window           # default: xr
     window: { width, height }
     xr:     { near_z, far_z }
     clear_color: [r, g, b, a]
     placements:
       cam:
         lock_mode: lazy         # world | head | lazy | gimbal
         distance: 1.5
         # size: [w_m, h_m]
         # stereo_plane_distance_cm: 0   # gap between the eyes' planes
         # shape: quad           # quad | cylinder | equirect (cylinder/equirect are XR-only)
         # equirect_yaw_deg: 0.0  # equirect: heading the middle of the feed points at
         # compositor: openxr    # openxr (default) | televiz — quads only
         # cylinder_radius_m: 2.0
         # cylinder_angle_deg: 90

See the :code-dir:`configs/ <examples/camera_viz/configs>` directory for a complete, commented
YAML per source kind.

Troubleshooting
---------------

- **The XR session fails to create** — check ``~/.cloudxr/logs/cxr_server.*.log`` and
  ``runtime_stderr.log`` for the startup failure. Pass ``--mode window`` to render to a
  desktop window instead (no runtime involved).
- **No window appears over SSH** — ``--mode window`` needs a local display; run on the machine
  you're sitting at, or use a video-capable remote desktop.
- **"video source: no such file"** — relative ``path:`` values resolve against the YAML's
  directory (``configs/``), not the directory you launched from.
- **A source fails asking for CuPy / CUDA** — check ``nvidia-smi`` works and setup completed;
  all sources allocate their frame buffers on the GPU.
- **No video on Orin.** When the CloudXR runtime runs on Jetson Orin, set
  **Video Codec** to **H.264** in the CloudXR web client. See
  :ref:`connect-xr-headset`.
- **Black camera plane on Orin** — with the CloudXR experimental runtime on Orin, the default
  OpenXR compositor shows the camera plane but leaves its contents black. Under the camera's
  ``display.placements`` entry, set ``compositor: televiz`` (quads only). The same feed displays
  correctly with this workaround:

  .. code-block:: yaml

     display:
       placements:
         cam:
           compositor: televiz

- **Split mode renders nothing** — check the sender is up (``./camera_viz.sh service-status``),
  ``$STREAMING_HOST`` was the workstation's IP at deploy time, and UDP ports (default 5000+)
  aren't firewalled.
- **Not sure which side is stuck?** — set ``verbose: true`` at the top of the YAML for periodic
  per-source breadcrumbs on both ends.

How it works
------------

The sample is organized so that capture, transport, and visualization are cleanly separated, with
Televiz as the compositor at the end of the chain:

.. code-block:: text
   :class: code-100col

   camera_viz/
   ├── camera_viz.sh        — CLI: setup / loopback / run / deploy / service-*
   ├── camera_viz.py        — receiver / viewer (drives a Televiz VizSession)
   ├── camera_streamer.py   — robot-side RTP sender (per-camera supervisor)
   ├── pipeline/            — source ABC + threaded runner
   ├── placements/          — XR lock-mode strategies (world / head / lazy / gimbal)
   ├── sources/             — V4L2 / OAK-D / ZED / video replay / synthetic / rtp_h264
   ├── transports/          — RTP sender + receiver (native + GStreamer)
   ├── codec/               — native NVENC / NVDEC pybind module
   ├── configs/             — one YAML per source kind
   ├── test_data/           — sample replay clip (Git LFS)
   └── scripts/             — installer + systemd unit template

- **Sources** (:code-dir:`sources/ <examples/camera_viz/sources>`) implement a common source ABC and
  hand frames to a threaded runner in :code-dir:`pipeline/ <examples/camera_viz/pipeline>`. Each
  source produces GPU frames where possible — e.g. the ZED source uses ``retrieve_image(MEM.GPU)`` so
  BGRA8 stays in VRAM and a CUDA kernel channel-swaps into contiguous RGBA with no host round-trip.
- **The viewer** (:code-file:`camera_viz.py <examples/camera_viz/camera_viz.py>`) creates a
  ``VizSession`` and adds one ``QuadLayer`` per enabled camera, then submits each frame to its layer
  and calls ``render()`` once per frame. Stereo cameras submit both eyes.
- **Transport** (:code-dir:`transports/ <examples/camera_viz/transports>`) carries the split mode:
  an RTP H.264 sender on the robot and a receiver on the workstation, with a native NVENC/NVDEC codec
  module (:code-dir:`codec/ <examples/camera_viz/codec>`) or a GStreamer fallback.
- **Placement** (:code-dir:`placements/ <examples/camera_viz/placements>`) holds the XR lock-mode
  strategies. Placement is application policy — Televiz only renders a layer at whatever pose the app
  sets each frame.

Sharing the XR session with TeleopSession
-----------------------------------------

Only one OpenXR session is allowed per process, so when this sample runs alongside teleoperation the
``VizSession`` owns the session and hands its handles to ``TeleopSession`` / ``DeviceIOSession``,
which then skip creating their own. The viewer builds the session with the trackers' required
extensions and forwards the handles through ``TeleopSessionConfig.oxr_handles``:

.. code-block:: python

   cfg.required_extensions = DeviceIOSession.get_required_extensions(trackers)
   viz_session = televiz.VizSession.create(cfg)

   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       oxr_handles=OpenXRSessionHandles(*viz_session.get_oxr_handles()),
   )

See the *Sharing the XR session* section of :doc:`/getting_started/televiz` for the full pattern
(imports, frame loop, and how the two sessions' lifecycles relate), and
:doc:`/getting_started/teleop_session` for the ``TeleopSession`` side.
