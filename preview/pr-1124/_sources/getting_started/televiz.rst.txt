.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Televiz
=======

Televiz (``isaacteleop.viz``) is the visualization module for Isaac Teleop. It composites what the
operator sees — camera and sensor feeds, plus 3D rendered content such as gsplat or nvblox — and
presents it in stereo to an XR headset over :doc:`CloudXR </references/cloudxr>`. Desktop-window and
offscreen output are also available, mainly for development and debugging.

It is a **compositor**, not a capture or streaming layer: it consumes GPU frames and assembles them
into a final image. Camera capture, decode, and network transport live in the application (see
:doc:`/references/camera_streaming`).

The compositor is implemented in C++ (``namespace viz``, built on Vulkan + OpenXR + CUDA with no
external rendering-framework dependency) and exposed through a pybind11 binding. This page uses the
Python API, which mirrors the C++ names one-to-one — see `C++ API`_.

.. contents:: On this page
   :local:
   :depth: 2

Installation
------------

Televiz ships inside the ``isaacteleop`` wheel — install it from PyPI:

.. code-block:: bash

   pip install isaacteleop

The published wheels (Linux x86_64 / aarch64, CPython 3.11–3.13) bundle the compiled
``isaacteleop.viz`` module, so **no source build is required**. Verify with:

.. code-block:: python

   import isaacteleop.viz as televiz

You only need to :doc:`build from source </getting_started/build_from_source/index>` when
developing Isaac Teleop itself — that build enables Televiz automatically when Vulkan and the
CUDA Toolkit are detected. ``glslangValidator`` is additionally required to compile the shaders:
if it is missing, configuration fails rather than quietly dropping the module — install it, or
pass ``-DBUILD_VIZ=OFF``.

Overview
--------

The central object is :code-file:`VizSession <src/viz/session/cpp/inc/viz/session/viz_session.hpp>`,
which owns the Vulkan context, the display target, the OpenXR session (in XR mode), and a registry
of **layers**. Content producers submit GPU buffers to layers; the session composites every layer
into one frame each time you call ``render()``.

Four layer types are available:

* :code-file:`QuadLayer <src/viz/layers/cpp/inc/viz/layers/quad_layer.hpp>` — a CUDA-fed 2D texture
  plane (mono or stereo), optionally placed in 3D space. Use it for camera feeds.
* :code-file:`CylinderLayer <src/viz/layers/cpp/inc/viz/layers/cylinder_layer.hpp>` — the same
  CUDA-fed texture curved onto the inside of a cylinder arc, so wide-FOV feeds keep a constant
  viewing distance edge to edge. XR-only.
* :code-file:`EquirectLayer <src/viz/layers/cpp/inc/viz/layers/equirect_layer.hpp>` — an
  equirectangular texture mapped onto the inside of a sphere, for 360°/180° panorama and VR-video
  sources. XR-only.
* :code-file:`ProjectionLayer <src/viz/layers/cpp/inc/viz/layers/projection_layer.hpp>` — a full-view
  RGBD layer for external renderers (gsplat, nvblox, neural reconstruction) that produce per-view
  ``(color, depth)`` buffers. Use it to present a rendered 3D scene from the current head pose.

A session holds **either** one ``ProjectionLayer`` **or** any number of texture layers
(``QuadLayer`` / ``CylinderLayer`` / ``EquirectLayer``), never both — see `Layers`_.

All symbols are imported from the top-level module::

   import isaacteleop.viz as televiz

Display modes
-------------

A session runs in exactly one display mode, set on the config:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``DisplayMode``
     - Behavior
   * - ``kXr``
     - OpenXR + Vulkan. Per-eye swapchains, stereo rendering, depth composition layer. Requires a
       running OpenXR runtime (e.g. CloudXR).
   * - ``kWindow``
     - GLFW desktop window. Layers are aspect-fit tiled; stereo layers show the left eye.
   * - ``kOffscreen``
     - No display. Composite to an internal target and pull pixels back with
       ``readback_to_host()``. Useful for tests and headless rendering.

Quick start
-----------

A minimal offscreen render-and-readback (no GPU display, no headset):

.. code-block:: python

   import cupy as cp
   import isaacteleop.viz as televiz

   viz_cfg = televiz.VizSessionConfig()
   viz_cfg.mode = televiz.DisplayMode.kOffscreen
   viz_cfg.window_width = 1024
   viz_cfg.window_height = 1024

   session = televiz.VizSession.create(viz_cfg)

   layer_cfg = televiz.QuadLayerConfig()
   layer_cfg.name = "cam"
   layer_cfg.resolution = televiz.Resolution(1024, 1024)
   layer = session.add_quad_layer(layer_cfg)

   # Any __cuda_array_interface__ array (CuPy / PyTorch / Numba) or a VizBuffer.
   frame = cp.zeros((1024, 1024, 4), dtype=cp.uint8)   # RGBA8
   layer.submit(frame)

   info = session.render()                # wait + composite + present
   img = session.readback_to_host()       # HostImage; numpy.asarray(img) for pixels

   session.destroy()

For a window or headset, set ``mode`` to ``DisplayMode.kWindow`` or ``DisplayMode.kXr`` instead. The
layer setup is identical; you just drive a frame loop (see `Frame loop`_) rather than the one-shot
``readback_to_host()``, which is offscreen-only.

Session configuration
---------------------

``VizSessionConfig`` fields:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Field
     - Default
     - Description
   * - ``mode``
     - ``DisplayMode.kOffscreen``
     - ``DisplayMode.kXr`` / ``kWindow`` / ``kOffscreen``. Set it explicitly; the default is
       offscreen.
   * - ``window_width`` / ``window_height``
     - ``1024``
     - Render size for window and offscreen modes. Ignored in XR (the runtime dictates per-eye
       resolution; query it with ``get_recommended_resolution()``).
   * - ``app_name``
     - ``"televiz"``
     - OpenXR application name.
   * - ``required_extensions``
     - ``[]``
     - Extra OpenXR instance extensions to enable when Televiz hosts the session and downstream
       components (e.g. ``TeleopSession`` trackers) need them. Televiz already enables its own
       rendering extensions. See `Sharing the XR session`_.
   * - ``xr_near_z`` / ``xr_far_z``
     - ``0.05`` / ``100.0``
     - Near / far planes for the XR projection.
   * - ``xr_system_wait_seconds``
     - ``0``
     - How long to wait for the OpenXR system (headset) to become available at create time.
       Zero fails fast when no headset is present.
   * - ``clear_color``
     - ``(0, 0, 0, 0)``
     - Background color as an ``(r, g, b, a)`` sequence in ``[0, 1]``. Transparent by default.
   * - ``gpu_timing``
     - ``False``
     - Enable GPU timestamp queries, surfaced via ``get_gpu_timing()``.

Construct the session with the factory; never call the class directly:

.. code-block:: python

   session = televiz.VizSession.create(cfg)

Layers
------

Layers come in two families, and a session holds one family or the other:

* **Texture layers** — ``QuadLayer``, ``CylinderLayer``, ``EquirectLayer``. A session can hold any
  number of them. All three take the same configuration, are fed by the same ``submit()``, and are
  composited the same way; they differ *only* in the surface the texture is mapped onto — a flat
  plane, a curved arc, or a sphere. The three sections at the end of this chapter cover just that
  difference; everything before them applies to all of them.
* **Projection layer** — one ``ProjectionLayer``, a full-view RGBD layer for an in-loop renderer.
  A session holds at most one, and never alongside a texture layer. See `ProjectionLayer`_.

Layers render in **insertion order** — the first added renders first (underneath). A layer is owned
by the session; ``add_quad_layer`` returns a **non-owning** handle, so don't keep it past the
session's lifetime.

.. _openxr-composition-layers:

Composition model
^^^^^^^^^^^^^^^^^

In XR, each texture layer is handed to the OpenXR runtime as its own composition layer
(``XrCompositionLayerQuad`` / ``CylinderKHR`` / ``Equirect2KHR``) rather than drawn by Televiz; the
runtime places, samples, and reprojects it at display rate. That keeps text and fine detail sharp
under head motion, and it lets CloudXR stream color-only video when every visible layer is
runtime-composited and opaque — the headset then rebuilds the composition from the layer geometry
at lower bandwidth.

Three consequences, in rough order of how often they bite:

* **Runtime-composited layers carry no depth.** They blend in insertion order, so a foreground
  overlay added before a camera feed renders *behind* it no matter where the two sit in 3D. Add
  backgrounds — an ``EquirectLayer`` panorama, say — first.
* **Alpha is ignored unless you ask for it.** ``alpha_blend`` (default off) makes the runtime honor
  the texture's alpha channel — turn it on for translucent content such as a HUD. Keep it off on
  opaque camera feeds: as long as no visible layer uses source alpha, CloudXR can stream the layers
  themselves and let the headset rebuild the composition, rather than sending a fully composited
  stereo frame. That is less to encode and less on the wire every frame — lower bandwidth, and
  lower latency on a constrained link. A single visible alpha-blended layer disqualifies the whole
  frame.
* **Mipmaps only exist on the built-in path.** ``generate_mipmaps`` applies when Televiz does the
  drawing; the runtime samples native layers itself. On the default XR path the flag has no effect.

``QuadLayer`` is the one layer with a choice, which is why ``openxr_composition`` and
``generate_mipmaps`` appear on its config and on no other. Set ``openxr_composition = False`` to
draw the quad with Televiz's built-in compositor instead, where 3D-placed quads depth-test against
each other in a shared render target — the way out of the no-depth constraint above. Cylinder and
equirect layers have no such switch: they exist only as native runtime layers, and therefore only in
XR. Window and offscreen modes always use the built-in compositor.

Shared configuration
^^^^^^^^^^^^^^^^^^^^

``QuadLayerConfig``, ``CylinderLayerConfig``, and ``EquirectLayerConfig`` all carry these fields:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Field
     - Default
     - Description
   * - ``name``
     - —
     - Layer name (used as the placement key in app config).
   * - ``resolution``
     - —
     - Source texture size, a ``Resolution``. Submitted buffers must match it.
   * - ``format``
     - —
     - ``PixelFormat`` of the source. ``kRGBA8`` is the only value texture layers accept;
       anything else raises ``ValueError``.
   * - ``placement``
     - —
     - Where and how big the surface is. Each layer has its own placement type describing its own
       geometry — see the per-layer sections below.
   * - ``stereo``
     - ``False``
     - Per-eye stereo. When ``True``, ``submit`` requires both eyes' buffers; view 0 (left) samples
       the left buffer, view 1 (right) the right. Memory doubles.
   * - ``stereo_baseline_mm``
     - ``0``
     - Horizontal disparity between the per-eye surfaces (mm), along the placement's local +x axis.
       ``0`` → both eyes see the same surface in the world, and any parallax comes from the frames
       themselves. XR + stereo only, and no effect on an infinite-radius equirect sphere.
   * - ``alpha_blend``
     - ``False``
     - Honor the texture's alpha channel (translucent content). Runtime composition only; ignored
       on the built-in compositor path.

Stereo works the same way on every surface, following the VR-video convention: per-eye textures on
the *same* surface, with ``stereo_baseline_mm`` adding an optional per-eye pose shift on top.

Submitting frames
^^^^^^^^^^^^^^^^^

Every texture layer (``QuadLayer`` / ``CylinderLayer`` / ``EquirectLayer``) takes content through
the same ``submit()``. What a frame must be:

* **GPU-resident.** A ``VizBuffer`` in device memory, or any object exposing
  ``__cuda_array_interface__`` (CuPy, PyTorch, Numba). A host buffer is rejected — the teleop hot
  path never round-trips through system memory.
* **RGBA8**, matching the layer's ``format``.
* **Exactly the layer's resolution.** Submitted dimensions are checked against ``resolution``, not
  scaled — resize upstream.

Three properties of ``submit()`` decide how you structure the producer around it:

**It is a latest-wins mailbox, not a queue.** Each layer owns a small ring of device images.
``submit()`` copies your pixels into a slot no in-flight frame is reading, then publishes it; the
renderer always picks up the most recent completed publish. A producer faster than the display
rate therefore **drops** frames rather than queueing or blocking, and a producer slower than the
display rate has its last frame re-presented. Neither side waits on the other, and no frame is
ever shown half-updated.

**It blocks until the copy completes.** ``submit()`` synchronizes ``stream`` before returning, so
your source buffer is free to reuse the moment it does — no fences to manage, no lifetime rules
past the call. The cost lands on the calling thread (see `Performance and diagnostics`_), not on
the render path, which is why capture threads should call ``submit()`` directly rather than
funnelling frames to the render thread.

**One producer per layer.** ``submit()`` is safe against the renderer consuming the same layer
concurrently, but *not* against a second thread submitting to the same layer. Give each producer
its own layer.

.. warning::

   **Stereo submits carry a stream precondition.** ``submit(left, right, stream=...)`` copies both
   eyes on the single ``stream`` you pass. CUDA orders work only within one stream, so if either
   buffer was produced on a *different* stream, synchronize that stream first — either
   ``cudaStreamSynchronize`` on the producer stream, or record an event there and
   ``cudaStreamWaitEvent`` it on ``stream``. Skip it and that eye can be copied mid-write: torn or
   stale pixels, with no error raised. The in-tree ZED and OAK-D sources synchronize per eye
   before publishing, which is what makes their plain ``submit(left, right)`` (stream 0) safe.


QuadLayer
^^^^^^^^^

A flat plane — the default surface, and the right one for normal-FOV camera feeds. Beyond the
`Shared configuration`_ fields, ``QuadLayerConfig`` adds the two composition knobs, which exist here
because the quad is the only layer that can be drawn either way:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Field
     - Default
     - Description
   * - ``generate_mipmaps``
     - ``True``
     - Allocate + regenerate a capped mip chain each frame; sampler uses trilinear filtering. Only
       applies on the built-in compositor path.
   * - ``openxr_composition``
     - ``True``
     - ``True`` = the OpenXR runtime composites the quad, ``False`` = Televiz's built-in
       compositor. See `Composition model`_.

Its placement is a ``QuadLayerPlacement`` — a ``pose`` plus ``size_meters``. It is optional:
a quad with no placement fills the window in window mode.

.. code-block:: python

   layer = session.add_quad_layer(layer_cfg)

   # Mono: pass exactly one buffer. Stereo: layer.submit(left, right).
   layer.submit(rgba_array)            # optional: stream=<cuda stream ptr>

   # 3D placement (XR). Pose is OpenXR stage space: position (x,y,z),
   # orientation quaternion (w,x,y,z). size_meters is (width, height).
   placement = televiz.QuadLayerPlacement(
       televiz.Pose3D(position=(0.0, 0.0, -1.5), orientation=(1.0, 0.0, 0.0, 0.0)),
       size_meters=(1.0, 0.5625),
   )
   layer.set_placement(placement)
   layer.set_visible(True)

Lock-mode placement strategies (``world`` / ``head`` / ``lazy`` / ``gimbal``) are **application
policy** and ship in the sample, not in the module.

CylinderLayer
^^^^^^^^^^^^^

The texture curved onto the inside of a vertical cylinder arc. Every point on the surface sits at
the same distance from the cylinder's axis, which makes it the natural surface for wide-FOV camera
feeds — a flat quad wide enough for a 110° image would put its edges much farther from the eye than
its center. ``CylinderLayerPlacement`` describes the arc:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Field
     - Default
     - Description
   * - ``pose``
     - identity
     - Cylinder center; the arc bows out along the pose's ``-z``, cylinder axis is ``+y``.
   * - ``radius_m``
     - ``1.0``
     - Radius in meters — the viewing distance to the surface. ``0`` / ``+inf`` = infinite.
   * - ``central_angle_rad``
     - ``π/2``
     - Visible arc in radians, ``(0, 2π)``.
   * - ``aspect_ratio``
     - ``0``
     - Arc width / height. ``0`` derives it from ``resolution`` (square texels).

XR only, and the runtime must support ``XR_KHR_composition_layer_cylinder`` (CloudXR does).
``add_cylinder_layer`` raises ``ValueError`` otherwise.

EquirectLayer
^^^^^^^^^^^^^

An equirectangular texture mapped onto the inside of a sphere centered on (by default) the
operator, for 360°/180° panoramas and VR-video sources. The ``EquirectLayerPlacement`` defaults
describe a full 360°×180° sphere at infinite radius, so a panorama needs no placement at all:

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Field
     - Default
     - Description
   * - ``pose``
     - identity
     - Sphere center; the texture's horizontal center maps to the pose's ``-z``.
   * - ``radius_m``
     - ``0``
     - Radius in meters; ``0`` / ``+inf`` = infinite sphere.
   * - ``central_horizontal_angle_rad``
     - ``2π``
     - Horizontal span (``π`` for VR180).
   * - ``upper_vertical_angle_rad`` / ``lower_vertical_angle_rad``
     - ``π/2`` / ``−π/2``
     - Vertical span from the horizon, upper > lower.

XR only, like the cylinder; the required extension here is
``XR_KHR_composition_layer_equirect2``.

Combining the two — a panorama background with a camera feed on an arc in front of it, background
added first so it composites underneath:

.. code-block:: python

   eq_cfg = televiz.EquirectLayerConfig()
   eq_cfg.name = "sky"
   eq_cfg.resolution = televiz.Resolution(4096, 2048)
   sky = session.add_equirect_layer(eq_cfg)          # default = full sphere

   cyl_cfg = televiz.CylinderLayerConfig()
   cyl_cfg.name = "cam"
   cyl_cfg.resolution = televiz.Resolution(1920, 1080)
   cyl_cfg.placement = televiz.CylinderLayerPlacement(radius_m=2.0)   # 2 m arc in front of the user
   cam = session.add_cylinder_layer(cyl_cfg)

   while running:
       sky.submit(panorama_rgba)
       cam.submit(camera_rgba)
       session.render()

ProjectionLayer
^^^^^^^^^^^^^^^

A full-view RGBD layer for **in-loop** renderers — gsplat, nvblox, or neural reconstruction engines
that produce per-view ``(color, depth)`` buffers. Configure it with ``ProjectionLayerConfig``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``name``
     - Layer name.
   * - ``view_resolution``
     - Per-view render resolution. **Must equal** ``session.get_recommended_resolution()`` — the
       layer's images are copied 1:1 into the presentation swapchains (per-eye in XR). A mismatch
       is rejected by ``add_projection_layer``.
   * - ``color_format``
     - ``PixelFormat.kRGBA8``.
   * - ``depth_format``
     - ``PixelFormat.kD32F`` (default) so the depth reaches the XR runtime for positional
       reprojection, or ``None`` to present color only.
   * - ``stereo``
     - ``True`` for per-eye buffers. A stereo (XR) display **requires** a stereo layer; a mono layer
       is rejected at ``add_projection_layer``.

Unlike ``QuadLayer``, a projection layer is **direct-present**: each view's ``(color, depth)`` is
copied straight into the presentation swapchains (no shared render target). Because of that a session
holds *either* one ``ProjectionLayer`` *or* any number of texture layers, never both.

The renderer runs **in-loop** with the frame loop: read the predicted view poses from the
``FrameInfo`` returned by ``begin_frame()``, render against them, then ``submit()`` before
``end_frame()``:

.. code-block:: python

   cfg = televiz.ProjectionLayerConfig()
   cfg.view_resolution = session.get_recommended_resolution()
   cfg.stereo = session.is_xr_mode()
   layer = session.add_projection_layer(cfg)

   while running:
       info = session.begin_frame()
       if info.should_render:
           # Render against THIS frame's per-eye poses (info.views[i].pose + .fov).
           color, depth = renderer.render(info.views)        # RGBA8 + D32F CUDA buffers
           if layer.stereo:
               layer.submit(left_color, left_depth, right_color, right_depth, stream=cuda_stream)
           else:
               layer.submit(color, depth, stream=cuda_stream)
       session.end_frame()

If the renderer is slower than display rate, the runtime / CloudXR paces the app via ``xrWaitFrame``
and reprojects the last submitted frame at display rate. In XR, a visible layer that does **not**
submit for a frame presents nothing (the swapchains are cleared) rather than reproject stale RGBD
under a new pose.

Frame loop
----------

Two API levels drive the frame loop. Both release the GIL during blocking waits.

**Convenience** — ``render()`` does wait + composite + present in one call and returns a
``FrameInfo``. Internally it checks ``should_render`` and skips the GPU pass when the runtime
says the frame won't be visible; producers' ``submit`` writes still land in the back buffer.

.. code-block:: python

   while running:
       cam_layer.submit(camera_frame)
       info = session.render()

**Explicit** — ``begin_frame()`` / ``end_frame()``, when the app needs the
``FrameInfo`` *before* submitting (e.g. to read the predicted view poses before rendering, or to
skip expensive decode when not visible):

.. code-block:: python

   while running:
       info = session.begin_frame()
       if info.should_render:
           cam_layer.submit(decode_camera())   # skip decode when not visible
       session.end_frame()

``FrameInfo`` carries ``frame_index``, ``predicted_display_time`` (XR time in ns; 0 outside
XR), ``delta_time`` (CPU wall-clock seconds — usable without any XR knowledge), ``should_render``,
``resolution``, and ``views``. Each ``ViewInfo`` in ``views`` has ``viewport``, ``fov``, and
``pose`` — 2 entries in XR stereo, 1 (identity pose) in window / offscreen.

Performance and diagnostics
---------------------------

``get_frame_timing_stats()`` is the first thing to reach for when the view stutters. It costs
nothing to call and needs no configuration:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``render_fps``
     - Smoothed frame rate over a recent window — what you are actually achieving.
   * - ``target_fps``
     - What you should be achieving: the headset's display rate in XR (60 / 72 / 90 / 120), the
       vsync rate in window mode.
   * - ``missed_frames``
     - Cumulative count of frames whose GPU work didn't finish inside the budget. Rising while
       ``render_fps`` holds means you are on the edge; rising with ``render_fps`` below
       ``target_fps`` means you are over it.
   * - ``avg_frame_time_ms``
     - Mean total frame time.
   * - ``gpu_time_ms``
     - Last frame's GPU-side render time.
   * - ``stale_layers``
     - Layers whose producer missed the stale timeout this frame — the signal that a *capture*
       thread, not the compositor, is behind. Non-zero here points you upstream.

``get_gpu_timing()`` breaks the GPU side into ``total_ms`` / ``render_pass_ms`` / ``post_pass_ms``.
It requires ``gpu_timing = True`` on ``VizSessionConfig`` (timestamp queries are off by default);
without it the fields read zero.

.. code-block:: python

   last_missed = 0

   while running:
       session.render()

       stats = session.get_frame_timing_stats()
       if stats.missed_frames > last_missed or stats.stale_layers:
           print(f"{stats.render_fps:.1f}/{stats.target_fps:.0f} fps, "
                 f"gpu {stats.gpu_time_ms:.2f} ms, stale layers {stats.stale_layers}")
       last_missed = stats.missed_frames

Session state
-------------

A session moves through ``SessionState``:

``kUninitialized → kReady → kRunning → kStopping → kLost → kDestroyed``

- ``kReady`` after ``create`` — add layers and submit content.
- ``kRunning`` once the frame loop is active.
- ``kStopping`` (XR) — the runtime is stopping; ``end_frame`` submits empty frames.
- ``kLost`` (XR) — the session was lost; ``render`` / ``begin_frame`` raise. Destroy and recreate
  the ``VizSession`` (Televiz supports clean in-process recreation).
- ``kDestroyed`` after ``destroy``.

Query it with ``get_state()``; in window mode ``should_close()`` reports the window-close
request. OpenXR events are polled inside ``begin_frame``, which drives the XR-specific transitions.

.. _sharing-the-xr-session:

Sharing the XR session
----------------------

Only one OpenXR session is allowed per process. In XR mode ``VizSession`` creates a **graphics-bound**
session (Isaac Teleop's own ``OpenXRSession`` is headless and cannot render). When you use Televiz
*and* ``TeleopSession`` together, let Televiz own the session and hand its live handles to
``TeleopSession`` so trackers attach to the same session — one CloudXR connection, synchronized
timing.

Declare the extensions your trackers need in ``required_extensions`` (Televiz adds its own rendering
extensions automatically), then pass the handles through:

.. code-block:: python

   import isaacteleop.viz as televiz
   from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
   from isaacteleop.deviceio import DeviceIOSession
   from isaacteleop.oxr import OpenXRSessionHandles

   viz_cfg = televiz.VizSessionConfig()
   viz_cfg.mode = televiz.DisplayMode.kXr
   # Aggregate the XR extensions downstream trackers need so they're present
   # on the XrInstance Televiz is about to create.
   viz_cfg.required_extensions = DeviceIOSession.get_required_extensions(trackers)
   viz_session = televiz.VizSession.create(viz_cfg)

   teleop_cfg = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       oxr_handles=OpenXRSessionHandles(*viz_session.get_oxr_handles()),
   )
   with TeleopSession(teleop_cfg) as session:
       while running:
           session.step()
           cam_layer.submit(camera_frame)
           viz_session.render()

``get_oxr_handles()`` returns ``(instance, session, space, proc_addr)`` as raw ``uint64``
values (or ``None`` outside ``kXr``); wrap them with ``OpenXRSessionHandles(*tuple)`` for
``TeleopSessionConfig.oxr_handles``. ``VizSession`` and ``TeleopSession`` keep **independent** state
machines and lifecycles — either can run without the other. In the unified pattern the underlying
OpenXR session is shared, so if it is lost both must be recreated. See
:doc:`teleop_session` for the ``TeleopSession`` side.

API reference
-------------

VizSession
^^^^^^^^^^

- ``create(config) -> VizSession`` *(static)* — validate config + initialize Vulkan / display backend.
- ``render() -> FrameInfo`` — wait + composite + present.
- ``begin_frame() -> FrameInfo`` / ``end_frame()`` — explicit two-phase frame loop.
- ``add_quad_layer(config) -> QuadLayer`` — construct + register a layer; returns a non-owning handle.
- ``add_cylinder_layer(config) -> CylinderLayer`` / ``add_equirect_layer(config) -> EquirectLayer`` —
  shaped texture layers (``kXr`` only; raise ``ValueError`` elsewhere).
- ``readback_to_host() -> HostImage`` — most recent frame as RGBA8 host pixels (``kOffscreen`` only).
- ``get_state() -> SessionState``, ``should_close() -> bool``, ``is_xr_mode() -> bool``.
- ``get_recommended_resolution() -> Resolution`` — runtime per-eye resolution (XR).
- ``head_pose_now() -> Optional[Pose3D]`` — current head pose (``kXr`` only; ``None`` on tracking loss).
- ``get_oxr_handles() -> Optional[tuple]`` — ``(instance, session, space, proc_addr)`` as raw ``uint64``.
- ``get_frame_timing_stats() -> FrameTimingStats`` / ``get_gpu_timing() -> GpuFrameTiming``.
- ``destroy()`` — release all resources (idempotent).
- Properties ``vk_device`` / ``vk_physical_device`` / ``vk_queue_family_index`` — raw handles for
  wiring Televiz into a foreign Vulkan app. Most users won't touch these.

QuadLayer / CylinderLayer / EquirectLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``submit(left, right=None, stream=0)`` — submit a frame (mono: ``left`` only; stereo: both).
- ``set_placement(placement)`` / ``placement()`` — placement swap, thread-safe vs the frame loop.
- ``set_stereo_baseline_mm(mm)`` / ``stereo_baseline_mm`` — live per-eye offset, thread-safe vs
  the frame loop; applies on the next frame. Inert while the layer is mono, since ``stereo`` is
  fixed at construction. Raises ``ValueError`` on a non-finite value.
- ``set_stereo_convergence_deg(deg)`` / ``stereo_convergence_deg`` (``CylinderLayer`` /
  ``EquirectLayer``) — the same idea as a per-eye yaw instead of a translation. Prefer it on a
  curved surface: translating one gives full disparity dead ahead and less toward the edges, and
  none at all at infinite radius, while a rotation is uniform across the arc and works at any
  radius.
  ``QuadLayer`` accepts ``None`` (fullscreen, window mode); the shaped layers validate and raise
  ``ValueError`` on bad shape parameters.
- ``set_visible(visible)`` / ``is_visible()``.
- Properties ``resolution``, ``format``, ``name`` (plus ``aspect_ratio`` on ``QuadLayer``).

Data types
^^^^^^^^^^

- ``VizBuffer`` — non-owning 2D pixel buffer descriptor. Device buffers expose
  ``__cuda_array_interface__`` (``cupy.asarray(buf)``); host buffers expose ``__array_interface__``
  (``numpy.asarray(buf)``).
- ``HostImage`` — owning host pixel buffer returned by ``readback_to_host``; wrap with
  ``numpy.asarray``.
- ``Resolution`` ``(width, height)``, ``Pose3D`` (``position``, ``orientation`` as
  ``(w, x, y, z)``), ``Fov``, ``ViewInfo``.
- Enums ``DisplayMode``, ``PixelFormat`` (``kRGBA8`` / ``kD32F``),
  ``MemorySpace``, ``SessionState``.

C++ API
-------

Televiz is a C++ library; ``isaacteleop.viz`` is a thin pybind11 binding over it. The Python and C++
APIs share the same type and method names (``VizSession``, ``QuadLayer``, ``submit``,
``set_placement``, ``DisplayMode::kXr``, …), so everything on this page maps directly to C++. All
symbols live in ``namespace viz``, and headers use nested include paths::

   #include <viz/session/viz_session.hpp>
   #include <viz/layers/quad_layer.hpp>
   #include <viz/core/viz_buffer.hpp>

``BUILD_VIZ`` auto-enables when Vulkan and the CUDA toolkit are detected
(force it with ``-DBUILD_VIZ=ON`` / ``-DBUILD_VIZ=OFF``); link the relevant CMake target:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Target
     - Alias
     - Provides
   * - ``viz_core``
     - ``viz::core``
     - Core types (``VizBuffer``, ``Pose3D``, ``HostImage``, ``DeviceImage``) and Vulkan / CUDA infrastructure
   * - ``viz_layers``
     - ``viz::layers``
     - The built-in layers (``QuadLayer``, ``CylinderLayer``, ``EquirectLayer``,
       ``ProjectionLayer``) and their shared ``ImageLayerBase`` mailbox
   * - ``viz_session``
     - ``viz::session``
     - ``VizSession``, the compositor, ``FrameInfo``, window / offscreen backends
   * - ``viz_xr``
     - ``viz::xr``
     - OpenXR backend — per-eye swapchains, depth composition layer

Public headers live under :code-dir:`src/viz/<module>/cpp/inc/viz/ <src/viz>`. One difference from
the Python bindings: in C++, layers are added with a single templated
``VizSession::add_layer<L>(args...)`` method, which also accepts your own ``LayerBase`` subclasses —
the route for plugging in a custom renderer. See :doc:`/references/build` for build options and
output locations.

More information
----------------

- :doc:`/references/camera_streaming` — the reference ``camera_viz`` sample built on Televiz
- :doc:`teleop_session` — how ``TeleopSession`` works and how to share its OpenXR session
- :code-dir:`src/viz/ <src/viz>` — module source, organized as ``core`` / ``layers`` / ``session`` /
  ``xr`` / ``shaders`` / ``python`` sub-modules
