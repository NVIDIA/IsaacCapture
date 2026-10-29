.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Teleop Session
==============

TeleopSessionManager provides a declarative, configuration-based API for
setting up complete teleop pipelines. It eliminates boilerplate code and makes
examples more readable by focusing on **what** you want to do rather than
**how** to set it up.

Overview
--------

The main component is :code-file:`TeleopSession <src/python/isaaccapture/teleop_session_manager/teleop_session.py>`, which manages the complete lifecycle
of a teleop session. It wraps the lower-level
:code-file:`DeviceIOSession <src/core/deviceio_session/cpp/inc/deviceio_session/deviceio_session.hpp>`
and :doc:`device trackers <../device/trackers/index>` so that callers don't need to
manage them directly:

#. Creates and configures trackers (head, hands, controllers)
#. Sets up OpenXR session with required extensions
#. Initializes and manages plugins
#. Runs retargeting pipeline with automatic updates
#. Handles cleanup via RAII

Quick Start
-----------

Here's a minimal example:

.. code-block:: python

   from isaaccapture.teleop_session_manager import (
       TeleopSession,
       TeleopSessionConfig,
   )
   from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaaccapture.retargeting_engine.interface import OutputCombiner
   from isaaccapture.retargeters import GripperRetargeter, GripperRetargeterConfig

   # Create source and build pipeline (one GripperRetargeter per side)
   controllers = ControllersSource(name="controllers")
   gripper_left = GripperRetargeter(
       GripperRetargeterConfig(hand_side="left"),
       name="gripper_left",
   )
   connected_left = gripper_left.connect({
       ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
   })
   gripper_right = GripperRetargeter(
       GripperRetargeterConfig(hand_side="right"),
       name="gripper_right",
   )
   connected_right = gripper_right.connect({
       ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
   })
   pipeline = OutputCombiner({
       "gripper_left": connected_left.output("gripper_command"),
       "gripper_right": connected_right.output("gripper_command"),
   })

   # Configure session
   config = TeleopSessionConfig(
       app_name="MyTeleopApp",
       pipeline=pipeline,
       trackers=[],  # Auto-discovered
   )

   # Run!
   with TeleopSession(config) as session:
       while True:
           result = session.step()
           # Access outputs
           left = result["gripper_left"][0]
           right = result["gripper_right"][0]

Configuration Classes
---------------------

.. note::
    Currently, to customize the teleop session, you need to configure it via Python code. We are
    working on a more declarative configuration API that takes a YAML file as input. Follow the
    following issue for progress: https://github.com/NVIDIA/IsaacCapture/issues/229

TeleopSessionConfig
^^^^^^^^^^^^^^^^^^^

The main configuration object:

.. code-block:: python

   @dataclass
   class TeleopSessionConfig:
       app_name: str                            # OpenXR application name
       pipeline: Any                            # Connected retargeting pipeline
       mode: SessionMode = SessionMode.LIVE     # LIVE, REPLAY or REMOTE
       trackers: List[Any] = []                 # Tracker instances (optional)
       plugins: List[PluginConfig] = []         # Plugin configurations (optional)
       verbose: bool = True                     # Print progress info
       oxr_handles: Optional[...] = None        # External OpenXR handles (optional)
       mcap_config: Optional[...] = None        # Required for REPLAY, optional for LIVE
       retargeting_execution: RetargetingExecutionConfig = ...
       remote_endpoint: Optional[str] = None    # Teleop Session Server (REMOTE only)
       remote_tick_rate_hz: float = 90.0        # Server ticks (tracker reads) per second
       remote_max_tick_age_s: float = 0.25      # Older samples count as missing
       remote_connect_timeout_s: Optional[float] = 60.0  # Wait for READY; None: no limit

When ``mode`` is ``SessionMode.REPLAY``, ``TeleopSession`` skips OpenXR
session creation and reads tracking data from the MCAP file specified in
``mcap_config`` (a ``McapReplayConfig``).  ``mcap_config`` is **required** in
replay mode — omitting it raises ``ValueError``.  In the default ``LIVE``
mode, ``mcap_config`` is optional; passing an ``McapRecordingConfig`` enables
recording to disk.  See :doc:`/references/mcap_record_replay` for full
details.

When ``oxr_handles`` is provided (live mode), ``TeleopSession`` uses the
supplied handles instead of creating its own OpenXR session. The caller is
responsible for the external session's lifetime. Construct handles with
``OpenXRSessionHandles(instance, session, space, proc_addr)`` where each
argument is a ``uint64`` handle value.

.. code-block:: python

   from isaaccapture.oxr import OpenXRSessionHandles

   handles = OpenXRSessionHandles(
       instance_handle, session_handle, space_handle, proc_addr
   )
   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       oxr_handles=handles,  # Skip internal OpenXR session creation
   )

.. _remote-sessions:

Remote sessions
"""""""""""""""

``SessionMode.REMOTE`` runs the robot app on a different machine from the
headset and CloudXR. A Teleop Session Server on the CloudXR machine reads the
trackers the robot app asks for and streams their data to it; ``step()``, the
retargeting pipeline and the robot commands stay on the robot machine. Run the
same Isaac Capture version on both machines: the robot app logs the server's
version when it connects and warns when it differs from its own.

Start the server on the CloudXR machine. It starts CloudXR if none is running
and listens on TCP port 47600 by default:

.. code-block:: bash

   python -m isaaccapture.teleop_session_server

Then point the robot app at it:

.. code-block:: python

   config = TeleopSessionConfig(
       app_name="MyRobotApp",
       pipeline=pipeline,
       plugins=[...],                       # started by the server
       mode=SessionMode.REMOTE,
       remote_endpoint="cloud-host:47600",  # or tcp://host:port, ws://host:port[/path]
   )
   with TeleopSession(config) as session:
       while True:
           result = session.step()

- **Requirements.** The robot machine needs no CloudXR and no OpenXR runtime,
  so do not wrap a remote session in ``CloudXRLauncher.launch_context`` as the
  live examples do. Both machines run Linux: remote sessions use POSIX sockets,
  the server starts only on Linux, and on Windows entering the session raises
  ``RuntimeError("Remote sessions require POSIX sockets; not supported on
  Windows")``.
- **Endpoint.** ``host:port`` and ``tcp://host:port`` connect to the server's
  TCP port (``--listen``, 47600 by default); ``ws://host:port[/path]`` connects
  to its WebSocket port (``--ws-listen``, usually 47601). With the wrong scheme
  for the port, entering the session fails at once: ``ws://`` on the TCP port
  with ``TimeoutError`` (``server did not answer (WebSocket upgrade to ...
  refused: HTTP/1.1 400 Bad Request)``), ``tcp://`` on the WebSocket port with
  ``RuntimeError("Remote session ended before HELLO_ACK: ...")``.
- **Startup.** Entering the session connects, sends the trackers and plugin
  requests, and returns once the server reports READY: plugins started and
  headset connected. All of it happens within ``remote_connect_timeout_s``
  (60 s; ``None`` waits without limit):

  - It retries while nothing accepts the connection, for example because the
    server is not up yet, and while the server closes or resets the connection
    before sending any message, for example because the server crashed or was
    killed before it started a session for this robot app. It logs ``Waiting
    for the Teleop Session Server at <endpoint> (<reason>)`` once and tries
    again about every 0.5 s. One TCP connect attempt takes at most 2 s; looking
    up the host name before it is not bounded.
  - Once connected, it waits for the server's answer. While the server is
    still ending the previous robot app's session, that lasts until the
    session's plugins have stopped, up to their ``shutdown_timeout_seconds``
    (``plugin.yaml``) added up; set ``remote_connect_timeout_s`` above that
    when a plugin needs long to stop.
  - When the time is up, it raises ``TimeoutError`` naming the stage it was
    stuck at: ``server did not answer`` (followed by the last error if the last
    attempt did not connect or was closed), ``connecting_openxr``,
    ``starting_plugins`` or ``waiting_for_headset``.
  - With ``remote_connect_timeout_s=None`` the wait has no limit. If the server
    machine disappears after accepting the connection but before answering,
    nothing tells the robot app until TCP gives up on the connection, which
    takes many minutes (over ``ws://`` it may never happen); Ctrl-C ends the
    wait.
  - It does not retry a host name that cannot be resolved (``RuntimeError``),
    a WebSocket upgrade the server refuses or answers with something other than
    HTTP (``TimeoutError``, ``server did not answer``), a server that sends part
    of a message and then closes or resets the connection (``RuntimeError``),
    or an error the server reports. The server's error is raised as
    ``RuntimeError`` with its text, for example ``busy: another robot app is
    connected to this Teleop Session Server``, ``The Teleop Session Server is
    shutting down`` or a required plugin it cannot find. A server stopped with
    Ctrl-C or ``SIGTERM``, for example for a restart, sends ``shutting down``
    to every robot app connected to it; a server that is killed, even with
    ``SIGKILL``, still sends it to a robot app it has started a session for. To
    wait out a busy or a restarting server, the robot app catches these errors
    and retries entering the session itself.
  - When the link is lost while it waits for READY, it raises
    ``RuntimeError("Remote session lost: ...")``, as ``step()`` does.
  - Ctrl-C raises ``KeyboardInterrupt`` within about 0.1 s, or at the end of the
    TCP connect attempt in progress (at most 2 s, plus any host name lookup), and
    closes the connection.

- **Ticks.** The server ticks ``remote_tick_rate_hz`` times per second (90 by
  default; it clamps to 1-240): each tick reads every tracker once and sends
  the samples to the robot app. ``step()`` never waits for the network: it
  uses the newest samples and reports their tick age, the time since the
  server read them, in ``session.last_step_info.remote_tick_age_s``. Samples
  whose tick age exceeds ``remote_max_tick_age_s`` (0.25 s) count as missing,
  so trackers report no data. A ``DefaultTeleopStateManager`` in
  ``teleop_control_pipeline`` whose buttons come from those trackers then
  switches to ``STOPPED`` (see :doc:`control_state_machine`). Without a
  ``teleop_control_pipeline`` the execution state stays ``RUNNING`` and the
  pipeline runs without tracker data, so the robot app decides what to do;
  ``remote_tick_age_s`` is ``None`` for such a step.
- **Lost link.** When nothing arrives for 3 s, ``step()`` raises
  ``RuntimeError("Remote session lost: ...")``; the session does not reconnect.
- **Plugins.** The server starts the enabled ``plugins`` by name from its own
  plugin directories. ``PluginConfig.search_paths`` must still be given but is
  ignored, and ``plugin_args``, paths included, are read on the server machine.
  A plugin that stops running on the server makes ``step()`` raise
  ``PluginCrashException``, as a local plugin crash does. As in live mode, a
  tracker reports no data once its plugin's data goes away, and keeps its last
  sample while the plugin runs but pushes nothing new.
- **Status.** ``get_status()`` shows the server's OpenXR session as the
  ``openxr/runtime`` provider and the headset as ``openxr/headset``, as in live
  mode; for example, the headset turns ``DISCONNECTED`` when the operator takes
  it off. With a controller input source, ``openxr/controller/left`` and
  ``openxr/controller/right`` follow the controller data, as in live mode.
  Each plugin the server started appears as ``plugin/<plugin_root_id>``
  with its process health only: its ``pid`` is a process id on the server
  machine, and device status from plugins (for example Manus gloves) is not
  forwarded yet.
- **Not available:** keyboard and message-channel sources, ``TactileSource``,
  MCAP recording (``mcap_config``), ``oxr_handles``, ``oxr_session_factory``,
  ``joint_publisher`` and ``session.oxr_session``. ``sinks`` run every step but
  are not flushed, so their data, for example haptics, is dropped; entering the
  session logs a warning. Trackers are limited to head, hands, controllers,
  full body and the plugin data trackers that read a ``collection_id`` and
  have no metadata. Tracker metadata is not forwarded yet, so creating the
  ``TeleopSession`` with a tracker that has metadata, such as
  ``TactileTracker``, raises ``ValueError``.
- **Security.** The link has no authentication and no encryption. Any peer that
  reaches the server's port can make it start its installed plugins with any
  arguments and receive tracker data. Run the server on a trusted network or a
  VPN, and open TCP 47600 (and 47601 for WebSocket) only toward the robot
  machine.

Teleop Session Server
"""""""""""""""""""""

The server runs on the CloudXR machine (Linux only) and serves one robot app at
a time:

.. code-block:: bash

   python -m isaaccapture.teleop_session_server \
       --listen tcp://0.0.0.0:47600 --ws-listen ws://0.0.0.0:47601 \
       --plugin-dir /opt/my_plugins

- ``--listen``: TCP address for robot apps (default ``tcp://0.0.0.0:47600``).
- ``--ws-listen``: also accept WebSocket connections, usually on port 47601
  (off by default). Robot apps cannot connect with ``wss://`` yet. To encrypt
  the link, run a tunnel between the two machines (VPN, SSH port forwarding or
  stunnel) and point ``remote_endpoint`` at the robot machine's end of it.
- ``--plugin-dir``: a directory of plugins that robot apps may start by name.
  The server searches the plugins bundled with Isaac Capture, then
  ``ISAAC_TELEOP_PLUGIN_PATH``, then each ``--plugin-dir``; if a plugin name is
  found in more than one, the last directory wins. Repeatable.
- The CloudXR options of the examples: ``--accept-eula``,
  ``--cloudxr-install-dir``, ``--cloudxr-env-config``,
  ``--cloudxr-device-profile``, ``--host-client`` / ``--no-host-client``, and
  ``--no-launch-cloudxr-runtime`` to use an OpenXR runtime that is already
  configured instead of attaching to or starting CloudXR.
- ``--verbose``: debug logging.

It prints ``Teleop Session Server listening on <address>`` for each address
once it accepts connections. Each robot app gets a session process that opens
the OpenXR session, reads the requested trackers and starts the requested
plugins; it ends, stopping those plugins, when the robot app disconnects.

- **One robot app at a time.** While a robot app is connected, the server
  refuses any other with ``busy: another robot app is connected to this Teleop
  Session Server``. A robot app that reconnects right after it left is served
  once its previous session has ended, which takes as long as that session's
  plugins need to stop. A robot app whose connection dropped without closing,
  for example when its machine lost the network, counts as connected until the
  server notices, which takes about 3 s; until then, a new connection gets the
  ``busy`` error.
- **Stopping.** Ctrl-C or ``SIGTERM`` stops the server and the session, which
  stops its plugins as ``TeleopSession`` does, and exits with code 0. The signal
  takes effect within about 0.2 s, or up to about 3 s while the server is
  refusing a connection that sends nothing. The robot app's ``step()`` then
  raises ``RuntimeError("The Teleop Session Server is shutting down")``, and so
  does entering the session for any other robot app connected to the server;
  neither is retried. From then on the server refuses connections, so a robot
  app that connects later keeps trying, within its
  ``remote_connect_timeout_s``, until a restarted server serves it. The server
  kills the session's processes only after its plugins had their
  ``shutdown_timeout_seconds`` (``plugin.yaml``, 2 s by default) to stop, one
  after another; a second Ctrl-C or ``SIGTERM`` kills them at once. When the
  server is killed instead, for example with ``SIGKILL``, robot apps waiting for
  a session see their connection close and keep trying, but the session process
  still stops its plugins and sends ``shutting down`` to its robot app. The
  server exits with code 1 when the CloudXR runtime it started or attached to
  stops.

Tracker vendor selection
""""""""""""""""""""""""

In live mode, a vendored source selects the backend ("vendor") for its tracker
-- for example which backend drives a ``FullBodySource``. The selection is
carried **on the source** via its ``vendor`` argument, a
``deviceio.TrackerVendor(id, params)``; sources left at the default use their
tracker's default vendor. Because the vendor travels with the source, it is part
of the pipeline: ``TeleopSession`` picks it up automatically, and
``get_required_oxr_extensions_from_pipeline(pipeline)`` reports the matching
OpenXR extensions with no extra argument (important for the external-``oxr_handles``
flow). See :ref:`vendor-selection` for the underlying DeviceIO mechanism and the
available vendor ids.

.. code-block:: python

   import isaaccapture.deviceio as deviceio
   from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource

   # Select the backend on the source; it flows through the pipeline into the session.
   full_body = FullBodySource(
       name="full_body", vendor=deviceio.TrackerVendor("body.pico-xr")
   )

Retargeting execution
"""""""""""""""""""""

``TeleopSession.step()`` runs retargeting synchronously by default: each call
updates DeviceIO, polls trackers, merges caller-provided inputs, runs optional
control logic, runs the main retarget pipeline, and returns the current frame's
``RetargeterIO`` before ``step()`` completes.

Set ``retargeting_execution=RetargetingExecutionConfig(mode="pipelined")`` to
opt into background retargeting. In pipelined mode, the application still calls
``step()`` once per simulation frame and supplies any external inputs, graph
time, or explicit execution events. IsaacTeleop owns one retarget worker thread.
After the first seed frame, that worker runs the normal synchronous step body
end-to-end.

Per application frame in pipelined mode, ``step()`` submits the current request
and returns the latest completed ``RetargeterIO``. On a normal hit this is often
the previous application frame's output. If that returned output came from
request N-1, then ``session.last_context`` also comes from N-1: tracker data,
control messages, execution events, outputs, and context stay self-consistent
for the completed frame that is returned. Use ``mode="sync"`` when the
application needs exact current-frame sample/compute/return behavior.

The return type is unchanged; inspect ``session.last_step_info`` for age and
timing metadata such as ``returned_frame_id``, ``submitted_frame_id``,
``returned_age_frames``, ``compute_duration_s``, ``ran_synchronously``, and
``dropped_submissions``. ``dropped_submissions`` is per-step; accumulate it in
the application when you need a run-wide total. ``frame_deadline_miss`` is a
per-step flag for outputs more than one submitted frame old, meaning the worker
missed the normal pipelined target of having a result ready for the next
application frame. Consumers that want totals should accumulate it.

``last_step_info`` fields:

- ``returned_frame_id`` / ``submitted_frame_id``: frame identifiers for the
  output returned and the request submitted by this call.
- ``returned_age_frames`` / ``returned_age_s``: how old the returned output is.
- ``compute_duration_s``: worker compute time for the returned output.
- ``ran_synchronously``: the returned output was computed on the application
  thread during this ``step()`` call. This is true in ``mode="sync"`` and for
  the pipelined seed frame.
- ``dropped_submissions``: unstarted pending requests replaced by this call;
  accumulate this field for a run-wide total.
- ``frame_deadline_miss``: returned output is more than one submitted frame old.
- ``worker_exception``: worker failure surfaced on this step, when present.
- ``remote_tick_age_s``: in ``SessionMode.REMOTE``, the tick age in seconds of
  the server samples the returned output was computed from, taken when that
  step updated the session; ``None`` when no fresh samples had arrived, and
  outside remote mode.

Because pipelined mode reuses completed frames, caller-provided external inputs
and returned outputs must be snapshot-copyable. ``TensorGroup.create_snapshot()``
handles standard scalar, array, and common DLPack-backed tensor cases; custom
opaque values should provide an equivalent ``create_snapshot()`` hook or run
with ``mode="sync"``. Pipelined mode returns a new snapshot of the cached
output on each ``step()`` call. Sync mode returns the pipeline result directly,
matching the historical ownership behavior.

.. code-block:: python

   from isaaccapture.teleop_session_manager import RetargetingExecutionConfig

   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       retargeting_execution=RetargetingExecutionConfig(mode="pipelined"),
   )

Key pipelined options:

These options affect pipelined mode only. In ``mode="sync"``, ``step()`` always
runs and returns the current frame directly.

- ``pacing=ImmediatePacingConfig()`` starts retarget work as soon as the worker
  receives a request. This is the default pacing when pipelined mode is enabled.
- ``pacing=DeadlinePacingConfig(safety_margin_s=0.015)`` delays worker
  requests toward the predicted next application frame using recent cadence and
  compute-spike estimates. This prepares results just in time for simulation
  consumption while preserving the pipelined submit-current/return-latest
  contract. The safety margin is the main tuning knob.

Pacing changes when the worker begins a submitted step request. It does not
change the one-worker/one-pending-request correctness model. Unstarted paced
work is coalesced so newer submissions replace older delayed requests.

``TeleopSession.step()`` is intended to be called by one application loop
thread. IsaacTeleop runs the retarget work on one background worker in
pipelined mode, but public session state such as ``frame_count``,
``last_context``, and ``last_step_info`` belongs to the application thread.

.. code-block:: python

   from isaaccapture.teleop_session_manager import (
       DeadlinePacingConfig,
       RetargetingExecutionConfig,
   )

   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       retargeting_execution=RetargetingExecutionConfig(
           mode="pipelined",
           pacing=DeadlinePacingConfig(safety_margin_s=0.015),
       ),
   )

Deadline pacing tuning
""""""""""""""""""""""

``DeadlinePacingConfig`` predicts the next application-frame deadline
from recent ``step()`` submissions, estimates retarget compute cost, and starts
the worker at roughly:

``predicted_next_step_time - estimated_retarget_cost - safety_margin_s``.

The goal is to avoid starting too early, which uses older inputs, while still
finishing before the application asks for the next action. If accumulated
``session.last_step_info.frame_deadline_miss`` counts rise, make the schedule
more conservative. If misses stay low but outputs use unnecessarily old inputs,
make the schedule less conservative.

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - What it changes
     - When to tune it
   * - ``safety_margin_s``
     - ``0.015``
     - Starts work this much earlier than the predicted deadline.
     - Tune this first. Increase it when ``frame_deadline_miss`` counts rise;
       decrease it when misses are near zero and you want more recent inputs.
   * - ``spike_guard_percentile``
     - ``0.90``
     - Chooses how conservative the compute-spike estimate is.
     - Raise it for rare slow retarget spikes; lower it for more recent inputs
       on stable workloads.
   * - ``spike_guard_window``
     - ``60``
     - Controls how many recent retarget durations feed the spike estimate.
     - Increase it when spikes repeat in bursts; decrease it when old spikes
       make the worker start early for too long.
   * - ``frame_period_adaptation``
     - ``0.2``
     - Controls how quickly the predicted application cadence follows changes.
     - Raise it when app frame rate changes quickly; lower it when cadence is
       stable but jittery.
   * - ``compute_cost_adaptation``
     - ``0.25``
     - Controls how quickly the estimated retarget cost follows recent compute
       time.
     - Raise it when retarget cost changes by mode/task; lower it when single
       slow frames cause overreaction.
   * - ``startup_frame_period_s``
     - ``0.022``
     - Initial frame-period guess before enough samples exist.
     - Rarely tune. Match it to expected app cadence only if startup behavior
       matters.
   * - ``startup_compute_cost_s``
     - ``0.005``
     - Initial retarget-cost guess before enough samples exist.
     - Rarely tune. Increase it only if the first few retargets are heavy.

Practical tuning:

- Light, stable load with near-zero misses: try
  ``DeadlinePacingConfig(safety_margin_s=0.005)`` to ``0.010``. This
  starts retargeting later so inputs are more recent.
- Typical interactive XR load: keep ``safety_margin_s=0.015``. This is the
  default balance between input recency and spike tolerance.
- Heavy retargeting or occasional compute spikes: try
  ``safety_margin_s=0.025`` to ``0.040``. If misses still come in bursts, also
  try ``spike_guard_percentile=0.95`` or ``spike_guard_window=120``.
- Variable app frame rate: try ``frame_period_adaptation=0.35`` to ``0.50`` so
  the predicted deadline catches up faster.
- Task modes that change retarget cost abruptly: try
  ``compute_cost_adaptation=0.40`` to ``0.60`` so the compute estimate catches
  up faster.

PluginConfig
^^^^^^^^^^^^

Configure plugins:

.. code-block:: python

   PluginConfig(
       plugin_name="controller_synthetic_hands",
       plugin_root_id="synthetic_hands",
       search_paths=[Path("/path/to/plugins")],
       enabled=True,
       plugin_args=["--arg1=val1", "--arg2=val2"],
       required=True,
   )

Plugins are optional by default: if none of their search paths exist or the
configured plugin is not discovered, session startup continues without them.
Set ``required=True`` to make either condition fail session startup with a
``RuntimeError``. Setting ``enabled=False`` always disables the plugin,
regardless of ``required``.

.. note::

   Any ``--plugin-root-id=...`` in ``plugin_args`` is ignored so that the
   ``plugin_root_id`` parameter cannot be overridden.

Plugins with device monitoring support appear as providers in the status
inventory. Other enabled plugins retain their existing launch behavior without
appearing there. See
:doc:`/references/device_provider_monitoring` for status identities, state
precedence, and plugin integration guidance.

API Reference
-------------

TeleopSession
^^^^^^^^^^^^^

Methods
"""""""

- ``step(*, external_inputs=None, graph_time=None, execution_events=None) -> Dict[str, TensorGroup]``
  -- Execute one step. In default sync mode, all work completes before
  ``step()`` returns. In pipelined mode, the first call seeds an initial output,
  then later calls submit DeviceIO update, tracker polling, ``external_inputs``
  merge, and retargeting work to the worker before returning the latest
  completed output; unstarted pending work may be replaced by a newer
  submission.
  ``graph_time`` can be provided explicitly; when omitted, monotonic time is used
  for both sim/real time. If ``execution_events`` is provided, it is injected into
  ``ComputeContext`` and ``teleop_control_pipeline`` is skipped for that step.
  Extra top-level external inputs and extra per-leaf keys are ignored. Raises
  ``ValueError`` if required external inputs are missing or collide with DeviceIO
  source names.
- ``get_external_input_specs() -> Dict[str, RetargeterIOType]`` -- Return the
  input specifications for all external (non-DeviceIO) leaf nodes that require
  caller-provided inputs in ``step()``.
- ``has_external_inputs() -> bool`` -- Whether this pipeline has external leaf
  nodes that require caller-provided inputs.
- ``get_status() -> StatusSnapshot`` -- Return the current immutable provider
  and device snapshot without performing I/O.
- ``get_provider_status(provider_id) -> ProviderStatus | None`` -- Return one
  cached provider record.
- ``get_device_status(device_id) -> DeviceStatus | None`` -- Return one cached
  device record.
- ``get_elapsed_time() -> float`` -- Get elapsed time since session started.

Example (explicit GraphTime + ExecutionEvents override):

.. code-block:: python

   import time

   from isaaccapture.retargeting_engine.interface.execution_events import (
       ExecutionEvents,
       ExecutionState,
   )
   from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime

   now_ns = time.monotonic_ns()
   result = session.step(
       graph_time=GraphTime(sim_time_ns=123_000_000, real_time_ns=now_ns),
       execution_events=ExecutionEvents(
           execution_state=ExecutionState.PAUSED,
           reset=False,
       ),
   )

Properties
""""""""""

- ``frame_count: int`` -- Current frame number
- ``start_time: float`` -- Session start time
- ``config: TeleopSessionConfig`` -- The configuration object
- ``oxr_session: Optional[OpenXRSession]`` -- The internal OpenXR session, or
  ``None`` when using external handles or outside ``LIVE`` mode (read-only)
- ``last_context: Optional[ComputeContext]`` -- Context corresponding to the
  most recent returned output
- ``last_step_info: RetargetingStepInfo`` -- Age and timing metadata for
  the most recent returned output

Helper Functions
^^^^^^^^^^^^^^^^

The module also exports two utility functions:

- ``get_required_oxr_extensions_from_pipeline(pipeline) -> List[str]`` --
  Discover the OpenXR extensions needed by a retargeting pipeline by
  traversing its DeviceIO source leaf nodes. Returns a sorted, deduplicated
  list of extension name strings. Extensions are vendor-dependent, but each
  source carries its own vendor selection, so the result already reflects them
  -- when you create the OpenXR session yourself and feed the handles in via
  ``oxr_handles``, the enabled extensions match the session that gets built
  with no extra argument. Select a vendor on the source (e.g.
  ``FullBodySource(name="full_body", vendor=deviceio.TrackerVendor("body.pico-xr"))``).

- ``create_standard_inputs(trackers) -> Dict[str, IDeviceIOSource]`` --
  Convenience function that creates ``HandsSource``, ``ControllersSource``,
  and/or ``HeadSource`` instances from a list of tracker objects.

Examples
--------

Complete Examples
^^^^^^^^^^^^^^^^^

#. **Simplified Gripper Example**: ``examples/teleop/python/gripper_retargeting_example_simple.py``
   -- Shows the minimal configuration approach and demonstrates auto-creation
   of input sources. Run it from the project root, with the environment set up
   as in the :doc:`Quick Start <quick_start>`:

   .. code-block:: bash

      python examples/teleop/python/gripper_retargeting_example_simple.py

Before vs After
^^^^^^^^^^^^^^^

**Before (verbose, manual setup):**

.. code-block:: python

   # Create trackers
   controller_tracker = deviceio.ControllerTracker()

   # Get extensions
   required_extensions = deviceio.DeviceIOSession.get_required_extensions([controller_tracker])

   # Create OpenXR session
   oxr_session = oxr.OpenXRSession("MyApp", required_extensions)
   oxr_session.__enter__()

   # Create DeviceIO session
   handles = oxr_session.get_handles()
   deviceio_session = deviceio.DeviceIOSession.run([controller_tracker], handles)
   deviceio_session.__enter__()

   # Setup plugins
   plugin_manager = pm.PluginManager([...])
   plugin_context = plugin_manager.start(...)

   # Setup pipeline
   controllers = ControllersSource(name="controllers")
   gripper_left = GripperRetargeter(
       GripperRetargeterConfig(hand_side="left"),
       name="gripper_left",
   )
   connected_left = gripper_left.connect({
       ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
   })
   gripper_right = GripperRetargeter(
       GripperRetargeterConfig(hand_side="right"),
       name="gripper_right",
   )
   connected_right = gripper_right.connect({
       ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
   })
   pipeline = OutputCombiner({
       "gripper_left": connected_left.output("gripper_command"),
       "gripper_right": connected_right.output("gripper_command"),
   })

   # Main loop
   while True:
       deviceio_session.update()
       # Manual data injection needed for new sources
       left_controller = controller_tracker.get_left_controller(deviceio_session)
       right_controller = controller_tracker.get_right_controller(deviceio_session)
       inputs = {
           "controllers": {
               "deviceio_controller_left": [left_controller],
               "deviceio_controller_right": [right_controller]
           }
       }
       result = pipeline(inputs)
       # ... error handling, cleanup ...

**After (declarative):**

.. code-block:: python

   # Configuration
   config = TeleopSessionConfig(
       app_name="MyApp",
       trackers=[controller_tracker],
       pipeline=pipeline,
   )

   # Run!
   with TeleopSession(config) as session:
       while True:
           result = session.step()  # Everything handled automatically!

Benefits
--------

#. **Reduced Boilerplate** -- ~70% reduction in code length
#. **Declarative** -- Focus on configuration, not implementation
#. **Auto-Initialization** -- Plugins and sessions all managed automatically
#. **Self-Documenting** -- Configuration structure makes intent clear
#. **Error Handling** -- Automatic error handling and cleanup
#. **Plugin Management** -- Built-in plugin lifecycle management
#. **Maintainable** -- Changes to setup logic happen in one place

Advanced Features
-----------------

Custom Update Logic
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   with TeleopSession(config) as session:
       while True:
           result = session.step()

           # Custom logic
           left_gripper = result["gripper_left"][0]
           if left_gripper > 0.5:
               print("Left gripper activated!")

           # Frame timing
           if session.frame_count % 60 == 0:
               print(f"Running at {60 / session.get_elapsed_time():.1f} FPS")
