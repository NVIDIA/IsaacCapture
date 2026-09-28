.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Teleop Session
==============

TeleopSessionManager provides a declarative, configuration-based API for
setting up complete teleop pipelines. It eliminates boilerplate code and makes
examples more readable by focusing on **what** you want to do rather than
**how** to set it up.

Overview
--------

The main component is :code-file:`TeleopSession <src/python/isaacteleop/teleop_session_manager/teleop_session.py>`, which manages the complete lifecycle
of a teleop session. It wraps the lower-level
:code-file:`DeviceIOSession <src/core/deviceio_session/cpp/inc/deviceio_session/deviceio_session.hpp>`
and :doc:`device trackers <../device/trackers>` so that callers don't need to
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

   from isaacteleop.teleop_session_manager import (
       TeleopSession,
       TeleopSessionConfig,
   )
   from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaacteleop.retargeting_engine.interface import OutputCombiner
   from isaacteleop.retargeters import GripperRetargeter, GripperRetargeterConfig

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
    following issue for progress: https://github.com/NVIDIA/IsaacTeleop/issues/229

TeleopSessionConfig
^^^^^^^^^^^^^^^^^^^

The main configuration object:

.. code-block:: python

   @dataclass
   class TeleopSessionConfig:
       app_name: str                            # OpenXR application name
       pipeline: Any                            # Connected retargeting pipeline
       mode: SessionMode = SessionMode.LIVE     # LIVE or REPLAY
       trackers: List[Any] = []                 # Tracker instances (optional)
       plugins: List[PluginConfig] = []         # Plugin configurations (optional)
       verbose: bool = True                     # Print progress info
       oxr_handles: Optional[...] = None        # External OpenXR handles (optional)
       mcap_config: Optional[...] = None        # Required for REPLAY, optional for LIVE
       retargeting_execution: RetargetingExecutionConfig = ...

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

   from isaacteleop.oxr import OpenXRSessionHandles

   handles = OpenXRSessionHandles(
       instance_handle, session_handle, space_handle, proc_addr
   )
   config = TeleopSessionConfig(
       app_name="MyApp",
       pipeline=pipeline,
       oxr_handles=handles,  # Skip internal OpenXR session creation
   )

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

   import isaacteleop.deviceio as deviceio
   from isaacteleop.retargeting_engine.deviceio_source_nodes import FullBodySource

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

Because pipelined mode reuses completed frames, caller-provided external inputs
and returned outputs must be snapshot-copyable. ``TensorGroup.create_snapshot()``
handles standard scalar, array, and common DLPack-backed tensor cases; custom
opaque values should provide an equivalent ``create_snapshot()`` hook or run
with ``mode="sync"``. Pipelined mode returns a new snapshot of the cached
output on each ``step()`` call. Sync mode returns the pipeline result directly,
matching the historical ownership behavior.

.. code-block:: python

   from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

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

   from isaacteleop.teleop_session_manager import (
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
- ``get_elapsed_time() -> float`` -- Get elapsed time since session started.

Example (explicit GraphTime + ExecutionEvents override):

.. code-block:: python

   import time

   from isaacteleop.retargeting_engine.interface.execution_events import (
       ExecutionEvents,
       ExecutionState,
   )
   from isaacteleop.retargeting_engine.interface.retargeter_core_types import GraphTime

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
  ``None`` when using external handles (read-only)
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
   of input sources.

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
