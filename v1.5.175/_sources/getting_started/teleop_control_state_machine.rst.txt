.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Teleop Control State Machine
============================

This page describes the default teleop control state machine used by
``DefaultTeleopStateManager`` and how to connect button inputs.

Overview
--------

``TeleopSession`` expects the optional ``teleop_control_pipeline`` to expose
this output contract:

- ``teleop_state``: one-hot execution state channels in this order:
  ``stopped``, ``paused``, ``running``
- ``reset_event``: bool pulse

When provided, ``TeleopSession`` decodes these outputs into
``ComputeContext.execution_events`` so downstream retargeters and high-level app
logic can read:

- ``context.execution_events.execution_state``
- ``context.execution_events.reset``

Isaac Teleop provides ``DefaultTeleopStateManager`` as a ready-to-use default
state machine. It should work for most setups, but you can also implement your
own control state machine as long as it preserves the same output contract.

Default State Machine Behavior
------------------------------

Inputs (all optional bool signals):

- ``kill_button``
- ``run_toggle_button``
- ``reset_button``

Transitions:

1. ``kill_button`` level-held:

   - while asserted, state is forced to ``STOPPED``
   - emits ``reset_event = True``

2. ``run_toggle_button`` rising edge:

   - ``STOPPED`` -> ``PAUSED``
   - ``PAUSED`` -> ``RUNNING``
   - ``RUNNING`` -> ``PAUSED``

3. ``reset_button`` rising edge:

   - state unchanged
   - emits ``reset_event = True``

Safety behavior:

- If a required control input (kill/run-toggle) is absent for the frame, manager
  fail-safes to ``STOPPED``.
- Loss-driven ``reset_event`` is edge-triggered on availability loss (emitted once
  on the first frame required inputs drop out, not unconditionally every frame).
- This fail-safe is immediate (same frame). If teleop control inputs are lost
  (for example controller disconnect, tracking loss, or selector output becomes
  unavailable), execution state is forced to ``STOPPED`` right away.

Connecting Inputs
-----------------

Use one debounced selector per signal (simple wiring, no multi-input lambdas):
see :code-file:`teleop_controls_simple_example.py <examples/teleop_session_manager/python/isaacteleop_examples/teleop_session_manager/teleop_controls_simple_example.py>`
for the full runnable setup.

.. code-block:: python

   from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex
   from isaacteleop.teleop_session_manager import (
       DefaultTeleopStateManager,
       create_bool_selector,
   )

   controllers = ControllersSource(name="controllers")
   left = controllers.output(ControllersSource.LEFT)

   kill_signal = create_bool_selector(
       left,
       name="kill_signal_selector",
       selector_fn=lambda selected: selected[ControllerInputIndex.SECONDARY_CLICK],
   )
   run_toggle_signal = create_bool_selector(
       left,
       name="run_toggle_signal_selector",
       selector_fn=lambda selected: selected[ControllerInputIndex.PRIMARY_CLICK],
   )
   reset_signal = create_bool_selector(
       left,
       name="reset_signal_selector",
       selector_fn=lambda selected: selected[ControllerInputIndex.THUMBSTICK_CLICK],
   )

   manager = DefaultTeleopStateManager(name="teleop_manager")
   control_pipeline = manager.connect(
       {
           manager.INPUT_KILL: kill_signal.output("value"),
           manager.INPUT_RUN_TOGGLE: run_toggle_signal.output("value"),
           manager.INPUT_RESET: reset_signal.output("value"),
       }
   )

Debounce behavior is configurable in ``create_bool_selector``:

- ``threshold`` / ``release_threshold`` for float hysteresis
- ``activate_frames`` / ``deactivate_frames`` for rising/falling debounce length
- automatic ``None`` propagation when upstream input is absent

Important availability note:

- ``create_bool_selector`` propagates unavailable upstream input as ``None``.
- ``DefaultTeleopStateManager`` treats ``None`` on any required control input
  as a safety fault and immediately transitions to ``STOPPED``; when required
  inputs first become unavailable it emits a single loss-driven
  ``reset_event = True`` pulse.

Using Context in High-Level App Logic
-------------------------------------

After ``session.step()``, use ``session.last_context`` to gate robot enablement
or other high-level behavior:

.. code-block:: python

   from isaacteleop.retargeting_engine.interface.execution_events import ExecutionState

   outputs = session.step()
   context = session.last_context
   if context is not None:
       robot_enabled = context.execution_events.execution_state != ExecutionState.STOPPED
       if context.execution_events.reset:
           handle_reset()
