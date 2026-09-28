.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Haptic Feedback
===============

Drive a haptic actuator from a simulator signal through the existing Isaac Teleop
retargeting pipeline. Motion-controller vibration (Quest, Vive Index, Pico, …) is
the first integration; the abstraction is vendor-neutral so haptic gloves and
force-feedback devices fit the same contract.

.. contents:: On this page
   :local:
   :depth: 2

Architecture
------------

Haptic output is a graph phase symmetric to the input phase: where an input
source feeds device data *into* the retargeting graph, a haptic **sink** consumes
graph outputs and writes them *out* to a device after the graph runs.

- **Device-side schemas** (``TensorGroupType``) describe what a device consumes.
  ``ControllerHapticPulse`` carries ``[amplitude, frequency_hz, duration_s]``;
  ``EndEffectorForce`` carries a 3-DoF force for future grounded devices. They
  live in ``isaacteleop.retargeting_engine.tensor_types``.
- **Retargeters** in ``isaacteleop.retargeters.tactile_retargeters`` map
  sim-side tactile data (a ``TactileVector`` / ``TactileHeatmap``) to a
  device-side schema — e.g. ``TactileVectorToControllerPulse``.
- **HapticSink** (``IDeviceIOSink``) is the vendor-neutral output node. It stores
  each frame's values per endpoint, and ``TeleopSession`` flushes it to the
  device after the graph, with the active session in scope. Register it with
  ``TeleopSessionConfig(sinks=[...])``.
- **IHapticDevice** is the vendor adapter. ``ControllerHapticDevice`` is the
  in-process reference; it drives the controller's vibration actuator through the
  same ``ControllerTracker`` that ``ControllersSource`` reads on the input side
  (via the per-side ``apply_left_haptic_feedback`` / ``apply_right_haptic_feedback``).
  OpenXR specifics stay in the live tracker layer, so the device, sink, and schema
  are runtime-neutral.

.. code-block:: text

   Sim / input signal      Retargeter                 Sink (IDeviceIOSink)
   (contact force,    -->  (TactileVectorTo-     -->   HapticSink
    trigger value)          ControllerPulse)            -> ControllerHapticDevice
                                                            (flushed after the graph)
                                                         -> ControllerTracker
                                                            .apply_left/right_haptic_feedback

Example
-------

``examples/haptic_feedback/python/controller_haptic_example.py`` is the minimal
end-to-end wiring: pull a controller's trigger and that **same** controller
rumbles. ``TriggerToTactile`` turns the trigger value into a ``TactileVector``,
``TactileVectorToControllerPulse`` turns that into a ``ControllerHapticPulse``,
and the ``HapticSink`` drives the controller. Swap ``TriggerToTactile`` for any
``TactileVector``-producing source (e.g. an Isaac Lab ``ContactSensor`` fetch) to
rumble from sim contact instead.

``ControllerHapticDevice`` must reuse the ``ControllerTracker`` owned by
``ControllersSource`` (pass ``controllers.get_tracker()``) so the session creates
a single controller tracker and there is no OpenXR action-set contention.

Running
-------

The example connects through the CloudXR / OpenXR runtime, which it starts
itself if nothing is serving (see :ref:`dedicated-cloudxr-runtime` to manage
that service yourself). Run it from the ``examples/haptic_feedback/python``
directory:

.. code-block:: bash

   uv run controller_haptic_example.py

No arguments — pull either trigger to rumble that controller. Press Ctrl+C to
exit. Runtimes that do not expose ``xrApplyHapticFeedback`` silently no-op rather
than tearing down the session.

Adding a new haptic device
---------------------------

A new haptic device implements ``IHapticDevice`` (``accepted_type``,
``endpoints``, ``apply``, ``flush``, ``get_tracker``) and is wired into a
``HapticSink``. Endpoints are named — ``"left"`` / ``"right"`` by convention, or
``"device"`` for a single grounded device — so single-actuator and
multi-actuator rigs share the same contract without a hardcoded handedness
assumption. Devices that run their vendor SDK in a separate process (haptic
gloves, exoskeletons) implement the same interface but exchange data with their
plugin over a tensor collection; those integrations land on top of this
foundation.

See also
--------

- Example + tests: ``examples/haptic_feedback/python/`` and
  ``tests/python/core/retargeting_engine/test_haptic_devices.py`` /
  ``test_haptic_sink.py``.
