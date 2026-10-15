.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Synthetic XR controls and recording reference
=============================================

These optional controls help inspect input and reproduce issues. The automated
Franka example drives the controller API itself; selecting a controller or
moving it with the panel is not a prerequisite. Start with
:doc:`synthetic_xr` for the task workflow and shared native path.

Requirements
------------

Complete the host setup in :doc:`quick_start`. The same CloudXR runtime, GPU,
network and certificate requirements apply. Isaac Lab is needed only for the
Franka simulation example. Desktop operation requires neither ADB nor a headset.
Native headset sessions retain their own XR runtime; this feature does not
replace it with IWER.

Build the client from this checkout following
:doc:`/development/build_from_source/webxr`. Start the service with that build:

.. code-block:: bash

   export TELEOP_WEB_CLIENT_STATIC_DIR="$PWD/deps/cloudxr/webxr_client/build"
   python -m isaaccapture.cloudxr.service start --host-client

Use the client URL printed by the service. Open **Synthetic XR controls** to
reload with the panel enabled, or add ``syntheticXR=1`` to the URL query.
For an optional desktop launcher, install the example from this checkout:

.. code-block:: bash

   python -m pip install -e ./examples/synthetic_xr
   synthetic-xr --client-url 'https://HOST:PORT/client/'

Replace the URL with the service's actual URL, keeping its query parameters and
application fragment. The launcher preserves them. It opens the client; it does
not start another runtime. For unattended hub setup without ADB, see
:doc:`/references/oob_teleop_control` and ``TELEOP_OOB_HUB_ONLY=1``.

Move controllers
----------------

#. Connect the WebXR client and start a native input viewer or robot application.
#. Select the **Left** or **Right** controller in the Synthetic XR panel.
#. Change X/Y/Z in metres or rotate around an axis using the ten-degree buttons.
#. Use **Grip / clutch** and **Trigger** according to the application's mapping.
#. Use **Stop input** to cancel synthetic playback and release both buttons.

**Reset virtual poses** resets input poses, not the simulator. **Stop input**
is an input stop, not an application emergency stop: use the robot application's
stop control to stop its task. Switching to a hidden browser tab also stops
synthetic playback and releases buttons.

Controller poses use IWER local coordinates: X right, Y up, Z backward, with
quaternions in x/y/z/w order. They describe the controller's target-ray pose;
its grip pose additionally includes the selected IWER profile's grip offset.
The browser produces poses and buttons; host retargeting and a simulation task's
IK solver determine robot joint motion.

Run and record a generated sequence
-----------------------------------

**Run controller sweep** generates a nine-second sequence with separate left
and right movement, a grip interval and a trigger interval. It runs once and
releases buttons on completion. It requires no captured manual demonstration.

Use **Record input** while moving the virtual controllers, then **Save input
trace**. Load that JSON with **Load trace**, then select **Replay once**.
This trace records complete synthetic input states at command times; playback
holds each state until the next timestamp. It is distinct from the client's
existing XR-frame recording JSON and from native MCAP.

Only one input producer may own synthetic control. Manual/API pose changes are
rejected while a trace is playing. The client suspends synthetic controls while
its existing browser-recording replay owns the CloudXR tracking adapter.

Native MCAP record and single replay
------------------------------------

Install the existing :doc:`/references/mcap_record_replay` example. With the
native runtime and desktop client connected, record a controller sequence:

.. code-block:: bash

   python -m isaaccapture_examples.mcap_record_replay.record_controller 15 controllers.mcap

Run the sweep during that recording. Then replay the MCAP once using this
package's wrapper around the native example's replay pipeline:

.. code-block:: bash

   python -m isaaccapture_examples.synthetic_xr.replay_once controllers.mcap

For an Isaac Teleop 1.5 installation, add ``--legacy-example-dir`` pointing to
that SDK checkout's ``examples/mcap_record_replay/python`` directory. This
explicitly selects its native replay implementation. The default command
requires the current Isaac Capture example package. The native example polls at
60 Hz; a lower-rate recording may exhaust its payloads before its duration timer
expires and then show invalid inputs. Payload equality and order are verified;
original sample timing is not part of this preview's qualification.

Open the printed viewer URL. The wrapper plays one sequence, leaves the viewer open for five additional
seconds and closes. This does not guarantee a held final valid pose. Native MCAP replay needs no live XR session.
The main MCAP example retains its existing repeated-playback behavior.

Browser input API
-----------------

In a desktop client opened with ``syntheticXR=1``, the preview API is available
as ``window.syntheticXR``. For example, in browser developer tools:

.. code-block:: javascript

   const input = window.syntheticXR;
   input.setController('right', {
     position: [0.3, 1.3, -0.4],
     quaternion: [0, 0, 0, 1],
     grip: 1,
     trigger: 0.5,
   });
   input.stop();

The reusable implementation is
:download:`src/syntheticXR/controller.ts <../../../deps/cloudxr/webxr_client/src/syntheticXR/controller.ts>`.
``setHead``, ``getState``, ``record``, ``finishRecording``, ``play`` and ``stop``
provide input generation and reproduction. ``play(trace)`` runs once;
``play(trace, {loop: true})`` repeats until stopped. ``lastFrame`` exposes observed
WebXR aim/grip poses and gamepad values for debugging, not host acknowledgements.

Inputs must be finite, positions within ten metres per axis, normalized
quaternions and button values in [0, 1]. Trace version 1 uses
``referenceSpace: 'iwer-local'`` and strictly increasing ``timeMs`` values
starting at zero, with complete ``head``, ``left`` and ``right`` state in each
frame. Traces are limited to ten minutes and 36,000 frames. Schedule button
transitions far enough apart to be observed by the application's frame loop.

Scope and troubleshooting
-------------------------

* **Controls unavailable:** check that desktop IWER loaded and the URL has
  ``syntheticXR=1``. Native immersive XR sessions intentionally keep their input.
* **No movement:** verify connection, fresh native poses, clutch and application
  start state. Moving a controller is not itself a robot task success signal.
* **Replay rejected:** synthetic traces, browser recordings and MCAP are separate
  formats. Use the corresponding replay path.
* **No WebGL:** use a browser/driver combination that supports WebGL2. Do not
  treat an unavailable renderer as a successful transport test.

This controller preview does not emulate physical tracking accuracy, haptics,
Pico/Quest device behavior or glove protocols. Hand/body adapters are outside
this release. Scripted simulation data is synthetic, not a human demonstration.
