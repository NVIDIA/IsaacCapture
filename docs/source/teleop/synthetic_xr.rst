.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Synthetic XR Teleoperation
==========================

**Build and automate teleoperation from your desktop, without an XR headset.**
Supply virtual controller inputs through IWER and use the same WebXR, CloudXR
and native robot-control path as a human operator. Start with Franka stacking,
then bring your own task logic.

Franka simulation example
-------------------------

The example locates a cube, approaches it, closes the gripper, lifts it and
places it on another cube. It repeats the transfer to build a three-cube stack
and stops after one successful native episode export. No recorded human
movement or manual controller positioning is required.

The host-side ``stack_logic.py`` reads cube and gripper poses from Lab and
chooses the next gripper target. A bridge converts that target into synthetic
XR controller input. **Lab's native IK calculates the joint commands.** The
Franka you see in the browser is the streamed result of the simulation.

One shared robot-control path
-----------------------------

A human observes the streamed scene and moves a physical controller. Automation
reads numerical scene feedback and generates virtual controller poses. Both
feed the same downstream path in this Franka example:

.. figure:: /_static/synthetic-xr-shared-flow.svg
   :alt: Human: streamed scene, visual decisions, physical controller movement and tracked poses. Automation: numerical Lab feedback, task logic, coordinate conversion and IWER poses. Both converge at WebXR and CloudXR, then native retargeting, Lab IK and simulated robot motion. Rendered images return to the browser; added numerical feedback returns to the script.
   :width: 100%
   :target: ../_static/synthetic-xr-shared-flow.svg

   Different input producers; shared transport, retargeting, IK and simulation.
   Open the diagram to view it at full size.

The numerical feedback is an extra, read-only observation path for the script.
In the human workflow, the operator sees the cubes and robot in streamed images;
this example's structured cube-pose feed is not part of that browser view.

The script generates intermediate **3D positions and orientations**, rather
than joint angles or directional button presses. The bridge accounts for robot,
world and XR coordinates and the controller/gripper offsets. Native retargeting
maps the resulting controller poses back to robot targets; IK acts inside Lab.
Trigger values close or open the gripper, just as a physical trigger would.

Try the example
---------------

Use a configured CloudXR and Isaac Lab host, this preview's built WebXR client
and the example's optional Franka dependencies. A headset and ADB are not needed;
CloudXR's GPU, network and certificate requirements still apply.

Follow the :download:`Franka setup and run commands <../../../examples/synthetic_xr/README.md>`:

#. Start the CloudXR service and Lab recording task with the read-only observer.
#. Connect the desktop WebXR client with ``syntheticXR=1`` enabled.
#. Run ``stack_logic``. It drives the synthetic controller through the browser
   API, including task start, movement and gripping.

The Synthetic XR panel lets you see the input being applied. **You do not need
to select Left/Right or operate its movement buttons for this example.**

The preview's native result is one successful seed-42 episode on the prepared
TeleOp 1.5 / CloudXR 6.3 / Lab compatibility stack. Targets use current cube
poses, but repeated randomized episodes have not been qualified. See the
:download:`validation scope <../../../examples/synthetic_xr/VALIDATION.txt>` for component
identities and evidence limits.

Bring your own task logic
-------------------------

Keep the virtual controller and native pipeline; change the decisions that
drive them. For example, a pick-and-place script could choose a destination
region instead of another cube as the placement target.

.. code-block:: text

   Your task logic
     observe object and gripper poses
     choose approach -> grasp -> lift -> destination -> release
     generate gripper targets and open/close commands
         |
         v
   Robot-specific pose bridge -> browser API / IWER
         |
         v
   Shared WebXR -> CloudXR -> retargeting -> IK -> robot simulation

Use these pieces as the starting point:

* :download:`stack_logic.py <../../../examples/synthetic_xr/python/isaaccapture_examples/synthetic_xr/franka/stack_logic.py>`
  chooses targets and checks grasp, placement and task completion. Replace these
  task decisions with your own; this is scripted logic, not a learned model.
* :download:`xr_controller_bridge.py <../../../examples/synthetic_xr/python/isaaccapture_examples/synthetic_xr/franka/xr_controller_bridge.py>`
  interpolates movement, converts Franka targets into XR controller poses and
  checks that the corresponding action reaches the native task. Reuse its
  mapping for the same configured Franka task; another robot needs its own
  retargeting and coordinate mapping.
* ``window.syntheticXR.setController(...)`` supplies controller poses and
  buttons. The :doc:`browser API reference <synthetic_xr_reference>` describes
  coordinates and optional inspection/recording tools.

The feedback source and task checks belong to your application. New tasks need
their own validation; the stacking example is not a general motion planner.
This preview covers controller input, not physical headset tracking, glove
protocols or hand/body input.

.. toctree::
   :hidden:

   synthetic_xr_reference
