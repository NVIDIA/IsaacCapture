.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Sharpa Avatar Glove
===================

A Linux-only plugin for integrating Sharpa Avatar data gloves into the
Isaac Teleop framework. The gloves stream IMU and encoder data through the
official Avatar SDK over USB. The plugin maps that stream into Isaac Teleop's
usual surfaces: OpenXR hand tracking for the HUMAN skeleton,
``JointStateOutput`` tensors for the RAW and ROBOT 22-DoF joint sets, and
inbound per-finger vibration on a haptic collection.

Any downstream consumer that already reads OpenXR hands (``HandsSource``) or
schema joint-state collections can use Avatar gloves through the same Isaac
Teleop interfaces. The plugin talks to the gloves directly, so do not run
``avatar-backend`` or Avatar Desktop at the same time.

Plugin sources and the operator README live under
:code-dir:`src/plugins/sharpa_avatar`. A TeleopSession visualization example
is :code-dir:`examples/sharpa_avatar`.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

The glove SDK exposes three datasets plus haptics. HUMAN landmarks are placed
at a fused wrist pose and injected as OpenXR hands; RAW and ROBOT stay as
named joint-state collections:

.. code-block:: text

   Sharpa Avatar gloves (USB)
           │
           ▼
   avatar_hand_plugin  (Avatar SDK, production package)
           │
           ├── HUMAN  ──► OpenXR /hand/left, /hand/right   (26 joints)
           ├── RAW    ──► avatar_raw_left / avatar_raw_right
           ├── ROBOT  ──► avatar_robot_left / avatar_robot_right
           └── haptic ◄── avatar_glove_haptic  (five finger vibration motors)

.. seealso::

   :doc:`/references/retargeting/sharpa` maps OpenXR hand tracking onto a
   Sharpa robot hand. Avatar HUMAN output is the same OpenXR layer that
   retargeter consumes.

Prerequisites
-------------

- **Linux x86_64** (tested on Ubuntu 22.04).
- **Pinned production Avatar SDK** (``avatar-sdk`` 1.7.3-17) from Sharpa's
  signed APT repository. The install script retrieves it; do not substitute
  ``avatar-sdk-dev`` or ``avatar-sdk-beta``.
- **Sharpa Avatar gloves** powered and connected over USB. Transport settings
  are selected in ``<sdk-root>/share/sdk_config.json`` (default root
  ``/opt/avatar-sdk``).
- **CloudXR runtime** on the plugin host, unless a ``TeleopSession`` example
  launches it for you.
- A built Isaac Teleop checkout. CMake 3.24 or newer is required.

The SDK stays external: headers, libraries, and runtime assets are not copied
into this repository or the plugin install prefix.

Installation
------------

From the Isaac Teleop root, ``install.sh`` checks (and if needed installs)
the pinned production SDK, then configures, builds, and installs the plugin:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install.sh

Pass ``--build-dir <path>`` to reuse a non-default CMake build directory.
``AVATAR_SDK_ROOT`` (or ``-DAVATAR_SDK_ROOT``) selects a non-default SDK
tree; that tree must already be complete. To install only the SDK:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install_avatar_sdk.sh

``-DBUILD_PLUGIN_SHARPA_AVATAR=ON`` without a usable SDK under the selected
root skips this plugin and the rest of Isaac Teleop still configures. See
:doc:`/getting_started/build_from_source/index`.

USB access (one-time, on the host)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

udev rules must run **on the host**, not inside a container:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install_udev_rules.sh
   # then unplug and reconnect the glove

Running the plugin
------------------

The plugin publishes through the CloudXR / OpenXR runtime. Start the runtime
and source its environment in the shell that launches the plugin:

.. code-block:: bash

   python -m isaacteleop.cloudxr.service start   # runs in the background
   source ~/.cloudxr/run/cloudxr.env
   ./install/plugins/sharpa_avatar/avatar_hand_plugin

See :ref:`dedicated-cloudxr-runtime` and
:ref:`load-cloudxr-environment-variables` for the full service setup.

``TeleopSession`` can launch the same binary through ``plugin.yaml``. The
visualization example does that for you (CloudXR and the plugin start unless
you opt out):

.. code-block:: bash

   uv pip install -e ./examples/sharpa_avatar
   python -m isaacteleop_examples.sharpa_avatar

Gloves reconnect from ``update()`` if they drop offline. Only one process
may hold the Avatar SDK at a time.

Datasets
--------

By default every path is on. Restrict them with ``--datasets=``
(comma-separated: ``human``, ``raw``, ``robot``, ``haptic``):

.. code-block:: bash

   ./install/plugins/sharpa_avatar/avatar_hand_plugin --datasets=human,raw,robot,haptic
   ./install/plugins/sharpa_avatar/avatar_hand_plugin --datasets=human

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Device or collection
     - Contents
   * - ``/hand/left``, ``/hand/right``
     - OpenXR 26-joint poses from Avatar HUMAN landmarks
   * - ``avatar_raw_left``, ``avatar_raw_right``
     - RAW 22-DoF encoder joint state
   * - ``avatar_robot_left``, ``avatar_robot_right``
     - ROBOT 22-DoF retargeted joint state
   * - ``avatar_glove_haptic``
     - Inbound per-finger vibration commands

HUMAN (OpenXR hands)
~~~~~~~~~~~~~~~~~~~~

HUMAN landmarks are mapped onto the 26 OpenXR hand slots (unsupported slots
stay invalid) and pushed with ``HandInjector``. Hosts read them through
``HandsSource`` like any other hand-tracking plugin.

RAW and ROBOT (joint-state tensors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each side pushes a ``JointStateOutput`` (tensor id ``joint_state``) with the
22 named DoF from the pinned SDK. Consume them with ``JointStateSource``
and the collection ids above.

Haptic
~~~~~~

The plugin reads ``HapticCommand`` on ``avatar_glove_haptic`` and drives
five finger vibration motors. The visualization example uses the same
per-finger pinch path as the generic glove haptic sample.

Wrist pose
----------

Avatar glove data is wrist-relative, so the plugin anchors HUMAN landmarks
with a pose from ``WristPoseSource`` before publishing OpenXR hands. It prefers
optical hand tracking and falls back to the controller aim pose when needed.
If no wrist pose is available yet, the plugin still publishes a local skeleton
with VALID-only flags; TRACKED is set once the selected wrist source is actively
tracked.

The example places that local skeleton into a stable left/right display
frame. Pass ``--world-frame`` to leave the OpenXR poses unmodified.

Troubleshooting
---------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Symptom
     - Fix
   * - Avatar SDK installation fails
     - Confirm access to Sharpa's production APT endpoint and re-run
       ``install_avatar_sdk.sh``.
   * - CMake skipped the plugin
     - The SDK is missing under the selected root. Run
       ``install_avatar_sdk.sh`` (or ``install.sh``) and reconfigure.
   * - CMake rejected the Avatar SDK
     - Install the pinned production package, or point ``AVATAR_SDK_ROOT``
       at a complete SDK tree. A ``-dev`` / ``-beta`` package at
       ``/opt/avatar-sdk`` is not the pin.
   * - USB glove is not detected
     - Run ``install_udev_rules.sh`` on the host, then unplug and reconnect
       the glove.
   * - Plugin binary is not found
     - Run ``install.sh``. The example looks under
       ``install/plugins/sharpa_avatar/``.
   * - HUMAN / RAW / ROBOT stay offline
     - Power the gloves, and stop Avatar Desktop, ``avatar-backend``, or
       any other process using the Avatar SDK.
   * - Viser shows an empty grid
     - HUMAN must be enabled (default). RAW-only is joint angles, not a
       3D skeleton. Confirm the example is not launched with
       ``--datasets`` that omit ``human``.
   * - No vibration
     - Keep ``haptic`` in ``--datasets`` and close a fingertip toward the
       thumb (example pinch path).
   * - CloudXR runtime errors
     - Check ``python -m isaacteleop.cloudxr.service status``, and source
       ``~/.cloudxr/run/cloudxr.env`` in the same terminal as the plugin.

License
-------

Plugin sources are Apache-2.0. The Avatar SDK is proprietary to Sharpa and
is subject to its own license; this project does not redistribute it.
