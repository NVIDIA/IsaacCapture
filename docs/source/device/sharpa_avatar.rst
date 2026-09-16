.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Sharpa Avatar Glove
===================

The Sharpa Avatar plugin connects Avatar gloves to Isaac Teleop. It publishes
hand poses and glove joint data through the standard Isaac Teleop interfaces,
and forwards haptic commands to the glove motors.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

The plugin uses the Avatar SDK to read both gloves and exposes three kinds of
data to the Teleop stack:

.. code-block:: text

   Avatar gloves ──► avatar_hand_plugin ──► TeleopSession ──► consumers
          ▲                  │
          └──── haptic ◄─────┘

The HUMAN data stream is converted to 26-joint OpenXR hand poses. The RAW and
ROBOT streams are published as 22-DoF joint-state tensors. Haptic commands are
sent back to the glove motors through the ``avatar_glove_haptic`` collection.

SDK availability
----------------

The installer retrieves the production ``avatar-sdk`` package, currently
pinned to ``1.7.3-17``, from Sharpa's signed production APT repository. It
never selects the ``avatar-sdk-dev`` or ``avatar-sdk-beta`` channels. The SDK
remains external under ``/opt/avatar-sdk``; its files are neither committed to
Isaac Teleop nor copied into the plugin installation.

Prerequisites
-------------

- Linux x86_64 (Ubuntu 22.04).
- A built Isaac Teleop checkout.
- Sharpa Avatar gloves connected through the USB dongle or wired Ethernet.
- CMake 3.24 or newer.

The plugin talks to the gloves directly. Do not run ``avatar-backend``, Avatar
Desktop, or ``avatar_hand_tracker_printer`` at the same time.

Installation
------------

Run the following command from the Isaac Teleop repository root:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install.sh

If ``/opt/avatar-sdk`` is absent, the installer configures the same production
APT channel used by the Sharpa host application and installs the pinned SDK.
To install only the dependency:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install_avatar_sdk.sh

If a development or beta SDK is already installed, the dependency installer
stops instead of replacing it implicitly. Remove that package explicitly before
installing production. The plugin build accepts an existing SDK under
``/opt/avatar-sdk`` but reports its version and warns when its ``BUILD_TYPE`` is
not ``Production``.

Install the device rules once on the host:

.. code-block:: bash

   ./src/plugins/sharpa_avatar/install_udev_rules.sh

Then unplug and reconnect the glove or dongle. udev does not run in containers,
so the rules installer must run on the host.

The plugin follows the Sharpa host application layout and uses the SDK,
configuration, and runtime assets directly from ``/opt/avatar-sdk``. Transport
selection, including wired Ethernet, is controlled by
``/opt/avatar-sdk/share/sdk_config.json``.

The installer builds ``avatar_hand_plugin`` and
``avatar_hand_tracker_printer`` under the normal Isaac Teleop build and install
directories.

Run the sample
--------------

Install the sample's Python dependencies once:

.. code-block:: bash

   uv pip install viser numpy

Then launch the sample from the repository root:

.. code-block:: bash

   .venv/bin/python src/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py

The sample starts CloudXR and the glove plugin, serves a viser view at
``http://127.0.0.1:8080``, and enables pinch-triggered haptic feedback. The
left hand is shown in cyan and the right hand in orange. Use ``--no-viz`` for
terminal output and haptics without the browser view.

Useful options include:

.. code-block:: text

   --no-haptic                   disable pinch feedback
   --no-viz                      terminal and haptic output only
   --host / --port               viser bind address and port
   --no-launch-plugin            connect to an already running plugin
   --no-launch-cloudxr-runtime   connect to an already running CloudXR runtime
   --world-frame                 display unmodified OpenXR poses

Published data
--------------

.. list-table::
   :header-rows: 1
   :widths: 2 3

   * - Collection or path
     - Contents
   * - ``/hand/left`` and ``/hand/right``
     - 26 OpenXR hand joints from Avatar HUMAN data
   * - ``avatar_raw_left/right``
     - RAW 22-DoF joint-state tensors
   * - ``avatar_robot_left/right``
     - ROBOT 22-DoF joint-state tensors
   * - ``avatar_glove_haptic``
     - Per-finger vibration commands

The gloves do not provide a world-space wrist pose. Without another tracked
wrist source, the sample places both skeletons in a stable local display frame.

Troubleshooting
---------------

* **Avatar SDK installation fails:** check access to Sharpa's production APT
  endpoint and rerun ``install_avatar_sdk.sh``.
* **The USB glove is not detected:** run ``install_udev_rules.sh`` on the host,
  then unplug and reconnect the glove or dongle.
* **The plugin binary is not found:** rerun ``install.sh``, or pass
  ``--plugin-search-path install/plugins`` to the sample.
* **Hand and joint streams stay offline:** power on the gloves and stop other
  processes using the Avatar SDK.
* **No vibration:** keep ``haptic`` in ``--datasets`` and move a fingertip
  toward the thumb.
* **CloudXR cannot start:** source ``~/.cloudxr/run/cloudxr.env``, or let the
  sample launch CloudXR.
