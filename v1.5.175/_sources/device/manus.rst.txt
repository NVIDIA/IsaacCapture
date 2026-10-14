.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

MANUS Gloves
============

A Linux-only plugin for integrating `MANUS <https://www.manus-meta.com/>`_ gloves
into the Isaac Teleop framework. It provides full hand-joint tracking via the
Manus SDK and injects the resulting poses into the OpenXR hand-tracking layer so
any downstream retargeter can consume them transparently. Optionally it also
publishes Manus flex-sensor (RawDeviceData) tip poses as ``JointStateOutput``
tensors and consumes inbound haptic commands for vibration gloves.

.. contents:: On this page
   :local:
   :depth: 2

Components
----------

- **Core library** (``manus_plugin_core``) — wraps the Manus SDK
  (``libIsaacTeleopPluginsManus.so``) and exposes per-joint tracking data.
- **Plugin executable** (``manus_hand_plugin``) — the main plugin binary that
  integrates with the Teleop system via CloudXR / OpenXR.
- **CLI tool** (``manus_hand_tracker_printer``) — a standalone diagnostic tool
  that prints tracked joint data to the terminal and opens a real-time
  **MANUS Data Visualizer** window showing the hand skeleton from two orthographic
  views per hand.

Prerequisites
-------------

- **Linux** — x86_64 or aarch64 (tested on Ubuntu 22.04 / 24.04).
- **Manus SDK** for Linux — downloaded automatically by the install script.
- **System dependencies** — the install script installs required packages automatically.

Installation
------------

MANUS access has two halves: **device permissions** (kernel/udev, lives on the
host) and **SDK + plugin build** (lives wherever you build, typically a
container). The two scripts below split along that line.

Step 1: grant the host access to the Manus dongle (one-time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this **on the host machine**, not inside a container. udev rules are
processed by ``systemd-udevd``, which does not run inside Docker — so
installing rules from a container has no effect.

.. code-block:: bash

   cd src/plugins/manus
   ./install_udev_rules.sh
   # then unplug + replug the Manus dongle

If you're using the Isaac ROS dev container (``isaac_ros run_dev``), it
bind-mounts ``/dev/bus/usb`` from the host, so once the host has the rules
applied the container will see the dongle with the right permissions.

Step 2: build the SDK and plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this **inside the build environment** (devcontainer or Isaac ROS container):

.. code-block:: bash

   cd src/plugins/manus
   ./install_manus.sh

Pass ``--build-dir <path>`` to reuse a non-default top-level CMake build
directory.

The script will:

1. Install the required system packages for MANUS Core Integrated.
2. Download MANUS SDK v3.2.0.
3. Extract and place the SDK for the host architecture in the correct location.
4. Build the plugin and the diagnostic tool.

When run inside a container, ``install_manus.sh`` skips the udev step and
reminds you to run ``install_udev_rules.sh`` on the host. Pass ``--container``
when the build environment does not expose standard container markers, such as
during a Docker BuildKit build.

Manual installation (fallback only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steps 1 and 2 above are the supported path — most users should never need
this section. Only follow it if ``install_manus.sh`` fails (for example, the
SDK download is unreachable from your network) and you need to place the SDK
by hand.

1. Download the MANUS Core SDK from
   `MANUS Downloads <https://docs.manus-meta.com/3.2.0/Resources/>`_.
2. From ``C++/SDKClient/ManusSDK`` in the archive, place ``include/`` and the
   library for your architecture (``lib/amd64/libManusSDK-amd64.so`` or
   ``lib/aarch64/libManusSDK-aarch64.so``, renamed to ``lib/libManusSDK.so``)
   into a ``ManusSDK`` folder inside ``src/plugins/manus/``, or point CMake at a
   different path by setting ``MANUS_SDK_ROOT``.
3. Follow the
   `MANUS Getting Started guide for Linux <https://docs.manus-meta.com/3.2.0/Plugins/SDK/Linux/>`_
   to install the dependencies and configure device permissions.

Expected directory layout after placing the SDK:

.. code-block:: text

   src/plugins/manus/
     app/
       main.cpp
     core/
       manus_hand_tracking_plugin.cpp
       inc/
         manus/
           manus_glove_collection.hpp
           manus_hand_tracking_plugin.hpp
     tools/
       manus_hand_tracker_printer.cpp
     ManusSDK/        <-- placed here
       include/
       lib/
         libManusSDK.so

Then build from the root:

.. code-block:: bash

   cd ../../..  # navigate to root
   cmake -S . -B build
   cmake --build build --target manus_hand_plugin manus_hand_tracker_printer -j
   cmake --install build --component manus

Running the Plugin
------------------

1. Start the CloudXR runtime and load its environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MANUS plugin connects to the Teleop session through the CloudXR / OpenXR
runtime, so the runtime must be running and its environment sourced in the
shell that launches the plugin.

Start the CloudXR runtime. It runs as a background service that outlives
this shell, until you stop it with ``python -m isaacteleop.cloudxr.service stop``:

.. code-block:: bash

   python -m isaacteleop.cloudxr.service start

In the shell you will use to run the plugin, source the environment file
that the runtime writes on startup. This points the OpenXR loader at CloudXR:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env

See :ref:`dedicated-cloudxr-runtime` for the full CloudXR service setup,
including EULA acceptance, and :ref:`whitelist-firewall-ports` in the Quick
Start for firewall configuration.

2. Verify with the CLI tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that the gloves are working using the CLI tool:

.. code-block:: bash

   ./build/bin/manus_hand_tracker_printer

The tool prints joint positions to the terminal and opens a **MANUS Data
Visualizer** window showing a top-down and side view of each hand.

3. Run the plugin
~~~~~~~~~~~~~~~~~~

The plugin is installed to the ``install`` directory. Ensure the CLI tool is
not running when you launch the plugin — only one process can hold the Manus
SDK connection at a time.

.. code-block:: bash

   ./install/plugins/manus/manus_hand_plugin

By default the plugin enables human hand injection, flex-sensor push, and
haptic read. Restrict datasets with ``--datasets=`` (comma-separated):

.. code-block:: bash

   ./install/plugins/manus/manus_hand_plugin --datasets=human,sensors,haptic
   ./install/plugins/manus/manus_hand_plugin --datasets=human          # skeleton only
   ./install/plugins/manus/manus_hand_plugin --datasets=sensors        # flex sensors only

Data paths
----------

Human (OpenXR hand injection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Manus raw skeleton stream is mapped to 26 OpenXR hand joints and pushed
through ``HandInjector`` (requires ``XR_NVX1_device_interface_base``). Downstream
hosts consume hands via ``HandsSource`` as with any other hand-tracking plugin.

Sensors (flex tips via SchemaPusher)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When gloves report at least five RawDeviceData flex sensors, the plugin pushes
a ``JointStateOutput`` (tensor id ``joint_state``) on:

- ``manus_sensors_left``
- ``manus_sensors_right``

Layout: 35 joints named ``j0``..``j34``. For sensor ``i`` in thumb→pinky order,
``j[7*i : 7*i+7]`` is ``[x, y, z, qx, qy, qz, qw]`` (meters, quaternion xyzw)
in the Manus SDK frame after the plugin's VUH coordinate setup. Poses are raw
Manus flex transforms; hosts that mask against a re-framed skeleton must apply
their own sensor-pose processing.

This path requires CloudXR tensor **push** extensions
(``XR_NVX1_push_tensor`` and ``XR_NVX1_tensor_data``). When sensors first become
available the plugin logs ``sensors=on`` once per side.

Host-side consumption example:

.. code-block:: python

   from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource

   SENSOR_JOINTS = [f"j{i}" for i in range(35)]
   left = JointStateSource(
       name="manus_sensors_left",
       collection_id="manus_sensors_left",
       joint_names=SENSOR_JOINTS,
   )
   # Decode thumb tip: joints j0..j6 -> [x, y, z, qx, qy, qz, qw]

Haptic (inbound vibration)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The plugin reads ``HapticCommand`` on collection ``manus_glove_haptic``
(``XR_NVX1_tensor_data``) and drives five finger motors via the Manus SDK.
See the haptic feedback example and
``isaacteleop.haptic_devices.glove.haptic_glove_device``.

Wrist Positioning — Controllers vs Optical Hand Tracking
---------------------------------------------------------

Two sources are available for positioning the MANUS gloves in 3D space:

- **Controller adapters** — attach Quest 3 controllers to the MANUS Universal
  Mount on the back of the glove. The controller pose drives wrist placement.
- **Optical hand tracking** — use the HMD's built-in optical hand tracking to
  position the hands. No physical controller adapter required.

The plugin selects the source automatically at runtime: optical hand tracking is
preferred when ``XR_MNDX_xdev_space`` is supported and the runtime reports an
actively tracked wrist pose; otherwise it falls back to the controller pose.

.. note::

   When using controller adapters it is advisable to disable the HMD's automatic
   hand-tracking–to–controller switching to avoid unexpected source changes
   mid-session.

Troubleshooting
---------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Symptom
     - Resolution
   * - SDK download fails
     - Check your internet connection and re-run the install script.
   * - ``uv not found. Please install uv: ...``
     - Install uv and re-run: ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
       It installs to ``~/.local/bin``, so also add that to ``PATH`` for the
       current shell (and ``~/.bashrc``, to make it stick):
       ``export PATH="$HOME/.local/bin:$PATH"``.
   * - ``Missing build tools: ... patchelf ...``
     - CMake's dependency preflight lists every missing tool and the exact
       install command in the same message, e.g.
       ``sudo apt-get install -y patchelf``. Run the printed command and
       re-run the install script.
   * - Manus SDK not found at build time
     - With manual installation, ensure ``ManusSDK`` is inside
       ``src/plugins/manus/`` or set ``MANUS_SDK_ROOT`` to your installation path.
   * - Manus SDK not found at runtime
     - The build configures RPATH automatically. If you moved the SDK after
       building, set ``LD_LIBRARY_PATH`` to its ``lib/`` directory.
   * - No data received
     - Ensure MANUS Core is running and the gloves are connected and calibrated.
   * - CloudXR runtime errors
     - Check that a runtime is serving with ``python -m isaacteleop.cloudxr.service status``,
       and that ``~/.cloudxr/run/cloudxr.env`` has been sourced in the same
       terminal as the plugin.
   * - Permission denied for USB devices
     - udev rules must be installed on the host. Run ``./install_udev_rules.sh``
       from the host (not inside a container), then unplug and replug the
       MANUS dongle. Verify on the host with ``ls -l /dev/hidraw*`` — entries for the
       dongle should be mode ``0666``.
   * - ``udevadm control --reload-rules`` fails with "No such file or directory"
     - You're inside a container. ``systemd-udevd`` doesn't run in containers,
       so this command can never succeed there. Run ``install_udev_rules.sh``
       on the host instead.
   * - Dongle not visible inside the Isaac ROS container (``lsusb`` doesn't
       show vendor ``3325``)
     - The container needs ``/dev/bus/usb`` bind-mounted from the host. The
       Isaac ROS dev container does this automatically; for a custom
       ``docker run``, add ``-v /dev/bus/usb:/dev/bus/usb`` (or
       ``--device=/dev/hidraw<N>`` for a specific device).

License
-------

Source files are covered by their stated licenses (Apache-2.0). The Manus SDK is
proprietary to MANUS and is subject to its own license; it is **not** redistributed
by this project.
