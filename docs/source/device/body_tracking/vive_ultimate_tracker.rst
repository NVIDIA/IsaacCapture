.. SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

VIVE Ultimate Tracker
=====================

Use `HTC VIVE Ultimate Trackers <https://www.vive.com/us/accessory/vive-ultimate-tracker/>`_
with the Isaac Teleop framework. The ``vive_se3_tracker`` plugin streams each tracker as its own
generic SE3 (6-DoF) pose collection, which can be read live with ``Se3Tracker``, recorded to MCAP
and replayed. Currently only Linux is supported, on x86_64 and arm64. No headset is required.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

Tracker data reaches Isaac Teleop through two pieces of software:

1. **VIVE Hub for Linux** (from HTC) pairs the trackers with the dongle, builds the tracking map,
   and runs the ``tracker_server`` daemon. The daemon publishes tracker poses over the VUT
   (VIVE Ultimate Tracker) SDK socket.
2. **The** ``vive_se3_tracker`` **plugin** (in Isaac Teleop) reads those poses through the VUT
   SDK and pushes them into the Teleop session as one SE3 tensor collection per tracker.

.. code-block:: text

   Ultimate Trackers -> USB dongle -> VIVE Hub tracker_server -> /tmp/vut.sock (VUT SDK)
       -> vive_se3_tracker_plugin -> "vive_tracker_<serial>" SE3 collections
       -> Se3Tracker readers / MCAP recording / replay

Components
----------

- **VIVE Hub for Linux**: desktop app plus ``tracker_server`` daemon, used for pairing, map
  building and pose streaming. It also ships the VUT C++ SDK (``sdk/include/vut/`` and
  ``sdk/lib/libvut_sdk.a``), which the plugin needs at build time.
- **Plugin executable** (``vive_se3_tracker_plugin``): reads tracker poses from VIVE Hub and
  pushes them into the Teleop system through CloudXR / OpenXR.
- **Examples**: ``record_se3_vive`` records every tracker to an MCAP file, and
  ``replay_se3_vive`` plays a recording back in a 3D browser view.

Prerequisites
-------------

- **Linux**: x86_64 or arm64, on a desktop (graphical) session. Ubuntu 20.04 or newer is
  recommended.
- **VIVE Ultimate Trackers** and the **VIVE Ultimate Tracker dongle**.
- **VIVE Hub for Linux 1.0.1 or newer.** Version 1.0.0 does not include the VUT SDK; the plugin
  will neither build nor connect with it.
- **Isaac Teleop built from source**. See :doc:`/development/build_from_source/index`.

Quick Start
-----------

Part 1 brings the trackers up with VIVE Hub. Part 2 builds the plugin and streams the trackers
into Isaac Teleop.

Part 1: Set up the trackers with VIVE Hub
-----------------------------------------

Step 1: Download VIVE Hub for Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the package for your architecture from the
`latest VIVE Hub for Linux release <https://github.com/ViveRoboticsSoftware/vivehub-linux/releases/latest>`_.
Run ``uname -m`` if you are unsure which one you need:

.. list-table::
   :header-rows: 1

   * - ``uname -m`` output
     - Package
   * - ``x86_64``
     - ``VIVEHub-Linux-<version>-x86_64.tar.gz``
   * - ``aarch64``
     - ``VIVEHub-Linux-<version>-arm64.tar.gz``

To verify the download, download the ``.sha256`` file next to the package and run:

.. code-block:: bash

   sha256sum -c VIVEHub-Linux-*.sha256

Step 2: Install VIVE Hub
~~~~~~~~~~~~~~~~~~~~~~~~

``libglfw`` and ``libhidapi-hidraw`` are bundled in the package. The C++ runtime and the
GPU/display stack come from the system. On a minimal or headless image, install the graphics stack
first:

.. code-block:: bash

   sudo apt install libgl1 libx11-6 libgl1-mesa-dri

Extract the package and set up USB permissions for the dongle. This is a one-time step:

.. code-block:: bash

   tar xzf VIVEHub-Linux-*.tar.gz
   cd VIVEHub-Linux
   sudo ./install-udev.sh

Then **unplug and replug the tracker dongle** so the new permissions take effect.

**arm64 only:** the tracking-map wizard is an x86_64 build that runs under box64. Install it once
(this works offline):

.. code-block:: bash

   sudo ./install-box64.sh

.. note::

   udev rules only take effect on the host machine. If you build Isaac Teleop in a container, run
   ``install-udev.sh`` on the host, not inside the container.

Step 3: Pair the trackers and build the tracking map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start VIVE Hub from the extracted folder:

.. code-block:: bash

   ./run-vivehub.sh

Always launch VIVE Hub with ``run-vivehub.sh``, which points the loader at the bundled libraries.
The UI starts the ``tracker_server`` daemon itself, so do not start the daemon separately.

In the VIVE Hub UI:

1. Pair each Ultimate Tracker with the dongle.
2. Press **Start setup** to open the map-building wizard, and follow it to build the tracking map.
3. Check that every tracker shows as tracking.

**Leave VIVE Hub running.** The plugin reads poses from its daemon.

Step 4 (optional): Check the trackers with the VUT SDK demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To confirm that poses are coming out of VIVE Hub before you involve Isaac Teleop, run the demo
that ships with the SDK while at least one tracker is tracking:

.. code-block:: bash

   cd VIVEHub-Linux/sdk/examples
   make && ./vut_demo

Part 2: Stream the trackers into Isaac Teleop
---------------------------------------------

Step 5: Build Isaac Teleop with the plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plugin links against the VUT SDK in the VIVE Hub package. Point ``VUT_SDK_DIR`` at its
``sdk`` folder when you configure the build. From the Isaac Teleop root:

.. code-block:: bash

   cmake -B build -DVUT_SDK_DIR=/path/to/VIVEHub-Linux/sdk
   cmake --build build --parallel
   cmake --install build

The executable is installed to ``install/plugins/vive_se3_tracker/``.

.. note::

   If ``VUT_SDK_DIR`` is not set, or ``lib/libvut_sdk.a`` is not found under it, CMake prints
   ``VIVEHub VUT SDK not found`` and skips the plugin, and the rest of the project still builds.
   If ``vive_se3_tracker_plugin`` is missing after the build, check this message in the configure
   output.

Step 6: Start the CloudXR runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the CloudXR runtime from the Python environment where ``isaaccapture`` is installed. It runs
as a background service until you stop it with
``python -m isaaccapture.cloudxr.service stop``:

.. code-block:: bash

   python -m isaaccapture.cloudxr.service start --accept-eula

See :doc:`/references/cloudxr` for details.

Step 7: Run the plugin
~~~~~~~~~~~~~~~~~~~~~~

With VIVE Hub still running and the trackers tracking, source the CloudXR environment and start
the plugin in a new terminal:

.. code-block:: bash

   cd IsaacTeleop/
   source ~/.cloudxr/run/cloudxr.env
   ./install/plugins/vive_se3_tracker/vive_se3_tracker_plugin

The plugin creates one collection per tracker the first time it receives a pose from that tracker,
and prints the collection ids. Press ``Ctrl+C`` to stop it.

Step 8: Verify the pose stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In another terminal, source the CloudXR environment and read one collection with the stock SE3
printer. Use a collection id printed by the plugin in Step 7:

.. code-block:: bash

   cd IsaacTeleop/
   source ~/.cloudxr/run/cloudxr.env
   ./install/examples/schemaio/se3_printer vive_tracker_<serial>

Step 9 (optional): Record and replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the examples package once, from the Isaac Teleop root:

.. code-block:: bash

   uv pip install -e ./examples/mcap_record_replay

While the plugin is running, record every tracker. With no arguments, the script records 10
seconds to ``./recordings/<timestamp>.mcap``:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   python -m isaaccapture_examples.mcap_record_replay.record_se3_vive

To record for a different duration or to a specific file, use
``record_se3_vive [duration_s] [output.mcap] [--collections a,b,c]``.

Replay the newest recording. Replay needs no CloudXR runtime and no hardware. Open the printed
URL (by default ``http://localhost:8080``) to see each tracker as a 3D coordinate frame:

.. code-block:: bash

   python -m isaaccapture_examples.mcap_record_replay.replay_se3_vive

Useful options are ``--loop``, ``--rate N``, ``--no-viz`` and ``--collections a,b,c``. See
:doc:`/references/mcap_record_replay` for the underlying MCAP tooling.

Using the Tracker Data
----------------------

Collection naming
~~~~~~~~~~~~~~~~~

Each tracker becomes its own collection, named after the tracker's physical serial number:

- ``vive_tracker_<serial>`` (for example ``vive_tracker_SN0001``) when the pose carries a serial.
- ``vive_tracker_<device_id>`` as a fallback if the serial is empty. The plugin logs when this
  happens.

The name is fixed by the first pose from a tracker. Because the serial is the stable identity of
a tracker, an MCAP file recorded today still shows which physical tracker each channel came from.
The plugin does not assign roles such as "left wrist". Map serials to roles in your own
application.

Collection auto-discovery
~~~~~~~~~~~~~~~~~~~~~~~~~

While the plugin runs, it writes its live collection ids, one per line, to
``$XDG_RUNTIME_DIR/vive_se3_collections.txt`` (or ``/tmp/vive_se3_collections.txt`` if
``XDG_RUNTIME_DIR`` is not set). The file is updated as trackers appear and removed when the plugin
exits. Readers such as ``record_se3_vive`` use it to find collections without being told the ids.

To read the poses in your own code, create one ``Se3Tracker`` per collection id. See
:doc:`/device/trackers`.

Reference frame
~~~~~~~~~~~~~~~

Poses are forwarded exactly as VIVE Hub reports them: right-handed, X right, Y up, -Z forward
(the same axis convention as OpenXR). The origin comes from the VIVE Hub tracking map. It is
**not** the OpenXR session base space, so align the frames downstream if you combine the trackers
with headset or controller data.

Timestamps and validity
~~~~~~~~~~~~~~~~~~~~~~~

VUT pose timestamps use the host ``CLOCK_MONOTONIC`` clock, the same local clock that Isaac Teleop
uses, so each sample keeps its original sample time. The plugin runs at about 90 Hz and, for each
tracker:

.. list-table::
   :header-rows: 1

   * - Condition
     - Action
   * - New sample since the last push
     - Push ``is_valid=true`` with the VUT sample time.
   * - No new sample, but still fresh
     - Push nothing. Readers keep the last pose, and MCAP gets no duplicate timestamps.
   * - Stale (older than ``VIVE_SE3_STALE_MS``)
     - Push ``is_valid=false`` with an identity pose on every tick.

Configuration
-------------

The plugin is configured with environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Meaning
   * - ``VIVE_VUT_SOCKET``
     - ``/tmp/vut.sock``
     - Path of the VIVE Hub VUT daemon socket.
   * - ``VIVE_SE3_STALE_MS``
     - ``250``
     - Time in ms (1 to 3600000) after which a tracker without new samples is pushed as invalid.
   * - ``VIVE_SE3_COLLECTIONS_FILE``
     - ``$XDG_RUNTIME_DIR/vive_se3_collections.txt``
     - Where live collection ids are advertised.
   * - ``VIVE_SE3_SYNTHETIC``
     - (unset)
     - Set to ``1`` to generate fake trackers without VIVE Hub, for a no-hardware smoke test.

Troubleshooting
---------------

- **The plugin was not built.** Check the CMake configure output for ``VIVEHub VUT SDK not found``
  and make sure ``-DVUT_SDK_DIR`` points at the ``sdk`` folder of VIVE Hub 1.0.1 or newer.
- **The plugin runs but no collections appear.** Make sure VIVE Hub is running, the trackers are
  tracking in its UI, and ``vut_demo`` (Step 4) prints poses. If VIVE Hub uses a different socket
  path, set ``VIVE_VUT_SOCKET``.
- **VIVE Hub does not see the dongle.** Run ``sudo ./install-udev.sh`` on the host, then unplug and
  replug the dongle.
- **Testing without hardware.** Run the plugin with ``VIVE_SE3_SYNTHETIC=1`` to check the Isaac
  Teleop side on its own.
- **Reporting a VIVE Hub problem.** Open an issue on the
  `vivehub-linux repository <https://github.com/ViveRoboticsSoftware/vivehub-linux/issues>`_ and
  attach the newest ``vivehub_*.log`` and ``tracker_server_*.log`` files from
  ``~/.local/share/HTC/VIVEUltimateTracker/logs/``. VIVE Hub keeps only the latest 10 logs for
  each component, so copy them before you launch it again several times.

License
-------

The ``vive_se3_tracker`` plugin and examples are covered by their stated licenses (Apache-2.0).
VIVE Hub for Linux and the VUT SDK are proprietary to HTC, are governed by the VIVE Product EULA,
and are **not** redistributed by this project.
