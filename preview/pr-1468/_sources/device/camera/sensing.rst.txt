.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

SENSING GMSL Camera
===================

SENSING GMSL2 cameras connect to a Jetson through the SG8A carrier and stream
through NVIDIA SIPL. Choose either setup below to preview both cameras with
``nvsipl_camera``, the vendor reference application. Run commands from a
terminal on the Jetson desktop, at the Isaac Capture repository root. The
preview appears on the Jetson monitor.

Prerequisites
-------------

This setup targets a Jetson AGX Orin 64 GB with JetPack 7.2.1 / L4T R39.2.1,
a SENSING **SG8A-AGON-G2Y-A1** carrier, and two Astra **SHW5G** cameras on
**CN1 CAM2/CAM3**. The ``SHW5G_2`` configuration captures 2560×1984 at 60 fps
per sensor through the ISP. Other camera populations are not tested.

Host setup needs network access, Git, and sudo. Container setup additionally
needs Docker with the NVIDIA container runtime configured on the Jetson.

.. warning::

   SIPL has exclusive access to the cameras. Stop any other camera client
   before running the check.

Host: setup and preview
-----------------------

**1. Install the host drivers.**

.. code-block:: bash

   src/plugins/sensing/scripts/setup.sh

Accept the installation and group-membership prompts. On first install,
setup opens Jetson-IO: choose **Configure Jetson AGX CSI Connector → Configure
for compatible hardware → Jetson Sensing SG8A-AGON-G2Y-A1 SIPL GMSL2x8**, then
save and reboot. Return to the repository root after reboot. If only group
membership changed, log out and back in before continuing.

**2. Preview both cameras.**

.. code-block:: bash

   nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 \
       -m "0x0000 0x1100" --enable-camera-hal -s -Z --egl-display

A live camera grid opens on the monitor. The terminal also reports FPS for
both sensors, approximately 60 fps each. Enter ``q`` in the terminal to stop.

Container: setup and preview
----------------------------

Complete host setup above first, including the overlay reboot and group
activation. Both setup scripts run **on the host**.

**1. Build and enter the camera container.**

.. code-block:: bash

   src/plugins/sensing/docker/setup_container.sh

This builds the runtime image and opens a shell at the repository root inside
it. Host camera tools, drivers, ISP tuning, device nodes, and the SENSING
source directory are mounted automatically. The desktop X11 socket and
authorization file are passed through for the preview. No SDK downloads or
manual Docker flags are needed. The container runs as root with privileged
device access;
the SENSING source and vendor files are mounted read-only.

**2. Run the same preview command inside that shell.**

.. code-block:: bash

   nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 \
       -m "0x0000 0x1100" --enable-camera-hal -s -Z --egl-display

The live camera grid appears on the same monitor as the host preview. Enter
``q`` to stop, then type ``exit`` to leave the container;
it is removed when you exit. Subsequent launches can use
``setup_container.sh --no-build`` to reuse the image.

Setup details
-------------

Host setup fetches the `SENSING driver package
<https://github.com/SENSING-Technology/nvidia-jetson-camera-drivers>`_ and
installs the SHW5G userspace drivers, camera HAL, ISP tuning, device-tree
overlay, and reference tools. It compares file contents before installing
updates and records the installed vendor commit in
``/var/lib/sensing/installed.json``. A userspace-only update needs no reboot.
There are no sensor kernel modules to load or per-boot setup commands.

Use ``setup.sh --pkg DIR`` for a local vendor package, ``--no-fetch`` to use
the cached package, or ``--yes`` to accept prompts unattended. Setup returns
a nonzero status if installation fails or a reboot/new login is still needed.
The scripts' ``--help`` output lists their options.

The tracked :code-file:`configs/shw5g.json
<src/plugins/sensing/configs/shw5g.json>` supplies the two-camera configuration.
The camera command uses ``--enable-camera-hal`` for this carrier, ``-Z`` to
skip unsupported sensor authentication, ``-s`` to report FPS, and
``--egl-display`` to open the live preview. Add
``-r 10`` for a ten-second run, or omit ``--egl-display`` for a terminal-only
capture check.

For custom images, :code-file:`docker/Dockerfile.example
<src/plugins/sensing/docker/Dockerfile.example>` shows the runtime dependencies.
``setup_container.sh --print-args`` prints shell-quoted Docker flags;
``--json`` emits the same flags as a JSON array. These runtime helpers do not
install the SDK headers needed to compile SIPL applications.

Troubleshooting
---------------

Run :code-file:`scripts/verify.sh <src/plugins/sensing/scripts/verify.sh>`
on the side that fails for a read-only runtime report:

.. code-block:: bash

   src/plugins/sensing/scripts/verify.sh

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Symptom
     - Action
   * - ``SetPlatformConfig (Camera HAL) failed. status: 10``
     - Check device permissions, the selected overlay, and vendor mounts with
       ``verify.sh``. Activate new host groups by logging in again.
   * - ``nvsipl_camera: command not found``
     - Complete host setup. It installs the tool in ``/usr/sbin``; ensure that
       directory is on your host's ``PATH``. Recreate containers through
       ``setup_container.sh`` to mount the installed tool.
   * - ``Failed to open file .../shw5g.json``
     - Run the command from the repository root.
   * - ``cannot open NITO file .../SHW5G.nito``
     - Rerun host setup to install ISP tuning, then recreate the container.
   * - No frames, or no modules after applying link masks
     - Check the cameras are on CN1 CAM2/CAM3 and use the masks shown above.
       Stop other SIPL clients before retrying.
   * - ``Failed to create EGLImage``
     - Check that ``DISPLAY`` does not point at Xvfb or another server that
       cannot use Tegra EGL. Use the physical desktop display.
   * - Missing ``DISPLAY`` or Xauthority
     - Over SSH, set ``DISPLAY`` to the Jetson desktop (``:1`` on the test rig).
       Container setup finds the current user's GDM authority file automatically
       if ``XAUTHORITY`` is unset and ``~/.Xauthority`` is absent. For other
       desktop managers, set ``XAUTHORITY`` explicitly. SSH X11 forwarding does
       not target the Jetson monitor.
   * - Low FPS or dropped frames
     - Check the Jetson power mode. Maximum performance can be selected with
       ``sudo nvpmodel -m 0 && sudo jetson_clocks``.

Build and run the camera producer
---------------------------------

``camera_plugin_sensing`` runs on the Orin. It captures the GMSL cameras with
SIPL, encodes H.264 on the hardware encoder and sends each camera's video over a
direct TCP link to the PC video stage (``video_stage_worker``), which publishes it
to LiveKit. The producer has no LiveKit client of its own. CaptureServer, the
egocap backend, recording and the browser run on the PC.

Prerequisites: an AGX Orin on JetPack 7.2.1 / L4T R39.2.1 with the SENSING
drivers installed and the cameras working, CUDA, CMake, and a network path to the
PC running ``video_stage_worker``. The build also needs two header trees; neither is fetched
by ``setup.sh``:

- ``JETSON_MMAPI_DIR``: the Jetson Multimedia API tree, containing
  ``include/nvbufsurface.h``. The encoder classes are copied into the repository
  and are not taken from it. CMake searches ``/usr/src/jetson_multimedia_api``.
- ``SIPL_API_DIR``: the Jetson SIPL API tree, containing
  ``sipl/include/NvSIPLCamera.hpp`` (the ``Jetson_SIPL_API_R<release>_aarch64.tbz2``
  package published with the L4T release). CMake searches ``/usr/src/jetson_sipl_api``,
  ``/opt/nvidia/jetson_sipl_api`` and ``$SIPL_API_DIR``.

.. code-block:: bash

   # Build.
   cmake -B build -DBUILD_PLUGIN_SENSING=ON -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF \
     -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=OFF \
     -DSIPL_API_DIR=... -DJETSON_MMAPI_DIR=...
   cmake --build build --target camera_plugin_sensing -j

   # Check the cameras are visible.
   build/src/plugins/sensing/app/camera_plugin_sensing --list-sensors

   # Send. Each stream's direct= is the PC address and port its video_stage_worker
   # --stream=PORT:TRACK listens on; start the worker first.
   build/src/plugins/sensing/app/camera_plugin_sensing \
     --add-stream=sensor=0,name=sensing-left,direct=PC_HOST:5600 \
     --add-stream=sensor=1,name=sensing-right,direct=PC_HOST:5601

The ``direct=`` ports must match the ``--stream`` ports of the PC's
``video_stage_worker``, whose ``TRACK`` names must equal the ``track`` names in
the PC's streams file. Stop with Ctrl-C. ``--help`` lists every option. The direct
link is plain TCP: use a wired link or a trusted network.

``SENSING_TSC_PROBE=1`` logs the SIPL capture counter beside the host clocks twice
a second, as separate uncalibrated samples. The counter's units are unconfirmed, so
do not subtract or fit them.

Cross-compile on a PC
---------------------

``scripts/setup_orin_crosscompile.py`` builds the producer for the Orin on an x86 Ubuntu PC, without sudo and
without touching the Orin. Everything goes under one directory (``--root``, default ``~/orin-cross-39.2.1``).

.. code-block:: bash

   src/plugins/sensing/scripts/setup_orin_crosscompile.py doctor   # what is present, what is missing
   src/plugins/sensing/scripts/setup_orin_crosscompile.py fetch    # sysroot and aarch64 compiler
   src/plugins/sensing/scripts/setup_orin_crosscompile.py syntax   # type-check every source in seconds
   src/plugins/sensing/scripts/setup_orin_crosscompile.py build    # camera_plugin_sensing for aarch64

``fetch`` unpacks the L4T libraries from the Jetson Linux archive, and the SIPL API, from the SDK Manager
download directory (``--sdk-downloads``). It takes the Multimedia API from there or from NVIDIA's apt repository,
the C library and EGL/V4L2 libraries from Ubuntu 24.04, spdlog at the revision the repository pins, and Arm's
GNU toolchain; each download is checked against the publisher's SHA-256. ``build`` also needs the CUDA toolkit
with its aarch64 cross target (``cuda-cross-sbsa``) and uses that toolkit's own ``nvcc``. The result is the same
binary a build on the Orin produces, but it has not been run: deploy it and test it on the Orin.
