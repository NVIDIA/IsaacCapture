.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

SENSING GMSL Camera
===================

Bring-up for SENSING GMSL2 cameras on a Jetson AGX Orin, where capture goes
through **SIPL** rather than Argus or V4L2. This page covers provisioning the
rig and confirming it streams with the vendor's own tool; scripts live in
:code-file:`src/plugins/sensing`.

.. contents:: On this page
   :local:
   :depth: 2

Supported hardware
------------------

This plugin is developed and verified against exactly one configuration:

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - Host
     - NVIDIA Jetson AGX Orin (``p3737-0000 + p3701-0005``, 64 GB)
   * - Software
     - JetPack 7.2.1 / L4T R39.2.1
   * - Carrier
     - SENSING **SG8A-AGON-G2Y-A1** (GMSL2)
   * - Cameras
     - Two Astra **SHW5G** (VB1940), 2560×1984 @ 60 fps, RAW10 → ISP
   * - Platform config
     - ``SHW5G_2`` on CN1 CAM2/CAM3, link masks ``0x0000 0x1100``
   * - Device tree
     - ``Jetson Sensing SG8A-AGON-G2Y-A1 SIPL GMSL2x8``

The carrier also takes S56C and SHF3L modules, and the vendor package ships
``S56C_1_SHF3L_2`` and ``SHF3L_2_SHW5G_2`` configs for them. Those remain
*addressable* — the config name and JSON path are arguments — but nothing is
built or tested for them. They would also need an ICP passthrough path, since
the SHF3L modules are YUV422 sensors that bypass the ISP entirely.

Drivers come from the vendor SIPL package, which supplies userspace device
drivers (``libnvuddf_*.so``), a device-tree overlay, NITO ISP tuning files and
the ``nvsipl_camera`` reference application. The scripts below wrap it; they do
not replace it.

SENSING publishes it in `nvidia-jetson-camera-drivers
<https://github.com/SENSING-Technology/nvidia-jetson-camera-drivers>`_ as a
directory rather than a release, so ``setup.sh`` takes a blobless, shallow,
sparse clone of just this board's tree — 3.6 MB of a ~700 MB repo — into a
gitignored ``.cache/``. Nothing needs sourcing by hand.

What SIPL changes
-----------------

If you know the previous Argus-based rig, almost every operational assumption
moved:

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * -
     - Argus (SG10A)
     - SIPL (SG8A)
   * - Sensor drivers
     - kernel ``.ko``, loaded every boot
     - userspace ``.so`` in ``/usr/lib/nvsipl_drv``
   * - Device nodes
     - ``/dev/video0..9``
     - none
   * - Arbitration
     - ``nvargus-daemon``, multi-client
     - in-process, **exclusive**
   * - Per-boot step
     - ``load_modules.sh`` + systemd unit
     - none
   * - Frame sync
     - PWM trigger + J19 strap
     - deserializer ``fsyncMode: osc_manual``
   * - Geometry
     - ``--width``/``--height``/``--sensor-mode``
     - read from the platform config
   * - Privilege
     - container user + ``/tmp/argus_socket``
     - container user + two groups

.. warning::

   SIPL takes **exclusive** ownership of the hardware. ``nvsipl_camera`` and
   ``camera_plugin_sensing`` cannot run at the same time; stop one before
   starting the other. There is no daemon to arbitrate between them.

Setup
-----

Drivers are installed on the **host**, always — a container cannot do it, and
does not need to. Whether you then run the camera on the host or in a container
only decides what comes after:

.. code-block:: bash

   # 1. On the Jetson, once: install the drivers.
   src/plugins/sensing/scripts/setup.sh

   # 2. Confirm the rig streams. Same script on the host and in a container.
   src/plugins/sensing/scripts/validate.sh

   # 3. Only if you want a container:
   src/plugins/sensing/scripts/run_docker_example.sh   # ready-made, drops you at a shell
   src/plugins/sensing/docker/setup_container.sh      # or adapt your own

:code-file:`scripts/ <src/plugins/sensing/scripts>` holds what you run —
on the host, or on either side in the case of ``validate.sh`` and
``verify.sh``. :code-file:`docker/ <src/plugins/sensing/docker>` holds what
runs *inside* a container: the example image and the two scripts that only
make sense there. ``verify.sh`` is a read-only report and is what to
paste into a bug. Every script names the privileged actions it will take before
the first ``sudo`` prompt and asks before each optional one; ``--yes`` accepts
them all, and a non-interactive stdin declines them all.

.. note::

   First install only: after the drivers land you must select the device-tree
   overlay and reboot before anything streams. ``setup.sh`` stops and says
   so. That is the one step no script can do for you.

Host
~~~~

.. code-block:: bash

   src/plugins/sensing/scripts/setup.sh

It refreshes the vendor package, compares every file ``install.sh`` would copy
against what is installed, and offers to run it when anything is missing *or
stale*. ``--pkg DIR`` uses a package you already have, ``--no-fetch`` forbids
the download, and ``--install-drivers`` reinstalls unconditionally.

Every file is classified by camera module, and only the modules this rig
populates — ``SHW5G`` — are compared or installed. A drop that touches only
S56C changes nothing here:

.. code-block:: text

   ==> Installed vs package  (modules: SHW5G)
     · package SG8A-AGON-G2Y-A1-JetPack7.2.1 @ f4a59f7 (installed)
     ✓ libnvuddf_max96712_library.so             [shared]
     - libnvuddf_s56c_cameramodule_library.so    [S56C] not on this rig
     ! libnvuddf_shw5g_cameramodule_library.so   [SHW5G] stale
     ! SHW5G.nito                                [SHW5G] stale

.. note::

   Staleness is checked by content, not by version. SENSING reuses the package
   name across drops and does not bump ``Version.md``, so a release that only
   rebuilds the drivers and retunes the ISP is invisible to any check that
   looks at names or timestamps. What was installed is recorded in
   ``/var/lib/sensing/installed.json`` — a per-module digest plus the upstream
   commit, which is the version to quote when reporting a problem to SENSING.

   Such a drop needs no reboot either: only files are copied, and the overlay
   is the sole artifact the kernel reads at boot. The script says so rather
   than sending you through ``jetson-io`` again.

.. warning::

   Setup copies the files itself rather than running the vendor
   ``install.sh``, which wipes ``/usr/lib/nvsipl_drv`` and installs every
   module's driver. The trade is that a new vendor artifact could go
   unnoticed, so each run first checks that every source ``install.sh`` copies
   is one setup knows about, and says so loudly if not.

Then select the overlay and reboot:

.. code-block:: bash

   sudo /opt/nvidia/jetson-io/jetson-io.py
   # Configure Jetson AGX CSI Connector
   #   -> Configure for compatible hardware
   #   -> Jetson Sensing SG8A-AGON-G2Y-A1 SIPL GMSL2x8
   sudo nvpmodel -m 0 && sudo jetson_clocks

.. note::

   There is **no per-boot step**. The vendor ``install.sh`` is copy-only, so
   once the overlay is selected the rig comes up on its own. The old
   ``sensing-load.sh`` and ``sensing-camera.service`` no longer exist.

Groups, not root
~~~~~~~~~~~~~~~~

SIPL needs no root, but it does need two supplementary groups beyond ``video``:

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Node
     - Group
     - Why
   * - ``/dev/i2c-9``, ``/dev/i2c-10``
     - ``i2c``
     - the two MAX96712 deserializers
   * - ``/dev/gpiochip0``, ``/dev/gpiochip1``
     - ``gpio``
     - deserializer power enable

.. code-block:: bash

   sudo usermod -aG i2c,gpio $USER   # then log out and back in

Omit either and SIPL fails with exactly one line — ``Master SetPlatformConfig
(Camera HAL) failed. status: 10`` — which names neither the node nor the
permission. ``verify.sh`` diagnoses it properly.

Container
~~~~~~~~~

To build against this rig from inside a container:

.. code-block:: bash

   src/plugins/sensing/docker/setup_container.sh

It checks the container can reach the device nodes, then provisions both build
SDKs — public downloads resolved from the version in ``/etc/nv_tegra_release``.
The **Jetson Multimedia API** comes from the apt pool; the **SIPL API** is a
tarball published alongside the L4T release, fetched by URL. To install that
one by hand instead:

.. code-block:: bash

   sudo tar xf Jetson_SIPL_API_R39.2.1_aarch64.tbz2 -C /usr/src/

Running the cameras needs none of that, only what the next two sections cover.

Validating in a container
~~~~~~~~~~~~~~~~~~~~~~~~~

Before adapting your own image, prove the rig works from a throwaway one:

.. code-block:: bash

   src/plugins/sensing/scripts/run_docker_example.sh
   # ... builds, then drops you at a shell in the container:
   ./validate.sh

The script builds :code-file:`docker/Dockerfile.example
<src/plugins/sensing/docker/Dockerfile.example>` — which does nothing but add
the EGL runtime and run ``image_setup.sh`` — and starts it with the flags no
Dockerfile can express. ``--print-args`` and ``--json`` emit those flags for
your own container instead; ``-- validate.sh`` runs the check and exits rather
than opening a shell.

Five ingredients, and leaving any one out fails in a way that names something
else:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Ingredient
     - What its absence looks like
   * - ``image_setup.sh`` groups
     - ``SetPlatformConfig ... status: 10``, naming no node
   * - ``--privileged`` (or ``--device-cgroup-rule``)
     - every node unreadable no matter which groups the process holds
   * - ``NVIDIA_*`` env with ``--runtime=nvidia``
     - ``libnvsipl.so: cannot open shared object file`` — the runtime injects
       the BSP libraries only when the capabilities are requested
   * - ``libegl1`` in the image
     - ``libEGL.so.1: cannot open shared object file``
   * - the two ``:ro`` mounts
     - ``SetPlatformConfig ... status: 10`` again — the CSV list carries no
       vendor-added driver or NITO

Adapting your own image
~~~~~~~~~~~~~~~~~~~~~~~

Group membership belongs in the **image**, not in ``runArgs``: Docker resolves
supplementary groups for the container user from the image's own
``/etc/group``, so putting the user in ``i2c``, ``gpio`` and ``video`` at build
time covers PID 1 and ``docker exec`` alike. Verified on this rig — with the
gids in the image and no ``--group-add``, all of ``/dev/i2c-9``,
``/dev/i2c-10``, ``/dev/gpiochip0`` and ``/dev/nvmap`` are readable. The
reverse does not hold: ``--group-add`` *alone* leaves an interactive shell
without them, because ``docker exec`` ignores ``HostConfig.GroupAdd``.

The gids are host-specific, so read them off the host rather than copying:

.. code-block:: console

   $ src/plugins/sensing/scripts/run_docker_example.sh --print-gids
   114:i2c 985:gpio 44:video

The image half is one ``RUN``. Expose the directory as a named build context —
``docker build`` cannot ``COPY`` across into a submodule:

.. code-block:: bash

   docker build --build-context sensing=path/to/src/plugins/sensing/docker ...

.. code-block:: dockerfile

   RUN --mount=type=bind,from=sensing,target=/mnt/sensing \
       /mnt/sensing/image_setup.sh "$USERNAME" 114:i2c 985:gpio 44:video

The bind mount leaves nothing behind, so the script is not baked into a layer.
To layer onto an existing image instead of editing its Dockerfile:

.. code-block:: bash

   docker build -f docker/Dockerfile.example --build-arg BASE=my-image:latest \
       --build-arg SENSING_USER=me -t my-image:sensing src/plugins/sensing/docker

The runtime half comes from the same script:

.. code-block:: bash

   docker run $(src/plugins/sensing/scripts/run_docker_example.sh --print-args) my-image:sensing ...

``--json`` emits the same flags as ``devcontainer.json`` ``runArgs`` elements.
``--with-groups`` adds ``--group-add`` for an image built *without*
``image_setup.sh``; it is redundant otherwise.

nvsipl_camera
-------------

The vendor reference application, which setup installs to ``/usr/sbin``. It is
the ground truth for "is the rig itself alive": it drives
the same SIPL API as any other client, so if it cannot stream, nothing else
will either. Reach for it before debugging anything of ours.

``validate.sh`` wraps it with this rig's arguments and runs the same way on the
host and inside a container:

.. code-block:: bash

   src/plugins/sensing/scripts/validate.sh              # stream both sensors for 10 s
   src/plugins/sensing/scripts/validate.sh --query      # ask the config, no hardware
   src/plugins/sensing/scripts/validate.sh -t 0         # until 'q'

It resolves the package itself and fails with the reason rather than a status
code when the config or the binary is missing. To drive the tool directly:

.. code-block:: bash

   cd src/plugins/sensing/.cache/SG8A-AGON-G2Y-A1-JetPack7.2.1
   nvsipl_camera -t query/sg8a_agth_g2a/shw5g.json -c SHW5G_2 \
                 -m "0x0000 0x1100" --enable-camera-hal -s -Z -r 10

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Why this rig needs it
   * - ``-t FILE``
     - platform config JSON, resolved against the working directory — hence the ``cd``
   * - ``-c NAME``
     - config within that file; ``SHW5G_2`` for two SHW5G, ``-l`` lists them
   * - ``-m MASKS``
     - one mask per deserializer in transport order: nothing on CSI-GH, links 2 and 3 on CSI-CD
   * - ``--enable-camera-hal``
     - the camera-HAL path; without it SIPL takes the legacy DevBlk path, which this carrier does not support
   * - ``-Z``
     - skip sensor authentication — these modules carry no auth keys
   * - ``-s``
     - print FPS every 2 s, which is how you tell streaming from merely starting
   * - ``-r N``
     - exit after N seconds, and disable the interactive menu
   * - ``-v 3``
     - raise verbosity when a failure names nothing

Expect ~60 fps per sensor with zero drops. Without ``-r`` it runs until ``q``.

``nvsipl_query`` answers what the config *declares*, with no hardware and no
capture:

.. code-block:: bash

   nvsipl_query -t query/sg8a_agth_g2a/shw5g.json -l   # config names
   nvsipl_query -t query/sg8a_agth_g2a/shw5g.json -c SHW5G_2

Troubleshooting
---------------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Symptom
     - Cause
   * - ``SetPlatformConfig (Camera HAL) failed. status: 10``
     - missing ``i2c`` or ``gpio`` group — run ``verify.sh``
   * - ``no modules left in '<config>' after applying the link masks``
     - ``--link-masks`` does not match the config
   * - ``cannot open NITO file .../SHW5G.nito``
     - drivers were never installed — re-run ``setup.sh``
   * - ``Failed to open file .../shw5g.json``
     - the fetched package ships no ``SHW5G_2`` config; ``setup.sh`` drops one in
   * - capture hangs, no frames, no error
     - another SIPL client holds the hardware — stop ``nvsipl_camera``
   * - ``Could not get EglImage from fd`` / ``Failed to create EGLImage``
     - ``DISPLAY`` names an X server Tegra EGL cannot drive; see below

DISPLAY and the EGL vendor
~~~~~~~~~~~~~~~~~~~~~~~~~~

Two X servers usually exist on this box: ``:1`` is a real server the Tegra
driver can drive, ``:99`` is Xvfb and it cannot. With both ``10_nvidia.json``
and ``50_mesa.json`` installed, GLVND hands out *Mesa's* EGL whenever
``DISPLAY`` names a server Tegra EGL cannot drive, and
``NvBufSurfaceMapEglImage()`` then fails with ``Could not get EglImage from
fd``.

The container defaults to ``DISPLAY=:99``, so anything that renders needs:

.. code-block:: bash

   export DISPLAY=:1     # nvsipl_camera --egl-display, or any viewer
