.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Haptikos Exoskeletons
=====================

.. raw:: html

   <div style="margin-top: 1rem;"></div>

.. image:: ../_static/haptikos-exoskeletons.gif
   :alt: Haptikos exoskeletons used with Isaac Teleop
   :class: align-center
   :width: 100%

.. raw:: html

   <div style="margin-bottom: 1.5rem;"></div>

Use the `Haptikos <https://haptikos.tech/>`_ Exoskeletons with the Isaac Teleop framework. Currently only Linux is supported. Tested on Meta Quest headsets. Other headsets with controllers may work as well.

.. contents:: On this page
   :local:
   :depth: 2

Components
----------

- **Core library** (``haptikos_core``) — wraps the Haptikos Core App(``libHaptikosCore.so``) and provides access to per-joint tracking data and haptics.
- **Plugin executable** (``haptikos_plugin``) — the primary plugin binary that enables integration with the Teleop system via CloudXR and OpenXR.

Prerequisites
-------------
- **Linux** — x86_64 (tested on Ubuntu 22.04 / 24.04).
- **Haptikos API** for Linux.


Quick Start
-----------

Step 1: Get the Haptikos API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Haptikos API must be downloaded separately due to licensing. Check our `website <https://haptikos.tech/>`_.

1. Obtain a Haptikos account.
2. Clone the `repository <https://github.com/Haptikostech/HaptikosAPI>`_.
3. Copy the ``HaptikosCpp_API_Shared`` folder from our HaptikosAPI repository into the ``src/plugins/haptikos`` folder. The folder structure should be the following:

.. code-block:: text

   src/plugins/haptikos/
   ├── CMakeLists.txt
   ├── HaptikosCpp_API_Shared
   │   ├── include
   │   └── lib
   ├── haptikos_hands_plugin.cpp
   ├── haptikos_hands_plugin.hpp
   ├── main.cpp
   ├── plugin.yaml
   └── README.md

Step 2: Build the plugin
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ../../..  # Navigate to TeleopCore root
   cmake -S . -B build
   cd build
   make haptikos_hands_plugin

Step 3: Build the entire project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   cmake -B build -DENABLE_CLANG_FORMAT_CHECK=OFF #From Project root
   cmake --build build --parallel 4
   cmake --install build

We use the ``-DENABLE_CLANG_FORMAT_CHECK=OFF`` because the headers in the HaptikosAPI library are not clang-formatted. You can clang format the headers to not include the flag by running:

.. code-block:: bash

   clang-format -i src/plugins/haptikos/HaptikosCpp_API_Shared/include/*

Step 4: Run the plugin
~~~~~~~~~~~~~~~~~~~~~~

The Haptikos plugin connects to the Teleop session through the CloudXR / OpenXR runtime, so the runtime must be running and its environment sourced in the shell that launches the plugin.

Start the CloudXR runtime. It runs as a background service that outlives this shell, until you stop it with ``python -m isaacteleop.cloudxr.service stop``:

.. code-block:: bash

   source isaac_teleop_env/bin/activate
   python -m isaacteleop.cloudxr.service start --accept-eula

In the shell that runs the plugin, source the environment file that the runtime writes on startup. This points the OpenXR loader at CloudXR:

.. code-block:: bash

   cd IsaacTeleop/
   source ~/.cloudxr/run/cloudxr.env

In the same terminal, run the plugin with the following:

.. code-block:: bash

   ./install/plugins/haptikos/haptikos_hands_plugin

Important Information
---------------------

1. To use the plugin properly it is necessary to attach the controllers on our exoskeletons using the included mount.

2. The orientation calibration defines the direction the Haptikos gloves define as forward. This needs to be aligned with the Head Mounted Display's forward direction.

3. To inject the hand tracking data and haptic feedback, the controllers, the Haptikos App and the exoskeletons need to be active.

4. The executable will be located in the ``build/src/plugins/haptikos``. If you installed the project by running ``cmake --install build`` the executable will also be found in the ``install/plugins/haptikos`` folder.


License
-------

Source files are covered by their stated licenses (Apache-2.0). The Haptikos API is
proprietary to Haptikos and is subject to its own license; it is **not** redistributed
by this project.
