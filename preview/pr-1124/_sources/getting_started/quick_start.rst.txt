.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Quick Start
===========

This guide walks through running a teleoperation session with an XR headset using CloudXR. By the
end you will have the retargeting pipeline processing live hand/controller data and printing gripper
commands to the terminal.

.. contents:: Steps
   :local:
   :depth: 1
   :backlinks: none

.. _try-on-brev:

Try It on Brev — Zero Setup
---------------------------

Experience **Isaac Teleop** remote teleoperation with no local install. Brev provisions the
server side — CloudXR runtime, retargeting pipeline, and an
`Isaac Lab <https://developer.nvidia.com/isaac/lab>`__ simulation — on a cloud GPU.
Just grab the instance IP and connect from your headset.

.. note::

   **Server (Brev):** CloudXR + Isaac Teleop retargeting + Isaac Lab sim on a cloud GPU.

   **Client (you):** open the `Isaac Teleop Web Client`_
   in your headset or browser, enter the Brev instance IP, accept the certificate, and click
   **Connect** — see step :ref:`connect-xr-headset`.

   Isaac Teleop is not limited to simulation; it drives real robots and ROS 2 pipelines too.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Launch on Brev — Isaac Lab 2.3
      :link: https://brev.nvidia.com/launchable/deploy?launchableID=env-3B3ETyweD8hOcZfNzaE12cnGNBq
      :link-type: url

      Stable

   .. grid-item-card:: Launch on Brev — Isaac Lab 3.0 beta 1
      :link: https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3B2jdZR8Ct0vQCzwseLuOZXoMyk
      :link-type: url

      Latest · recommended upgrade path

----

.. _check-out-code-base:

1. Check out code base (for examples)
-------------------------------------

Clone the repository and enter the project directory:

.. code-block:: bash

   git clone https://github.com/NVIDIA/IsaacTeleop.git
   cd IsaacTeleop

As a quick start guide, we don't need to build the code base from source. However, we still need
to clone the repository for a couple quick samples to run.

.. _install-isaacteleop-pip-package:

2. Install the ``isaacteleop`` pip package
-------------------------------------------

In a new terminal, activate your preferred virtual or conda environment, then install the latest
stable release from PyPI:

.. code-block:: bash

   pip install 'isaacteleop[cloudxr,retargeters]~=1.0'

Pre-release builds are published between stable releases. Picking one up takes the NVIDIA index
and an opt-in to pre-release versions — shown here with `uv <https://docs.astral.sh/uv/>`__:

.. code-block:: bash

   uv pip install 'isaacteleop[cloudxr,retargeters]~=1.0' \
         --extra-index-url https://pypi.nvidia.com --prerelease=allow

The two commands above track the newest 1.x release. To follow this version of the documentation
(|version|) exactly, pin to the series it describes instead:

.. parsed-literal::

   uv pip install 'isaacteleop[cloudxr,retargeters]\ |pip_version_pin|\ ' \\
         --extra-index-url https://pypi.nvidia.com --prerelease=allow

Instead of installing the package from PyPI, you can build from source and install the local wheel.
See :doc:`build_from_source/index` for more details.

.. _run-cloudxr-server:

3. Configure CloudXR (optional)
-------------------------------

The teleop examples in this guide bring CloudXR up for you when they connect —
you do **not** need to start the runtime in a separate terminal or source any
environment file. If nothing is serving yet, the example starts a CloudXR
service in the background and prints how to stop it; that service outlives the
example, so the headset stays connected from one run to the next.
The first launch downloads the CloudXR Web Client SDK and asks you to review and
accept the EULA on the terminal; answer the prompt once and the acceptance is
remembered for subsequent runs.

.. TODO: ``Quest3`` is the profile every WebXR client uses — Quest, PICO, and the
   desktop-browser IWER emulator — so the name reads as a hardware restriction it
   does not impose. Either rename the profile or add a generic alias (e.g.
   ``webxr``) in the runtime, then update the default and the values listed below.

The CloudXR runtime uses the ``Quest3`` device profile by default. The name is
narrower than the profile: ``Quest3`` serves every WebXR client — Quest, PICO,
and the desktop-browser IWER emulator — so PICO users should keep it. Apple
Vision Pro connects natively and needs ``auto-native`` instead; the default is
not derived from the connected headset, so set it yourself. The profile is set
on the *runtime*, so applications inherit whatever the runtime they connect to
was started with. To override it, or any other setting, write a ``KEY=value``
env file and start the service with it:

.. code-block:: bash

   echo 'NV_DEVICE_PROFILE=auto-native' > custom.env
   python -m isaacteleop.cloudxr.service start --cloudxr-env-config ./custom.env

The same ``--cloudxr-env-config`` flag is available on the teleop examples under
``examples/teleop/python/``, which register CloudXR's launcher arguments through
``CloudXRLauncher.add_launcher_arguments()`` — but it applies only when the
example is the one starting the runtime. With a service already running the
example attaches to it instead, and prints which settings it had to ignore (see
:doc:`/references/cloudxr`). (The ROS 2 example takes the equivalent
``cloudxr_env_config`` ROS parameter instead.)
To inspect the resolved settings after startup:

.. code-block:: bash

   cat ~/.cloudxr/run/cloudxr.env

.. note::

   To manage the service yourself — check what is running, stop it, or use
   launch modes like ``--host-client`` and ``--setup-oob`` — see
   :doc:`/references/cloudxr`.

.. list-table:: Environment variables
   :header-rows: 1
   :widths: 25 15 35 25

   * - Variable
     - Default
     - Description
     - Values
   * - ``NV_DEVICE_PROFILE``
     - ``Quest3``
     - Device profile
     - ``auto-webrtc``, ``auto-native``, ``Quest3``, ``AppleVisionPro``
   * - ``NV_CXR_ENABLE_PUSH_DEVICES``
     - ``true``
     - Push device overseer for hand tracking
     - ``true``, ``false``
   * - ``NV_CXR_FILE_LOGGING``
     - ``true``
     - File-based logging (disable to print to console)
     - ``true``, ``false``

On Jetson Orin the launcher selects the experimental CloudXR runtime and
main-thread join automatically. You normally do not need to set
``ISAAC_TELEOP_CLOUDXR_EXP`` or ``ISAAC_TELEOP_CLOUDXR_JOIN_MAIN``; those are
overrides for debugging or forcing the stable path. See
:doc:`/references/cloudxr`.

.. _whitelist-firewall-ports:

4. Whitelist ports for Firewall
---------------------------------

CloudXR requires certain network ports to be open. Depending on your firewall configuration, you
might need to whitelist them manually.

.. dropdown:: Meta Quest and Pico headsets
   :open:

   For **Quest and Pico headsets** (web client), at the minimum, you need to whitelist the ports
   for the CloudXR runtime and wss proxy:

   .. code-block:: bash

      sudo ufw allow 47998/udp
      sudo ufw allow 49100,48322/tcp

   If you are running the web client from source (dev server), open both
   ports:

   .. code-block:: bash

      sudo ufw allow 8080,8443/tcp

.. dropdown:: Vision Pro client

   For **Vision Pro client**, you need to whitelist the ports for the CloudXR runtime and wss proxy:

   .. code-block:: bash

      sudo ufw allow 48010,48322/tcp
      sudo ufw allow 47998:48000,48005,48008,48012/udp

Please see the `CloudXR network setup`_ for more details for other network configurations (such as
running the CloudXR runtime and wss proxy in containerized environment; or using Vision Pro client).

.. _connect-xr-headset:

5. Connect an XR headset
------------------------

.. _connect-quest-pico:

.. dropdown:: Meta Quest, PICO headset, or desktop browser
   :open:

   No physical headset required for a quick test: open
   `nvidia.github.io/IsaacTeleop/client`_
   in a **desktop browser** — IWER (Immersive Web Emulator Runtime) loads automatically and
   emulates a Meta Quest 3 headset.

   For a real headset, open the same URL in your **Meta Quest or PICO browser**.

   .. important::

      If using a physical headset, make sure it is updated to the latest firmware before connecting.
      Older firmware may ship an outdated WebXR runtime that fails to connect or streams with reduced
      functionality.

   .. note::

      Out of the box, Quest headsets stream at 72 FPS and 25 Mbps (Pico 4 Ultra at 90 FPS
      and 100 Mbps). You can raise both under **Advanced settings** in the control panel,
      up to 120 FPS and 200 Mbps, but on a typical 5 GHz Wi-Fi link the higher bitrates
      saturate the connection and you get reprojection judder within a few minutes. The
      defaults are the values that held stable over long sessions in our testing.

      The client remembers your settings between sessions. A saved profile other than
      ``Custom`` picks up the current recommended values on the next load, so updated
      defaults reach you without clearing browser storage. Switch the profile to
      ``Custom`` if you want your manual values to stick.

   .. note::

      If GitHub Pages is unreachable (corporate network, air-gapped machine), you can serve the web
      client locally from the CloudXR proxy and open ``https://<your-ip>:48322/client/`` instead of
      the GitHub Pages URL. Port 48322 is already whitelisted in step
      :ref:`whitelist-firewall-ports`. See :doc:`/references/oob_teleop_control` for how to serve the
      client from the proxy.

   .. tab-set::
      .. tab-item:: CloudXR web client

         .. figure:: ../_static/cloudxr-web-client-howto.png
            :alt: CloudXR web client usage instruction
            :align: center

            **Figure:** CloudXR web client usage instruction

      .. tab-item:: Privacy warning

         .. figure:: ../_static/cloudxr_accept_cert_not_private.png
            :alt: Browser privacy warning for self-signed certificate

            **Figure:** Browser privacy warning for self-signed certificate

      .. tab-item:: Certificate accepted

         .. figure:: ../_static/cloudxr_accept_cert_accepted.png
            :alt: Certificate accepted page

            **Figure:** Certificate accepted page

   As illustrated in the figure above, there are 3 steps to connect to your headset:

   1. Enter the IP address of the workstation running CloudXR
   2. Accept the self-signed SSL certificate, which was created automatically during :ref:`run-cloudxr-server`:

      - Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
      - In the new tab, you will see a **"Your connection is not private"** warning. Click **Advanced**, then **Proceed to <ip> (unsafe)**.
      - Once accepted, the page will show **Certificate Accepted**. Navigate back to the CloudXR.js client page.

   .. important::

      **Jetson Orin:** when the CloudXR runtime runs on Orin, set **Video Codec** to
      **H.264** in the CloudXR web client.

   3. Click **Connect** to begin teleoperation.

   .. note::
      For advanced usage and troubleshooting of CloudXR, see the `CloudXR documentation`_ for more
      details.

   .. dropdown:: Offline / air-gapped use

      On **first run**, the launcher syncs the published web client into
      ``~/.cloudxr/static-client/`` (override with ``TELEOP_WEB_CLIENT_STATIC_DIR``):
      ``index.html``, ``bundle.js``, and ``bundle.emulator.js``. Subsequent runs
      are offline once those files are cached.

      For a **true air-gapped machine**, copy the full ``build/`` output (or the
      matching directory from `nvidia.github.io/IsaacTeleop/client`_) into
      ``~/.cloudxr/static-client/`` on the air-gapped host before the first run.

   The source code for the web client is in the :code-dir:`deps/cloudxr/webxr_client/` directory.
   To build the web client from source, see :doc:`build_from_source/webxr`.

.. _connect-apple-vision-pro:

.. dropdown:: Apple Vision Pro

   For Apple Vision Pro, you will need to build and install the Isaac XR Teleop Sample Client. Follow
   the instructions in the `Isaac XR Teleop Sample Client for Apple Vision Pro`_ repository to build
   and install the sample client on your Apple Vision Pro.

   .. note::

      You will need v3.0.0 or newer of the `Isaac XR Teleop Sample Client for Apple Vision Pro`_
      to connect to Isaac Teleop.


.. _run-teleop-example:

6. Run a teleop example
------------------------

Run the simplified gripper retargeting example. This demonstrates the full
pipeline: reading XR controller input via CloudXR, retargeting it through the
``GripperRetargeter``, and printing the resulting gripper command values:

.. code-block:: bash

   python examples/teleop/python/gripper_retargeting_example_simple.py

Once running, squeeze the controller triggers on your XR headset to control
the gripper. You should see periodic status output:

.. code-block:: text

   ============================================================
   Gripper Retargeting - Squeeze triggers to control grippers
   ============================================================

   [  0.5s] Right: 0.00
   [  1.0s] Right: 0.73
   [  1.5s] Right: 1.00
   ...

The example runs for 20 seconds and then exits. To try other examples, see
``examples/teleop/python/`` — for instance:

- ``se3_retargeting_example.py`` — maps hand or controller poses to
  end-effector poses (absolute or relative)
- ``dex_bimanual_example.py`` — bimanual dexterous hand retargeting
- ``gripper_retargeting_example.py`` — full gripper example with more
  configuration options

Next steps
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card::

      .. image:: ../_static/isaaclab.jpg
         :alt: Isaac Lab

      ^^^^^^^^^^^^^

      **Teleoperation in Isaac Lab**

      Follow instructions in `Teleoperation and Imitation Learning with Isaac Lab Mimic`_ to know
      more about how to collect demonstrations with Isaac Lab and how to augment them with Isaac
      Lab Mimic and train imitation learning policies.

      If you are new to Isaac Lab, follow instructions in `Isaac Lab Quick Start`_ to get started.

   .. grid-item-card::

      .. image:: ../_static/isaacros.png
         :alt: Isaac ROS

      ^^^^^^^^^^^^^

      **Teleoperation with Isaac ROS**

      For a complete `Isaac ROS`_ pipeline, follow `Teleoperation with Isaac GR00T and Unitree G1`_
      — an end-to-end workflow that combines Isaac Teleop, CloudXR, and ROS 2 to teleoperate a
      Unitree G1 humanoid. You validate the setup in MuJoCo, then deploy on real hardware over a
      Jetson AGX Thor, as a precursor to data collection and imitation learning.

      .. rst-class:: trademark-notice

      *ROS is a trademark of Open Robotics.*

More Information
----------------

- :doc:`teleop_session` — learn how ``TeleopSession`` works and how to build
  custom retargeting pipelines
- :doc:`televiz` — visualize robot camera and sensor feeds in an XR headset with
  the Televiz compositor (and share a single XR session with your teleop pipeline)
- :doc:`build_from_source/index` — build the C++ core, Python bindings, and plugins
  from source

..
   References
.. _`CloudXR documentation`: https://docs.nvidia.com/cloudxr-sdk/latest/index.html
.. _`Isaac XR Teleop Sample Client for Apple Vision Pro`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
.. _`Isaac Lab Quick Start`: https://isaac-sim.github.io/IsaacLab/develop/source/setup/quickstart.html
.. _`Teleoperation and Imitation Learning with Isaac Lab Mimic`: https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/teleop_imitation.html#teleoperation-imitation-learning
.. _`CloudXR network setup`: https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html#ports-and-firewalls
.. _`Isaac ROS`: https://nvidia-isaac-ros.github.io
.. _`Teleoperation with Isaac GR00T and Unitree G1`: https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-teleop.html
