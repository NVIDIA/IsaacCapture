.. SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RealHand FFG Glove
==================

The Linux RealHand FFG Glove plugin reads one or two gloves over USB serial and publishes each
glove as a standard 26-joint OpenXR ``hand_tracker``. It discovers serial ports and
baud rates, queries the firmware-reported side, and reconnects after a disconnect. Either side can
operate alone; a missing glove does not control the opposite hand.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

.. code-block:: text

   FFG USB -> calibrated human skeleton -> HandInjector -> HandsSource -> HandInput retargeter
           21 angle sensors               26 OpenXR joints                any robot hand

The serial protocol follows the Apache-2.0
`RealHand FFG Glove pure-Python SDK
<https://github.com/RealHand-Robotics/FFG_realhand_pure_python>`_. The plugin implements the protocol
directly in C++, so the Python SDK is not redistributed or required at runtime.

The primary device paths are ``/hand/left`` and ``/hand/right``. They are consumed by
``HandsSource`` and the same ``HandInput`` retargeters used for optical hand tracking,
including ``DexHandRetargeter``, Sharpa, and Wuji. The device plugin contains
no robot-specific retargeting. Human joint poses are estimated from glove angles
and approximate hand dimensions.

Build
-----

The plugin is enabled by default on POSIX systems. Configure and build IsaacCapture normally:

.. code-block:: console

   $ cmake -S . -B build
   $ cmake --build build --target realhand_ffg_glove_plugin
   $ cmake --install build --component realhand_ffg_glove

Set ``-DBUILD_PLUGIN_REALHAND_FFG_GLOVE=OFF`` to exclude it from a POSIX build. The serial backend
is not built by default on Windows.

USB permissions
---------------

The current user must be able to open the glove's serial device. On Ubuntu, add the user to the
``dialout`` group, then log out and back in:

.. code-block:: console

   $ sudo usermod -aG dialout "$USER"

Do not run IsaacCapture as root. Check permissions with ``ls -l /dev/ttyUSB*`` and membership with
``id -nG``.

Run
---

Start the CloudXR runtime, source its OpenXR environment, and launch the installed plugin:

.. code-block:: console

   $ python -m isaaccapture.cloudxr.service start
   $ source ~/.cloudxr/run/cloudxr.env
   $ ./install/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin

Automatic discovery only probes ``/dev/serial/by-id/`` entries whose names contain
``RealHand``, ``RealForce``, or ``FFG`` (case insensitive). A generic USB UART identifier
does not distinguish a glove from other hardware. For these devices, explicitly
select the verified glove ports:

.. code-block:: console

   $ ./install/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin \
       --ports=/dev/ttyUSB0,/dev/ttyUSB1 \
       --baudrates=2000000,460800

Use only verified glove ports and close other applications reading the same devices.

When a Python example starts the plugin with ``--plugin-path``, pass
``--realhand-ffg-glove-ports=/dev/ttyUSB0,/dev/ttyUSB1`` to select its ports.
If the plugin is already running separately, omit ``--plugin-path`` instead of
starting another process on the same ports.

The supplementary collections ``realhand_ffg_glove_left`` and
``realhand_ffg_glove_right`` publish ``sensor_0`` through ``sensor_20`` in radians.
These ``JointStateSource`` inputs are for calibration and diagnostics, not robot
control. Mechanical hands consume the standard human skeleton through ``HandsSource``.

Wrist placement
---------------

The plugin uses the shared ``WristPoseSource`` utility, as in the Wuji Glove plugin.
``--wrist-source=auto`` (default) prefers native hand tracking, then falls back to the
controller aim pose. Use ``controller`` or ``handtracking`` to select one source explicitly.
The combined FFG example selects controller wrists; P7 arm targets still use controller
grip poses.

The default offsets support PICO 4 Ultra controllers mounted upright on the back of
each FFG glove using the RealHand bracket, with mirrored left/right installation.
For a different mount, override them with
``--left-aim-to-wrist=px,py,pz,qx,qy,qz,qw`` and
``--right-aim-to-wrist=px,py,pz,qx,qy,qz,qw``. Translation is in meters in the aim frame;
quaternions use XYZW order. ``--hand-scale=1.0`` adjusts the estimated hand size
(supported range 0.5 to 2.0).

In controller mode, the hand becomes inactive if wrist tracking is lost or jumps.
Disconnected gloves and stale sensor streams also become inactive.

Retargeting and calibration
---------------------------

Human skeleton calibration describes the device, not a target mechanical hand. One calibration
can feed L6, O6, L20, or another manufacturer's ``HandInput`` retargeter. Wear the glove
in its final fit. The command-line workflow waits for Enter before every step, then gives
8 seconds to release the keyboard and prepare, followed by a 6-second hold. It requires
at least 120 samples per hold. Change ``--prepare-seconds``, ``--hold-seconds``, or
``--samples`` if needed.

Capture these five endpoint poses for each connected side:

* **Open palm:** wrist neutral, palm flat, four fingers straight and together; thumb points sideways.
* **Full fist:** curl the four fingers fully; keep the thumb outside the fist, without forcing it.
* **Thumb-only curl:** keep the four fingers straight; fully bend the thumb MCP and IP joints.
* **Thumb opposition:** keep the four fingers straight; sweep the thumb across the palm toward
  the little-finger base, without forcing fingertip contact.
* **Finger spread:** keep the fingers straight and spread them sideways as far as comfortable.

Then lightly touch the thumb tip to the **index, middle, ring, and little fingertip**, one
at a time. Keep the unused fingers relaxed, without squeezing. Release and repeat each
of these four contacts, then repeat the original **open palm** pose. This is one 14-step
calibration sequence; when both sides are selected, do the same pose with both hands.
The PC calibration window in the hand demos uses this same sequence, sample reduction,
and fitting. L6, O6, and L20 reuse one human calibration without a model-specific step.

Calibration requires NumPy and SciPy in the Capture interpreter. Fingers remain
independently driven; contact poses are used for calibration, not runtime snapping.

Capture a calibration from the standard DeviceIO stream:

.. code-block:: console

   $ pip install -e ./examples/teleop
   $ python -m isaaccapture_examples.teleop.realhand_ffg_glove_calibration \
       --side both \
       --plugin-path install/plugins \
       --realhand-ffg-glove-ports=/dev/ttyUSB0,/dev/ttyUSB1 \
       --output /path/to/ffg_human.yml

Use ``--side left`` or ``--side right`` when one glove is connected. A later run for the other
side updates the same YAML and preserves the existing side. Pass the resulting
path with ``--realhand-ffg-glove-calibration``
when starting the plugin from the combined FFG example.
For a separately launched plugin, restart it with ``--calibration=/path/to/ffg_human.yml``.
Without calibration the plugin publishes raw sensors for capture but no active human hand.
Recalibrate after re-wearing, power-cycling, or materially shifting a glove. Failed
captures leave the previous calibration intact. Files use
``realhand-ffg-glove-human-calibration-v1``; robot-specific calibration files are incompatible.
The demo loads existing human calibrations, including those used before CTS, without
refitting or rewriting them. Calibration itself does not certify CTS conformance.

.. seealso::

   :doc:`/references/retargeting/realhand` describes the P7 and RealHand output pipelines.
   For device verification, see the official `Hand CTS workflow
   <https://nvidia.github.io/IsaacCapture/main/conformance/hand.html>`_.
