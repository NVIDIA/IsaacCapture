.. SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RealHand FFG Glove
==================

The Linux RealHand FFG Glove plugin reads one or two gloves over USB serial and publishes each
glove as a standard Isaac Teleop ``JointStateOutput`` collection. It discovers serial ports and
baud rates, queries the firmware-reported side, and reconnects after a disconnect. Either side can
operate alone; a missing glove does not control the opposite hand.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

.. code-block:: text

   RealHand FFG Glove USB ─► realhand_ffg_glove plugin ─► JointStateSource ─► RealHandFFGGloveRetargeter
                         side discovery       21 sensors       L6 / O6 / L20 joints

The serial protocol follows the Apache-2.0
`RealHand FFG Glove pure-Python SDK
<https://github.com/RealHand-Robotics/FFG_realhand_pure_python>`_. The plugin implements the protocol
directly in C++, so the Python SDK is not redistributed or required at runtime.

Build
-----

The plugin is enabled by default on POSIX systems. Configure and build Isaac Teleop normally:

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

Do not run Isaac Teleop as root. Check permissions with ``ls -l /dev/ttyUSB*`` and membership with
``id -nG``.

Run
---

Start the CloudXR runtime, source its OpenXR environment, and launch the installed plugin:

.. code-block:: console

   $ python -m isaaccapture.cloudxr.service start
   $ source ~/.cloudxr/run/cloudxr.env
   $ ./install/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin

By default the plugin scans ``/dev/ttyUSB*``, ``/dev/ttyACM*``, ``/dev/ttyXRUSB*``, and
``/dev/ttyOBC*`` using the supported baud rates. Restrict discovery when diagnosing a connection:

.. code-block:: console

   $ ./install/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin \
       --ports=/dev/ttyUSB0,/dev/ttyUSB1 \
       --baudrates=2000000,460800

The left and right collections are ``realhand_ffg_glove_left`` and
``realhand_ffg_glove_right``. Each publishes ``sensor_0`` through ``sensor_20``.

Retargeting and calibration
---------------------------

RealHand FFG Glove samples are glove measurements, not robot joint commands. Use one
``RealHandFFGGloveRetargeter`` per connected side and select ``l6``, ``o6``, or ``l20``. Calibration
must be captured while wearing the glove in its final fit. Hold each pose steadily and avoid
pressing the fingertips together hard enough to deform the glove.

For L6 and O6, capture these poses:

* **Open palm:** wrist neutral, palm flat, all five digits naturally straight and separated.
* **Full fist:** all four fingers fully flexed; thumb folded naturally across the index and middle
  fingers rather than forced into the palm.
* **Thumb-only curl:** four fingers stay open; move the thumb through opposition and flexion toward
  the palm without closing another finger.
* **Thumb to index:** touch the two fingertip pads lightly while the other fingers stay relaxed.
* **Thumb to middle:** touch the two fingertip pads lightly while the other fingers stay relaxed.

For L20, additionally capture thumb-to-ring and thumb-to-pinky poses. These samples describe the
thumb's calibrated response surface. At runtime, thumb outputs use only thumb sensors, and every
other digit uses only its own sensors; there is no contact snapping or cross-finger pose trigger.
Samples that continue beyond a calibrated endpoint saturate at that endpoint instead of reopening
the corresponding mechanical-hand joints.

Capture a calibration from the standard DeviceIO stream:

.. code-block:: console

   $ python examples/teleop/python/realhand_ffg_glove_calibration.py \
       --hand-model l20 --side both \
       --plugin-path install/plugins \
       --output /path/to/calibration_l20.yml

Use ``--side left`` or ``--side right`` when one glove is connected. A later run for the other
side updates the same YAML and preserves the existing side. Calibration files contain
user- and glove-specific measurements. Keep them outside the source tree and pass the resulting
path explicitly with ``--realhand-ffg-glove-calibration`` when running an FFG example.

.. seealso::

   :doc:`/references/retargeting/realhand` describes the P7 and RealHand output pipelines.
