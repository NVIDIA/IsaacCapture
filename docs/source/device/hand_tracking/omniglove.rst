.. SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
.. SPDX-License-Identifier: Apache-2.0

AgiLink OmniGlove
=================

OmniGlove is a USB glove that measures the operator's finger skeleton. Its
Linux C++ plugin wraps the AgiLink OmniGlove SDK and publishes standard OpenXR
hand tracking for Isaac Capture. Robot-specific joint mapping runs in the
host's hand retargeter.

.. contents:: On this page
   :local:
   :depth: 2

Hardware and prerequisites
--------------------------

- Linux x86_64 or aarch64; the current hardware validation used x86_64.
- The Isaac Capture source-build dependencies and the
  `AgiLink OmniGlove SDK <https://github.com/AgibotTech/agilink_omniglove_sdk>`_.
- One left and one right OmniGlove for two-hand CTS, connected over USB.
  USB identifiers are ``cafe:4001`` (left) and ``cafe:4002`` (right).
- Host USB permission, CloudXR and a supported XR headset.
- Optical wrist tracking, or controllers rigidly mounted to the gloves with
  separately measured left/right controller-aim-to-wrist transforms.

Data flow
---------

The SDK supplies ``OMNIGLOVE_STREAM_POST_DATA`` in DIRECT mode at a nominal
125 Hz: a wrist-relative skeleton with 21 landmarks. The plugin converts SDK
joint axes, combines the skeleton with a world-space wrist pose from
``plugin_utils::WristPoseSource``, and pushes each hand through its own
``plugin_utils::HandInjector``. The host reads the injected hands using the
standard ``HandsSource``.

.. code-block:: text

   OmniGlove SDK (finger skeleton) ─┐
                                  ├─► OmniGlove plugin ─► OpenXR ─► HandsSource
   Optical hand / controller pose ─┘

Installation
------------

Install the USB rule on the host, then build in the Isaac Capture checkout:

.. code-block:: bash

   ./src/plugins/omniglove/install_udev_rules.sh
   ./src/plugins/omniglove/install.sh

The build script uses a sibling ``agilink_omniglove_sdk`` checkout when one is
available. Set ``OMNIGLOVE_SDK_ROOT`` or pass ``--sdk-root`` for a different
checkout. Otherwise it creates and updates a managed checkout. It validates
that SDK against its ``SHA256SUMS``, runs the API probe and conversion tests,
and installs the executable, manifest, runtime libraries and license notices
under ``install/plugins/omniglove``.

For an existing Python 3.12 build used by CTS:

.. code-block:: bash

   ./src/plugins/omniglove/install.sh --build-dir build-py312

Running
-------

Start CloudXR in one terminal and connect the headset:

.. code-block:: bash

   python -m isaaccapture.cloudxr.service run

In another terminal, source the runtime environment, select the wrist source
as described below, and launch the installed plugin:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   ./install/plugins/omniglove/omniglove_plugin

Wrist source and calibration
----------------------------

``OMNIGLOVE_WRIST_SOURCE`` selects ``hand_tracking`` (default), ``controller``
or ``auto``. Optical mode needs the headset to track the wrist while the glove
is worn. Controller mode uses controller aim poses and per-hand installation
offsets. Auto mode allows both sources; validate each source and the transition
before using it.

Rigidly attach each controller using its final mount. From simultaneous
controller aim and trusted wrist poses, calculate:

.. code-block:: text

   T_aim_to_wrist = inverse(T_world_aim) * T_world_wrist
   T_world_joint = T_world_aim * T_aim_to_wrist * T_wrist_joint

Use multiple poses to estimate the fixed transform, then verify it with new
positions and wrist rotations. Keep the controller mounted rigidly; changes
to its installation require recalibration. Each transform is a translation
in metres followed by a unit quaternion in ``(x, y, z, w)`` order:

.. code-block:: bash

   export OMNIGLOVE_WRIST_SOURCE=controller
   export OMNIGLOVE_AIM_TO_WRIST_LEFT="px,py,pz,qx,qy,qz,qw"
   export OMNIGLOVE_AIM_TO_WRIST_RIGHT="px,py,pz,qx,qy,qz,qw"

These are format placeholders; replace them with measured values. Controller
and Auto mode reject missing transforms. The tested mount's offsets and
calibration evidence are in the :code-file:`validation record
<src/plugins/omniglove/VALIDATION.md>`; those values apply only to that mount.

Published collections and joint contract
----------------------------------------

``plugin.yaml`` declares ``/hand/left`` and ``/hand/right``, both with
``type: hand_tracker``. The plugin publishes ``XR_EXT_hand_tracking`` joints;
the host records them as ``core.HandPoseRecord`` from ``hand.fbs``.

- Each frame has ``XrHandJointLocationEXT[26]`` slots in ``XrHandJointEXT``
  order. The wrist and four joints per finger are valid: 21 required joints.
- Palm and the four non-thumb metacarpals remain invalid because the SDK
  does not supply them. No neighbouring joint is copied into an absent slot.
- Valid joints have both ``POSITION_VALID`` and ``ORIENTATION_VALID`` set.
  Tracking bits additionally follow the validity of the world-space wrist.
- Positions are metres and orientations are unit ``(x, y, z, w)``
  quaternions, in the base space passed to ``HandInjector``.
- OpenXR joint ``+Z`` points away from the fingertip and ``+Y`` toward the
  back of the hand. SDK finger orientations receive a local Y half-turn;
  the right wrist basis is converted separately. Wrist and finger poses are
  composed with the same world-space wrist transform.
- SDK monotonic sample times are converted into runtime ``XrTime`` through
  ``XrTimeConverter``; the wrist is queried at that same time.
- Unmapped, unconverged IK, invalid bone data, non-finite poses, zero
  quaternions and stale frames are withheld. A sample is pushed only once.
  After 200 ms without fresh data the injector is destroyed, reporting the
  hand inactive. A fresh valid sample and wrist pose recreate it.

The plugin publishes the operator's skeleton. It emits no robot joint
commands or hand ``JointStateOutput`` data and defines no vendor hand schema.

Validation and CTS
------------------

.. Keep dated take results and acceptance status in the plugin validation record.

See the :code-file:`validation record <src/plugins/omniglove/VALIDATION.md>`
for hardware test evidence and acceptance status.

Follow :doc:`/conformance/setup` to prepare CTS. From a clean plugin commit
already pushed to the integration PR, load the mount configuration and record:

.. code-block:: bash

   ./src/plugins/omniglove/record_cts.sh --wrist-config /path/to/mount.env

Use ``--no-launch-cloudxr-runtime`` if CloudXR is already running. For
development takes, ``--allow-dirty`` retains the recorder's dirty provenance.
Include headset, controller, firmware and mount details in ``--note``.

Review the take in the official panel, then package it using the panel's
Submission button or the wrapper that uses the same bundle builder:

.. code-block:: bash

   src/python/isaaccapture/cts/hand/checker/.venv/bin/python -m hand_cts.panel /path/to/take-hand.mcap
   src/python/isaaccapture/cts/hand/checker/.venv/bin/python src/plugins/omniglove/package_cts.py /path/to/take-hand.mcap

Follow :doc:`/conformance/submit` to upload the archive and associate it with
the integration PR. Keep the archive outside the Git repository.

Troubleshooting
---------------

- **No USB glove:** install the udev rule on the host, reconnect the device
  and check user access to ``cafe:4001`` / ``cafe:4002``.
- **No world-space hand:** connect the headset, load ``cloudxr.env`` and
  check that the selected wrist source is available.
- **Misplaced or rotated hand:** verify the final controller mount and its
  aim-to-wrist transform, multiplication order, units and quaternion order.
- **Inactive hand:** the plugin logs ``hand inactive`` when SDK data or the
  wrist becomes invalid. On recovery, ``hand active`` includes the SDK
  connection generation and source sequence. Do not expect a frozen pose.
- **Persistent IK rejection:** check glove fit and SDK/firmware state;
  invalid skeleton frames are intentionally withheld.
- **Header/runtime ABI mismatch:** rebuild against the selected SDK and
  verify that the installed executable loads its accompanying runtime library.
