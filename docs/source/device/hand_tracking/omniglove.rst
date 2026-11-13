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

- Linux x86_64 or aarch64 with the system libusb runtime (``libusb-1.0-0``
  on Ubuntu); the current hardware validation used x86_64.
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

Install the USB rule on the host:

.. code-block:: bash

   ./src/plugins/omniglove/install_udev_rules.sh

CMake retrieves the pinned SDK into the build directory and validates its
release checksums. The target architecture selects the SDK package. The SDK
runtime library and its license notices are installed with the plugin. libusb
is supplied by the operating system.

.. code-block:: bash

   cmake -S . -B build -DBUILD_PLUGIN_OMNIGLOVE=ON
   cmake --build build
   cmake --install build --component omniglove

Existing build directories using ``OMNIGLOVE_SDK_ROOT`` or ``OMNIGLOVE_SDK_REF``
should remove those cache entries. Local SDK checkouts use CMake's standard
``-DFETCHCONTENT_SOURCE_DIR_OMNIGLOVE_SDK=/path/to/agilink_omniglove_sdk`` override.

Running
-------

Start CloudXR and connect the headset:

.. code-block:: bash

   python -m isaaccapture.cloudxr.service start

Source the runtime environment, select the wrist source
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

These are format placeholders; replace them with measured values for the
current mount. Controller and Auto mode reject missing transforms. Unknown
modes and invalid offsets are errors. Optical mode does not read the offsets.

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
  Tracking bits additionally follow ``wrist.tracked``. A valid but untracked
  wrist is still published with the tracking bits cleared.
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
  Each hand is reported inactive when fresh, usable glove data or a valid
  wrist pose is unavailable. Tracking resumes when both are available.

The plugin publishes the operator's skeleton. It emits no robot joint
commands or hand ``JointStateOutput`` data and defines no vendor hand schema.

Device status and CTS
---------------------

The shared device status publisher reports left and right hand availability
under the configured plugin root. A glove that stops streaming reports a
disconnected hand. A glove still streaming without a usable skeleton or wrist
reports a degraded hand. A transient rejected sample retains the last usable
pose until it becomes stale. Stream configuration or startup failures and
incompatible SDK sample contracts stop the plugin with an error.

Follow :doc:`/conformance/setup`, :doc:`/conformance/hand` and
:doc:`/conformance/submit` for recording, review and submission.

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

License
-------

The plugin source is Apache-2.0. The AgiLink OmniGlove SDK is licensed under
MulanPSL-2.0; its shared library is redistributed with the plugin, together with
``LICENSE.omniglove-sdk`` and ``THIRD_PARTY_NOTICES.omniglove-sdk.md``.

The SDK depends on libusb, licensed under LGPL-2.1-or-later. This plugin uses
the operating system's libusb and does not redistribute the SDK's bundled
libusb binary. The SDK notices describe the vendor's complete release;
only the SDK runtime library is copied into this plugin's install tree.
