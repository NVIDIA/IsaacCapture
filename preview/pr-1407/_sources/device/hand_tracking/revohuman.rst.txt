.. SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RevoHuman Glove
===============

A Linux-only plugin for BrainCo RevoHuman exoskeleton gloves. It reads each
glove's 21 encoder angles and its tactile forces through the RevoHuman C SDK,
maps the SDK's hand points to the 26 OpenXR hand joints and injects them into
the OpenXR hand-tracking layer. It also pushes the encoder angles as joint
states and the forces with the common tactile contract, laid out with the SDK's
taxel layout.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

.. code-block:: text

   RevoHuman glove (USB) ──► RevoHuman C SDK ─┐
                                              ├─► revohuman plugin ──► CloudXR OpenXR runtime
   Optical hand / controller ─────────────────┘   (hand points,        (hand joints, joint states,
                                                   wrist-pose fusion)   tactile forces)

The plugin computes no kinematics of its own: the SDK turns the angles into hand
points for the glove's mechanical model, and the plugin maps those points to
OpenXR order with corrections reviewed for that model. No URDF is loaded.

Outputs
-------

Per hand, with the default collection prefix ``revohuman``
(``--collection-prefix=NAME`` changes it):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Output
     - Contents
   * - OpenXR hand ``/hand/left``, ``/hand/right``
     - 26 joints relative to the wrist, placed at the wrist pose described below
   * - ``revohuman_<side>_joint_state`` / tensor ``joint_state``
     - ``JointStateOutput`` with the 21 encoder angles in radians, in SDK order;
       see :doc:`/device/joint_space/index`
   * - ``revohuman_<side>_tactile`` / tensor ``tactile``
     - ``TactileDataArray`` taxel forces in newtons, as the SDK reports them,
       along each taxel's outward normal; see
       :doc:`/device/tactile/index`
   * - ``revohuman_<side>_tactile/metadata``
     - The ``TactileMetadata`` layout the forces belong to

Started by hand, the plugin publishes ``<id>/device_status`` only with
``--plugin-root-id=<id>``.

Prerequisites
-------------

- **Linux** — x86_64 or aarch64.
- **Source-build prerequisites** — the toolchain and system packages in
  :doc:`/development/build_from_source/index`.
- **RevoHuman C SDK 0.1.3** — the ``c-sdk-v0.1.3`` release of
  `brainco-revohuman-sdk <https://github.com/BrainCoTech/brainco-revohuman-sdk>`_,
  which extracts to ``include/revohuman.h``, ``lib/librevohuman.so`` with its
  versioned names, and ``LICENSE``. The CMake configure downloads it
  automatically. A Python SDK wheel cannot replace it.
- **RevoHuman gloves** connected over USB, with access to their serial ports.
- **CloudXR runtime** running on the plugin host.

Installation
------------

With ``BUILD_PLUGIN_REVOHUMAN=ON``, the CMake configure downloads the pinned C
SDK release into the build directory and verifies it against a
per-architecture SHA-256. Build and install the plugin with the SDK library and
its licence next to it:

.. code-block:: bash

   cmake -B build -DBUILD_PLUGIN_REVOHUMAN=ON
   cmake --build build --target revohuman_plugin
   cmake --install build --component revohuman

``-DREVOHUMAN_SDK_ROOT=<dir>`` uses an SDK package you already extracted
instead of the download.

The installed plugin finds the SDK library next to itself, so it needs neither
the download nor the build directory afterwards. Keep the library's symlinks.

Running the Plugin
------------------

The plugin reaches the Teleop session through the CloudXR / OpenXR runtime, so
start the runtime first and source its environment in the shell that launches
the plugin:

.. code-block:: bash

   python -m isaaccapture.cloudxr.service start   # runs in the background
   source ~/.cloudxr/run/cloudxr.env
   ./install/plugins/revohuman/revohuman_plugin

See :ref:`dedicated-cloudxr-runtime` and
:ref:`load-cloudxr-environment-variables` for the full service setup.
Applications can also launch it through its ``plugin.yaml`` using
``TeleopSession``. The gloves are its only source, so it takes no source
argument; an unknown argument such as ``--live`` or ``--replay`` is ignored with
a warning.

The plugin neither records nor plays back: record its collections with
``McapRecordingConfig`` and play them back with ``McapReplayConfig``, as for any
other tracker (see :doc:`/references/mcap_record_replay`).

The plugin finds the gloves through the SDK's autokit, at most one left and one
right, with nothing to declare; the SDK opens, retries and reopens them in the
background. Gloves plugged in at start-up open together in one two-hand SDK kit,
synced on one USB frame where the SDK can; otherwise each opens on its own, and
a missing glove is looked for again at the SDK autokit's retry interval
(``retry_ms`` in ``revohuman.h``) with the SDK's reason in its status. The
SDK reconnects a glove after a short dropout and re-syncs a pair itself. When a
stream of the pair ends anyway, both gloves close and reopen as a pair once both
are back; if only one is back for the autokit's ``pair_wait_ms`` it opens on its
own, the other follows on its own, and the two stay unpaired until the plugin
restarts. Gloves streaming without hardware sync to each other (paired in host
time, unpaired after a lost pair, or opened separately) report their ``/glove``
status as degraded with the reason.

From start-up until a hand's first glove sample, for at most 5 s
(``kStartupHoldNs`` in the plugin's pipeline), the plugin injects that hand with
no valid joints. A client reading OpenXR hand tracking then gets no joints while the
gloves open, not the headset's optical hand followed by the glove's.

Each glove's mechanical model follows the product the SDK identifies: a PTV1
glove uses the DV1 model and a P1 glove the P model. A P1 glove has no tactile
sensors, so it streams the hand and joint states without tactile outputs. A
glove of the wrong product, with incompatible firmware or with a hand model the
plugin is not fitted to is reported as failed; the plugin keeps running. The
plugin is fitted to the models of the C SDK release that its CMake pins.

Wrist pose
----------

The glove stream contains wrist-relative joints but no absolute 6DoF wrist
pose. Before injection, the plugin places the hand at a wrist pose from optical
hand tracking or from a controller aim pose combined with a rigid per-hand
offset.

Set ``REVOHUMAN_WRIST_SOURCE`` (or pass ``--wrist-source``) to ``auto`` (the
default), ``hand_tracking``, or ``controller``. In ``auto`` mode, optical
tracking is preferred; the controller is used when optical tracking provides no
valid pose. The plugin refuses a ``--wrist-source`` that disagrees with the
variable.

A controller reports its aim pose, not the wrist. No controller mount is
calibrated for these gloves yet, so the aim-to-wrist offsets default to
identity. Set them with:

.. code-block:: bash

   ./install/plugins/revohuman/revohuman_plugin \
       --aim-to-wrist-left=px,py,pz,qx,qy,qz,qw \
       --aim-to-wrist-right=px,py,pz,qx,qy,qz,qw

Position values are in meters and quaternion values use ``(x, y, z, w)``.

Without a tracked wrist the hand is still injected, without tracking flags, at
the last valid wrist pose (or the stage origin), and its status is degraded.

The glove's wrist is a reference point of its mechanical model: the origin of
its ``palm_link``, on the back of the hand near the end of the forearm, moved
12.6 mm towards the palm. It is not the anatomical wrist
centre that the OpenXR wrist joint stands for. Nominally the wrist centre is
about 4 cm further towards the fingers and about 0.2 to 0.7 cm further towards
the palm; these figures come from typical adult hand dimensions, not from
measurements on the glove, and vary with the wearer. With optical hand tracking,
the injected hand therefore sits that much further towards the fingers and the
palm than the tracked hand. With a controller, set the aim-to-wrist offsets to
end at the glove's wrist point. Verify the alignment for the wearer.

Tactile layout
--------------

When a glove opens, the plugin builds its hand's ``TactileMetadata`` from the C
SDK's taxel layout for the glove's model (``rh_hand_model_taxel_layout``); there
is no layout file or option. Each taxel is bound rigidly to the OpenXR joint its
SDK link moves with, with its +Z along the outward normal. That joint is rigid
on the link, so the skinned taxels stay on the SDK's taxels as the fingers bend.
The log names the taxel count and whether the metadata was built; the debug log
names the SDK's layout source. When the SDK has no layout for the glove's taxel
count, the plugin warns why, pushes no tactile data and reports the hand's
tactile as failed; the other outputs are unaffected.

A force sample is pushed only while it is calibrated, recent and newer than the
last one, and its status names the reason when it is not. Consumers must check
that status as well as the presence of data.

Troubleshooting
---------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Symptom
     - Fix
   * - CMake configuration fails with a hash mismatch for ``revohuman_sdk``
     - The downloaded package does not match the pinned release. Reconfigure to
       rule out a corrupted download; if it persists, do not bypass the check
       — verify the release assets upstream.
   * - CMake configuration fails naming the RevoHuman SDK
     - The SDK at ``REVOHUMAN_SDK_ROOT`` is incomplete: its library, headers or
       licence are missing. Point it at a complete C SDK release, or clear it
       to download the pinned one, and reconfigure.
   * - A glove is never opened and its status names a busy port
     - Another process, such as the SDK viewer, holds the glove's serial port.
       Close it; the plugin opens the glove on its next attempt.
   * - A glove is reported as failed
     - The status carries the reason: a product or variant the plugin does not
       support, incompatible firmware, or a hand model the plugin is not fitted
       to, as from a C SDK release other than the one the plugin's CMake
       pins. These are not retried.
   * - Hands appear but a downstream client sees no tactile data
     - A P1 glove has no tactile sensors. Otherwise check the tactile status:
       failed means there is no ``TactileMetadata`` (the start-up log warns
       why, such as an SDK without the glove's taxel layout); otherwise it names
       the reason a sample was not pushed.
