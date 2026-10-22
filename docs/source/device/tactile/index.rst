.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Tactile devices
===============

Tactile devices capture contact measurements. For actuator output, see
:doc:`/references/haptic_feedback`.

The common tactile interface separates each frame's force measurements from the
device's static layout. A *taxel* is one tactile sensing element: its force has
three components, and its layout describes where it is attached to a skeleton.
The current geometry mapping is linear blend skinning (LBS) over hand joints.

.. contents:: On this page
   :local:
   :depth: 2

Data contract
-------------

The contract is defined by
:code-file:`src/core/schema/fbs/tactile_array.fbs` and
:code-file:`src/core/schema/fbs/tactile_metadata.fbs`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Meaning
   * - ``TactileDataArray``
     - One force vector per taxel, in newtons, ordered exactly as the metadata.
       The Python view is an ``(N, 3)`` float32 array.
   * - ``TactileMetadata``
     - The static ``geometry_mapping`` and ``segment_source`` for that stream.
       Currently these select ``LBS`` and ``HandSkeletonRef``.
   * - ``HandSkeletonRef``
     - An ordered list of unique ``HandJoint`` identifiers from
       :code-file:`src/core/schema/fbs/hand.fbs`. ``NUM_JOINTS`` is a count,
       not a usable joint.
   * - ``LBS``
     - ``J`` segment bind poses, ``N`` taxel bind poses, and four segment indices
       and weights per taxel. An index addresses the ordered segment list,
       rather than a ``HandJoint`` ordinal.

Python pose arrays have shape ``(count, 7)``: position ``(x, y, z)`` in meters,
then a unit quaternion ``(x, y, z, w)``. Identity orientation is ``(0, 0, 0, 1)``.
All frames are right-handed. A pose maps coordinates from its local frame into
its reference frame: ``p_reference = R(q) * p_local + position``.

Coordinate conventions
----------------------

Hand joints and the skeleton root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HandSkeletonRef`` uses ``HandJoint.WRIST`` as the skeleton root. Its origin
is the wrist pivot. The joint names and local orientations follow
`OpenXR hand-joint conventions
<https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#convention-of-hand-joints>`_:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Root axis
     - Anatomical direction
   * - ``+Y``
     - Out through the back of the hand at the wrist.
   * - ``+Z``
     - Along the line through the wrist and middle-finger metacarpal, pointing
       toward the forearm, away from the fingertips.
   * - ``+X``
     - ``+Y cross +Z``, completing the right-handed basis.

Finger-joint local ``+Z`` follows its bone away from the fingertip; local ``+Y``
points toward the nail/back side in the fully open hand. The palm uses the
middle metacarpal for its longitudinal axis. These local frames move with the
joints when the hand bends; their axes are not fixed world directions.

``segment_bind_poses[j]`` maps joint-local coordinates into the wrist frame in
the calibrated bind configuration. All segments share that root; the poses are
not relative to each joint's parent. If WRIST appears in the segment list, its
bind pose is identity. ``taxel_bind_poses`` use the same root and bind
configuration. The bind configuration must match the device's calibration;
the schema does not prescribe joint angles or a particular hand size.

**HandPose is not automatically wrist-relative.** The live ``HandTracker``
copies OpenXR joint poses in the session base space, currently ``STAGE``.
To obtain current joint poses in the tactile skeleton's wrist frame, use the
wrist and joint poses from the same valid hand sample:

.. code-block:: text

   T_root_joint(t) = inverse(T_base_wrist(t)) * T_base_joint(t)

Here ``T_A_B`` maps a point from frame B to frame A. This conversion uses both
the wrist translation and rotation. Subtracting the wrist position alone is
insufficient. A downstream consumer must perform this conversion when using
root-relative joint poses; ``TactileSource`` does not rebase a hand stream.

Taxel frames
~~~~~~~~~~~~

A taxel pose's origin is its sensing location. Its orientation defines:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Taxel axis
     - Direction
   * - ``+Z = N``
     - Unit normal pointing out of the sensing surface, away from the device.
   * - ``+X = T``
     - A calibrated unit tangent perpendicular to ``N``.
   * - ``+Y``
     - ``N cross T``. Thus ``+X cross +Y = +Z``.

The producer chooses and calibrates ``T`` for each taxel, then stores the full
orientation in ``taxel_bind_poses``. A normal alone cannot determine the tangent
direction. In particular, taxel ``+Z`` is a surface normal, whereas hand-joint
``+Z`` follows a bone; the two local frames generally differ.

Forces ``(fx, fy, fz)`` are components in these taxel-local axes, rather than in
the wrist or world frame. Positive ``fz`` points outward; this does not imply
that a vendor's positive compression reading can be copied directly. The schema
does not identify which body the reported force acts on. A device integration
must document that choice and calibrate its units, axes, and signs accordingly.
Raw pressure or ADC counts require conversion to force before publication.

Left and right hands
~~~~~~~~~~~~~~~~~~~~

Both hands use the same joint identifiers, right-handed algebra, units,
quaternion ordering, and taxel normal/tangent rules. Hand side does not reverse
the cross product or automatically negate a force component. Anatomical geometry
and the calibrated poses can differ between hands. For example, with an open
hand, wrist ``+X`` points toward the thumb side on the left hand and the little
finger side on the right hand.

``HandSkeletonRef`` carries neither a side nor the identity of a hand tracker.
The application must associate each tactile stream with the correct hand stream
and calibration, using configuration and distinct collection IDs such as
``tactile/left`` and ``tactile/right``. The runtime does not infer side from
these names. Mirroring positions alone is insufficient: a reflection cannot be
encoded as a rotation quaternion, so each hand needs a proper right-handed
orientation for every joint and taxel.

Mapping taxels onto the moving hand
-----------------------------------

For a taxel's homogeneous bind position ``p_bind`` in the skeleton root, LBS
computes its current root-relative position as:

.. code-block:: text

   p_root(t) = sum_k w[k] * C[j[k]](t) * inverse(B[j[k]]) * p_bind

``B[j]`` is ``segment_bind_poses[j]`` and ``C[j](t)`` is the current pose of
``joint_ids[j]`` in the current wrist frame. Each taxel has four slots; weights
must be finite, nonnegative, and sum to one (validation tolerance ``1e-5``).
Every slot's index must be valid, including slots with zero weight. A taxel
rigidly attached to one segment can use weights ``(1, 0, 0, 0)`` and repeat that
segment's index in all four slots.

To express the resulting position in the hand tracker's base space:

.. code-block:: text

   p_base(t) = T_base_wrist(t) * p_root(t)

LBS specifies blended positions; it does not define a unique rigid taxel frame
after blending multiple segment rotations. A consumer that needs animated
normals or world-space forces must choose an orientation/normal mapping rule.
For a single rigid attachment, the taxel transform is unambiguous:
``C[j](t) * inverse(B[j]) * T_root_taxel_bind``. Transform force directions using
rotation only, without adding the pose translation.

Publishing and consuming
------------------------

A C++ plugin publishes serialized samples with ``SchemaPusher``; a Python
producer can use ``TensorPushTracker``. The paired streams are:

.. code-block:: text

   <collection_id>           tensor "tactile"  -> TactileDataArray
   <collection_id>/metadata  tensor "metadata" -> TactileMetadata

Publish metadata on startup and periodically so consumers that start later can
discover it. Publish the same encoded metadata throughout the tracker session.
``TactileTracker`` validates and adopts the first metadata, records it once,
and exposes it through ``get_metadata()``. It withholds force data until metadata
is confirmed. Different or invalid metadata causes an error; after a producer
restart, its metadata must be confirmed again. See :ref:`tracker-metadata`.

The default force-buffer size is 16,384 bytes; larger frames require a matching
``max_flatbuffer_size`` on the producer and consumer. Metadata has its own
``TRACKER_METADATA_MAX_SIZE`` limit of 68,864 bytes, including FlatBuffer overhead.
The data array length must match the metadata's taxel count.

Applications can read ``TactileTracker`` directly for live or replay data.
Retargeting graphs can instead use ``TactileSource(name, collection_id,
num_segments, num_taxels)`` from
``isaaccapture.retargeting_engine.deviceio_source_nodes``. It emits
``taxel_forces`` and ``tactile_layout`` arrays; constructor counts fix their
shapes. It adapts the frame and static layout to graph tensors; the application
supplies hand motion and evaluates the geometry mapping separately.

The runnable feeder, direct tracker consumer, and MCAP replay commands are in
:code-file:`examples/tactile/README.md`. The feeder uses synthetic geometry to
exercise transport; it is not an anatomical calibration model. Live operation
requires an OpenXR runtime and an available HMD system, including when using a
headless session; replay does not require the runtime.

Vendor integrations
-------------------

The guides below describe individual devices. The existing OGLO integration
uses its own ``OgloGloveSample`` schema with raw measurements; it does not
automatically publish the common force-and-layout contract described above.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   oglo
