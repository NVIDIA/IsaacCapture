.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Tactile Schema
==============

A tactile device publishes through the generated ``TactileTracker``: a ``TactileMetadata`` once per
session as the tracker's :ref:`metadata <tracker-metadata>`, and a ``TactileData`` every frame. The
schemas are :code-file:`src/core/schema/fbs/tactile_metadata.fbs` and
:code-file:`src/core/schema/fbs/tactile.fbs`; the checks below are
:code-file:`src/core/schema/cpp/inc/schema/tactile_validation.hpp`.

Device description
------------------

``TactileMetadata`` holds:

- ``device`` — vendor, model, serial number, firmware, ``side`` (a ``BodySide``), nominal rate and
  calibration id.
- ``shape`` — the rigid ``segments`` of the device, each with its bind pose, and the ``taxels``, each
  with its pose and up to four skinning weights over the segments. Both are in the device frame at
  the reference configuration. ``taxel_areas`` optionally gives each taxel's area in m².
- ``binding`` — a :doc:`skeleton_binding` naming each segment in the skeleton that poses it. Without
  one the segments are not posed by anything and the taxels stay at their bind poses.
- ``channels`` — which ``TactileData`` vectors the device fills.
- ``extensions`` — vendor data, each named by a registry id such as ``com.example.glove``.

A taxel's frame has +Z along the outward surface normal and +X along the shear x axis. Its position
at run time is

.. code-block:: text

   p = sum_k weights[k] * T_current[segments[k]] * inverse(T_bind[segments[k]]) * p_bind

where ``T_current`` is the segment's pose from the bound skeleton, and ``p`` comes out in the frame
that skeleton is posed in. The taxel's axes blend the same way and are renormalized.

Per-frame data
--------------

Entry *i* of each ``TactileData`` vector is taxel *i* of the shape. Forces are what the contacted
surface exerts on the taxel, in newtons:

- ``normal_force`` — along the taxel normal; positive presses into the surface.
- ``shear_force`` — along the taxel frame's +X and +Y.
- ``raw`` — ``channels.raw_values_per_taxel`` uncalibrated readings per taxel.
- ``taxel_state`` — ``VALID``, ``SATURATED``, ``INVALID`` or ``UNAVAILABLE``; when absent, every
  taxel is ``VALID``.
- ``sequence`` — the device's frame counter, when it has one.

Rules
-----

``validate_tactile_metadata(metadata)`` returns an empty string for a valid description, otherwise
the first rule it breaks. The Python ``TactileMetadata`` constructor raises ``ValueError`` with the
same message.

#. ``shape`` has at least one segment and one taxel.
#. Every bind pose and taxel pose is finite, with an orientation of norm 1 within 1e-3.
#. Each taxel's weights are finite and nonnegative and sum to 1 within 1e-5. A slot with a nonzero
   weight names an existing segment that no other slot of the taxel names; a zero-weight slot
   holds segment 0.
#. ``taxel_areas`` is empty or holds one finite, positive value per taxel.
#. ``channels`` is present and declares normal force, shear force or raw readings.
#. ``binding``, when present, passes ``validate_skeleton_binding`` for the shape's segments and
   ``device.side``.
#. Extension ids are registry ids and unique.

``validate_tactile_data(data, metadata)`` checks that a frame fills exactly the declared vectors,
with one entry per taxel (``raw_values_per_taxel`` per taxel for ``raw``).
