.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Skeleton Binding
================

A device that describes rigid segments — the links a tactile glove's taxels are skinned to,
for example — carries a ``SkeletonBinding`` that names each segment in a skeleton whose pose
some other source provides at run time. The schema is
:code-file:`src/core/schema/fbs/skeleton_binding.fbs`; the registry and the checks below are
:code-file:`src/core/schema/cpp/inc/schema/skeleton_binding.hpp`.

``segment_names[i]`` names segment ``i`` of the device. ``profile`` says which skeleton the names
come from; without it the binding is *generic* and the names are labels only. ``asset`` identifies
an external description, such as a URDF, that a profile names segments in.

Registered profiles
-------------------

.. list-table::
   :header-rows: 1

   * - ``id``
     - ``version``
     - Segment names
     - Requires
   * - ``openxr.hand``
     - 1
     - ``HandJoint`` names (``PALM`` … ``LITTLE_TIP``), the XR_EXT_hand_tracking joints
     - ``side`` LEFT or RIGHT
   * - ``openxr.body.bd``
     - 1
     - ``BodyJoint`` names (``PELVIS`` … ``RIGHT_HAND``), the XR_BD_body_tracking joints
     - —
   * - ``urdf.link``
     - 1
     - ``<link name>`` values of the URDF in ``asset``
     - ``asset``

A profile's names never change within a version; a change takes a new ``version``. Ids outside
this table are valid and read as generic bindings. Vendor profiles take a reverse-DNS prefix,
such as ``com.example.glove``.

Rules
-----

``validate_skeleton_binding(binding, segment_count, side)`` returns an empty string for a valid
binding, otherwise the first rule it breaks:

#. ``segment_names`` has exactly ``segment_count`` entries, each non-empty and unique.
#. ``asset.digest``, when present, is an RFC 6920 SHA-256 name: ``ni:///sha-256;`` and the
   43-character unpadded base64url digest of the asset's bytes.
#. An ``asset`` needs a ``profile``.
#. ``profile.id`` is two or more dot-separated tokens of ``[a-z0-9_]``, each starting with a
   letter; ``profile.version`` is at least 1.
#. For a registered profile: ``asset`` is present exactly when the profile requires one, ``side``
   is LEFT or RIGHT when the profile requires it, and every name is one of the profile's names.

Checking names against an asset needs its bytes, so it is a separate step in Python:
``check_skeleton_asset(binding, data)`` verifies the digest and, for ``urdf.link``, that every name
is a link of the URDF. ``asset_digest(data)`` computes the digest to store.

.. code-block:: python

   from isaaccapture.schema import (
       AssetRef, SkeletonBinding, SkeletonProfileRef, asset_digest, check_skeleton_asset,
       validate_skeleton_binding,
   )

   urdf = open("glove.urdf", "rb").read()
   binding = SkeletonBinding(
       ["base_link", "index_tip"],
       profile=SkeletonProfileRef("urdf.link", 1),
       asset=AssetRef(asset_digest(urdf), "package://glove/glove.urdf", "application/xml"),
   )
   assert validate_skeleton_binding(binding, segment_count=2) == ""
   assert check_skeleton_asset(binding, urdf) == ""
