.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

SOMA Pose Transport
===================

Isaac Teleop can transport SOMA body and hand data through FlatBuffers and
``SchemaPusher``. Producers publish either joint rotations or evaluated joint
poses. Consumer-side sources normalize both representations to global joint
poses for retargeting or visualization.

The schemas and examples are defined and verified against the SOMA-X v0.3.1
pose interface. See the `SOMA-X repository <https://github.com/NVlabs/SOMA-X>`_
for the upstream model and assets.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

Body and hand data use separate schemas and collections. Applications can
select their body and hand sources independently.

.. code-block:: text

   vendor body SDK -> SOMA body FlatBuffer -> SchemaPusher -> SOMA body tracker
   vendor hand SDK -> SOMA hand FlatBuffer -> SchemaPusher -> SOMA hand tracker

The body schema contains SOMA's complete body control set, including its hand
controls. The standalone hand schemas provide independent left and right hand
streams. Isaac Teleop does not automatically fuse standalone hands with the
hand controls in a body payload.

Choose a representation
-----------------------

Use the representation closest to the producer's native data.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Representation
     - FlatBuffer types
     - Behavior
   * - ``joint-rotations``
     - ``SomaBodyJointRotations`` and ``SomaHandJointRotations``
     - Publishes SOMA joint rotations and the global Hips or Wrist translation.
       The consumer evaluates forward kinematics against a prepared SOMA
       identity.
   * - ``joint-poses``
     - ``SomaBodyJointPoses`` and ``SomaHandJointPoses``
     - Publishes global position and orientation for each provided joint. The
       consumer maps these poses directly and does not run per-frame forward
       kinematics.

``joint-rotations`` most directly matches the SOMA-X pose-control interface.
``joint-poses`` is an Isaac Teleop transport profile for producers that already
have evaluated global joint transforms.

Both representations use positions in meters and unit XYZW quaternions. They
use a right-handed reference frame with positive Y up and positive Z forward.
The publisher must convert its native skeleton and reference frame before
serialization. See :ref:`tracker-reference` for the complete field and
validity contracts.

Consumers normalize finite quaternions whose norm is greater than ``1e-12``. A
zero, degenerate, or non-finite quaternion, position, or global translation is
unavailable. Quaternion signs are not canonicalized, so ``q`` and ``-q`` remain
equivalent.

Rotation consumers resolve the SOMA v0.3.1 reference through SOMA-X and pass it
explicitly during forward kinematics. The prepared identity and subject scale
remain caller-selected calibration and are not constrained by this check.

Body and hand joint values are sorted, unique keyed vectors. Every entry carries
an explicit ``SomaBodyJoint`` or ``SomaHandJoint`` identifier. A missing joint is
unavailable for that frame; producers do not serialize placeholder values for
joints they cannot provide. Consumers expand the keyed values into SOMA-X's
canonical dense order when invoking its body or hand helpers.

Build the examples
------------------

Start with a configured Isaac Teleop checkout. See
:doc:`../getting_started/build_from_source/index` for the complete build
prerequisites. Install the live viewer and both deterministic publishers, then
build their native pushers:

.. code-block:: bash

   uv pip install -e ./examples/deviceio_live_view
   uv pip install -e ./examples/soma_body_publisher
   uv pip install -e ./examples/soma_hand_publisher

   .venv/bin/cmake -S . -B build
   .venv/bin/cmake --build build \
     --target isaacteleop_python soma_body_pusher soma_hand_pusher \
     -j4

The example packages install ``py-soma-x==0.3.1``. SOMA-X resolves the model
assets needed by rotation consumers. The demo publishers additionally require a
matching SOMA-X asset directory containing ``example_animation.npy``. The hand
publisher also requires ``SOMAHand.npz``.

Run the body example
--------------------

Start the viewer first. It launches the CloudXR runtime and prints the viewer
URL. Open that URL before starting the publisher. On a remote host, replace
``localhost`` in the URL with the host name or IP address.

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.deviceio_live_view \
     --accept-eula \
     --body-schema soma

In another terminal, load the runtime environment and start the body
publisher:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   uv run --no-sync python -m isaaccapture_examples.soma_body_publisher \
     --data-root /path/to/SOMA-X/assets \
     --pusher ./build/examples/soma_body_publisher/cpp/soma_body_pusher/soma_body_pusher \
     --loop

These commands use ``joint-rotations``. To publish evaluated joint poses, add
``--body-representation joint-poses`` to the publisher and
``--soma-body-representation joint-poses`` to the viewer.

Run the hand example
--------------------

Start the viewer with SOMA hands:

.. code-block:: bash

   uv run --no-sync python -m isaaccapture_examples.deviceio_live_view \
     --accept-eula \
     --hand-schema soma

In another terminal, load the runtime environment and start the hand
publisher:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   uv run --no-sync python -m isaaccapture_examples.soma_hand_publisher \
     --data-root /path/to/SOMA-X/assets \
     --pusher ./build/examples/soma_hand_publisher/cpp/soma_hand_pusher/soma_hand_pusher \
     --loop

These commands use ``joint-rotations``. To publish evaluated joint poses, add
``--hand-representation joint-poses`` to the publisher and
``--soma-hand-representation joint-poses`` to the viewer.

Collection IDs and schema identifiers
-------------------------------------

A collection ID pairs a producer with a consumer and is not encoded in the
FlatBuffer payload. The tensor identifier selects the representation carried by
that collection. The demo publishers use these fixed collection IDs; another
producer can choose different IDs as long as its consumers use the same values.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Data
     - Collection ID
     - Viewer override
   * - Body
     - ``soma_body_demo``
     - ``--soma-body-collection-id``
   * - Left hand
     - ``soma_hand_left_demo``
     - ``--soma-left-hand-collection-id``
   * - Right hand
     - ``soma_hand_right_demo``
     - ``--soma-right-hand-collection-id``

The representation selected by the publisher and consumer must match. The
transport identifiers are ``soma_body_joint_rotations``,
``soma_body_joint_poses``, ``soma_hand_joint_rotations``, and
``soma_hand_joint_poses``.

Vendor integration
------------------

A vendor integration converts its native skeleton to one of the checked-in
SOMA contracts and publishes it through ``SchemaPusher``. Use these schemas as
the serialization source:

- :code-file:`src/core/schema/fbs/soma_common.fbs`
- :code-file:`src/core/schema/fbs/soma_body_common.fbs`
- :code-file:`src/core/schema/fbs/soma_body_joint_rotations.fbs`
- :code-file:`src/core/schema/fbs/soma_body_joint_poses.fbs`
- :code-file:`src/core/schema/fbs/soma_hand_joint_rotations.fbs`
- :code-file:`src/core/schema/fbs/soma_hand_joint_poses.fbs`

The FlatBuffer transport and generated trackers do not depend on SOMA-X or
PyTorch. A rotation consumer uses SOMA-X and its resolved neutral identity to
evaluate forward kinematics. A joint-pose consumer does not need SOMA-X for
per-frame evaluation.

Current limitations
-------------------

- The examples evaluate rotations against a neutral identity. They do not
  perform subject calibration.
- Body and standalone hand streams are not fused automatically.
- A producer and consumer must select the same representation.
- The live viewer selects sources at launch. It does not switch schemas while
  running.

Troubleshooting
---------------

No module named ``isaaccapture_examples``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the corresponding example with ``uv pip install -e`` and launch it
with ``uv run --no-sync`` from the repository root.

The publisher cannot connect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the viewer, open its client, and then source
``~/.cloudxr/run/cloudxr.env`` in the publisher terminal. The environment file
identifies the active CloudXR OpenXR runtime.

The viewer receives no skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the publisher and viewer use the same representation and
collection names. For a demo publisher, also confirm that its asset directory
matches SOMA-X v0.3.1 and contains the required animation and model files.

See also
--------

- :code-file:`examples/soma_body_publisher/README.md`
- :code-file:`examples/soma_hand_publisher/README.md`
- :code-file:`examples/deviceio_live_view/README.md`
- :doc:`trackers`
