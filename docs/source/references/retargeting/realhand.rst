.. SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RealHand L6 / O6 / L20 Hands
============================

L6, O6, and L20 use the official ``DexHandRetargeter`` with packaged left/right
URDF kinematics and vector-retargeting YAML. No particular arm or glove is
required. Standard OpenXR hand input can come from optical tracking or any
compatible device plugin.

Install
-------

Follow :ref:`build-from-source-install-package` to install the built Isaac
Capture package, then install the example:

.. code-block:: console

   $ uv pip install -e ./examples/realhand_hands

In-tree installation and tests are opt-in with
``-DBUILD_EXAMPLE_REALHAND_HANDS=ON``. Robot model files are installed in this
example's wheel, not the core Isaac Capture package.

Run
---

.. code-block:: console

   $ python -m isaaccapture_examples.realhand_hands.hand_example --hand-model l6
   $ python -m isaaccapture_examples.realhand_hands.hand_example --hand-model o6
   $ python -m isaaccapture_examples.realhand_hands.hand_example --hand-model l20

The default ``--input-mode handtracking`` reads ``HandsSource``. Start and
calibrate a glove plugin using its own device instructions; this example does
not open glove ports or fit glove calibration. ``--input-mode controller``
linearly scales each controller's trigger to that side's closed posture.

The CLI launches CloudXR and prints targets, not a simulator or hardware
controller. Use ``--dry-run`` to validate the graph without OpenXR.

Robot Data and Outputs
----------------------

``hand_config.get_hand_config(model, side)`` returns a public
``DexHandRetargeterConfig``. The solver uses relative bone and fingertip
vectors, with no gesture classification or runtime contact snapping.

``hand_example.build_pipeline(mode, model)`` returns named ``left_hand`` and
``right_hand`` joint outputs through official ``JointRateLimiter`` nodes.
Joint order and limits come from the packaged YAML and URDF. Bind these joint
names to the host robot; neither an arm nor a particular glove is assumed.

The mesh-free URDFs are retargeting kinematics, not visual/collision assets.
Self-collision, physics, and actuator control belong to the consuming environment.
The optional complete simulation application and robot assets are available in
`RealHand Teleop on Hugging Face <https://huggingface.co/realhandinc/realhand-teleop>`_.
They are not required to run the packaged retargeting models.
