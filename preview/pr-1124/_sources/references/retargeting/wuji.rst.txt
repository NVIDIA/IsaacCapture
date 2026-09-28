.. SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeter: Wuji Glove to Wuji Hand
====================================

``WujiHandRetargeter`` maps hand tracking from any OpenXR
``XR_EXT_hand_tracking`` source to the 20 joint commands of a Wuji Hand or
Wuji Hand 2. It wraps the retargeting support in ``wuji_sdk`` and does not
depend on the Wuji glove plugin.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - What happens
   * - Input
     - A 26-joint OpenXR ``HandInput`` for the configured side.
   * - Mapping
     - The OpenXR joints are reduced to the 21-joint MediaPipe hand layout
       (positions only; joint orientations are not used).
   * - Retargeting
     - ``wuji_sdk.retargeting.RetargetSession`` maps the hand pose to the
       selected Wuji hand model.
   * - Output
     - ``hand_joints`` contains 20 joint angles in firmware order;
       ``hand_valid`` indicates whether they came from a valid input frame.
   * - Tracking loss
     - Invalid input resets the retargeting session. The hardware demo holds
       the last valid command instead of forwarding the invalid output.

Installation
------------

Install the Wuji retargeting extra:

.. code-block:: console

   $ pip install 'isaacteleop[wuji]'

Use it from Python
------------------

.. code-block:: python

   from isaacteleop.retargeters import (
       WujiHandRetargeter,
       WujiHandRetargeterConfig,
   )

   config = WujiHandRetargeterConfig(
       model="wuji_hand_2",
       hand_side="right",
   )
   retargeter = WujiHandRetargeter(config, name="wuji_right")

``model`` accepts ``"wuji_hand"`` or ``"wuji_hand_2"``. ``hand_side`` accepts
``"left"`` or ``"right"``.

Run the example
---------------

The example supports a device-free synthetic pose, recorded keypoints, and a
real Wuji hand:

.. code-block:: console

   $ uv pip install -e ./examples/retargeting

   # Device-free smoke test
   $ python -m isaacteleop_examples.retargeting.wuji_hand_retargeter_demo \
       --mode synthetic --model wuji_hand_2 --hand right

   # Replay MediaPipe-format keypoints
   $ python -m isaacteleop_examples.retargeting.wuji_hand_retargeter_demo \
       --mode replay --replay my_take.pkl --model wuji_hand_2 --hand right

   # Drive real hardware and let TeleopSession launch the Wuji glove plugin
   $ python -m isaacteleop_examples.retargeting.wuji_hand_retargeter_demo \
       --mode drive --hand right --plugin-path build/src/plugins

Drive mode needs a running CloudXR runtime, an OpenXR hand-tracking source, and
a reachable Wuji hand.

``--model`` can be omitted: the model is detected from the connected hand, and
passing one that disagrees with it is an error. A hand whose reported side
differs from ``--hand`` produces a mirrored-motion warning. Without
``--plugin-path``, the OpenXR hand source must already be running.

Validate
--------

The tests import the Python package staged by a full build, so build the project
first — the glove plugin's ``install.sh`` only builds the plugin target:

.. code-block:: console

   $ cmake --build build
   $ ctest --test-dir build \
       -R 'retargeting_test_wuji_(hand_retargeter|mapping_tables)' \
       --output-on-failure

Both tests run by default — CTest installs the ``wuji`` extra for the
end-to-end retargeter test. Configure with ``-DTEST_WUJI_RETARGETER=OFF`` to
keep the Wuji SDK out of the test environment; the retargeter test then skips,
while the mapping-table cross-check still runs because it needs no SDK.

.. seealso::

   :doc:`/device/wuji_glove` -- install and run the Wuji glove input plugin.

   :ref:`isaac-teleop-pipeline-builder` -- connect the hand source and retargeter
   into a complete pipeline.
