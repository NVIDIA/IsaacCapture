.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

LeRobot and SO-101
==================

With Isaac Teleop, we got an end-to-end data collection and training pipeline for SO-101 in both sim and real teleoperation.

The `SO-101 <https://github.com/TheRobotStudio/SO-ARM100>`_ is a low-cost, open-source robot arm
that has become a popular platform in the `LeRobot <https://github.com/huggingface/lerobot>`_
community. Isaac Teleop lets you drive an SO-101 from more than one teleoperation device to collect
demonstrations — in simulation with `Isaac Lab <https://isaac-sim.github.io/IsaacLab>`_ and on real
hardware with `SO-101 support in LeRobot <https://huggingface.co/docs/lerobot/en/so101>`_ — then train a GR00T N1.7
manipulation policy on the result and close the loop from sim to real.

The same SO-101 task runs in Isaac Lab and on the real arm, each recording a LeRobot dataset. The two
panels below stack on narrow screens and sit side by side on wider ones, and can be replaced
independently:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::

      .. figure:: ../../_static/lerobot/so-101-isaac-teleop-sim.gif
         :alt: SO-101 task teleoperated in Isaac Lab (simulation)
         :width: 100%

         In Isaac Lab (simulation)

   .. grid-item::

      .. figure:: ../../_static/lerobot/so-101-isaac-teleop-real.gif
         :alt: SO-101 task teleoperated on the real arm
         :width: 100%

         On the real arm

End-to-end workflow
-------------------

The same SO-101 embodiment runs in simulation and on real hardware, so a single workflow carries
you from teleoperation to a deployed policy:

#. **Teleoperate and collect.** Drive the SO-101 with the :doc:`XR controller or the SO-101 Leader
   <devices>` and record demonstrations — in :doc:`real <data_collection_real>` and in
   :doc:`simulation <data_collection_sim>`. Both produce datasets in the LeRobot format.

   .. figure:: ../../_static/lerobot/xr-so-101-full.gif
      :alt: Teleoperation and data collection with LeRobot
      :width: 600px
      :align: center

      Teleoperation and data collection with LeRobot.

#. **Train.** Fine-tune a :doc:`GR00T N1.7 <training_groot>` policy on the collected dataset.

   .. Source: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/sim-teleop-example-huggingface.gif
   .. figure:: ../../_static/lerobot/sim-teleop-example-huggingface.gif
      :alt: Data collection with LeRobot
      :width: 600px
      :align: center

      Data set preview with LeRobot.

#. **Deploy.** Take the policy from sim to real — see the `Sim-to-Real SO-101 learning path`_ —
   addressing the sim-to-real gap with domain randomization, sim/real co-training, and
   actuator-gap compensation.

   .. Source: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/so101_vial_to_rack_task.gif
   .. figure:: ../../_static/lerobot/so101_vial_to_rack_task.gif
      :alt: Trained GR00T N1.7 policy running on the SO-101
      :width: 600px
      :align: center

      The trained GR00T N1.7 policy running autonomously on the SO-101.

In this section
---------------

- :doc:`devices` — the supported teleop devices: the XR controller and the SO-101 Leader.
- :doc:`data_collection_real` — record demonstrations on a physical SO-101.
- :doc:`data_collection_sim` — record demonstrations in Isaac Lab.
- :doc:`training_groot` — fine-tune a GR00T N1.7 policy on the collected data.
- :doc:`rebot` — bring up a reBot B601-RS arm on a Jetson AGX Orin over built-in CAN.

New to XR teleoperation? Start with the :doc:`Isaac Teleop Quick Start </getting_started/quick_start>`
to set up CloudXR and connect a headset.

Pending Tasks
-------------

The guides below are still being written:

| ☑ Implement teleop devices: XR controller and SO-101 Leader
| ☑ Data collection in real
| ☑ Data collection in sim (XR controller)
| ☑ Data collection in sim (SO-101 Leader)
| ☐ Export sim demos to the LeRobot dataset format
| ☐ Model training with GR00T N1.7
| ☐ Sim-to-Real — update the `Sim-to-Real SO-101 learning path`_ to use Isaac Teleop

See also: the `Sim-to-Real SO-101 learning path`_.

.. toctree::
   :hidden:

   devices
   data_collection_real
   data_collection_sim
   training_groot
   rebot

..
   References
.. _Sim-to-Real SO-101 learning path: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
