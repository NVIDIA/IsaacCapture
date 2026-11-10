<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LeRobot and SO-101

With Isaac Teleop, we got an end-to-end data collection and training pipeline for SO-101 in both sim and real teleoperation.

The [SO-101](https://github.com/TheRobotStudio/SO-ARM100) is a low-cost, open-source robot arm
that has become a popular platform in the [LeRobot](https://github.com/huggingface/lerobot)
community. Isaac Teleop lets you drive an SO-101 from more than one teleoperation device to collect
demonstrations — in simulation with [Isaac Lab](https://isaac-sim.github.io/IsaacLab) and on real
hardware with [SO-101 support in LeRobot](https://huggingface.co/docs/lerobot/en/so101) — then train a GR00T N1.7
manipulation policy on the result and close the loop from sim to real.

The same SO-101 task runs in Isaac Lab and on the real arm, each recording a LeRobot dataset. The two
panels below stack on narrow screens and sit side by side on wider ones, and can be replaced
independently:

![SO-101 task teleoperated in Isaac Lab (simulation)](_static/lerobot/so-101-isaac-teleop-sim.gif)

In Isaac Lab (simulation)

![SO-101 task teleoperated on the real arm](_static/lerobot/so-101-isaac-teleop-real.gif)

On the real arm

## End-to-end workflow

The same SO-101 embodiment runs in simulation and on real hardware, so a single workflow carries
you from teleoperation to a deployed policy:

1. **Teleoperate and collect.** Drive the SO-101 with the [XR controller or the SO-101 Leader](devices.md) and record demonstrations — in [real](data_collection_real.md) and in
   [simulation](data_collection_sim.md). Both produce datasets in the LeRobot format.
   ![Teleoperation and data collection with LeRobot](_static/lerobot/xr-so-101-full.gif)

   Teleoperation and data collection with LeRobot.
2. **Train.** Fine-tune a [GR00T N1.7](training_groot.md) policy on the collected dataset.
   <!-- Source: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/sim-teleop-example-huggingface.gif -->![Data collection with LeRobot](_static/lerobot/sim-teleop-example-huggingface.gif)

   Data set preview with LeRobot.
3. **Deploy.** Take the policy from sim to real — see the [Sim-to-Real SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) —
   addressing the sim-to-real gap with domain randomization, sim/real co-training, and
   actuator-gap compensation.
   <!-- Source: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/so101_vial_to_rack_task.gif -->![Trained GR00T N1.7 policy running on the SO-101](_static/lerobot/so101_vial_to_rack_task.gif)

   The trained GR00T N1.7 policy running autonomously on the SO-101.

## In this section

- [Supported Teleop devices](devices.md) — the supported teleop devices: the XR controller and the SO-101 Leader.
- [Data Collection in Real](data_collection_real.md) — record demonstrations on a physical SO-101.
- [Data Collection in Sim](data_collection_sim.md) — record demonstrations in Isaac Lab.
- [Model Training with GR00T](training_groot.md) — fine-tune a GR00T N1.7 policy on the collected data.
- [reBot B601-RS on Jetson AGX Orin](rebot.md) — bring up a reBot B601-RS arm on a Jetson AGX Orin over built-in CAN.

New to XR teleoperation? Start with the [Isaac Teleop Quick Start](../../teleop/quick_start.md)
to set up CloudXR and connect a headset.

## Pending Tasks

The guides below are still being written:

☑ Implement teleop devices: XR controller and SO-101 Leader
<br/>
☑ Data collection in real
<br/>
☑ Data collection in sim (XR controller)
<br/>
☑ Data collection in sim (SO-101 Leader)
<br/>
☐ Export sim demos to the LeRobot dataset format
<br/>
☐ Model training with GR00T N1.7
<br/>
☐ Sim-to-Real — update the [Sim-to-Real SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) to use Isaac Teleop
<br/>

See also: the [Sim-to-Real SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

<!-- References -->
