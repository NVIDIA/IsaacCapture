<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="ecosystem"></a>

# Ecosystem

The Isaac Capture ecosystem brings together the input devices, data services, and cloud
platforms used with the shared capture infrastructure. Browse every entry and
follow the link straight to its landing page, integration guide, or acquisition channel.

- **Devices** — Headsets, controllers, gloves, pedals, and master manipulators.
- **Data workflows** — Tools and services for recording, processing, and managing
  teleoperation data.
- **Teams building on Isaac Teleop** — Robot makers shipping teleoperation in their own
  product, and labs collecting data.

<!-- Explicit targets on the counted sections below: the count pill is part of the heading text,
so without them the slug carries the count — #supported-devices picks up whatever that
count currently is, and changes every time a row is added or removed. -->

<a id="supported-devices"></a>

<a id="supported-devices-15"></a>

## Supported Devices 15

A standardized device interface removes custom integrations and their maintenance. Input
modes determine which retargeters and control schemes are available.
[Add a new device →](../device/add_device.md)

### XR Headsets and Tracking Peripherals

* [Apple Vision Pro](https://www.apple.com/apple-vision-pro/): Hand tracking (26 joints), spatial controllers. Set up: [Sample Client](https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple); [Connect Apple Vision Pro](../teleop/quick_start.md#connect-apple-vision-pro)
* [Meta Quest 2 / 3 / 3S](https://www.meta.com/quest/): Motion controllers, hand tracking. Set up: [How To Connect](../teleop/quick_start.md#connect-quest-pico); Isaac Teleop Web Client — runs in the headset browser
* [PICO 4 Ultra](https://www.picoxr.com/products/pico4-ultra): Motion controllers, hand tracking. Set up: [How To Connect](../teleop/quick_start.md#connect-quest-pico); Isaac Teleop Web Client — runs in the headset browser
* [PICO Motion Tracker](https://www.picoxr.com/global/products/pico-motion-tracker): Full body tracking. Set up: [Body Tracking guide](../device/body_tracking/index.md); Isaac Teleop Web Client
* [VIVE Ultimate Tracker](https://www.vive.com/us/accessory/vive-ultimate-tracker/): Motion capture, body tracking. Set up: [VIVE Tracker Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/vive_se3_tracker); [VIVE Ultimate Tracker docs](../device/trackers/vive_ultimate_tracker.md); [Developer Guide](https://github.com/ViveRoboticsSoftware/vive-ultimate-tracker-sdk/blob/main/VIVE-Ultimate-Tracker-SDK-Guide-EN.md)

### Standalone Input Devices

* [MANUS Metagloves Pro](https://www.manus-meta.com/products/metagloves-pro): High-fidelity finger tracking. Set up: [MANUS Gloves Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/manus); [MANUS Gloves docs](../device/hand_tracking/manus.md)
* [Haptikos Exoskeletons](https://haptikos.tech/product/haptikos/): Finger tracking, haptic feedback. Set up: [Haptikos Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/haptikos); [Haptikos docs](../device/hand_tracking/haptikos.md)
* [OGLO Tactile Glove](https://www.opengraphlabs.com/): Tactile contact sensing (80 taxels), 6-axis IMU. Set up: [OGLO Tactile Glove Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/oglo_tactile); [OGLO Tactile Glove docs](../device/tactile/oglo.md)
* [WUJI Glove](https://wuji.tech/en/glove): High-fidelity hand tracking. Set up: [WUJI Glove docs](../device/hand_tracking/wuji_glove.md); [WUJI Glove Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/wuji_glove); [WUJI Hand retargeting](../references/retargeting/wuji.md)
* [Noitom Robotics Motion Capture](https://www.noitomrobotics.com): Full-body motion capture. Set up: [Noitom Mocap Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/noitom_mocap)
* [j-mex AgileMaster](https://jmex.com.tw/agilemaster/): Full-body motion capture. Set up: [j-mex AgileMaster Plugin](https://github.com/NVIDIA/IsaacCapture/tree/main/src/plugins/jmex)
* [Logitech Rudder Pedals](https://www.logitechg.com/en-us/products/flight/flight-simulator-rudder-pedals.html): 3-axis foot pedal. Set up: [Generic 3-axis Pedal Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/generic_3axis_pedal)
* [3Dconnexion SpaceMouse](https://3dconnexion.com/product/spacemouse-compact/): 6-DoF end effector, base velocity. Set up: [SpaceMouse Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/spacemouse); [SpaceMouse docs](../device/spatial_controllers/spacemouse.md)
* [Gamepads](https://github.com/mdqinc/SDL_GameControllerDB): 6-DoF end effector, base velocity. Set up: [Gamepad Plugin](https://github.com/NVIDIA/IsaacCapture/tree/main/src/plugins/gamepad); [Gamepad docs](../device/gamepad.md)
* [OAK-D Camera](https://shop.luxonis.com/products/oak-d): Offline data recording. Set up: [OAK-D Camera Plugin](https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/oak); [OAK Camera docs](../device/cameras/oak.md)

* Planned: [3Dconnexion 3D Space Mouse](https://3dconnexion.com/us/spacemouse/) — HID input over USB, CLI tool. ([tracking](https://github.com/NVIDIA/IsaacTeleop/issues/276))

<a id="data-factory"></a>

<a id="data-factory-3"></a>

## Data Factory 3

Frameworks and services for collecting teleoperation data and turning it into training
datasets.

* [LeRobot](https://huggingface.co/docs/lerobot): Robot learning framework, dataset collection. Set up: [LeRobot integration](../integrations/lerobot/index.md); [LeRobot on GitHub](https://github.com/huggingface/lerobot)
* [Lightwheel](https://lightwheel.ai/): Ego data capture, simulation assets
* [Noitom Robotics](https://www.noitomrobotics.com): Human-centric datasets

<a id="cloud-infrastructure"></a>

<a id="cloud-infrastructure-1"></a>

## Cloud Infrastructure 1

Cloud platforms available for running Isaac Capture workloads.

* [Volcano Engine](https://www.volcengine.com/): Enterprise AI cloud platform

<a id="become-part-of-isaac-capture"></a>

## Become Part of Isaac Capture

Two ways in, both on [GitHub](https://github.com/NVIDIA/IsaacCapture): bring a device to the
stack, or build on the stack itself.

**Add your device**

Devices connect through a plugin process, so your SDK stays in your own repository under
your own license. Build the plugin, open a pull request, and the device joins the tables
above.

[Add a New Device](../device/add_device.md)

**Build on Isaac Teleop**

Collect teleoperation data with it, or run it as the teleoperation stack inside the robot
you are building. One interface covers every device listed above.

[Quick Start](../teleop/quick_start.md)

Looking to go further together? Tell us what you are building. We review submissions on a
rolling basis and will be in touch if we see an opportunity to work more closely together.

[Get in Touch](https://forms.gle/Fo5nRUHZivGN1itg9)
