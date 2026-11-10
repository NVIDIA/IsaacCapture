<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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

## Supported Devices 14

A standardized device interface removes custom integrations and their maintenance. Input
modes determine which retargeters and control schemes are available.
[Add a new device →](../device/add_device.md)

<!-- The device tables are interactive HTML layout; the Markdown build for agents leaves them out. -->

<a id="data-factory"></a>

## Data Factory 3

Frameworks and services for collecting teleoperation data and turning it into training
datasets.

<a id="cloud-infrastructure"></a>

## Cloud Infrastructure 1

Cloud platforms available for running Isaac Capture workloads.

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
