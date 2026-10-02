.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Ecosystem
=========

The Isaac Teleop ecosystem brings together the input devices, data services, and cloud
platforms that work seamlessly with the unified teleoperation stack. Browse every entry and
follow the link straight to its landing page, integration guide, or acquisition channel.

- **Devices** — Headsets, controllers, gloves, pedals, and master manipulators.
- **Data workflows** — Tools and services for recording, processing, and managing
  teleoperation data.
- **Teams building on Isaac Teleop** — Robot makers shipping teleoperation in their own
  product, and labs collecting data.

.. Explicit targets on the counted sections below: the count pill is part of the heading text,
   so without them the slug carries the count — #supported-devices picks up whatever that
   count currently is, and changes every time a row is added or removed.

.. _supported-devices:

.. rst-class:: eco-section

Supported Devices :device-count:`input`
---------------------------------------

A standardized device interface removes custom integrations and their maintenance. Input
modes determine which retargeters and control schemes are available.
:doc:`Add a new device → </device/add_device>`

.. device-matrix::
   :section: input

.. _data-factory:

.. rst-class:: eco-section

Data Factory :device-count:`data_factory`
-----------------------------------------

Frameworks and services for collecting teleoperation data and turning it into training
datasets.

.. device-matrix::
   :section: data_factory

.. _cloud-infrastructure:

.. rst-class:: eco-section

Cloud Infrastructure :device-count:`cloud`
------------------------------------------

Cloud platforms available for running Isaac Teleop workloads.

.. device-matrix::
   :section: cloud

.. rst-class:: eco-section

Become Part of Isaac Teleop
---------------------------

Two ways in, both on `GitHub <https://github.com/NVIDIA/IsaacTeleop>`_: bring a device to the
stack, or build on the stack itself.

.. eco-block:: eco-paths

   .. eco-block:: eco-path

      **Add your device**

      Devices connect through a plugin process, so your SDK stays in your own repository under
      your own license. Build the plugin, open a pull request, and the device joins the tables
      above.

      .. eco-block:: eco-path-action

         :doc:`Add a New Device </device/add_device>`

   .. eco-block:: eco-path

      **Build on Isaac Teleop**

      Collect teleoperation data with it, or run it as the teleoperation stack inside the robot
      you are building. One interface covers every device listed above.

      .. eco-block:: eco-path-action

         :doc:`Quick Start </getting_started/quick_start>`

.. eco-block:: eco-cta

   Looking to go further together? Tell us what you are building. We review submissions on a
   rolling basis and will be in touch if we see an opportunity to work more closely together.

   .. eco-block:: eco-cta-actions

      `Get in Touch <https://forms.gle/Fo5nRUHZivGN1itg9>`_
