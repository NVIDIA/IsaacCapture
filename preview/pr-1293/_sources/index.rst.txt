.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Welcome to Isaac Capture
========================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: development/license.html
   :alt: License
   :class: license-badge

**NVIDIA Isaac Capture** is a shared library for teleoperation and egocentric data
capture. It brings device plugins, retargeters, and data utilities together so
applications can collect and use human demonstration data across robot and
simulation environments.

**Isaac Teleop** is the robot-in-the-loop component within Isaac Capture. It
provides teleoperation sessions and workflows for controlling robots and
collecting demonstrations. The **Ego Pipeline** covers human and egocentric data
capture and post-processing. Both build on the shared capture infrastructure.

See :doc:`overview/architecture` for the component relationships and
:doc:`overview/ecosystem` for devices and integrations. Source code and Python
packages are distributed through `GitHub <https://github.com/NVIDIA/IsaacCapture>`_
and `PyPI <https://pypi.org/project/isaaccapture/>`_.

.. tip::

   To run Isaac Teleop, follow :doc:`teleop/quick_start` for installation,
   headset setup, and your first session. Existing users can consult
   :doc:`development/migration` for the Python package migration.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :titlesonly:

   overview/architecture
   overview/isaac_teleop
   overview/ecosystem

.. toctree::
   :maxdepth: 2
   :caption: Isaac Teleop

   teleop/quick_start
   teleop/synthetic_xr
   teleop/teleop_session
   teleop/control_state_machine

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/build_from_source/index
   development/televiz
   development/migration
   development/contributing
   development/license

.. toctree::
   :maxdepth: 2
   :caption: Integrations
   :titlesonly:

   integrations/isaac_lab_sim
   integrations/isaac_ros
   integrations/lerobot/index

.. toctree::
   :maxdepth: 2
   :caption: Devices
   :titlesonly:

   device/xr/index
   device/hand_tracking/index
   Body tracking <device/body_tracking/index>
   device/cameras/index
   Joint-space devices <device/joint_space/index>
   device/tactile/index
   device/spatial_controllers/index
   device/gamepads/index
   Spatial trackers <device/trackers>
   device/add_device

.. toctree::
   :maxdepth: 2
   :caption: References

   references/requirements
   references/build
   references/generated_trackers
   references/retargeting/index
   references/device_provider_monitoring
   references/camera_streaming
   references/mcap_record_replay
   references/cloudxr
   references/oob_teleop_control
   references/haptic_feedback
   references/egocentric_hand_reconstruction

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
