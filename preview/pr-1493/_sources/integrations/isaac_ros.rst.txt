.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Isaac ROS
=========

Isaac ROS Teleop consumes ``teleop_ros2``, the reference ROS 2 publisher for Isaac Teleop
data. It runs in a Docker image and publishes hand, end-effector, root, head, controller,
finger-joint, and TF data on ``xr_teleop/*`` topics.

Build the image from the repository root:

.. code-block:: bash

   docker build -f examples/teleop_ros2/Dockerfile -t teleop_ros2_ref .

:code-file:`examples/teleop_ros2/README.md` covers running the container (including the
Jetson ``docker run`` flag), hand-tracking providers and retargeters, parameters, published
topics, and MCAP replay.
