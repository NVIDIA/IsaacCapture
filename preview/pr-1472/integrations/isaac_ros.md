<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="isaac-ros"></a>

# Isaac ROS

Isaac ROS Teleop consumes `teleop_ros2`, the reference ROS 2 publisher for Isaac Teleop
data. It runs in a Docker image and publishes hand, end-effector, root, head, controller and
finger-joint data on `xr_teleop/*` topics, and TF on `/tf`.

Build the image from the repository root:

```bash
docker build -f examples/teleop_ros2/Dockerfile -t teleop_ros2_ref .
```

[examples/teleop_ros2/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop_ros2/README.md) covers running the container (including the
Jetson `docker run` flag), hand-tracking providers and retargeters, parameters, published
topics, and MCAP replay.
