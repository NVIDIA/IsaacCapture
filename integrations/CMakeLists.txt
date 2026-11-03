# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Integrations with third-party hardware, robots and frameworks
# ==============================================================================
# First-party code, so the project warning set applies to the whole tree. Enabled
# here rather than at the root for the reasons in src/CMakeLists.txt.
isaac_teleop_enable_compiler_warnings()

if(BUILD_EXAMPLES)
    add_subdirectory(teleop_ros2)
    add_subdirectory(sharpa_avatar)
    add_subdirectory(vehicle_teleop)
elseif(BUILD_EXAMPLE_TELEOP_ROS2)
    # Only the ROS 2 reference integration (e.g. for Docker).
    add_subdirectory(teleop_ros2)
endif()
