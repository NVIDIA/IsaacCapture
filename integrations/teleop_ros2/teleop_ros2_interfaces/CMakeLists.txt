# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.8)
project(teleop_ros2_interfaces)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/NamedPoseArray.msg"
    DEPENDENCIES
        geometry_msgs
        std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
