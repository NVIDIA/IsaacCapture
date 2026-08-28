# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(teleop_ros2_ref)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)

add_subdirectory(cpp/integration_tests)

install_python_example(DESTINATION examples/teleop_ros2)

install(FILES README.md Dockerfile
    DESTINATION examples/teleop_ros2
)

install(DIRECTORY configs assets
    DESTINATION examples/teleop_ros2
)

install(DIRECTORY teleop_ros2_interfaces
    DESTINATION examples/teleop_ros2
)
