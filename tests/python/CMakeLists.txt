# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(core)

if(BUILD_PYTHON_BINDINGS AND NOT SKBUILD)
    add_subdirectory(packaging)
endif()

if(BUILD_PYTHON_BINDINGS)
    # The only suite that puts the compat stage dir on PYTHONPATH. The others
    # must stay unable to resolve `isaacteleop`: an in-tree import of the old
    # name is a bug, and reaching one would warn us against ourselves.
    add_subdirectory(compat)
endif()

if(BUILD_EXAMPLE_TELEOP_ROS2)
    add_subdirectory(examples/teleop_ros2)
endif()

if(BUILD_VIZ AND BUILD_PYTHON_BINDINGS)
    add_subdirectory(viz)
endif()

if(BUILD_EXAMPLES AND BUILD_VIZ)
    add_subdirectory(examples)
endif()
