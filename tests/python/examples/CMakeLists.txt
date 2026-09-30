# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(BUILD_VIZ)
    add_subdirectory(camera_viz)
endif()

if(BUILD_PYTHON_BINDINGS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(vehicle_teleop)
endif()
