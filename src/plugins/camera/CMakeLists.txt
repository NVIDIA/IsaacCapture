# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The camera plugin: one process that owns the cameras and serves their frames to readers.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping camera plugin (Linux only)")
    return()
endif()

add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(app)
add_subdirectory(bench)

install(FILES plugin.yaml README.md DESTINATION plugins/camera)
