# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The camera plugin: one process that owns the cameras and serves their frames to readers.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping camera plugin (Linux only)")
    return()
endif()

# app/ copies the manifest next to the executable, which makes that directory one the plugin
# manager can start the plugin from.
set(CAMERA_PLUGIN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml")

add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(app)
add_subdirectory(bench)

install(FILES plugin.yaml README.md DESTINATION plugins/camera)
