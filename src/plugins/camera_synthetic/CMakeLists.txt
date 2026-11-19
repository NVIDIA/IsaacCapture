# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The synthetic camera plugin: cameras that need no hardware, served to readers like any
# vendor's. It needs no SDK, so it has no build option of its own and is built with the plugins.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping camera_synthetic plugin (Linux only)")
    return()
endif()

# app/ copies the manifest next to the executable, which makes that directory one the plugin
# manager can start the plugin from.
set(CAMERA_SYNTHETIC_PLUGIN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml")

add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(tools)

install(FILES plugin.yaml README.md DESTINATION plugins/camera_synthetic)
