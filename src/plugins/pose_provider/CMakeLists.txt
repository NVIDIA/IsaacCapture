# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source only: under pip, the root install walk would put the plugin and its tests into the wheel.
if(SKBUILD)
    message(FATAL_ERROR "The Pose Provider plugin cannot be part of a pip build. Build it with CMake directly.")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/PoseEngine.cmake)

add_subdirectory(engine_api)
add_subdirectory(engines/stub)

# Sets POSE_ENGINE_TARGET to the adapter that executables link, the one enabled adapter or the stub, and
# POSE_ENGINE_TARGET_DISTRIBUTION to its distribution. Add install rules only for a redistributable adapter.
pose_engine_select(POSE_ENGINE_TARGET pose_provider::pose_engine_stub)

add_subdirectory(core)

# Build-tree plugin directory: plugin.yaml beside pose_provider_plugin, so <build>/src/plugins is a search path.
configure_file(plugin.yaml ${CMAKE_CURRENT_BINARY_DIR}/plugin.yaml COPYONLY)
set(POSE_PROVIDER_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(app)
