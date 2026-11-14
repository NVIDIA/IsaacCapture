# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(plugin_utils)

if(TARGET avatar_plugin_core)
    add_subdirectory(sharpa_avatar)
endif()

# The SpaceMouse and gamepad plugins (and their core libraries) are Linux only.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(spacemouse)
    add_subdirectory(gamepad)
endif()

if(BUILD_PLUGIN_OGLO)
    add_subdirectory(oglo_tactile)
endif()

# Defined on Linux whether or not the Manus SDK is present (see src/plugins/manus_calibration/CMakeLists.txt).
if(TARGET manus_calibration_core)
    add_subdirectory(manus_calibration)
endif()

# Skipped with the plugin: off Linux, or on an arch with no RevoHuman C SDK release.
if(BUILD_PLUGIN_REVOHUMAN AND TARGET revohuman_plugin::revohuman_core)
    add_subdirectory(revohuman)
endif()
