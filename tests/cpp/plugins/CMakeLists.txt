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
