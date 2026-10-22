# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(plugin_utils)

# The SpaceMouse and gamepad plugins (and their core libraries) are Linux only.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(spacemouse)
    add_subdirectory(gamepad)
endif()

if(BUILD_PLUGIN_OGLO)
    add_subdirectory(oglo_tactile)
endif()

if(BUILD_PLUGIN_EGO_CAMERA OR BUILD_EGO_MEDIA_TOOLS)
    add_subdirectory(ego)
endif()
