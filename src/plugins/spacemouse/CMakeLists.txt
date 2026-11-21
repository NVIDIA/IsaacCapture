# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reads a 3Dconnexion SpaceMouse from /dev/hidraw* and pushes SpaceMouseOutput via OpenXR.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping spacemouse plugin (Linux only)")
    add_custom_target(spacemouse_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping spacemouse: Linux only")
    return()
endif()

# app/ also stages the plugin with these files into the Python package, so a wheel install gets
# its manifest and the udev setup.
set(SPACEMOUSE_PLUGIN_BUNDLE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/70-spacemouse.rules"
    "${CMAKE_CURRENT_SOURCE_DIR}/install_udev_rules.sh"
)

add_subdirectory(core)
add_subdirectory(app)

install(FILES plugin.yaml README.md 70-spacemouse.rules DESTINATION plugins/spacemouse)
install(PROGRAMS install_udev_rules.sh DESTINATION plugins/spacemouse)
