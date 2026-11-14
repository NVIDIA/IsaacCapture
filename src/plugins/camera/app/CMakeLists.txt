# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_executable(camera_plugin
    main.cpp
)

target_link_libraries(camera_plugin PRIVATE
    camera::plugin_core
    camera::plugin_backends
    camera::camera_ipc
    log_bridge::log_bridge_core
    Threads::Threads
)

add_custom_command(TARGET camera_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CAMERA_PLUGIN_MANIFEST}" "$<TARGET_FILE_DIR:camera_plugin>/plugin.yaml"
    COMMENT "Putting the camera plugin's manifest next to its executable")

install(TARGETS camera_plugin RUNTIME DESTINATION plugins/camera)
