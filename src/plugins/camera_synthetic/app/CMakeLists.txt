# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(camera_synthetic_plugin
    main.cpp
)

target_link_libraries(camera_synthetic_plugin PRIVATE
    camera_synthetic::camera_synthetic_core
    camera::camera
)

add_custom_command(TARGET camera_synthetic_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CAMERA_SYNTHETIC_PLUGIN_MANIFEST}" "$<TARGET_FILE_DIR:camera_synthetic_plugin>/plugin.yaml"
    COMMENT "Putting the camera_synthetic plugin's manifest next to its executable")

install(TARGETS camera_synthetic_plugin RUNTIME DESTINATION plugins/camera_synthetic)
