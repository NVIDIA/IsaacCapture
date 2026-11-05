# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(camera_plugin_sensing main.cpp)

set_target_properties(camera_plugin_sensing PROPERTIES INSTALL_RPATH "$ORIGIN")

target_link_libraries(camera_plugin_sensing PRIVATE sensing::core)

# The plugin resolves configs/shw5g.json relative to its own executable, so the
# config has to sit beside the binary in the build tree as well as the install
# tree -- otherwise it only works after `cmake --install`.
add_custom_command(TARGET camera_plugin_sensing POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/../configs"
            "$<TARGET_FILE_DIR:camera_plugin_sensing>/configs"
    COMMENT "Staging SIPL platform configs beside camera_plugin_sensing"
)

install(TARGETS camera_plugin_sensing
    RUNTIME DESTINATION plugins/sensing_camera
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../configs/"
    DESTINATION plugins/sensing_camera/configs
)
