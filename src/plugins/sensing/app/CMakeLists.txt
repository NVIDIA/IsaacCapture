# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(camera_plugin_sensing main.cpp)

set_target_properties(camera_plugin_sensing PROPERTIES INSTALL_RPATH "$ORIGIN")

target_link_libraries(camera_plugin_sensing PRIVATE sensing::core sensing::control)

# The plugin resolves configs/shw5g.json relative to its own executable, so the
# config has to sit beside the binary in the build tree as well as the install
# tree -- otherwise it only works after `cmake --install`.
add_custom_command(TARGET camera_plugin_sensing POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SENSING_CONFIGS_DIR}"
            "$<TARGET_FILE_DIR:camera_plugin_sensing>/configs"
    COMMENT "Staging SIPL platform configs beside camera_plugin_sensing"
)

# Everything below is the `producer` install component: `cmake --install <build> --component producer` stages
# exactly what scripts/deploy_orin.py packages for an Orin.
install(TARGETS camera_plugin_sensing
    RUNTIME DESTINATION plugins/sensing_camera
    COMPONENT producer
)
install(DIRECTORY "${SENSING_CONFIGS_DIR}/"
    DESTINATION plugins/sensing_camera/configs
    COMPONENT producer
)

# The vendored encoder classes are compiled into the binary, so their license text ships with it.
install(FILES "${SENSING_MMAPI_VENDORED_DIR}/LICENSE"
    DESTINATION plugins/sensing_camera/licenses/jetson-multimedia-api
    COMPONENT producer
)
