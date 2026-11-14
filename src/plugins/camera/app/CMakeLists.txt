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

# Ship the plugin inside the isaaccapture package, so a reader in a wheel install can start it.
# The binary links nothing beyond libc/libstdc++.
if(TARGET python_package)
    set(_bundle_dir "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/plugins/camera")
    add_custom_command(TARGET camera_plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_bundle_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:camera_plugin>" "${CAMERA_PLUGIN_MANIFEST}" "${_bundle_dir}/"
        COMMENT "Bundling camera_plugin into the isaaccapture package")
    # Files staged into the package tree must exist before the wheel is built from it.
    add_dependencies(python_package camera_plugin)
endif()
