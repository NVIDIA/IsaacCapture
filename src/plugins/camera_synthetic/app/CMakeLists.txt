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

# Ship the plugin inside the isaaccapture package, so an application in a wheel install can
# start it through the plugin manager. The binary links nothing beyond libc/libstdc++, which
# is why it may be bundled: a vendor's plugin copied from this one leaves this block out,
# because the wheel does not ship its SDK.
if(TARGET python_package)
    set(_bundle_dir "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/plugins/camera_synthetic")
    add_custom_command(TARGET camera_synthetic_plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_bundle_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:camera_synthetic_plugin>" "${CAMERA_SYNTHETIC_PLUGIN_MANIFEST}" "${_bundle_dir}/"
        COMMENT "Bundling camera_synthetic_plugin into the isaaccapture package")
    # Files staged into the package tree must exist before the wheel is built from it.
    add_dependencies(python_package camera_synthetic_plugin)
endif()
