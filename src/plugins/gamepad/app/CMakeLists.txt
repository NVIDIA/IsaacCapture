# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(gamepad_plugin
    main.cpp
    gamepad_plugin.cpp
    gamepad_plugin.hpp
)

target_link_libraries(gamepad_plugin PRIVATE
    gamepad::gamepad_core
    Teleop::plugin_utils
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    log_bridge::log_bridge_core
)

install(TARGETS gamepad_plugin RUNTIME DESTINATION plugins/gamepad)

# Ship the plugin inside the isaaccapture package, so a wheel install can start it (PluginConfig with
# search path BUNDLED_PLUGINS_DIR). The binary links nothing beyond libc/libstdc++.
if(TARGET python_package)
    set(_bundle_dir "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/plugins/gamepad")
    add_custom_command(TARGET gamepad_plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_bundle_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:gamepad_plugin>" "${GAMEPAD_PLUGIN_MANIFEST}" "${_bundle_dir}/"
        COMMENT "Bundling gamepad_plugin into the isaaccapture package")
    # Files staged into the package tree must exist before the wheel is built from it.
    add_dependencies(python_package gamepad_plugin)
endif()
