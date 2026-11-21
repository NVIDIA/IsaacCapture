# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(spacemouse_plugin
    main.cpp
    spacemouse_plugin.cpp
    spacemouse_plugin.hpp
)

target_link_libraries(spacemouse_plugin PRIVATE
    spacemouse::spacemouse_core
    Teleop::plugin_utils
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    log_bridge::log_bridge_core
)

install(TARGETS spacemouse_plugin RUNTIME DESTINATION plugins/spacemouse)

# Ship the plugin inside the isaaccapture package, so a wheel install can start it (PluginConfig with
# search path BUNDLED_PLUGINS_DIR). The binary links nothing beyond libc/libstdc++.
if(TARGET python_package)
    set(_bundle_dir "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/plugins/spacemouse")
    add_custom_command(TARGET spacemouse_plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_bundle_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:spacemouse_plugin>" ${SPACEMOUSE_PLUGIN_BUNDLE_FILES} "${_bundle_dir}/"
        COMMENT "Bundling spacemouse_plugin into the isaaccapture package")
    # Files staged into the package tree must exist before the wheel is built from it.
    add_dependencies(python_package spacemouse_plugin)
endif()
