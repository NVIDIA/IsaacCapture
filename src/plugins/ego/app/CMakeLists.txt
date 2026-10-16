# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_camera_plugin
    main.cpp
)

target_link_libraries(ego_camera_plugin
    PRIVATE
        ego::camera_core
        log_bridge::log_bridge_core
)

target_compile_definitions(ego_camera_plugin
    PRIVATE
        EGO_ENABLE_PREVIEW=$<BOOL:${EGO_ENABLE_PREVIEW}>
)

set_target_properties(ego_camera_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EGO_PLUGIN_OUTPUT_DIR}"
    BUILD_RPATH "${ORBBEC_SDK_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN"
)

configure_file("${EGO_PLUGIN_MANIFEST}" "${EGO_PLUGIN_OUTPUT_DIR}/plugin.yaml" COPYONLY)
install(TARGETS ego_camera_plugin RUNTIME DESTINATION plugins/ego COMPONENT ego)
