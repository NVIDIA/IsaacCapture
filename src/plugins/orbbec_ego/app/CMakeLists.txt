# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(orbbec_ego_camera_plugin
    main.cpp
)

target_link_libraries(orbbec_ego_camera_plugin
    PRIVATE
        ego::camera_core
        log_bridge::log_bridge_core
)

target_compile_definitions(orbbec_ego_camera_plugin
    PRIVATE
        ORBBEC_EGO_ENABLE_PREVIEW=$<BOOL:${ORBBEC_EGO_ENABLE_PREVIEW}>
)

set_target_properties(orbbec_ego_camera_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EGO_PLUGIN_OUTPUT_DIR}"
    BUILD_RPATH "${ORBBEC_SDK_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN"
)

configure_file("${EGO_PLUGIN_MANIFEST}" "${EGO_PLUGIN_OUTPUT_DIR}/plugin.yaml" COPYONLY)
install(TARGETS orbbec_ego_camera_plugin RUNTIME DESTINATION plugins/orbbec_ego COMPONENT orbbec_ego)
