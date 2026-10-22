# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET ego::mcap_merge)
    add_subdirectory(mcap)
endif()

if(TARGET ego_mcap_export_media AND UNIX)
    add_subdirectory(media)
endif()

if(TARGET ego::camera_core)
    add_subdirectory(camera)
endif()

add_test(
    NAME ego_script_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_script.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/ego/ego.sh"
)

add_test(
    NAME ego_install_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_install.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/ego/install.sh"
)
