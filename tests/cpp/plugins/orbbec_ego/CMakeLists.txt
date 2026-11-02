# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(camera)

add_test(
    NAME ego_script_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_script.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/orbbec_ego/orbbec_ego.sh"
)

add_test(
    NAME ego_install_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_install.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/orbbec_ego/install.sh"
)
