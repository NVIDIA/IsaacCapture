# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# IsaacTeleop tests (pytest + Catch2/CTest)
# ==============================================================================
#
# Production libraries live under src/ and examples/; pytest and Catch2
# suites live here under tests/python/ and tests/cpp/, and configure-time
# tests of CMake helpers under tests/cmake/.

if(BUILD_TESTING)
    add_subdirectory(cpp)
    add_subdirectory(python)
    if(BUILD_PLUGINS AND BUILD_PLUGIN_POSE_PROVIDER)
        add_subdirectory(cmake/pose_provider)
    endif()
endif()
