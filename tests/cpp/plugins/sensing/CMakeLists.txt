# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Query-only: needs the vendor JSON on disk but no camera, and links
# sensing_core purely for SiplCamera::query. sipl_camera.hpp is pimpl'd, so this
# target never sees a SIPL or NvSci header.
add_executable(test_sipl_query test_sipl_query.cpp)
target_link_libraries(test_sipl_query PRIVATE sensing::core)
# The in-tree config, so this runs unconditionally -- a test that skips when the
# vendor package is absent is a test that does not exist in CI.
target_compile_definitions(test_sipl_query PRIVATE
    SENSING_PLATFORM_CONFIG="${CMAKE_SOURCE_DIR}/src/plugins/sensing/configs/shw5g.json"
)
add_test(NAME sipl_query COMMAND test_sipl_query)
