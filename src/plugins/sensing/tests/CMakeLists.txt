# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Backend selection and the frame lease are pure C++; compile camera_backend.cpp directly so the test links
# neither sensing_core (CUDA, SIPL, the Multimedia API) nor any vendor SDK.
add_executable(sensing_backend_tests
    test_camera_backend.cpp
    ../core/camera_backend.cpp
)

# Reaches ../core headers as "core/..." without adding the plugin's own core/ include path.
target_include_directories(sensing_backend_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(sensing_backend_tests PRIVATE Catch2::Catch2WithMain)

catch_discover_tests(sensing_backend_tests)

# The clock-sync wire format is header-only and LiveKit-free, so the test needs only its include path.
add_executable(sensing_clock_sync_tests
    test_clock_sync.cpp
)

target_include_directories(sensing_clock_sync_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../livekit/inc)

target_link_libraries(sensing_clock_sync_tests PRIVATE Catch2::Catch2WithMain)

catch_discover_tests(sensing_clock_sync_tests)
