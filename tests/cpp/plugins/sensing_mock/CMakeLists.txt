# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercises the plugin on the MOCK backend: synthetic frames and a stand-in
# encoder, but the real SensingCamera, sinks, CUDA kernels and CUDA IPC. Needs an
# NVIDIA GPU, not a Jetson.
find_package(CUDAToolkit REQUIRED)

add_executable(sensing_mock_tests
    gif_fixture.hpp
    mock_test_support.hpp
    test_mock_camera.cpp
    test_mock_encoder.cpp
    test_mock_gif.cpp
    test_mock_pipeline.cpp
    test_cuda_ipc_consumer.cpp
)

target_link_libraries(sensing_mock_tests
    PRIVATE
        Catch2::Catch2WithMain
        sensing::core
        CUDA::cudart
        CUDA::cuda_driver
)

target_compile_definitions(sensing_mock_tests PRIVATE
    SENSING_PLATFORM_CONFIG="${CMAKE_SOURCE_DIR}/src/plugins/sensing/configs/shw5g.json"
    SENSING_MOCK_RIG_CONFIG="${CMAKE_SOURCE_DIR}/src/plugins/sensing/mock_assets/mock_rig.json"
    SENSING_BUNDLED_GIF_DIR="${CMAKE_SOURCE_DIR}/src/plugins/sensing/mock_assets"
)

catch_discover_tests(sensing_mock_tests ADD_TAGS_AS_LABELS)
