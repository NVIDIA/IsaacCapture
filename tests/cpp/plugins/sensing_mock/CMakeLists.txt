# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercises the plugin on the MOCK backend: synthetic frames and an NVENC
# encoder, but the real SensingCamera, sinks, CUDA kernels and CUDA IPC. Needs an
# NVIDIA GPU, not a Jetson.
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec libavutil)

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
        PkgConfig::AVCODEC
)

target_compile_definitions(sensing_mock_tests PRIVATE
    SENSING_PLATFORM_CONFIG="${CMAKE_SOURCE_DIR}/src/plugins/sensing/configs/shw5g.json"
    SENSING_MOCK_RIG_CONFIG="${CMAKE_SOURCE_DIR}/src/plugins/sensing/mock_assets/mock_rig.json"
    SENSING_BUNDLED_GIF_DIR="${CMAKE_SOURCE_DIR}/src/plugins/sensing/mock_assets"
)

# LiveKit streaming tests. The unit tests use a fake track; the integration tests also need a
# livekit-server binary, named by LIVEKIT_SERVER, and skip without it.
if(SENSING_LIVEKIT)
    find_package(LiveKit CONFIG REQUIRED PATHS "${SENSING_LIVEKIT_SDK_DIR}" NO_DEFAULT_PATH)
    find_package(OpenSSL REQUIRED)
    target_sources(sensing_mock_tests PRIVATE
        livekit_test_support.hpp
        test_livekit_cli.cpp
        test_livekit_output.cpp
        test_livekit_stream.cpp
    )
    target_link_libraries(sensing_mock_tests
        PRIVATE
            sensing::livekit
            LiveKit::livekit
            OpenSSL::Crypto
    )    # test_livekit_cli.cpp runs the plugin binary, so it has to be built first.
    add_dependencies(sensing_mock_tests camera_plugin_sensing)
    target_compile_definitions(sensing_mock_tests PRIVATE SENSING_PLUGIN_BINARY="$<TARGET_FILE:camera_plugin_sensing>")
endif()

catch_discover_tests(sensing_mock_tests ADD_TAGS_AS_LABELS)
