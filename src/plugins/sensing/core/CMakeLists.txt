# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Capture, encoder and output sinks. The JETSON backend needs the Jetson SDKs,
# so everything that does lives in this directory and tools/ stays buildable
# without them. The MOCK backend replaces SiplCamera and JetsonEncoder with
# synthetic ones that need only CUDA.
set(SENSING_COMMON_SOURCES
    sensing_camera.cpp
    frame_sink.cpp
    rawdata_writer.cpp
    yuv_to_rgba.cu
    rgba_to_nv12.cu
)

if(SENSING_BACKEND_RESOLVED STREQUAL "JETSON")
    set(SENSING_BACKEND_SOURCES
        sipl_camera.cpp
        jetson_encoder.cpp
        ${JETSON_MMAPI_SOURCES}
    )
else()
    # Parses the vendor JSON itself, so a MOCK build has no SIPL query library.
    find_package(nlohmann_json 3 QUIET)
    if(NOT nlohmann_json_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.3
            GIT_SHALLOW TRUE
        )
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        isaac_teleop_third_party_scope_begin()
        FetchContent_MakeAvailable(nlohmann_json)
        isaac_teleop_third_party_scope_end()
    endif()
    set(SENSING_BACKEND_SOURCES
        mock_sipl_camera.cpp
        mock_platform_config.cpp
        mock_nv12_source.cu
        mock_jetson_encoder.cpp
    )
endif()

add_library(sensing_core STATIC
    ${SENSING_COMMON_SOURCES}
    ${SENSING_BACKEND_SOURCES}
)
add_library(sensing::core ALIAS sensing_core)

# RESOLVE_DEVICE_SYMBOLS as well as SEPARABLE_COMPILATION: without it the
# device-link step only happens for consumers in a directory where CUDA is an
# enabled language, so linking sensing_core from tests/ fails on
# __cudaRegisterLinkedBinary_*. Resolving here makes the archive self-contained.
set_target_properties(sensing_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(sensing_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(sensing_core
    PUBLIC
        # sensing_types.hpp is public and includes the generated schema header.
        sensing::cuda_ipc
        isaacteleop_schema
    PRIVATE
        mcap::mcap
        oxr::oxr_utils
)

if(SENSING_BACKEND_RESOLVED STREQUAL "JETSON")
    target_include_directories(sensing_core
        PRIVATE
            # sipl_camera.hpp pimpls the SIPL and NvSci types away, so these stay
            # private: they drag in NvMedia and X11-style macros that collide with
            # mcap's StatusCode, which is what sensing_types.hpp exists to avoid.
            ${SIPL_INCLUDE_DIRS}
            ${EGL_INCLUDE_DIRS}
            ${JETSON_MMAPI_DIR}/include
    )

    # The Multimedia API reference sources carry legacy warnings.
    target_compile_options(sensing_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder -Wno-uninitialized -Wno-unused-variable
                                  -Wno-unused-parameter -Wno-missing-field-initializers -Wno-sign-compare>
    )

    target_link_libraries(sensing_core
        PRIVATE
            ${SIPL_LIBRARIES}
            ${EGL_LIBRARIES}
            ${NVBUFSURFACE_LIBRARY}
            ${V4L2_LIBRARY}
    )
else()
    target_link_libraries(sensing_core PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(sensing_core PUBLIC SENSING_MOCK_BACKEND=1)
endif()
