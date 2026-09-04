# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SIPL capture, the V4L2 M2M encoder wrapper, and the output sinks. Everything
# that needs the Jetson SDKs lives here, so tools/ stays buildable without them.
add_library(sensing_core STATIC
    sensing_camera.cpp
    frame_sink.cpp
    rawdata_writer.cpp
    sipl_camera.cpp
    yuv_to_rgba.cu
    jetson_encoder.cpp
    rgba_to_nv12.cu
    ${JETSON_MMAPI_SOURCES}
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
    PUBLIC
        # sensing_types.hpp is public and includes the generated schema header.
        sensing::cuda_ipc
        isaacteleop_schema
    PRIVATE
        ${SIPL_LIBRARIES}
        ${EGL_LIBRARIES}
        ${NVBUFSURFACE_LIBRARY}
        ${V4L2_LIBRARY}
        mcap::mcap
        oxr::oxr_utils
)
