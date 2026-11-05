# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SIPL capture, colour conversion, the hardware encoder and the LiveKit track output.
add_library(sensing_core STATIC
    sensing_camera.cpp
    sipl_camera.cpp
    sipl_capture.cpp
    camera_backend.cpp
    jetson_encoder.cpp
    livekit_video_output.cpp
    yuv_to_rgba.cu
    rgba_to_nv12.cu
    rgba_box_downscale.cu
    ${JETSON_MMAPI_SOURCES}
)
add_library(sensing::core ALIAS sensing_core)

# RESOLVE_DEVICE_SYMBOLS as well as SEPARABLE_COMPILATION: without it the device-link step only happens
# for consumers in a directory where CUDA is an enabled language. Resolving here makes the archive
# self-contained.
set_target_properties(sensing_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(sensing_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    PRIVATE
        # sipl_camera.hpp pimpls the SIPL and NvSci types away, so these stay private: they drag in
        # NvMedia and X11-style macros.
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
        sensing::livekit
    PRIVATE
        ${SIPL_LIBRARIES}
        ${EGL_LIBRARIES}
        ${NVBUFSURFACE_LIBRARY}
        ${V4L2_LIBRARY}
)
