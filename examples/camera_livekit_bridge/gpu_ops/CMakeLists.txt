# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPU operations on NV12 pictures, run between the IPC slot and the encoder input.
add_library(gpu_ops STATIC box_downscale.cu)

target_include_directories(gpu_ops
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(gpu_ops
    PUBLIC
        CUDA::cudart
)

add_library(camera_livekit_bridge::gpu_ops ALIAS gpu_ops)
