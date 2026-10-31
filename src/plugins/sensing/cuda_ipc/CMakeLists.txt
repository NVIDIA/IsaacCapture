# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CUDA IPC publisher and its wire format. Deliberately depends on CUDA alone so
# tools/ can build a producer without SIPL or the V4L2 encoder.
add_library(sensing_cuda_ipc STATIC
    cuda_ipc_publisher.cpp
)
add_library(sensing::cuda_ipc ALIAS sensing_cuda_ipc)

target_include_directories(sensing_cuda_ipc PUBLIC inc)

target_link_libraries(sensing_cuda_ipc
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)
