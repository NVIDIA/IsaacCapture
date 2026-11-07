# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# SENSING camera plugin — CUDA IPC transport
# ==============================================================================
# Two leaf targets, one per directory (cmake/cmake-structure.md rule 1):
#   cuda_ipc/  sensing_cuda_ipc     — publisher + wire format, CUDA only
#   tools/     sensing_ipc_testsrc  — synthetic producer for the same protocol
#
# Neither needs a camera, a capture SDK or an encoder: the transport is CUDA and
# a Unix socket. Keep the capture-side SDK checks in the directories that need
# them so this half stays buildable on its own.
#
# Configure with:
#   cmake -B build -DBUILD_PLUGIN_SENSING=ON
# ==============================================================================

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Orin is 87, Thor is 101; the rest cover desktop dev boxes.
    set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89;90")
endif()

find_package(CUDAToolkit REQUIRED)

add_subdirectory(cuda_ipc)
add_subdirectory(tools)
