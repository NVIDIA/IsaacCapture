# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Animated test pattern on the IPC socket, so a consumer can be exercised with
# no camera attached. Links sensing::cuda_ipc only, so the binary needs CUDA at
# runtime and neither SIPL nor the encoder. Configuring this directory still
# goes through the plugin's Jetson SDK checks in ../CMakeLists.txt.
add_executable(sensing_ipc_testsrc ipc_testsrc.cu)

set_target_properties(sensing_ipc_testsrc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INSTALL_RPATH "$ORIGIN"
)

target_link_libraries(sensing_ipc_testsrc PRIVATE sensing::cuda_ipc)
