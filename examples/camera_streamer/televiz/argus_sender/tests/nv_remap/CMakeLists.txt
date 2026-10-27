# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_nv_remap
    ${ARGUS_SENDER_ROOT}/tests/test_nv_remap.cpp
    ${ARGUS_SENDER_ROOT}/src/NvRemap.cpp
)
argus_sender_add_cpp_test(argus_sender_test_nv_remap nv_remap)
