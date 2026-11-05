# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_avs1_wire
    ${CAMERA_STREAMER_TEST_ROOT}/cpp/test_avs1_wire.cpp
    ${ARGUS_SENDER_ROOT}/src/Avs1Wire.cpp
)
argus_sender_add_cpp_test(argus_sender_test_avs1_wire avs1_wire)
