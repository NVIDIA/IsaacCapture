# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_tcp_transport
    ${CAMERA_STREAMER_TEST_ROOT}/cpp/test_tcp_transport.cpp
    ${ARGUS_SENDER_ROOT}/src/TcpTransport.cpp
    ${ARGUS_SENDER_ROOT}/src/StereoTiming.cpp
    ${ARGUS_SENDER_ROOT}/src/Runtime.cpp
)
argus_sender_add_cpp_test(argus_sender_test_tcp_transport tcp_transport)
set_tests_properties(tcp_transport PROPERTIES LABELS "loopback")
