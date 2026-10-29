# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_bundle_assembler
    ${CAMERA_STREAMER_TEST_ROOT}/cpp/test_bundle_assembler.cpp
    ${ARGUS_SENDER_ROOT}/src/FrameBundleAssembler.cpp
    ${ARGUS_SENDER_ROOT}/src/Avs1Wire.cpp
    ${ARGUS_SENDER_ROOT}/src/StereoTiming.cpp
    ${ARGUS_SENDER_ROOT}/src/Runtime.cpp
)
argus_sender_add_cpp_test(argus_sender_test_bundle_assembler bundle_assembler)
target_compile_definitions(argus_sender_test_bundle_assembler PRIVATE ARGUS_SENDER_TEST_SUBMISSION_HOOK)
