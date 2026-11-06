# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_capture_domain_plan
    ${CAMERA_STREAMER_TEST_ROOT}/cpp/test_capture_domain_plan.cpp
    ${ARGUS_SENDER_ROOT}/src/CaptureDomainPlan.cpp
)
argus_sender_add_cpp_test(argus_sender_test_capture_domain_plan capture_domain_plan)
