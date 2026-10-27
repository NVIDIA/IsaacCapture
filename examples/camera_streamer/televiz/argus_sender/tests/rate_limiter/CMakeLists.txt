# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_rate_limiter
    ${ARGUS_SENDER_ROOT}/tests/test_rate_limiter.cpp
    ${ARGUS_SENDER_ROOT}/src/StereoTiming.cpp
    ${ARGUS_SENDER_ROOT}/src/MixedRateScheduler.cpp
)
argus_sender_add_cpp_test(argus_sender_test_rate_limiter rate_limiter)
