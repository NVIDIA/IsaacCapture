# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_geometry_scheduler
    ${ARGUS_SENDER_ROOT}/tests/test_geometry_scheduler.cpp
    ${ARGUS_SENDER_ROOT}/src/OutputGeometry.cpp
    ${ARGUS_SENDER_ROOT}/src/MixedRateScheduler.cpp
)
argus_sender_add_cpp_test(argus_sender_test_geometry_scheduler geometry_scheduler)
