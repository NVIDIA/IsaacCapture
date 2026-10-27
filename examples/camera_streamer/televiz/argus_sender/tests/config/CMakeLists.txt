# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_config
    ${ARGUS_SENDER_ROOT}/tests/test_config.cpp
    ${ARGUS_SENDER_ROOT}/src/Config.cpp
)
argus_sender_add_cpp_test(argus_sender_test_config config)
target_link_libraries(argus_sender_test_config PRIVATE yaml-cpp::yaml-cpp)
