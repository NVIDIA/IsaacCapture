# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender_test_priority_frame_handoff
    ${ARGUS_SENDER_ROOT}/tests/test_priority_frame_handoff.cpp
)
argus_sender_add_cpp_test(argus_sender_test_priority_frame_handoff priority_frame_handoff)
