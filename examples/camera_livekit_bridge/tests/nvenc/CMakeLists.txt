# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Needs a CUDA device and libnvidia-encode; the test exits 77 (skipped) without them.
add_executable(test_nvenc test_nvenc_encoder.cpp)
target_link_libraries(test_nvenc PRIVATE camera_livekit_bridge::nvenc)
add_test(NAME nvenc COMMAND test_nvenc)
set_tests_properties(nvenc PROPERTIES SKIP_RETURN_CODE 77)
