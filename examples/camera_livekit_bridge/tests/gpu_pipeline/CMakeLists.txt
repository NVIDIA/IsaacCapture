# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Needs a CUDA device; the test exits 77 (skipped) without one.
add_executable(test_gpu_pipeline test_gpu_pipeline.cpp)
target_link_libraries(test_gpu_pipeline PRIVATE camera_livekit_bridge::gpu_ops CUDA::cudart)
add_test(NAME gpu_pipeline COMMAND test_gpu_pipeline)
set_tests_properties(gpu_pipeline PROPERTIES SKIP_RETURN_CODE 77)
