# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${CMAKE_SOURCE_DIR}/cmake/PoseEngine.cmake)

add_subdirectory(engine_api)
add_subdirectory(engines/stub)

# Sets POSE_ENGINE_TARGET to the adapter that executables link, the one enabled adapter or the stub, and
# POSE_ENGINE_TARGET_DISTRIBUTION to its distribution. Add install rules only for a redistributable adapter.
pose_engine_select(POSE_ENGINE_TARGET pose_provider::pose_engine_stub)
