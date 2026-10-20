# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# IsaacTeleop tests (pytest + Catch2/CTest)
# ==============================================================================
#
# Production libraries live under src/ and examples/; pytest and Catch2
# suites live here under tests/python/ and tests/cpp/.

if(BUILD_TESTING)
    add_subdirectory(cpp)
    add_subdirectory(python)
endif()
