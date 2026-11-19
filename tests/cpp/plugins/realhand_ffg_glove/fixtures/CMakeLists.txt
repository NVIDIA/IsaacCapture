# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_skeleton_fixtures
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_skeleton.cpp
)
target_link_libraries(realhand_ffg_skeleton_fixtures PRIVATE realhand_ffg_test_support)
