# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The live source against a fake of the C SDK library, so it is tested without gloves.
add_executable(revohuman_live_tests
    fake_sdk/fake_revohuman.cpp
    fake_sdk/fake_revohuman.hpp
    test_live_source.cpp
    test_rh_frame.cpp
)

target_link_libraries(revohuman_live_tests
    PRIVATE
        Catch2::Catch2WithMain
        revohuman_plugin::revohuman_live
)

catch_discover_tests(revohuman_live_tests)
