# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The reviewed hand geometry against the SDK and the manifest. The hand kinematics library is only
# built when the live source links against the C SDK, so a build without the SDK library skips this suite.
if(TARGET revohuman_plugin::revohuman_hand_kinematics)
    add_executable(revohuman_integration_tests test_integration.cpp)
    target_link_libraries(revohuman_integration_tests
        PRIVATE
            Catch2::Catch2WithMain
            revohuman_plugin::revohuman_live
            revohuman_plugin::revohuman_hand_kinematics
    )
    target_compile_definitions(revohuman_integration_tests PRIVATE
        REVOHUMAN_MANIFEST="${CMAKE_SOURCE_DIR}/src/plugins/revohuman/plugin.yaml"
        REVOHUMAN_TEST_SDK_POINTS=$<BOOL:${REVOHUMAN_HAS_SDK_POINTS}>
        REVOHUMAN_TEST_SDK_VARIANT_MODEL=$<BOOL:${REVOHUMAN_HAS_SDK_VARIANT_MODEL}>
        REVOHUMAN_TEST_SDK_TAXEL_LAYOUT=$<BOOL:${REVOHUMAN_HAS_SDK_TAXEL_LAYOUT}>)
    # Reads plugin.yaml from the source tree at run time, so it is not installed.
    set_target_properties(revohuman_integration_tests PROPERTIES ISAACTELEOP_NEEDS_SOURCE_TREE TRUE)
    # The point set and taxel checks call the SDK directly; its imported target is scoped to the plugin directory.
    if(REVOHUMAN_HAS_SDK_POINTS OR REVOHUMAN_HAS_SDK_TAXEL_LAYOUT)
        target_link_libraries(revohuman_integration_tests PRIVATE "${REVOHUMAN_SDK_LIBRARY}")
    endif()
    catch_discover_tests(revohuman_integration_tests)
endif()

# The hand capture labels a take with REVOHUMAN_WRIST_SOURCE, so the plugin refuses a value it does not know and a
# --wrist-source that disagrees with it. Both are refused before the plugin opens anything. Without the
# SDK library revohuman_plugin is only a placeholder target, so there is nothing to run.
if(REVOHUMAN_LIVE_AVAILABLE)
    add_test(NAME revohuman_wrist_source_variable_unknown COMMAND revohuman_plugin)
    set_tests_properties(revohuman_wrist_source_variable_unknown PROPERTIES
        ENVIRONMENT "REVOHUMAN_WRIST_SOURCE=hand-tracking"
        PASS_REGULAR_EXPRESSION "REVOHUMAN_WRIST_SOURCE must be auto, hand_tracking or controller, got 'hand-tracking'")
    add_test(NAME revohuman_wrist_source_flag_disagrees
        COMMAND revohuman_plugin --wrist-source=controller)
    set_tests_properties(revohuman_wrist_source_flag_disagrees PROPERTIES
        ENVIRONMENT "REVOHUMAN_WRIST_SOURCE=hand_tracking"
        PASS_REGULAR_EXPRESSION "--wrist-source=controller disagrees with REVOHUMAN_WRIST_SOURCE=hand_tracking")
endif()
