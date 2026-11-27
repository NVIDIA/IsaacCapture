# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each test configures the fixture in its own build directory. A "pass" test must configure without a CMake error
# and print <expected>; a "fail" test must stop with a CMake error followed by <expected>.
set(fixture ${CMAKE_CURRENT_SOURCE_DIR}/fixture)

function(pose_engine_test name outcome expected)
    set(build ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(NAME pose_engine_${name}
        COMMAND ${CMAKE_COMMAND} --fresh -S ${fixture} -B ${build}
                -DHELPER=${CMAKE_SOURCE_DIR}/cmake/PoseEngine.cmake ${ARGN}
    )
    if(outcome STREQUAL "pass")
        set_tests_properties(pose_engine_${name} PROPERTIES
            PASS_REGULAR_EXPRESSION "${expected}"
            FAIL_REGULAR_EXPRESSION "CMake Error"
        )
    else()
        set_tests_properties(pose_engine_${name} PROPERTIES PASS_REGULAR_EXPRESSION "CMake Error.*${expected}")
    endif()
    set_tests_properties(pose_engine_${name} PROPERTIES LABELS pose_provider)
endfunction()

# The fixture writes a complete engine library directory to <build>/engine.
set(dir ${CMAKE_CURRENT_BINARY_DIR})

pose_engine_test(no_adapter pass "Selected adapter: stub_adapter \\(redistributable\\)")
pose_engine_test(redistributable pass "Selected adapter: redist_adapter \\(redistributable\\)"
    -DPOSE_ENGINE_REDIST=ON -DREDIST_ROOT=${dir}/redistributable/engine)
pose_engine_test(user_built_only pass "Selected adapter: user_built_adapter \\(user-built-only\\)"
    -DPOSE_ENGINE_USER_BUILT=ON -DUSER_BUILT_ROOT=${dir}/user_built_only/engine)
pose_engine_test(root_not_set fail "REDIST_ROOT is not set"
    -DPOSE_ENGINE_REDIST=ON)
pose_engine_test(relative_root fail "REDIST_ROOT must be an absolute path"
    -DPOSE_ENGINE_REDIST=ON -DREDIST_ROOT:PATH=engine)
pose_engine_test(missing_paths fail "absent_include/.*include/absent_engine\\.h.*lib/libabsent_engine\\.so"
    -DPOSE_ENGINE_PARTIAL=ON -DPARTIAL_ROOT=${dir}/missing_paths/engine)
pose_engine_test(two_adapters fail "More than one Pose Provider engine adapter"
    -DPOSE_ENGINE_REDIST=ON -DREDIST_ROOT=${dir}/two_adapters/engine
    -DPOSE_ENGINE_USER_BUILT=ON -DUSER_BUILT_ROOT=${dir}/two_adapters/engine)
pose_engine_test(plain_and_helper fail "More than one Pose Provider engine adapter"
    -DPOSE_ENGINE_REDIST=ON -DREDIST_ROOT=${dir}/plain_and_helper/engine -DPLAIN_ADAPTER=ON)
pose_engine_test(user_built_in_pip_build fail "is user-built-only"
    -DPOSE_ENGINE_USER_BUILT=ON -DUSER_BUILT_ROOT=${dir}/user_built_in_pip_build/engine -DSKBUILD=ON)
pose_engine_test(plain_user_built_in_pip_build fail "is user-built-only"
    -DPLAIN_ADAPTER=ON -DPLAIN_DISTRIBUTION=user-built-only -DSKBUILD=ON)
pose_engine_test(unknown_distribution fail "DISTRIBUTION must be"
    -DPLAIN_ADAPTER=ON -DPLAIN_DISTRIBUTION=approved)
pose_engine_test(misspelled_keyword fail "unexpected arguments"
    -DMISSPELLED_KEYWORD=ON)
pose_engine_test(late_registration fail "registered after pose_engine_select"
    -DLATE_ADAPTER=ON)
