# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(omniglove_api_compat_probe main.cpp)
target_link_libraries(omniglove_api_compat_probe PRIVATE OmnigloveSDK::api)

set_target_properties(omniglove_api_compat_probe PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:OmnigloveSDK::api>;$<TARGET_FILE_DIR:OmnigloveSDK::api>/omniglove"
)

if(BUILD_TESTING)
    add_test(NAME omniglove_api_compat_probe COMMAND omniglove_api_compat_probe)
endif()
