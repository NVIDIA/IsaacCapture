# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")

add_subdirectory(core)
add_subdirectory(app)

# Install the verified target artifact; SDK linker aliases may not be in its manifest.
get_target_property(_omniglove_sdk_soname OmnigloveSDK::api IMPORTED_SONAME)
install(FILES "${OMNIGLOVE_SDK_LIBRARY}"
    DESTINATION plugins/omniglove
    RENAME "${_omniglove_sdk_soname}"
    COMPONENT omniglove
)

install(FILES "${OMNIGLOVE_SDK_RELEASE_ROOT}/LICENSE"
    DESTINATION plugins/omniglove
    RENAME LICENSE.omniglove-sdk
    COMPONENT omniglove
)
install(FILES "${OMNIGLOVE_SDK_RELEASE_ROOT}/THIRD_PARTY_NOTICES.md"
    DESTINATION plugins/omniglove
    RENAME THIRD_PARTY_NOTICES.omniglove-sdk.md
    COMPONENT omniglove
)

install(FILES plugin.yaml README.md 60-omniglove-usb.rules
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
install(PROGRAMS install_udev_rules.sh
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
