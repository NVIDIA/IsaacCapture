# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")

add_subdirectory(core)
add_subdirectory(app)

get_filename_component(_omniglove_sdk_library_dir "${OMNIGLOVE_SDK_LIBRARY}" DIRECTORY)
file(GLOB _omniglove_sdk_libraries "${_omniglove_sdk_library_dir}/libomniglove.so*")
file(GLOB _omniglove_libusb_libraries "${_omniglove_sdk_library_dir}/omniglove/libusb-1.0.so*")

if(NOT _omniglove_sdk_libraries)
    message(FATAL_ERROR "No libomniglove.so files found beside ${OMNIGLOVE_SDK_LIBRARY}")
endif()
if(NOT _omniglove_libusb_libraries)
    message(FATAL_ERROR "No bundled libusb runtime found under ${_omniglove_sdk_library_dir}/omniglove")
endif()

install(FILES ${_omniglove_sdk_libraries}
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
install(FILES ${_omniglove_libusb_libraries}
    DESTINATION plugins/omniglove/omniglove
    COMPONENT omniglove
)

if(OMNIGLOVE_SDK_RELEASE_ROOT)
    install(FILES
        "${OMNIGLOVE_SDK_RELEASE_ROOT}/LICENSE"
        DESTINATION plugins/omniglove
        RENAME LICENSE.omniglove-sdk
        COMPONENT omniglove
    )
    install(FILES
        "${OMNIGLOVE_SDK_RELEASE_ROOT}/THIRD_PARTY_NOTICES.md"
        DESTINATION plugins/omniglove
        RENAME THIRD_PARTY_NOTICES.omniglove-sdk.md
        COMPONENT omniglove
    )
    install(FILES
        "${OMNIGLOVE_SDK_RELEASE_ROOT}/licenses/libusb-COPYING"
        DESTINATION plugins/omniglove
        RENAME LICENSE.libusb
        COMPONENT omniglove
    )
else()
    message(WARNING "OMNIGLOVE_SDK_RELEASE_ROOT is unset; SDK license files will not be installed")
endif()

install(FILES plugin.yaml README.md 60-omniglove-usb.rules
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
install(PROGRAMS install_udev_rules.sh
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
