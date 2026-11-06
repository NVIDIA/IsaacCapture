# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

find_package(OmnigloveSDK CONFIG REQUIRED)

if(NOT TARGET OmnigloveSDK::api)
    message(FATAL_ERROR "OmnigloveSDK package did not define OmnigloveSDK::api")
endif()

get_target_property(OMNIGLOVE_SDK_LIBRARY OmnigloveSDK::api IMPORTED_LOCATION)
if(NOT OMNIGLOVE_SDK_LIBRARY OR NOT EXISTS "${OMNIGLOVE_SDK_LIBRARY}")
    message(FATAL_ERROR "OmnigloveSDK::api has no usable IMPORTED_LOCATION")
endif()

if(NOT OMNIGLOVE_SDK_RELEASE_ROOT)
    get_filename_component(_omniglove_platform_root "${OmnigloveSDK_DIR}/.." ABSOLUTE)
    get_filename_component(_omniglove_release_candidate "${_omniglove_platform_root}/.." ABSOLUTE)
    if(EXISTS "${_omniglove_release_candidate}/LICENSE")
        set(OMNIGLOVE_SDK_RELEASE_ROOT "${_omniglove_release_candidate}")
    endif()
endif()

if(OMNIGLOVE_SDK_RELEASE_ROOT AND
   (NOT OMNIGLOVE_SDK_GIT_COMMIT OR OMNIGLOVE_SDK_GIT_COMMIT STREQUAL "unknown"))
    find_package(Git QUIET)
    if(GIT_FOUND)
        # A release archive inside another repository must not inherit its commit.
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${OMNIGLOVE_SDK_RELEASE_ROOT}" rev-parse --show-toplevel
            OUTPUT_VARIABLE _omniglove_git_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        file(REAL_PATH "${OMNIGLOVE_SDK_RELEASE_ROOT}" _omniglove_release_root)
        if(_omniglove_git_root STREQUAL _omniglove_release_root)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${OMNIGLOVE_SDK_RELEASE_ROOT}" rev-parse HEAD
                RESULT_VARIABLE _omniglove_git_result
                OUTPUT_VARIABLE _omniglove_git_commit
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_omniglove_git_result EQUAL 0)
                set(OMNIGLOVE_SDK_GIT_COMMIT "${_omniglove_git_commit}")
            endif()
        endif()
    endif()
endif()

add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(compat_probe)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

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

install(FILES plugin.yaml README.md VALIDATION.md 60-omniglove-usb.rules
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
install(PROGRAMS install_udev_rules.sh
    DESTINATION plugins/omniglove
    COMPONENT omniglove
)
