# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reads a gamepad through SDL3 and pushes GamepadSnapshot, in the standard gamepad layout, via OpenXR.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping gamepad plugin (Linux only)")
    add_custom_target(gamepad_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping gamepad: Linux only")
    return()
endif()

# ------------------------------------------------------------------------------
# SDL3 (zlib): device support and the SDL_GameControllerDB mappings that turn each pad's own
# numbering into the standard layout. Built static with only the joystick and HIDAPI subsystems, so
# the bundled plugin still needs nothing beyond libc/libstdc++; libudev is loaded at runtime.
# ------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        829a65d769d935c4852f8159e964312c0957260a # release-3.4.18
)
foreach(_sdl_option SDL_SHARED SDL_TEST_LIBRARY SDL_TESTS SDL_EXAMPLES SDL_INSTALL
                    SDL_AUDIO SDL_VIDEO SDL_HAPTIC SDL_POWER SDL_SENSOR SDL_DIALOG SDL_TRAY
                    SDL_DBUS SDL_IBUS SDL_HIDAPI_LIBUSB)
    set(${_sdl_option} OFF CACHE BOOL "" FORCE)
endforeach()
# SDL_UNIX_CONSOLE_BUILD: no windows, so no X11 or Wayland development packages are needed.
foreach(_sdl_option SDL_STATIC SDL_JOYSTICK SDL_HIDAPI SDL_UNIX_CONSOLE_BUILD)
    set(${_sdl_option} ON CACHE BOOL "" FORCE)
endforeach()
isaac_teleop_third_party_scope_begin()
FetchContent_MakeAvailable(SDL3)
isaac_teleop_third_party_scope_end()

# app/ also stages the plugin with its manifest into the Python package.
set(GAMEPAD_PLUGIN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml")

add_subdirectory(core)
add_subdirectory(app)

install(FILES plugin.yaml README.md DESTINATION plugins/gamepad)
