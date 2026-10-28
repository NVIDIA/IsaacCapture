# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping io_exo_glove_ros2 plugin (Linux only)")
    add_custom_target(io_exo_glove_ros2_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping io_exo_glove_ros2: Linux only")
    return()
endif()

# rclcpp / sensor_msgs / tf2_msgs are ordinary ament CMake packages: find_package() locates them once
# the ROS 2 (Humble) environment is sourced (CMAKE_PREFIX_PATH from setup.bash), same as any other
# ROS 2 C++ node -- no colcon/ament build wrapper needed for a plain consumer like this plugin.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)

add_executable(io_exo_glove_ros2_plugin
    main.cpp
    io_exo_glove_ros2_aim.cpp
    io_exo_glove_ros2_config.cpp
    io_exo_glove_ros2_hand.cpp
    io_exo_glove_ros2_kinematics.cpp
    io_exo_glove_ros2_pack.cpp
    io_exo_glove_ros2_plugin.cpp
)

target_link_libraries(io_exo_glove_ros2_plugin PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    # The injected OpenXR hand skeleton, and the wrist pose that places it in the room.
    Teleop::plugin_utils
    Teleop::openxr_extensions
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    oxr::oxr_utils
    isaacteleop_schema
    rclcpp::rclcpp
    # ${<pkg>_TARGETS} rather than each package's <pkg>::<pkg> aggregate target: the aggregate is
    # written into the message package's own CMake config at its build time, so `sensor_msgs::sensor_msgs`
    # only exists from ros-humble-sensor-msgs 4.9.2 (2026-08-07) onwards. These variables work on every
    # Humble.
    ${sensor_msgs_TARGETS}
    ${tf2_msgs_TARGETS}
    yaml-cpp::yaml-cpp
)

# rclcpp initializes ROS's own spdlog-based logger in this process, and this project statically
# links the spdlog it fetches itself. Both define the same symbols, so unless ours are hidden the
# linker binds ROS's undefined references to our copy and the two implementations free each other's
# objects -- rcl_logging_external_initialize() aborts with "free(): invalid pointer" before this
# plugin runs a single line. Hidden visibility plus excluding archive symbols keeps ROS on its own
# spdlog; nothing links against this executable, so hiding its symbols costs nothing.
set_target_properties(io_exo_glove_ros2_plugin PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_options(io_exo_glove_ros2_plugin PRIVATE "LINKER:--exclude-libs,ALL")

install(TARGETS io_exo_glove_ros2_plugin RUNTIME DESTINATION plugins/io_exo_glove_ros2)
install(FILES plugin.yaml README.md DESTINATION plugins/io_exo_glove_ros2)
install(FILES config/io_exo_glove_ros2.yaml DESTINATION plugins/io_exo_glove_ros2/config)
